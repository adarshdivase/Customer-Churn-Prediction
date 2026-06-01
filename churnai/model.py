"""Training pipelines: fast cached mode and accurate tuned mode."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

from churnai.config import MODEL_DIR, PIPELINE_PATH
from churnai.evaluation import cross_validate_model, find_optimal_threshold
from churnai.features import engineer_features, feature_columns

METADATA_PATH = MODEL_DIR / "model_metadata.json"

FAST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.08,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
}


def risk_label(probability: float, threshold: float = 0.5) -> str:
    if probability >= max(threshold, 0.65):
        return "High Risk"
    if probability >= max(threshold * 0.6, 0.35):
        return "Medium Risk"
    return "Low Risk"


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num = X.select_dtypes(include=np.number).columns.tolist()
    return ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
    ])


def _scale_pos_weight(y: pd.Series) -> float:
    classes = np.array(sorted(y.unique()))
    if 0 in classes and 1 in classes:
        weights = compute_class_weight("balanced", classes=classes, y=y)
        return float(weights[1] / weights[0])
    return 1.0


def _make_xgb(spw: float, **kwargs) -> XGBClassifier:
    return XGBClassifier(
        eval_metric="logloss",
        random_state=42,
        scale_pos_weight=spw,
        **kwargs,
    )


def _compare_algorithms(X_train, y_train, preprocessor) -> tuple[str, Pipeline]:
    """Quick comparison; returns best algorithm name and unfitted pipeline template."""
    spw = _scale_pos_weight(y_train)
    candidates = {
        "XGBoost": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", _make_xgb(spw, n_estimators=150, max_depth=4, learning_rate=0.1)),
        ]),
        "RandomForest": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=200, max_depth=8, class_weight="balanced", random_state=42, n_jobs=1,
            )),
        ]),
        "LogisticRegression": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ]),
    }
    best_name, best_pipe, best_auc = "XGBoost", candidates["XGBoost"], -1.0
    for name, pipe in candidates.items():
        scores = cross_validate_model(pipe, X_train, y_train, cv=3)
        if scores["roc_auc_mean"] > best_auc:
            best_auc = scores["roc_auc_mean"]
            best_name = name
            best_pipe = pipe
    return best_name, best_pipe


def train_pipeline(
    df: pd.DataFrame,
    *,
    use_cache: bool = True,
    mode: str = "fast",
    compare_models: bool = False,
    calibrate: bool = True,
) -> dict | None:
    df = engineer_features(df)

    if use_cache and PIPELINE_PATH.exists() and METADATA_PATH.exists():
        try:
            pipeline = joblib.load(PIPELINE_PATH)
            meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
            if meta.get("mode") == mode or mode == "fast":
                return _pack(df, pipeline, meta, from_cache=True)
        except Exception:
            pass

    if df is None or "Exited" not in df.columns or len(df) < 50:
        return None
    if len(df["Exited"].unique()) < 2:
        return None

    cols = [c for c in feature_columns() if c in df.columns]
    X = df[cols]
    y = df["Exited"]

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = _build_preprocessor(X_train)
    spw = _scale_pos_weight(y_train)

    if compare_models and mode == "accurate":
        algo_name, base_pipe = _compare_algorithms(X_train, y_train, preprocessor)
    else:
        algo_name = "XGBoost"
        base_pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", _make_xgb(spw, **FAST_PARAMS)),
        ])

    if mode == "accurate" and algo_name == "XGBoost":
        base_pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", _make_xgb(spw)),
        ])
        grid = GridSearchCV(
            base_pipe,
            {
                "classifier__n_estimators": [150, 250, 350],
                "classifier__max_depth": [3, 5, 7],
                "classifier__learning_rate": [0.05, 0.1, 0.15],
                "classifier__subsample": [0.8, 1.0],
            },
            cv=3,
            scoring="roc_auc",
            n_jobs=1,
        )
        grid.fit(X_train, y_train)
        fitted = grid.best_estimator_
        best_params = grid.best_params_
    else:
        fitted = base_pipe
        fitted.fit(X_train, y_train)
        best_params = FAST_PARAMS

    if calibrate and mode == "accurate":
        calibrated = CalibratedClassifierCV(fitted, method="isotonic", cv=3)
        calibrated.fit(X_train, y_train)
        model = calibrated
    else:
        model = fitted

    y_proba = model.predict_proba(X_test)[:, 1]
    thresh_info = find_optimal_threshold(y_test, y_proba, strategy="balanced")
    cv_scores = cross_validate_model(fitted, X_train, y_train, cv=5 if mode == "accurate" else 3)

    meta = {
        "mode": mode,
        "algorithm": algo_name,
        "calibrated": calibrate and mode == "accurate",
        "best_params": best_params,
        "optimal_threshold": thresh_info["threshold"],
        "threshold_metrics": thresh_info,
        "cv_scores": cv_scores,
        "test_roc_auc": float(roc_auc_score(y_test, y_proba)),
        "test_pr_auc": float(average_precision_score(y_test, y_proba)),
        "feature_count": len(cols),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, PIPELINE_PATH)
    METADATA_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return _pack(df, model, meta, from_cache=False, X_test=X_test, y_test=y_test, X_train=X_train)


def _pack(df, pipeline, meta: dict, from_cache: bool, X_test=None, y_test=None, X_train=None) -> dict:
    if from_cache:
        df = engineer_features(df)
        cols = [c for c in feature_columns() if c in df.columns]
        X = df[cols]
        y = df["Exited"]
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return {
        "pipeline": pipeline,
        "X_test": X_test,
        "y_test": y_test,
        "X_train": X_train,
        "X_train_cols": X_train.columns.tolist(),
        "from_cache": from_cache,
        "meta": meta,
        "threshold": meta.get("optimal_threshold", 0.5),
    }
