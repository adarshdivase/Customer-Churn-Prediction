"""Compare baseline vs RetainAI accurate pipeline on the default dataset."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from churnai.data import load_dataset
from churnai.features import engineer_features
from churnai.model import train_pipeline


def baseline_pipeline(df: pd.DataFrame) -> dict:
    """Original-style model: raw features only, small grid, no calibration."""
    X = df.drop("Exited", axis=1)
    y = df["Exited"]
    cat = X.select_dtypes(include=["object"]).columns.tolist()
    num = X.select_dtypes(include=np.number).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    pipe = Pipeline([
        ("preprocessor", ColumnTransformer([
            ("num", StandardScaler(), num),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
        ])),
        ("classifier", XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, eval_metric="logloss", random_state=42)),
    ])
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
        "f1": f1_score(y_test, pred, zero_division=0),
        "features": len(X.columns),
    }


def accurate_pipeline(df: pd.DataFrame) -> dict:
    artifact = train_pipeline(df, use_cache=False, mode="accurate", compare_models=False, calibrate=True)
    if not artifact:
        raise RuntimeError("Accurate training failed")
    pipe = artifact["pipeline"]
    y_test = artifact["y_test"]
    proba = pipe.predict_proba(artifact["X_test"])[:, 1]
    thresh = artifact.get("threshold", 0.5)
    pred = (proba >= thresh).astype(int)
    meta = artifact.get("meta", {})
    return {
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
        "f1": f1_score(y_test, pred, zero_division=0),
        "features": meta.get("feature_count", 0),
        "cv_roc_mean": meta.get("cv_scores", {}).get("roc_auc_mean"),
        "cv_roc_std": meta.get("cv_scores", {}).get("roc_auc_std"),
        "threshold": thresh,
        "algorithm": meta.get("algorithm"),
        "calibrated": meta.get("calibrated"),
    }


def main():
    df, source = load_dataset()
    print(f"Dataset: {source} ({len(df)} rows, churn rate {df['Exited'].mean():.1%})\n")

    raw_df = pd.read_csv(ROOT / "customer_churn_with_added_features.csv")

    print("Training BASELINE (raw features, basic XGBoost)...")
    base = baseline_pipeline(raw_df)

    print("Training ACCURATE (engineered features + grid + calibration)...")
    acc = accurate_pipeline(raw_df)

    print("\n" + "=" * 55)
    print(f"{'Metric':<22} {'Baseline':>12} {'Accurate':>12} {'Delta':>8}")
    print("=" * 55)
    for key in ("roc_auc", "pr_auc", "f1"):
        b, a = base[key], acc[key]
        delta = a - b
        sign = "+" if delta >= 0 else ""
        print(f"{key.upper():<22} {b:>12.4f} {a:>12.4f} {sign}{delta:>7.4f}")
    print("=" * 55)
    print(f"Features: baseline={base['features']}, accurate={acc['features']}")
    print(f"CV ROC (accurate): {acc.get('cv_roc_mean', 0):.4f} ± {acc.get('cv_roc_std', 0):.4f}")
    print(f"Optimal threshold: {acc.get('threshold', 0.5):.3f}")

    out = {"baseline": base, "accurate": acc, "dataset_rows": len(raw_df)}
    out_path = ROOT / "models" / "comparison_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
