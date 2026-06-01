import numpy as np
import pandas as pd

from churnai.features import align_customer_row, engineer_features


def _inner_pipeline(model):
    if hasattr(model, "calibrated_classifiers_"):
        return model.calibrated_classifiers_[0].estimator
    return model


def predict_customer(pipeline, customer_row: pd.DataFrame, train_columns: list[str], threshold: float = 0.5):
    aligned = align_customer_row(customer_row, train_columns)
    proba = float(pipeline.predict_proba(aligned)[0, 1])
    pred = int(proba >= threshold)

    inner = _inner_pipeline(pipeline)
    preprocessor = inner.named_steps["preprocessor"]
    classifier = inner.named_steps["classifier"]
    names = preprocessor.get_feature_names_out()

    if hasattr(classifier, "feature_importances_"):
        imp = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        imp = np.abs(classifier.coef_).ravel()
        if len(imp) != len(names):
            return proba, pred, None
    else:
        return proba, pred, None

    analysis = pd.DataFrame({"feature": names, "importance": imp})
    analysis["cleaned_feature"] = analysis["feature"].apply(_clean_name)
    analysis = analysis.sort_values("importance", ascending=False).reset_index(drop=True)
    return proba, pred, analysis


def batch_predict(pipeline, df: pd.DataFrame, train_columns: list[str], threshold: float) -> tuple[pd.DataFrame, np.ndarray]:
    work = df.copy()
    if "Exited" in work.columns:
        work = work.drop(columns=["Exited"])
    for col in train_columns:
        if col not in work.columns:
            work[col] = 0
    work = engineer_features(work)
    aligned = work.reindex(columns=train_columns, fill_value=0).fillna(0)
    proba = pipeline.predict_proba(aligned)[:, 1]
    out = df.copy()
    out["Churn_Probability"] = proba
    out["Churn_Prediction"] = (proba >= threshold).astype(int)
    return out, proba


def _clean_name(name: str) -> str:
    if name.startswith("num__"):
        return name.replace("num__", "")
    if name.startswith("cat__"):
        parts = name.replace("cat__", "").split("_")
        return f"{parts[0].title()}: {parts[1].title()}" if len(parts) > 1 else name.replace("cat__", "").title()
    return name.replace("_", " ").title()
