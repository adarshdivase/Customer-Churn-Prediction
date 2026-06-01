"""Model evaluation, threshold tuning, and business metrics."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score


def cross_validate_model(pipeline, X, y, cv: int = 5) -> dict:
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    roc = cross_val_score(pipeline, X, y, cv=skf, scoring="roc_auc", n_jobs=1)
    f1 = cross_val_score(pipeline, X, y, cv=skf, scoring="f1", n_jobs=1)
    return {
        "roc_auc_mean": float(roc.mean()),
        "roc_auc_std": float(roc.std()),
        "f1_mean": float(f1.mean()),
        "f1_std": float(f1.std()),
        "folds": cv,
    }


def find_optimal_threshold(y_true, y_proba, strategy: str = "f1") -> dict:
    prec, rec, thresholds = precision_recall_curve(y_true, y_proba)
    thresholds = np.append(thresholds, 1.0)

    if strategy == "balanced":
        # Weight recall higher for churn (minority class retention)
        scores = 0.4 * prec + 0.6 * rec
    else:
        scores = 2 * prec * rec / (prec + rec + 1e-9)

    best_idx = int(np.argmax(scores[:-1]))
    return {
        "threshold": float(thresholds[best_idx]),
        "precision": float(prec[best_idx]),
        "recall": float(rec[best_idx]),
        "f1": float(f1_score(y_true, (y_proba >= thresholds[best_idx]).astype(int), zero_division=0)),
    }


def apply_threshold(y_proba: np.ndarray, threshold: float) -> np.ndarray:
    return (y_proba >= threshold).astype(int)


def business_impact(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    *,
    avg_revenue_per_customer: float = 1200.0,
    retention_cost_per_customer: float = 80.0,
    retention_success_rate: float = 0.35,
) -> dict:
    """Estimate savings from targeting predicted churners."""
    pred_churn = y_proba >= threshold
    true_churn = y_true == 1

    tp = int(np.sum(pred_churn & true_churn))
    fp = int(np.sum(pred_churn & ~true_churn))
    fn = int(np.sum(~pred_churn & true_churn))

    targeted = int(pred_churn.sum())
    campaign_cost = targeted * retention_cost_per_customer
    saved_customers = tp * retention_success_rate
    revenue_saved = saved_customers * avg_revenue_per_customer
    net_benefit = revenue_saved - campaign_cost

    return {
        "customers_targeted": targeted,
        "true_churners_caught": tp,
        "missed_churners": fn,
        "false_alarms": fp,
        "campaign_cost": campaign_cost,
        "revenue_saved": revenue_saved,
        "net_benefit": net_benefit,
        "roi_pct": (net_benefit / campaign_cost * 100) if campaign_cost else 0.0,
    }


def segment_churn_table(df: pd.DataFrame, group_col: str = "Geography") -> pd.DataFrame:
    if group_col not in df.columns or "Exited" not in df.columns:
        return pd.DataFrame()
    return (
        df.groupby(group_col)
        .agg(customers=("Exited", "count"), churn_rate=("Exited", "mean"))
        .reset_index()
        .sort_values("churn_rate", ascending=False)
    )
