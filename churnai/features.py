"""Automated feature engineering for stronger churn signals."""

import numpy as np
import pandas as pd

BASE_FEATURES = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure",
    "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary",
]

ENGINEERED_FEATURES = [
    "BalancePerProduct",
    "SalaryPerAge",
    "TenureRatio",
    "IsZeroBalance",
    "IsSenior",
    "LowEngagement",
    "CreditTenureScore",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features used by the model (idempotent on already-engineered data)."""
    out = df.copy()

    if "BalancePerProduct" in out.columns:
        return out

    out["BalancePerProduct"] = out["Balance"] / out["NumOfProducts"].clip(lower=1)
    out["SalaryPerAge"] = out["EstimatedSalary"] / out["Age"].clip(lower=1)
    out["TenureRatio"] = out["Tenure"] / out["Age"].clip(lower=1)
    out["IsZeroBalance"] = (out["Balance"] <= 0).astype(int)
    out["IsSenior"] = (out["Age"] >= 55).astype(int)
    out["LowEngagement"] = ((out["IsActiveMember"] == 0) | (out["NumOfProducts"] <= 1)).astype(int)
    out["CreditTenureScore"] = (out["CreditScore"] / 850.0) * (out["Tenure"] + 1)

    return out


def feature_columns() -> list[str]:
    return BASE_FEATURES + ENGINEERED_FEATURES


def align_customer_row(row: pd.DataFrame, train_columns: list[str]) -> pd.DataFrame:
    """Align a single customer row and apply engineering."""
    aligned = row.reindex(columns=[c for c in BASE_FEATURES if c in row.columns], fill_value=0)
    full = engineer_features(aligned)
    return full.reindex(columns=train_columns, fill_value=0).fillna(0)
