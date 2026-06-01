import os
from pathlib import Path

import numpy as np
import pandas as pd

from churnai.config import DEFAULT_DATA
from churnai.features import BASE_FEATURES, engineer_features


def create_sample_data(n: int = 2000) -> pd.DataFrame:
    np.random.seed(42)
    data = {
        "CreditScore": np.random.randint(300, 850, n),
        "Geography": np.random.choice(["France", "Spain", "Germany"], n),
        "Gender": np.random.choice(["Male", "Female"], n),
        "Age": np.random.randint(18, 80, n),
        "Tenure": np.random.randint(0, 11, n),
        "Balance": np.random.uniform(0, 250000, n).round(2),
        "NumOfProducts": np.random.randint(1, 5, n),
        "HasCrCard": np.random.randint(0, 2, n),
        "IsActiveMember": np.random.randint(0, 2, n),
        "EstimatedSalary": np.random.uniform(10000, 200000, n).round(2),
    }
    df = pd.DataFrame(data)
    prob = np.clip(
        0.2
        + (df["Age"] > 45) * 0.15
        + (df["Balance"] == 0) * 0.2
        + (df["CreditScore"] < 600) * 0.15
        + (df["Tenure"] <= 1) * 0.1
        + (df["IsActiveMember"] == 0) * 0.1,
        0.05,
        0.85,
    )
    df["Exited"] = np.random.binomial(1, prob)
    return df


def load_dataset(uploaded_file=None, default_path: str = DEFAULT_DATA) -> tuple[pd.DataFrame, str]:
    """Load CSV from upload, default path, or synthetic fallback."""
    source = "sample"
    df = None

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            source = getattr(uploaded_file, "name", "upload")
        except Exception:
            df = None

    if df is None and os.path.exists(default_path):
        df = pd.read_csv(default_path)
        source = default_path

    if df is None:
        df = create_sample_data()
        source = "generated"

    if not _validate(df):
        df = create_sample_data()
        source = "generated (invalid input)"

    return engineer_features(df), source


def _validate(df: pd.DataFrame) -> bool:
    if "Exited" not in df.columns:
        return False
    return all(c in df.columns for c in BASE_FEATURES)
