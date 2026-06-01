from pathlib import Path

APP_NAME = "RetainAI Enterprise"
DEFAULT_DATA = "customer_churn_with_added_features.csv"

FEATURE_COLUMNS = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure",
    "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary",
]

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
PIPELINE_PATH = MODEL_DIR / "xgb_pipeline.joblib"

RISK_THRESHOLDS = {"high": 0.7, "medium": 0.3}
