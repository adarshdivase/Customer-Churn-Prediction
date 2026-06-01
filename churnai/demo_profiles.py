"""Preset customer profiles for live demo."""

DEMO_PROFILES = {
    "High-risk senior": {
        "CreditScore": 520,
        "Geography": "Germany",
        "Gender": "Female",
        "Age": 62,
        "Tenure": 1,
        "Balance": 120000.0,
        "NumOfProducts": 1,
        "HasCrCard": 0,
        "IsActiveMember": 0,
        "EstimatedSalary": 45000.0,
    },
    "Loyal active member": {
        "CreditScore": 780,
        "Geography": "France",
        "Gender": "Male",
        "Age": 34,
        "Tenure": 8,
        "Balance": 95000.0,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 110000.0,
    },
    "New low-balance": {
        "CreditScore": 610,
        "Geography": "Spain",
        "Gender": "Female",
        "Age": 28,
        "Tenure": 0,
        "Balance": 0.0,
        "NumOfProducts": 1,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 52000.0,
    },
    "Multi-product VIP": {
        "CreditScore": 710,
        "Geography": "France",
        "Gender": "Male",
        "Age": 45,
        "Tenure": 6,
        "Balance": 180000.0,
        "NumOfProducts": 3,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 145000.0,
    },
}

DEMO_QUESTIONS = [
    "Explore churn rate by geography",
    "Review model AUC and confusion matrix",
    "Score the high-risk senior profile",
    "Run batch export on sample customers",
]
