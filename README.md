# RetainAI Enterprise — Customer Churn Intelligence

Enterprise-grade **customer churn prediction** with XGBoost, explainable scoring, batch export, and an interactive Streamlit command center.

![Demo preview](docs/demo-preview.svg)

## Features

- **7 engineered features** (balance ratios, engagement flags, credit-tenure score)
- **Fast vs Accurate training** — cached quick demos or full grid search + calibration
- **Algorithm comparison** — XGBoost vs Random Forest vs Logistic (accurate mode)
- **Optimal threshold** — tuned for precision/recall balance, not default 0.5
- **PR-AUC, calibration curves**, cross-validated ROC scores
- **Business ROI tab** — retention campaign net benefit estimator
- **5 workspaces**: Explorer, Performance, Live Score, Batch, Business ROI

## One-command demo

```powershell
.\scripts\start-demo.ps1
```

macOS/Linux: `chmod +x scripts/start-demo.sh && ./scripts/start-demo.sh`

Presenter script: **[DEMO.md](DEMO.md)**

## Manual setup

```bash
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate
pip install -r requirements.txt
python scripts/train_model.py
streamlit run app.py
```

## Project layout

```
ChurnAI/
├── app.py                 # Streamlit UI
├── churnai/               # Core package
│   ├── config.py
│   ├── data.py
│   ├── model.py
│   ├── metrics.py
│   ├── predict.py
│   └── demo_profiles.py
├── models/                # Cached pipeline (gitignored)
├── scripts/
│   ├── train_model.py
│   └── start-demo.ps1
├── DEMO.md
└── customer_churn_with_added_features.csv
```

## Docker

```bash
docker build -t retainai .
docker run -p 8501:8501 retainai
```

## Data columns

`CreditScore`, `Geography`, `Gender`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`, `Exited` (training only)
