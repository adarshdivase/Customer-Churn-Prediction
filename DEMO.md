# RetainAI Enterprise — 60-Second Demo Script

## Start

```powershell
.\scripts\start-demo.ps1
```

Opens **http://127.0.0.1:8501**

---

## 0:00 — Intro (10s)

> "This is **RetainAI** — an enterprise churn intelligence platform. XGBoost scores customers, and every prediction is explainable with feature drivers."

Point to KPI cards: customers, churn rate, data source.

---

## 0:10 — Data Explorer (15s)

Tab **Data Explorer** → show age/geography charts.

> "We explore churn patterns before modeling."

---

## 0:25 — Model Performance (15s)

Tab **Performance** → ROC + **PR-AUC** + calibration curve + optimal threshold.

> "Accurate mode uses engineered features, grid search, and isotonic calibration — not a vanilla classifier."

---

## 0:40 — Live Scoring (20s)

Tab **Live Scoring** → sidebar **High-risk senior** preset → **Score customer**.

Show probability, risk band, importance chart.

> "Product teams get actionable risk tiers, not just a black-box score."

---

## 0:50 — Business ROI (10s)

Tab **Business ROI** → show net benefit and campaign ROI at optimal threshold.

## 0:55 — Batch (5s)

Tab **Batch** → CSV export for CRM campaigns.

---

## Demo profiles to try

| Profile | Expected |
|---------|----------|
| High-risk senior | Higher churn probability |
| Loyal active member | Lower probability |
| New low-balance | Medium risk |

---

## One-liner close

> "Stack: Python, Streamlit, XGBoost, Plotly — modular `churnai` package with cached models for instant demos."
