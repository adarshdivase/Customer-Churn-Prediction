"""
RetainAI Enterprise — customer churn intelligence platform.
Run: streamlit run app.py
"""

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from churnai.config import APP_NAME
from churnai.data import load_dataset
from churnai.demo_profiles import DEMO_PROFILES
from churnai.evaluation import business_impact, find_optimal_threshold, segment_churn_table
from churnai.features import ENGINEERED_FEATURES, engineer_features
from churnai.metrics import (
    calibration_fig,
    confusion_fig,
    customer_importance_chart,
    explorer_charts,
    feature_importance,
    performance_metrics,
    pr_curve_fig,
    radar_chart,
    roc_fig,
    threshold_analysis_fig,
)
from churnai.model import METADATA_PATH, risk_label, train_pipeline
from churnai.predict import batch_predict, predict_customer
from churnai.theme import hero_html, inject_theme, style_fig

st.set_page_config(
    page_title=f"{APP_NAME} | Churn Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_theme()
st.markdown(
    hero_html(
        APP_NAME,
        "Engineered features · Calibrated XGBoost · Optimal thresholds · Business ROI",
        "🎯",
    ),
    unsafe_allow_html=True,
)

for key, default in {"artifact": None, "demo_profile": None, "upload_id": None}.items():
    if key not in st.session_state:
        st.session_state[key] = default

with st.sidebar:
    st.markdown("### ⚙️ Control Center")
    train_mode = st.radio("Training mode", ["fast", "accurate"], format_func=lambda x: "⚡ Fast (demo)" if x == "fast" else "🎯 Accurate (best metrics)")
    compare_models = st.checkbox("Compare algorithms (accurate only)", value=False)
    uploaded = st.file_uploader("Upload training CSV", type=["csv"])

    if st.button("🔄 Retrain model", use_container_width=True):
        st.session_state.artifact = None
        st.cache_resource.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("### 🎬 Demo profiles")
    profile_name = st.selectbox("Load preset", ["—"] + list(DEMO_PROFILES.keys()))
    if profile_name != "—":
        st.session_state.demo_profile = DEMO_PROFILES[profile_name]

    st.markdown("---")
    st.caption("7 engineered features · Isotonic calibration in accurate mode")

mode_key = (train_mode, compare_models)
if st.session_state.get("mode_key") != mode_key:
    st.session_state.mode_key = mode_key
    st.session_state.artifact = None

if uploaded is not None:
    uid = f"{uploaded.name}_{uploaded.size}"
    if st.session_state.upload_id != uid:
        st.session_state.upload_id = uid
        st.session_state.artifact = None

df, source = load_dataset(uploaded)
threshold = 0.5

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Customers", f"{len(df):,}")
c2.metric("Churned", f"{int(df['Exited'].sum()):,}")
c3.metric("Churn rate", f"{df['Exited'].mean():.1%}")
c4.metric("Regions", df["Geography"].nunique())
c5.metric("Features", len([c for c in ENGINEERED_FEATURES if c in df.columns]) + 10)
c6.metric("Source", source[:14])

@st.cache_resource(show_spinner="Training model…")
def _train(df_: pd.DataFrame, mode: str, compare: bool):
    return train_pipeline(df_, use_cache=True, mode=mode, compare_models=compare, calibrate=(mode == "accurate"))

if st.session_state.artifact is None:
    st.session_state.artifact = _train(df, train_mode, compare_models and train_mode == "accurate")

artifact = st.session_state.artifact
if not artifact:
    st.error("Training failed — need 50+ rows and both classes in `Exited`.")
    st.stop()

pipe = artifact["pipeline"]
meta = artifact.get("meta", {})
threshold = st.session_state.get("custom_threshold", artifact.get("threshold", meta.get("optimal_threshold", 0.5)))

with st.sidebar:
    if artifact.get("from_cache"):
        st.success("✅ Cached model loaded")
    if meta:
        st.markdown("### 📊 Model quality")
        st.write(f"**Algorithm:** {meta.get('algorithm', 'XGBoost')}")
        st.write(f"**ROC-AUC:** {meta.get('test_roc_auc', 0):.3f}")
        st.write(f"**PR-AUC:** {meta.get('test_pr_auc', 0):.3f}")
        cv = meta.get("cv_scores", {})
        if cv:
            st.write(f"**CV ROC:** {cv.get('roc_auc_mean', 0):.3f} ± {cv.get('roc_auc_std', 0):.3f}")

tabs = st.tabs([
    "🔍 Explorer",
    "📈 Performance",
    "🎯 Live Score",
    "📊 Batch",
    "💼 Business ROI",
])

with tabs[0]:
    charts = explorer_charts(df)
    a, b = st.columns(2)
    with a:
        st.plotly_chart(charts["age"], use_container_width=True)
        st.plotly_chart(charts["geography"], use_container_width=True)
    with b:
        st.plotly_chart(charts["balance"], use_container_width=True)
        st.plotly_chart(charts["correlation"], use_container_width=True)

    st.markdown("#### Churn by segment")
    seg = segment_churn_table(df, "Geography")
    st.dataframe(seg, use_container_width=True, hide_index=True)

    with st.expander("Engineered features preview"):
        st.dataframe(df[ENGINEERED_FEATURES].describe(), use_container_width=True)

with tabs[1]:
    y_test = artifact["y_test"]
    y_proba = pipe.predict_proba(artifact["X_test"])[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    metrics = performance_metrics(y_test, y_pred, y_proba)

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    for col, (k, v) in zip([m1, m2, m3, m4, m5, m6], metrics.items()):
        col.metric(k, f"{v:.3f}")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(radar_chart(metrics), use_container_width=True)
        st.plotly_chart(pr_curve_fig(y_test, y_proba), use_container_width=True)
        st.plotly_chart(confusion_fig(y_test, y_pred), use_container_width=True)
    with col2:
        st.plotly_chart(roc_fig(y_test, y_proba), use_container_width=True)
        st.plotly_chart(calibration_fig(y_test, y_proba), use_container_width=True)
        st.plotly_chart(threshold_analysis_fig(y_test, y_proba), use_container_width=True)

    opt = find_optimal_threshold(y_test, y_proba)
    st.info(f"Recommended threshold: **{opt['threshold']:.2f}** (F1={opt['f1']:.3f}, Precision={opt['precision']:.3f}, Recall={opt['recall']:.3f})")

    fig_bi, fig_perm, summary = feature_importance(pipe, artifact["X_test"], y_test)
    if fig_bi:
        st.plotly_chart(fig_bi, use_container_width=True)
    if fig_perm:
        st.plotly_chart(fig_perm, use_container_width=True)

    if METADATA_PATH.exists():
        with st.expander("Model metadata (JSON)"):
            st.json(json.loads(METADATA_PATH.read_text(encoding="utf-8")))

with tabs[2]:
    st.markdown("Score a customer using the **optimal threshold** and global feature drivers.")
    threshold = st.slider("Decision threshold", 0.05, 0.95, float(artifact.get("threshold", 0.5)), 0.01)
    st.session_state.custom_threshold = threshold

    defaults = st.session_state.get("demo_profile") or DEMO_PROFILES["Loyal active member"]
    col1, col2 = st.columns(2)
    with col1:
        credit = st.slider("Credit Score", 300, 850, int(defaults["CreditScore"]))
        balance = st.number_input("Balance ($)", 0.0, 300000.0, float(defaults["Balance"]))
        products = st.slider("Products", 1, 4, int(defaults["NumOfProducts"]))
        has_card = st.selectbox("Credit card", [0, 1], index=int(defaults["HasCrCard"]), format_func=lambda x: "Yes" if x else "No")
        active = st.selectbox("Active member", [0, 1], index=int(defaults["IsActiveMember"]), format_func=lambda x: "Yes" if x else "No")
        salary = st.number_input("Salary ($)", 0.0, 250000.0, float(defaults["EstimatedSalary"]))
    with col2:
        age = st.slider("Age", 18, 80, int(defaults["Age"]))
        tenure = st.slider("Tenure", 0, 10, int(defaults["Tenure"]))
        geo = st.selectbox("Geography", ["France", "Spain", "Germany"], index=["France", "Spain", "Germany"].index(defaults["Geography"]))
        gender = st.selectbox("Gender", ["Male", "Female"], index=0 if defaults["Gender"] == "Male" else 1)

    if st.button("🔮 Score customer", type="primary", use_container_width=True):
        row = pd.DataFrame([{
            "CreditScore": credit, "Balance": balance, "NumOfProducts": products,
            "HasCrCard": has_card, "IsActiveMember": active, "EstimatedSalary": salary,
            "Age": age, "Tenure": tenure, "Geography": geo, "Gender": gender,
        }])
        proba, pred, analysis = predict_customer(pipe, row, artifact["X_train_cols"], threshold)
        risk = risk_label(proba, threshold)
        css = "risk-high" if "High" in risk else "risk-medium" if "Medium" in risk else "risk-low"
        st.markdown(f"""
        <div class="{css}">
          <h3>Churn probability: {proba:.1%}</h3>
          <p><strong>{risk}</strong> · Predicted: {"Churn" if pred else "Retain"} (threshold {threshold:.2f})</p>
        </div>
        """, unsafe_allow_html=True)
        if analysis is not None:
            st.plotly_chart(customer_importance_chart(analysis), use_container_width=True)

with tabs[3]:
    batch_file = st.file_uploader("Batch CSV", type=["csv"], key="batch")
    sample = pd.DataFrame([DEMO_PROFILES["High-risk senior"], DEMO_PROFILES["Loyal active member"]])
    st.download_button("📥 Sample batch CSV", sample.to_csv(index=False).encode(), "sample_batch.csv")

    if batch_file:
        batch = pd.read_csv(batch_file)
        if st.button("🚀 Score batch", type="primary"):
            out, proba = batch_predict(pipe, batch, artifact["X_train_cols"], threshold)
            out["Risk_Level"] = [risk_label(p, threshold) for p in proba]
            st.dataframe(out, use_container_width=True)
            st.download_button("📥 Export", out.to_csv(index=False).encode(), "churn_scores.csv", use_container_width=True)
            counts = out["Risk_Level"].value_counts()
            st.plotly_chart(px.pie(names=counts.index, values=counts.values, title="Risk mix"), use_container_width=True)

with tabs[4]:
    st.markdown("### Retention campaign ROI (test-set simulation)")
    y_test = artifact["y_test"]
    y_proba = pipe.predict_proba(artifact["X_test"])[:, 1]

    rev = st.number_input("Avg annual revenue / customer ($)", 500, 5000, 1200)
    cost = st.number_input("Retention cost / targeted customer ($)", 10, 500, 80)
    success = st.slider("Retention success rate", 0.05, 0.8, 0.35)

    impact = business_impact(y_test, y_proba, threshold, avg_revenue_per_customer=rev, retention_cost_per_customer=cost, retention_success_rate=success)

    i1, i2, i3, i4 = st.columns(4)
    i1.metric("Targeted", impact["customers_targeted"])
    i2.metric("Churners caught", impact["true_churners_caught"])
    i3.metric("Net benefit", f"${impact['net_benefit']:,.0f}")
    i4.metric("Campaign ROI", f"{impact['roi_pct']:.0f}%")

    st.markdown("""
    **How to read this:** Among customers flagged at the current threshold on the holdout set,
    we estimate how many true churners you'd save vs. campaign spend.
    """)

st.caption(f"{APP_NAME} · Engineered features + calibrated probabilities")
