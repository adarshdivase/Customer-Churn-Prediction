import logging

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.inspection import permutation_importance
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from churnai.predict import _clean_name

logger = logging.getLogger(__name__)


def performance_metrics(y_test, y_pred, y_proba):
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
        "PR-AUC": average_precision_score(y_test, y_proba),
    }


def pr_curve_fig(y_test, y_proba) -> go.Figure:
    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name=f"PR-AUC = {ap:.3f}", line=dict(color="#8b5cf6", width=3)))
    fig.update_layout(title="Precision-Recall Curve", xaxis_title="Recall", yaxis_title="Precision", height=400)
    return fig


def calibration_fig(y_test, y_proba) -> go.Figure:
    prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=8, strategy="quantile")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prob_pred, y=prob_true, mode="lines+markers", name="Model"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect", line=dict(dash="dash", color="#94a3b8")))
    fig.update_layout(title="Calibration Curve", xaxis_title="Predicted", yaxis_title="Observed", height=400)
    return fig


def threshold_analysis_fig(y_test, y_proba) -> go.Figure:
    prec, rec, thresholds = precision_recall_curve(y_test, y_proba)
    f1s = 2 * prec * rec / (prec + rec + 1e-9)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=thresholds, y=f1s[:-1], mode="lines", name="F1", line=dict(color="#6366f1")))
    fig.update_layout(title="Threshold vs F1", xaxis_title="Threshold", yaxis_title="F1", height=350)
    return fig


def radar_chart(metrics: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=list(metrics.values()), theta=list(metrics.keys()),
        fill="toself", name="Model", line=dict(color="#6366f1", width=3),
        fillcolor="rgba(99, 102, 241, 0.25)",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Model Performance Metrics", height=400,
    )
    return fig


def confusion_fig(y_test, y_pred) -> go.Figure:
    cm = confusion_matrix(y_test, y_pred)
    fig = go.Figure(data=go.Heatmap(
        z=cm, x=["Pred: Stay", "Pred: Churn"], y=["Actual: Stay", "Actual: Churn"],
        colorscale="Blues", text=cm, texttemplate="%{text}", textfont={"size": 18, "color": "white"},
    ))
    fig.update_layout(title="Confusion Matrix", height=400)
    return fig


def roc_fig(y_test, y_proba) -> go.Figure:
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC = {auc:.3f}", line=dict(color="#6366f1", width=3)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash", color="#94a3b8")))
    fig.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR", height=400)
    return fig


def explorer_charts(df: pd.DataFrame) -> dict:
    charts = {}
    charts["age"] = px.histogram(df, x="Age", color="Exited", nbins=20, title="Age by Churn",
                                 color_discrete_map={0: "#38bdf8", 1: "#f87171"})
    charts["balance"] = px.box(df, x="Exited", y="Balance", color="Exited", title="Balance by Churn",
                               color_discrete_map={0: "#38bdf8", 1: "#f87171"})
    geo = df.groupby(["Geography", "Exited"]).size().reset_index(name="count")
    charts["geography"] = px.bar(geo, x="Geography", y="count", color="Exited", barmode="group",
                                   title="Churn by Geography", color_discrete_map={0: "#38bdf8", 1: "#f87171"})
    num = df.select_dtypes(include=np.number)
    corr = num.corr()
    charts["correlation"] = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale="RdBu", zmid=0))
    charts["correlation"].update_layout(title="Feature Correlations", height=450)
    return charts


def _unwrap(pipeline):
    if hasattr(pipeline, "calibrated_classifiers_"):
        return pipeline.calibrated_classifiers_[0].estimator
    return pipeline


def feature_importance(pipeline, X_test, y_test):
    try:
        inner = _unwrap(pipeline)
        pre = inner.named_steps["preprocessor"]
        clf = inner.named_steps["classifier"]
        names = pre.get_feature_names_out()
        imp = clf.feature_importances_
        built = pd.DataFrame({"feature": names, "importance": imp})
        built["cleaned_feature"] = built["feature"].apply(_clean_name)
        built = built.sort_values("importance", ascending=True)

        fig_bi = go.Figure(go.Bar(
            x=built["importance"], y=built["cleaned_feature"], orientation="h",
            marker=dict(color=built["importance"], colorscale="Viridis"),
        ))
        fig_bi.update_layout(title="XGBoost Feature Importance", height=max(500, len(built) * 28))

        perm = permutation_importance(inner, X_test, y_test, n_repeats=6, random_state=42, scoring="roc_auc", n_jobs=1)
        perm_df = pd.DataFrame({
            "feature": X_test.columns,
            "perm_importance": perm.importances_mean,
            "perm_std": perm.importances_std,
        })
        perm_df["cleaned_feature"] = perm_df["feature"].apply(_clean_name)
        perm_df = perm_df.sort_values("perm_importance", ascending=True)

        fig_perm = go.Figure(go.Bar(
            x=perm_df["perm_importance"], y=perm_df["cleaned_feature"], orientation="h",
            error_x=dict(type="data", array=perm_df["perm_std"]),
            marker=dict(color=perm_df["perm_importance"], colorscale="Cividis"),
        ))
        fig_perm.update_layout(title="Permutation Importance (AUC)", height=max(500, len(perm_df) * 28))

        summary = perm_df.sort_values("perm_importance", ascending=False).reset_index(drop=True)
        return fig_bi, fig_perm, summary
    except Exception as exc:
        logger.exception("Feature importance failed: %s", exc)
        return None, None, None


def customer_importance_chart(analysis_df: pd.DataFrame) -> go.Figure:
    top = analysis_df.head(10).sort_values("importance", ascending=True)
    fig = go.Figure(go.Bar(
        x=top["importance"], y=top["cleaned_feature"], orientation="h",
        marker=dict(color=top["importance"], colorscale="Plasma"),
    ))
    fig.update_layout(title="Top Drivers for This Customer", height=450)
    return fig
