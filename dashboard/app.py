"""
BeatCheck Analytics \u2014 Streamlit Dashboard

Four-tab analytics dashboard for the Heart on Your Wrist capstone project.
Tabs:
  1. Model Performance  \u2014 SVM test metrics, ROC curve, confusion matrix, threshold slider
  2. Cross-Model Comparison \u2014 validation metrics, grouped bar chart, overfitting table
  3. Layer 2 Validation  \u2014 MIMIC PERform AF results, gap analysis, stress tests
  4. Screening Demo     \u2014 live Apple Health file upload and risk tier display

Run with:
    conda activate cvd_project
    streamlit run dashboard/app.py
"""

import json
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    auc,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Path setup \u2014 must be before any src/ or app/ imports
# ---------------------------------------------------------------------------
_DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_DASHBOARD_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from app.pipeline import (  # noqa: E402
    parse_apple_health_export,
    parse_apple_health_xml,
    process_and_predict,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MODELS_DIR = os.path.join(_PROJECT_ROOT, "outputs", "models")
_LAYER2_DIR = os.path.join(_PROJECT_ROOT, "outputs", "layer2")
_DATA_DIR = os.path.join(_PROJECT_ROOT, "data", "processed")

FEATURES = [
    "rmssd", "sdnn", "mean_rr", "pnn50",
    "hr_mean", "hr_std", "rr_skewness", "rr_kurtosis",
]

FEATURE_LABELS = {
    "rmssd": "RMSSD",
    "sdnn": "SDNN",
    "mean_rr": "Mean RR",
    "pnn50": "pNN50",
    "hr_mean": "HR Mean",
    "hr_std": "HR Std Dev",
    "rr_skewness": "RR Skewness",
    "rr_kurtosis": "RR Kurtosis",
}

# Colour palette
C_GREEN  = "#2ecc71"
C_RED    = "#e74c3c"
C_BLUE   = "#3498db"
C_ORANGE = "#f39c12"
C_GREY   = "#95a5a6"
C_DARK   = "#2c3e50"

PLOTLY_TEMPLATE = "plotly_white"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="BeatCheck Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_data
def load_evaluation_report() -> dict:
    path = os.path.join(_MODELS_DIR, "evaluation_report.json")
    with open(path, "r") as f:
        return json.load(f)


@st.cache_data
def load_rf_importance() -> pd.DataFrame:
    return pd.read_csv(os.path.join(_MODELS_DIR, "rf_feature_importance.csv"))


@st.cache_data
def load_physionet_features() -> pd.DataFrame:
    return pd.read_csv(os.path.join(_DATA_DIR, "physionet_features.csv"))


@st.cache_data
def load_gap_quantification() -> pd.DataFrame:
    return pd.read_csv(os.path.join(_LAYER2_DIR, "gap_quantification_mimic.csv"))


@st.cache_data
def load_probability_scores_mimic() -> pd.DataFrame:
    return pd.read_csv(os.path.join(_LAYER2_DIR, "probability_scores_mimic.csv"))


@st.cache_data
def load_cross_model_comparison() -> pd.DataFrame:
    return pd.read_csv(os.path.join(_LAYER2_DIR, "cross_model_comparison_mimic.csv"))


@st.cache_data
def load_threshold_recalibration() -> pd.DataFrame:
    return pd.read_csv(os.path.join(_LAYER2_DIR, "threshold_recalibration_mimic.csv"))


@st.cache_data
def load_stress_test_results() -> pd.DataFrame:
    return pd.read_csv(os.path.join(_LAYER2_DIR, "stress_test_results_mimic.csv"))


@st.cache_resource
def load_model_and_scaler():
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    model  = joblib.load(os.path.join(_MODELS_DIR, "selected_model.joblib"))
    scaler = joblib.load(os.path.join(_MODELS_DIR, "scaler.joblib"))
    return model, scaler


# ---------------------------------------------------------------------------
# Helper: recreate test split and compute test probabilities
# (mirrors the training notebook split exactly)
# ---------------------------------------------------------------------------

@st.cache_data
def get_test_probabilities() -> tuple:
    """Return (y_test, probs_test) on the held-out test split."""
    df = load_physionet_features()
    model, scaler = load_model_and_scaler()

    X = df[FEATURES].values
    y = df["binary_label"].values

    # 80 / 10 / 10 stratified split \u2014 matches training notebook
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.10, stratify=y, random_state=42
    )
    # val = 1/9 of remaining 90% \u2192 ~10% of total
    _, X_val, _, y_val = train_test_split(
        X_temp, y_temp, test_size=1 / 9, stratify=y_temp, random_state=42
    )

    X_test_scaled = scaler.transform(X_test)  # transform only \u2014 never fit_transform
    probs_test = model.predict_proba(X_test_scaled)[:, 1]

    return y_test, probs_test


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar():
    with st.sidebar:
        st.title("BeatCheck Analytics")
        st.caption("Heart on Your Wrist \u2014 Capstone Project")
        st.markdown("---")
        st.markdown(
            "Evaluating consumer wearables as cardiac screening proxies "
            "using machine learning trained on clinical ECG data."
        )
        st.markdown("---")
        st.subheader("Key Results")
        st.metric("Layer 1 AUROC (SVM)", "0.9080")
        st.metric("Layer 2 AUROC (MIMIC)", "0.8586")
        st.metric("Layer 1 Sensitivity", "84.4%")
        st.metric("Layer 1 Specificity", "87.3%")
        st.markdown("---")
        st.caption(
            "Dataset: Physionet 2017 AF Challenge (8,187 ECG records). "
            "Layer 2: MIMIC PERform AF (35 subjects, PPG). "
            "Model: SVM with threshold 0.34."
        )


# ---------------------------------------------------------------------------
# Tab 1: Model Performance
# ---------------------------------------------------------------------------

def render_tab_model_performance():
    st.header("Layer 1 \u2014 SVM Model Performance (Held-Out Test Set)")

    # --- Metric cards ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sensitivity", "84.4%", help="True positive rate on held-out test set")
    c2.metric("Specificity", "87.3%", help="True negative rate on held-out test set")
    c3.metric("AUROC",       "0.9080", help="Area under the ROC curve")
    c4.metric("F1 Score",    "0.8243", help="Harmonic mean of precision and recall (Abnormal class)")

    st.markdown("---")

    # --- ROC curve + Confusion matrix ---
    col_roc, col_cm = st.columns(2)

    y_test, probs_test = get_test_probabilities()
    fpr, tpr, thresholds_roc = roc_curve(y_test, probs_test)
    roc_auc = auc(fpr, tpr)

    with col_roc:
        st.subheader("ROC Curve")
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode="lines",
            line=dict(color=C_BLUE, width=2.5),
            name=f"SVM (AUC = {roc_auc:.4f})",
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            line=dict(color=C_GREY, width=1.5, dash="dash"),
            name="Random classifier",
            showlegend=True,
        ))
        fig_roc.update_layout(
            template=PLOTLY_TEMPLATE,
            xaxis_title="False Positive Rate (1 \u2212 Specificity)",
            yaxis_title="True Positive Rate (Sensitivity)",
            title="ROC Curve \u2014 SVM on Held-Out Test Set",
            legend=dict(x=0.55, y=0.15),
            height=420,
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    with col_cm:
        st.subheader("Confusion Matrix")
        report = load_evaluation_report()
        cm_data = report["test_report"]["confusion_matrix"]
        tp = cm_data["tp"]
        tn = cm_data["tn"]
        fp = cm_data["fp"]
        fn = cm_data["fn"]

        cm_matrix = [[tn, fp], [fn, tp]]
        cm_labels = ["Normal (0)", "Abnormal (1)"]

        # Custom text annotations with count + label
        annotations = [
            ["TN = " + str(tn), "FP = " + str(fp)],
            ["FN = " + str(fn), "TP = " + str(tp)],
        ]

        fig_cm = go.Figure(data=go.Heatmap(
            z=cm_matrix,
            x=cm_labels,
            y=cm_labels,
            text=annotations,
            texttemplate="%{text}",
            colorscale=[[0, "#eaf4fb"], [1, C_BLUE]],
            showscale=False,
        ))
        fig_cm.update_layout(
            template=PLOTLY_TEMPLATE,
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
            title="Confusion Matrix \u2014 SVM at Threshold 0.34",
            height=420,
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("---")

    # --- Feature importance ---
    st.subheader("Feature Importance (Random Forest \u2014 Supplementary)")
    fi_df = load_rf_importance().sort_values("importance_pct", ascending=True)
    fi_df["feature_label"] = fi_df["feature"].map(FEATURE_LABELS)

    fig_fi = go.Figure(go.Bar(
        x=fi_df["importance_pct"],
        y=fi_df["feature_label"],
        orientation="h",
        marker_color=C_BLUE,
        text=fi_df["importance_pct"].apply(lambda v: f"{v:.2f}%"),
        textposition="outside",
    ))
    fig_fi.update_layout(
        template=PLOTLY_TEMPLATE,
        xaxis_title="Importance (%)",
        yaxis_title="Feature",
        title="Random Forest Feature Importance",
        height=380,
        margin=dict(r=80),
    )
    st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown("---")

    # --- Threshold slider ---
    st.subheader("Interactive Threshold Explorer")
    st.caption(
        "Adjust the decision threshold to see the live impact on sensitivity and specificity. "
        "The fixed operational threshold is 0.34 (optimised on the validation set)."
    )
    threshold_val = st.slider(
        "Classification threshold",
        min_value=0.01,
        max_value=0.99,
        value=0.34,
        step=0.01,
        format="%.2f",
    )

    preds_at_thresh = (probs_test >= threshold_val).astype(int)
    cm_thresh = confusion_matrix(y_test, preds_at_thresh)
    tn_t, fp_t, fn_t, tp_t = cm_thresh.ravel()
    sens_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0.0
    spec_t = tn_t / (tn_t + fp_t) if (tn_t + fp_t) > 0 else 0.0
    ppv_t  = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0.0
    f1_t   = (
        2 * ppv_t * sens_t / (ppv_t + sens_t)
        if (ppv_t + sens_t) > 0 else 0.0
    )

    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Sensitivity", f"{sens_t * 100:.1f}%")
    sc2.metric("Specificity", f"{spec_t * 100:.1f}%")
    sc3.metric("PPV (Precision)", f"{ppv_t * 100:.1f}%")
    sc4.metric("F1 Score", f"{f1_t:.4f}")

    sc5, sc6, sc7, sc8 = st.columns(4)
    sc5.metric("TP", str(tp_t))
    sc6.metric("FP", str(fp_t))
    sc7.metric("FN", str(fn_t))
    sc8.metric("TN", str(tn_t))


# ---------------------------------------------------------------------------
# Tab 2: Cross-Model Comparison
# ---------------------------------------------------------------------------

def render_tab_cross_model():
    st.header("Cross-Model Comparison \u2014 Validation Set")

    report = load_evaluation_report()
    val_reports = report["validation_reports"]

    # --- Metrics table ---
    st.subheader("Validation Set Metrics (All Four Models)")
    rows = []
    for r in val_reports:
        rows.append({
            "Model":         r["name"],
            "Threshold":     r["threshold"],
            "Sensitivity":   f"{r['sensitivity'] * 100:.1f}%",
            "Specificity":   f"{r['specificity'] * 100:.1f}%",
            "AUROC":         f"{r['auroc']:.4f}",
            "F1 (Abnormal)": f"{r['f1_abnormal']:.4f}",
            "AF Sensitivity":f"{r['af_sensitivity'] * 100:.1f}%",
            "Meets Criterion": "\u2713" if r["meets_criterion"] else "\u2717",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("---")

    # --- Grouped bar chart ---
    st.subheader("Metric Comparison \u2014 Grouped Bar Chart")
    metric_names = ["Sensitivity", "Specificity", "AUROC", "F1 (Abnormal)", "AF Sensitivity"]
    model_names  = [r["name"] for r in val_reports]

    metric_map = {
        "Sensitivity":    [r["sensitivity"]    for r in val_reports],
        "Specificity":    [r["specificity"]    for r in val_reports],
        "AUROC":          [r["auroc"]          for r in val_reports],
        "F1 (Abnormal)":  [r["f1_abnormal"]   for r in val_reports],
        "AF Sensitivity": [r["af_sensitivity"] for r in val_reports],
    }

    colours = [C_BLUE, C_GREEN, C_ORANGE, C_RED]
    fig_bar = go.Figure()
    for i, model_name in enumerate(model_names):
        fig_bar.add_trace(go.Bar(
            name=model_name,
            x=metric_names,
            y=[metric_map[m][i] for m in metric_names],
            marker_color=colours[i],
            text=[f"{metric_map[m][i]:.3f}" for m in metric_names],
            textposition="outside",
        ))

    fig_bar.update_layout(
        template=PLOTLY_TEMPLATE,
        barmode="group",
        yaxis=dict(range=[0, 1.15], title="Score (0\u20131)"),
        xaxis_title="Metric",
        title="Validation Set Performance \u2014 All Models",
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    # --- Overfitting table ---
    st.subheader("Overfitting Check \u2014 Train vs Test Accuracy Gap")
    overfit_data = {
        "Model":           ["Logistic Regression", "Random Forest", "XGBoost", "SVM"],
        "Train Accuracy":  ["83.2%", "100.0%", "99.8%", "85.6%"],
        "Test Accuracy":   ["85.6%", "86.6%", "86.2%", "86.8%"],
        "Gap":             ["\u22122.4%", "+13.4%", "+13.6%", "\u22121.2%"],
        "Verdict":         [
            "No overfitting",
            "Significant overfitting",
            "Significant overfitting",
            "No overfitting",
        ],
    }
    overfit_df = pd.DataFrame(overfit_data)
    st.dataframe(overfit_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # --- Selection rationale ---
    st.subheader("Model Selection Rationale")
    sel = report["selection"]
    st.info(
        f"**Selected model: {sel['selected_name']}**\n\n"
        f"{sel['reason']}\n\n"
        "Five-criterion natural selection framework was applied. "
        "Criterion 2 (highest specificity among survivors) resolved selection. "
        "SVM has the fewest false positives (60 on validation set) and no overfitting "
        "(\u22121.2% train\u2013test gap), making it the most defensible choice for a "
        "screening context where false positives carry real-world consequences."
    )


# ---------------------------------------------------------------------------
# Tab 3: Layer 2 Validation
# ---------------------------------------------------------------------------

def render_tab_layer2():
    st.header("Layer 2 Validation \u2014 MIMIC PERform AF (N=35 Subjects, PPG)")

    # --- Summary cards ---
    st.subheader("Primary Results \u2014 SVM at Fixed Threshold 0.34")
    lc1, lc2, lc3, lc4 = st.columns(4)
    lc1.metric("AUROC",       "0.8586", help="Discriminative signal transfers across modalities")
    lc2.metric("Sensitivity", "100%",   help="19/19 AF cases correctly flagged \u2014 zero false negatives")
    lc3.metric("Specificity", "12.5%",  help="2/16 NSR correct \u2014 threshold miscalibration from modality gap")
    lc4.metric("F1 Score",    "0.7308")

    st.caption(
        "Specificity failure is mechanistically explained: all 8 features show Large KS distances "
        "from the Physionet training distribution. Threshold 0.34 was calibrated on ECG data. "
        "When applied to ICU PPG data whose baseline features appear abnormal to the model, "
        "almost everything crosses threshold. AUROC 0.86 confirms discriminative signal is retained."
    )

    st.markdown("---")

    # --- KS distances bar chart ---
    st.subheader("Feature Distribution Gap \u2014 KS Statistics vs Physionet Training Data")
    gap_df = load_gap_quantification()
    gap_df = gap_df.sort_values("ks_statistic", ascending=True)
    gap_df["feature_label"] = gap_df["feature"].map(FEATURE_LABELS)

    colours_ks = [C_RED if v >= 0.3 else C_ORANGE for v in gap_df["ks_statistic"]]
    fig_ks = go.Figure(go.Bar(
        x=gap_df["ks_statistic"],
        y=gap_df["feature_label"],
        orientation="h",
        marker_color=colours_ks,
        text=gap_df.apply(
            lambda row: f"{row['ks_statistic']:.4f} ({row['direction']})", axis=1
        ),
        textposition="outside",
    ))
    fig_ks.add_vline(x=0.3, line_dash="dash", line_color=C_DARK, annotation_text="Large gap threshold (0.3)")
    fig_ks.update_layout(
        template=PLOTLY_TEMPLATE,
        xaxis_title="KS Statistic",
        yaxis_title="Feature",
        title="KS Distance: MIMIC PERform AF vs Physionet ECG Training Distribution",
        height=400,
        margin=dict(r=200),
    )
    st.plotly_chart(fig_ks, use_container_width=True)

    st.markdown("---")

    # --- Probability score strip plot ---
    st.subheader("Predicted Probability Scores by Subject")
    prob_df = load_probability_scores_mimic()
    prob_df["Group"] = prob_df["label"].map({1: "AF (Abnormal)", 0: "NSR (Normal)"})

    fig_strip = go.Figure()
    group_colours = {"AF (Abnormal)": C_RED, "NSR (Normal)": C_GREEN}
    for group, colour in group_colours.items():
        sub = prob_df[prob_df["Group"] == group]
        fig_strip.add_trace(go.Box(
            y=sub["prob_abnormal"],
            name=group,
            boxpoints="all",
            jitter=0.4,
            pointpos=0,
            marker=dict(color=colour, size=8, opacity=0.8),
            line=dict(color=colour),
        ))
    fig_strip.add_hline(
        y=0.34,
        line_dash="dash",
        line_color=C_DARK,
        annotation_text="Fixed threshold (0.34)",
        annotation_position="right",
    )
    fig_strip.add_hline(
        y=0.8368,
        line_dash="dot",
        line_color=C_BLUE,
        annotation_text="App threshold (0.8368)",
        annotation_position="right",
    )
    fig_strip.update_layout(
        template=PLOTLY_TEMPLATE,
        yaxis_title="Predicted Probability (Abnormal)",
        xaxis_title="Subject Group",
        title="SVM Probability Scores \u2014 MIMIC PERform AF Subjects",
        height=460,
        showlegend=False,
    )
    st.plotly_chart(fig_strip, use_container_width=True)

    st.markdown("---")

    # --- Cross-model comparison ---
    st.subheader("Cross-Model Comparison on MIMIC PERform AF (Fixed Layer 1 Thresholds)")
    cmc_df = load_cross_model_comparison()
    cmc_display = cmc_df.copy()
    cmc_display["sensitivity"] = (cmc_display["sensitivity"] * 100).map("{:.1f}%".format)
    cmc_display["specificity"] = (cmc_display["specificity"] * 100).map("{:.1f}%".format)
    cmc_display["auroc"]       = cmc_display["auroc"].map("{:.4f}".format)
    cmc_display["f1"]          = cmc_display["f1"].map("{:.4f}".format)
    cmc_display = cmc_display.rename(columns={
        "model": "Model", "threshold": "Threshold",
        "sensitivity": "Sensitivity", "specificity": "Specificity",
        "auroc": "AUROC", "f1": "F1",
        "TP": "TP", "FP": "FP", "FN": "FN", "TN": "TN",
    })
    st.dataframe(cmc_display, use_container_width=True, hide_index=True)
    st.caption(
        "Specificity failure is systemic across all four models \u2014 not a model selection problem. "
        "SVM retains highest AUROC (0.8586). Model selection confirmed correct."
    )

    st.markdown("---")

    # --- Threshold recalibration ---
    st.subheader("Threshold Recalibration \u2014 Youden's J Optimal Threshold (SVM)")
    recal_df = load_threshold_recalibration()
    row = recal_df.iloc[0]

    rc1, rc2 = st.columns(2)
    with rc1:
        st.markdown("**Fixed threshold (0.34)**")
        st.metric("Sensitivity", f"{row['fixed_sens'] * 100:.1f}%")
        st.metric("Specificity", f"{row['fixed_spec'] * 100:.1f}%")
        st.metric("F1",          f"{row['fixed_f1']:.4f}")
        conf_fixed = (
            f"TP={int(row['fixed_TP'])} | FP={int(row['fixed_FP'])} | "
            f"FN={int(row['fixed_FN'])} | TN={int(row['fixed_TN'])}"
        )
        st.caption(conf_fixed)
    with rc2:
        st.markdown(f"**Recalibrated threshold ({row['optimal_threshold']:.4f})**")
        st.metric("Sensitivity", f"{row['optimal_sens'] * 100:.1f}%")
        st.metric("Specificity", f"{row['optimal_spec'] * 100:.1f}%")
        st.metric("F1",          f"{row['optimal_f1']:.4f}")
        conf_opt = (
            f"TP={int(row['optimal_TP'])} | FP={int(row['optimal_FP'])} | "
            f"FN={int(row['optimal_FN'])} | TN={int(row['optimal_TN'])}"
        )
        st.caption(conf_opt)

    st.info(
        "Specificity recovers from 12.5% to 93.8% with domain-adapted threshold. "
        "Optimal threshold 0.8424 vs fixed 0.34 \u2014 gap quantifies the magnitude of the modality shift. "
        "Caveat: threshold found and evaluated on the same 35 subjects. Stress-tested below."
    )

    st.markdown("---")

    # --- Stress test results ---
    st.subheader("Stress Test Results")
    stress_df = load_stress_test_results()

    # Bootstrap AUROC
    auroc_row = stress_df[stress_df["test"] == "Bootstrap AUROC"]
    if not auroc_row.empty:
        r = auroc_row.iloc[0]
        st.markdown("**Test 1 \u2014 Bootstrap AUROC (n=1,000 resamples)**")
        bc1, bc2 = st.columns(2)
        bc1.metric("Mean AUROC", f"{float(r['value']):.4f}")
        bc2.metric("95% CI", f"[{float(r['ci_lower']):.4f}, {float(r['ci_upper']):.4f}]")

    # Bootstrap recalibrated metrics
    boot_recal = stress_df[stress_df["test"] == "Bootstrap Recalibrated"]
    if not boot_recal.empty:
        st.markdown("**Test 2 \u2014 Bootstrap Recalibrated Metrics (n=1,000 resamples)**")
        br_cols = st.columns(3)
        for i, (_, r) in enumerate(boot_recal.iterrows()):
            ci_lower = r["ci_lower"]
            ci_upper = r["ci_upper"]
            ci_str = (
                f"[{float(ci_lower):.4f}, {float(ci_upper):.4f}]"
                if pd.notna(ci_lower) and pd.notna(ci_upper)
                else "N/A"
            )
            br_cols[i].metric(
                r["metric"].title(),
                f"{float(r['value']):.4f}",
                help=f"95% CI: {ci_str}",
            )
            br_cols[i].caption(f"95% CI: {ci_str}")

    # LOOCV results
    st.markdown("**Test 3 \u2014 Leave-One-Out Cross-Validation (LOOCV)**")

    loocv_data = {
        "Method": ["Youden's J LOOCV", "Sensitivity-targeted LOOCV", "Pre-registered criterion"],
        "Sensitivity": ["68.4%", "78.9%", "\u226580%"],
        "Specificity": ["81.2%", "81.2%", "\u226575%"],
        "F1": ["0.7429", "0.8108", "\u2014"],
    }
    st.dataframe(pd.DataFrame(loocv_data), use_container_width=True, hide_index=True)
    st.caption(
        "Sensitivity-targeted LOOCV (mean per-fold threshold = 0.8368) improves sensitivity by "
        "10.5 percentage points with no specificity cost. Both methods hold specificity at 81.2%, "
        "clearing the pre-registered criterion. Sensitivity 1.1pp below 80% criterion \u2014 "
        "consistent with N=19 AF variance in leave-one-out evaluation."
    )


# ---------------------------------------------------------------------------
# Tab 4: Screening Demo
# ---------------------------------------------------------------------------

def render_tab_screening_demo():
    st.header("Screening Demo \u2014 BeatCheck")

    # --- Disclaimer ---
    st.warning(
        "\u26a0\ufe0f  **Not a medical device.** "
        "BeatCheck is a research demonstration tool only. "
        "It is not validated for clinical use, does not constitute a medical diagnosis, "
        "and should not replace professional medical advice. "
        "If you have cardiac concerns, consult a qualified healthcare professional. "
        "Results are for informational and research purposes only."
    )

    st.markdown(
        "Upload an Apple Health export file to analyse your heart rate data. "
        "BeatCheck will extract the most recent 90 days of heart rate records, "
        "build 30-minute windows with 15-minute overlap, compute HRV features, "
        "and apply the trained SVM at threshold 0.8368 (LOOCV mean)."
    )

    # --- File uploader ---
    st.subheader("Upload Apple Health Export")
    uploaded = st.file_uploader(
        "Choose a file (CSV export or XML export.xml)",
        type=["csv", "xml"],
        help=(
            "CSV: exported from Apple Health as a CSV file. "
            "XML: the export.xml file from an Apple Health full export."
        ),
    )

    if uploaded is None:
        st.info("No file uploaded yet. Upload a CSV or XML Apple Health export to run the demo.")
        return

    # --- Process file ---
    with st.spinner("Parsing file and running inference..."):
        try:
            file_bytes = uploaded.read()
            fname = uploaded.name.lower()

            if fname.endswith(".xml"):
                df_hr = parse_apple_health_xml(file_bytes)
            else:
                df_hr = parse_apple_health_export(file_bytes)

            result = process_and_predict(df_hr)

        except ValueError as e:
            st.error(f"Could not process file: {e}")
            return
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            return

    # --- Display results ---
    tier        = result["risk_tier"]
    pct_flagged = result["pct_flagged"]
    total_win   = result["total_windows"]
    flagged_win = result["flagged_windows"]
    days        = result["days_analysed"]

    tier_colours = {
        "Low":          C_GREEN,
        "Intermediate": C_ORANGE,
        "High":         C_RED,
    }
    tier_explanations = {
        "Low": (
            "Fewer than 10% of your heart rate windows were flagged as potentially abnormal. "
            "Your rhythm variability patterns are consistent with what the model learned from "
            "normal ECG recordings."
        ),
        "Intermediate": (
            "Between 10% and 40% of your heart rate windows were flagged as potentially abnormal. "
            "This is an intermediate signal \u2014 it does not confirm a cardiac abnormality, "
            "but it may be worth discussing with a healthcare professional if you have concerns."
        ),
        "High": (
            "More than 40% of your heart rate windows were flagged as potentially abnormal. "
            "This is a high signal \u2014 it does not confirm a cardiac abnormality. "
            "This model is a research tool, not a diagnostic instrument. "
            "Please consult a qualified healthcare professional."
        ),
    }

    colour = tier_colours.get(tier, C_GREY)

    st.markdown("---")
    st.subheader("Results")

    # Risk tier badge \u2014 styled with markdown + HTML
    st.markdown(
        f"""
        <div style="
            background-color: {colour};
            color: white;
            padding: 24px 32px;
            border-radius: 12px;
            text-align: center;
            font-size: 2rem;
            font-weight: bold;
            letter-spacing: 0.05em;
            margin-bottom: 1rem;
        ">
            {tier.upper()} RISK
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Metric row
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Windows Flagged", f"{pct_flagged}%")
    mc2.metric("Flagged / Total",  f"{flagged_win} / {total_win}")
    mc3.metric("Days Analysed",    str(days))
    mc4.metric("Threshold Used",   "0.8368")

    # Explanation
    st.markdown(
        f"**What this means:** {tier_explanations.get(tier, '')}"
    )

    # Risk tier legend
    with st.expander("Risk tier thresholds"):
        tier_table = pd.DataFrame({
            "Tier":           ["Low", "Intermediate", "High"],
            "Windows Flagged":["< 10%", "10% \u2013 40%", "> 40%"],
            "Interpretation": [
                "Rhythm patterns consistent with normal ECG training data",
                "Elevated signal \u2014 discuss with a clinician if concerned",
                "High proportion flagged \u2014 seek clinical evaluation",
            ],
        })
        st.dataframe(tier_table, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.caption(
        "Methodology: 30-minute sliding windows, 15-minute step, minimum 10 readings per window. "
        "Features: RMSSD, SDNN, Mean RR, pNN50, HR Mean, HR Std Dev, RR Skewness, RR Kurtosis. "
        "Model: SVM trained on Physionet 2017 AF Challenge (8,187 ECG records). "
        "RMSSD approximated from HR time series as 60000/HR. "
        "This is a research prototype. Results should not be used for clinical decisions."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    render_sidebar()

    tab1, tab2, tab3, tab4 = st.tabs([
        "Model Performance",
        "Cross-Model Comparison",
        "Layer 2 Validation",
        "Screening Demo",
    ])

    with tab1:
        render_tab_model_performance()

    with tab2:
        render_tab_cross_model()

    with tab3:
        render_tab_layer2()

    with tab4:
        render_tab_screening_demo()


if __name__ == "__main__":
    main()
