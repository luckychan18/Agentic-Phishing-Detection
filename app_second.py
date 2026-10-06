import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import shap

from scipy.io import arff


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Agentic Phishing Detector",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "Training Dataset.arff"
)

RF_PATH = os.path.join(
    BASE_DIR,
    "models",
    "second_dataset_random_forest.pkl"
)

XGB_PATH = os.path.join(
    BASE_DIR,
    "models",
    "second_dataset_xgboost.pkl"
)

CALIBRATOR_PATH = os.path.join(
    BASE_DIR,
    "models",
    "second_dataset_calibrator.pkl"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 38px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #777;
    margin-bottom: 30px;
}

.decision-box {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    margin: 10px 0px 20px 0px;
}

.decision-title {
    font-size: 32px;
    font-weight: 700;
}

.decision-score {
    font-size: 20px;
    margin-top: 10px;
}

.reason-box {
    padding: 18px;
    border-radius: 10px;
    border: 1px solid #ddd;
    margin-top: 10px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    data, meta = arff.loadarff(DATA_PATH)

    df = pd.DataFrame(data)

    for col in df.columns:

        if df[col].dtype == object:

            df[col] = df[col].apply(
                lambda x:
                x.decode("utf-8")
                if isinstance(x, bytes)
                else x
            )

    X = df.drop(
        columns=["Result"]
    ).astype(float)

    y = df["Result"].astype(int)

    return X, y


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    rf = joblib.load(RF_PATH)
    xgb = joblib.load(XGB_PATH)
    calibrator = joblib.load(CALIBRATOR_PATH)

    explainer = shap.TreeExplainer(rf)

    return rf, xgb, calibrator, explainer


X, y = load_data()

rf, xgb, calibrator, explainer = load_models()


# ============================================================
# EVIDENCE GROUPS
# ============================================================

EVIDENCE_GROUPS = {

    "Security Evidence": [
        "SSLfinal_State",
        "HTTPS_token",
        "port"
    ],

    "URL Structure": [
        "URL_Length",
        "having_IP_Address",
        "Prefix_Suffix",
        "having_Sub_Domain",
        "Shortening_Service",
        "having_At_Symbol",
        "double_slash_redirecting"
    ],

    "Link Structure": [
        "URL_of_Anchor",
        "Links_in_tags",
        "Request_URL",
        "Links_pointing_to_page",
        "IFrame"
    ],

    "Domain Reputation": [
        "web_traffic",
        "Domain_registration_length",
        "age_of_domain",
        "DNSRecord",
        "Google_Index",
        "Page_Rank"
    ],

    "Page Behavior": [
        "SFH",
        "popUpWindow",
        "Redirect",
        "on_mouseover",
        "RightClick"
    ],

    "Other Evidence": [
        "Statistical_report",
        "Submitting_to_email",
        "Abnormal_URL",
        "Favicon"
    ]
}


# ============================================================
# SHAP
# ============================================================

def calculate_shap(row):

    sample = row.to_frame().T

    values = explainer.shap_values(sample)

    if isinstance(values, list):

        values = values[1][0]

    else:

        values = np.asarray(values)

        if values.ndim == 3:
            values = values[0, :, 1]

        elif values.ndim == 2:
            values = values[0]

    return values


# ============================================================
# EVIDENCE SELECTION
# ============================================================

def select_evidence(shap_values):

    shap_df = pd.DataFrame({
        "Feature": X.columns,
        "SHAP": shap_values
    })

    shap_df["Absolute"] = np.abs(
        shap_df["SHAP"]
    )

    scores = {}

    for group, features in EVIDENCE_GROUPS.items():

        available = [
            f for f in features
            if f in X.columns
        ]

        if not available:

            scores[group] = 0

        else:

            scores[group] = shap_df[
                shap_df["Feature"].isin(available)
            ]["Absolute"].sum()

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked


# ============================================================
# FIND DISAGREEMENT
# ============================================================

def find_disagreement():

    rf_raw = rf.predict_proba(X)[:, 1]

    rf_calibrated = calibrator.predict_proba(
        rf_raw.reshape(-1, 1)
    )[:, 1]

    xgb_prob = xgb.predict_proba(X)[:, 1]

    rf_pred = rf_calibrated >= 0.5
    xgb_pred = xgb_prob >= 0.5

    disagreement = np.where(
        rf_pred != xgb_pred
    )[0]

    if len(disagreement) == 0:
        return None

    differences = np.abs(
        rf_calibrated[disagreement]
        - xgb_prob[disagreement]
    )

    return int(
        disagreement[
            np.argmax(differences)
        ]
    )


# ============================================================
# ANALYSIS
# ============================================================

def analyze(row):

    sample = row.to_frame().T

    # --------------------------------------------------------
    # MODEL PREDICTIONS
    # --------------------------------------------------------

    rf_raw = float(
        rf.predict_proba(sample)[0, 1]
    )

    rf_calibrated = float(
        calibrator.predict_proba(
            np.array([[rf_raw]])
        )[0, 1]
    )

    xgb_probability = float(
        xgb.predict_proba(sample)[0, 1]
    )

    rf_prediction = rf_calibrated >= 0.5

    xgb_prediction = xgb_probability >= 0.5

    model_agreement = (
        rf_prediction == xgb_prediction
    )

    probability_gap = abs(
        rf_calibrated - xgb_probability
    )

    average_probability = (
        rf_calibrated + xgb_probability
    ) / 2

    initial_confidence = max(
        average_probability,
        1 - average_probability
    )

    # --------------------------------------------------------
    # SHAP
    # --------------------------------------------------------

    shap_values = calculate_shap(row)

    ranked_evidence = select_evidence(
        shap_values
    )

    selected_evidence = ranked_evidence[:3]

    # --------------------------------------------------------
    # AGENT DECISION
    # --------------------------------------------------------

    uncertain = (
        not model_agreement
        or probability_gap > 0.20
        or initial_confidence < 0.70
    )

    if uncertain:

        shap_df = pd.DataFrame({
            "Feature": X.columns,
            "SHAP": shap_values
        })

        shap_df["Absolute"] = np.abs(
            shap_df["SHAP"]
        )

        top_features = shap_df.sort_values(
            "Absolute",
            ascending=False
        ).head(5)

        positive = top_features[
            top_features["SHAP"] > 0
        ]["Absolute"].sum()

        negative = top_features[
            top_features["SHAP"] < 0
        ]["Absolute"].sum()

        shap_signal = (
            1 if positive > negative
            else 0
        )

        final_score = (
            0.70 * average_probability
            + 0.30 * shap_signal
        )

        triggered_shap = True

        if not model_agreement:

            action = (
                "Model conflict detected. "
                "The agent triggered additional "
                "SHAP analysis."
            )

        else:

            action = (
                "Model confidence was insufficient. "
                "The agent triggered additional "
                "SHAP analysis."
            )

    else:

        top_features = None

        final_score = average_probability

        triggered_shap = False

        action = (
            "Models agree and provide sufficient "
            "evidence. No additional analysis required."
        )

    # --------------------------------------------------------
    # FINAL DECISION
    # --------------------------------------------------------

    if final_score >= 0.70:

        decision = "PHISHING"

    elif final_score <= 0.30:

        decision = "LEGITIMATE"

    else:

        decision = "SUSPICIOUS"

    final_confidence = max(
        final_score,
        1 - final_score
    )

    return {
        "rf_raw": rf_raw,
        "rf_calibrated": rf_calibrated,
        "xgb": xgb_probability,
        "rf_prediction": rf_prediction,
        "xgb_prediction": xgb_prediction,
        "agreement": model_agreement,
        "gap": probability_gap,
        "initial_confidence": initial_confidence,
        "selected_evidence": selected_evidence,
        "triggered_shap": triggered_shap,
        "top_features": top_features,
        "final_score": final_score,
        "final_confidence": final_confidence,
        "decision": decision,
        "action": action
    }


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🛡️ Agentic Phishing Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Explainable phishing detection using ML, SHAP and adaptive evidence orchestration'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Agent Controls")

    st.write(
        "Select how the agent should choose a sample."
    )

    mode = st.radio(
        "Analysis mode",
        [
            "Select sample",
            "Find model disagreement"
        ]
    )

    if mode == "Select sample":

        sample_number = st.number_input(
            "Dataset sample",
            min_value=0,
            max_value=len(X) - 1,
            value=100,
            step=1
        )

    else:

        st.info(
            "The agent will automatically search "
            "for a sample where RF and XGBoost disagree."
        )


# ============================================================
# ANALYZE BUTTON
# ============================================================

st.markdown("### 🔎 Analysis Input")

if mode == "Select sample":

    st.write(
        f"Analyzing dataset sample **#{sample_number}**"
    )

else:

    st.write(
        "The agent will dynamically locate a "
        "**model-conflict case**."
    )


analyze_button = st.button(
    "🚀 ANALYZE WITH AGENT",
    use_container_width=True
)


if analyze_button:

    if mode == "Select sample":

        index = int(sample_number)

    else:

        with st.spinner(
            "Agent searching for model disagreement..."
        ):

            index = find_disagreement()

        if index is None:

            st.error(
                "No model disagreement found."
            )

            st.stop()

        st.success(
            f"Agent found a disagreement at sample #{index}"
        )

    row = X.iloc[index]

    result = analyze(row)

    # ========================================================
    # FINAL DECISION
    # ========================================================

    st.markdown("---")

    st.subheader("🎯 Agent Decision")

    decision = result["decision"]

    if decision == "PHISHING":

        st.error(
            f"## 🚨 {decision}"
        )

    elif decision == "LEGITIMATE":

        st.success(
            f"## ✅ {decision}"
        )

    else:

        st.warning(
            f"## ⚠️ {decision}"
        )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Agent Risk Score",
            f"{result['final_score'] * 100:.2f}%"
        )

    with col2:

        st.metric(
            "Agent Confidence",
            f"{result['final_confidence'] * 100:.2f}%"
        )

    with col3:

        st.metric(
            "Evidence Modules",
            len(result["selected_evidence"])
        )

    # ========================================================
    # MODEL EVIDENCE
    # ========================================================

    st.markdown("---")

    st.subheader("🤖 Model Evidence")

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Random Forest",
            f"{result['rf_calibrated'] * 100:.2f}%"
        )

        st.caption(
            "Calibrated phishing score"
        )

    with c2:

        st.metric(
            "XGBoost",
            f"{result['xgb'] * 100:.2f}%"
        )

        st.caption(
            "Phishing score"
        )

    with c3:

        if result["agreement"]:

            st.success("✓ MODELS AGREE")

        else:

            st.error("⚠ MODEL CONFLICT")

    # ========================================================
    # AGENT REASONING
    # ========================================================

    st.markdown("---")

    st.subheader("🧠 Agent Reasoning")

    if result["triggered_shap"]:

        st.warning(
            "⚠️ Additional analysis was triggered."
        )

    else:

        st.success(
            "✓ No additional analysis was required."
        )

    st.info(
        result["action"]
    )

    # ========================================================
    # EVIDENCE SELECTION
    # ========================================================

    st.subheader(
        "🔍 Evidence Selected by Agent"
    )

    evidence_df = pd.DataFrame(
        result["selected_evidence"],
        columns=[
            "Evidence Type",
            "SHAP Evidence Score"
        ]
    )

    evidence_df["SHAP Evidence Score"] = (
        evidence_df["SHAP Evidence Score"]
        .round(4)
    )

    st.dataframe(
        evidence_df,
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # SHAP
    # ========================================================

    if result["triggered_shap"]:

        st.markdown("---")

        st.subheader(
            "🔬 Additional XAI Evidence — SHAP"
        )

        shap_df = result["top_features"].copy()

        shap_df["Direction"] = np.where(
            shap_df["SHAP"] > 0,
            "PHISHING",
            "LEGITIMATE"
        )

        display_df = shap_df[
            [
                "Feature",
                "SHAP",
                "Direction"
            ]
        ].copy()

        display_df["SHAP"] = display_df[
            "SHAP"
        ].round(4)

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # SHAP BAR
        # ----------------------------------------------------

        chart_df = shap_df[
            ["Feature", "SHAP"]
        ].copy()

        chart_df = chart_df.set_index(
            "Feature"
        )

        st.bar_chart(
            chart_df
        )

    # ========================================================
    # PIPELINE
    # ========================================================

    st.markdown("---")

    st.subheader(
        "🔄 Agentic Decision Pipeline"
    )

    if result["triggered_shap"]:

        st.success(
            """
            Input Evidence
            ↓
            Random Forest + XGBoost
            ↓
            ⚠ Model Conflict / Uncertainty
            ↓
            Agent Triggers SHAP
            ↓
            XAI Evidence
            ↓
            Evidence Fusion
            ↓
            Final Agent Decision
            """
        )

    else:

        st.success(
            """
            Input Evidence
            ↓
            Random Forest + XGBoost
            ↓
            ✓ Models Agree
            ↓
            Evidence Sufficient
            ↓
            Final Agent Decision
            """
        )

    # ========================================================
    # RESEARCH CONTRIBUTION
    # ========================================================

    st.markdown("---")

    st.subheader(
        "📌 Research Gap Addressed"
    )

    st.write(
        """
        **Static ML workflow**
        → Model prediction
        → Explanation

        **Our agentic workflow**
        → Compare multiple models
        → Evaluate uncertainty
        → Dynamically select evidence
        → Trigger additional XAI analysis
        → Treat XAI output as evidence
        → Fuse evidence
        → Produce an interpretable final decision
        """
    )

else:

    st.info(
        "Select a sample or let the agent find a "
        "model-conflict case, then click "
        "**ANALYZE WITH AGENT**."
    )