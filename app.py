import streamlit as st
import os
import sys
import re
import glob
import pandas as pd
import numpy as np
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.url_predict import predict_url
from src.xai import explain_url


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Agentic Phishing Detection",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Agentic Phishing Detection")
st.caption(
    "Adaptive evidence selection + ML + XAI + conflict resolution"
)


# ============================================================
# RAW URL ANALYZER
# ============================================================

def analyze_raw_url(url):

    original_url = url.strip()

    if not original_url.startswith(("http://", "https://")):
        url = "http://" + original_url
    else:
        url = original_url

    parsed = urlparse(url)

    hostname = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""

    text = url.lower()

    score = 0
    reasons = []

    # HTTPS
    if parsed.scheme == "http":
        score += 15
        reasons.append("Connection does not use HTTPS")

    # Suspicious keywords
    keywords = [
        "login",
        "signin",
        "verify",
        "verification",
        "security",
        "secure",
        "account",
        "password",
        "credential",
        "update",
        "confirm",
        "banking",
        "wallet",
        "payment",
        "recover",
        "unlock"
    ]

    found = [x for x in keywords if x in text]

    if found:
        score += min(25, len(found) * 8)
        reasons.append(
            "Security/login keywords detected: "
            + ", ".join(found[:6])
        )

    # Hyphens
    hyphens = hostname.count("-")

    if hyphens >= 2:
        score += 15
        reasons.append("Multiple hyphens in domain")

    elif hyphens == 1 and any(
        x in hostname for x in
        ["login", "secure", "verify", "account", "security"]
    ):
        score += 5
        reasons.append(
            "Security-related term combined with hyphenated domain"
        )

    # Subdomains
    parts = hostname.split(".")

    if len(parts) >= 5:
        score += 15
        reasons.append("Large number of subdomains")

    # IP address
    ip_pattern = r"^(?:\d{1,3}\.){3}\d{1,3}$"

    if re.match(ip_pattern, hostname):
        score += 25
        reasons.append(
            "IP address used instead of a domain name"
        )

    # @ symbol
    if "@" in url:
        score += 20
        reasons.append(
            "URL contains @ symbol"
        )

    # Punycode
    if "xn--" in hostname:
        score += 20
        reasons.append(
            "Punycode domain detected"
        )

    # Long URL
    if len(url) > 100:
        score += 10
        reasons.append(
            "Unusually long URL"
        )

    # Query
    if query:
        score += 5
        reasons.append(
            "Query parameters detected"
        )

    # Credential path + query
    credential_words = [
        "login",
        "signin",
        "password",
        "credential",
        "verify"
    ]

    if (
        any(x in path for x in credential_words)
        and query
    ):
        score += 10
        reasons.append(
            "Credential-related path combined with query parameters"
        )

    score = min(score, 100)

    if score >= 70:
        decision = "PHISHING"
    elif score >= 35:
        decision = "SUSPICIOUS"
    else:
        decision = "LEGITIMATE"

    if not reasons:
        reasons.append(
            "No major suspicious raw-URL patterns detected"
        )

    return {
        "score": score,
        "decision": decision,
        "reasons": reasons,
        "hostname": hostname,
        "path": path,
        "query": query
    }


# ============================================================
# MODEL PREDICTIONS
# ============================================================

def get_model_predictions(url):

    result = predict_url(url)

    ml_probability = float(
        result.get(
            "ml_phishing_probability",
            0
        )
    )

    rule_probability = float(
        result.get(
            "rule_probability",
            0
        )
    )

    features = result.get(
        "features",
        {}
    )

    return {
        "result": result,
        "ml_probability": ml_probability,
        "rule_probability": rule_probability,
        "features": features
    }


# ============================================================
# AGENT DECISION
# ============================================================

def fuse_evidence(
    ml_probability,
    rule_probability,
    raw_score,
    xai_probability
):

    ml = ml_probability * 100
    rule = rule_probability * 100
    raw = raw_score
    xai = xai_probability * 100

    conflict = False

    # Strong raw evidence overrides weak ML
    if ml < 30 and raw >= 50:
        conflict = True

        decision = "SUSPICIOUS"

        risk = (
            0.20 * ml +
            0.25 * rule +
            0.40 * raw +
            0.15 * xai
        )

        reasoning = (
            "The dataset-trained model indicates low phishing "
            "probability, but independent URL analysis identifies "
            "strong suspicious characteristics. The agent detected "
            "a conflict and conservatively classified the URL as "
            "SUSPICIOUS."
        )

        return decision, risk, conflict, reasoning

    # Strong agreement for phishing
    if (
        ml >= 70
        and raw >= 50
        and rule >= 50
    ):

        decision = "PHISHING"

        risk = (
            0.40 * ml +
            0.20 * rule +
            0.25 * raw +
            0.15 * xai
        )

        reasoning = (
            "Multiple independent evidence sources strongly "
            "support a phishing classification."
        )

        return decision, risk, conflict, reasoning

    # Strong ML + XAI
    if (
        ml >= 70
        and xai >= 50
    ):

        decision = "PHISHING"

        risk = (
            0.50 * ml +
            0.15 * rule +
            0.15 * raw +
            0.20 * xai
        )

        reasoning = (
            "The dataset-trained model and SHAP evidence "
            "strongly support phishing."
        )

        return decision, risk, conflict, reasoning

    # Rule + raw evidence conflict
    if (
        rule >= 50
        and raw >= 35
        and ml < 50
    ):

        conflict = True

        decision = "SUSPICIOUS"

        risk = (
            0.20 * ml +
            0.25 * rule +
            0.40 * raw +
            0.15 * xai
        )

        reasoning = (
            "The ML evidence conflicts with independent "
            "structural and rule-based evidence. The agent "
            "requested additional XAI evidence and classified "
            "the URL as SUSPICIOUS."
        )

        return decision, risk, conflict, reasoning

    # Strong legitimate agreement
    if (
        ml < 30
        and raw < 20
        and rule < 30
        and xai < 30
    ):

        decision = "LEGITIMATE"

        risk = (
            0.35 * ml +
            0.20 * rule +
            0.20 * raw +
            0.25 * xai
        )

        reasoning = (
            "The major evidence sources consistently support "
            "a legitimate classification."
        )

        return decision, risk, conflict, reasoning

    # Default uncertain state
    decision = "SUSPICIOUS"

    risk = (
        0.25 * ml +
        0.25 * rule +
        0.30 * raw +
        0.20 * xai
    )

    reasoning = (
        "The evidence sources are not sufficiently consistent "
        "for a definitive classification. The agent therefore "
        "classified the URL as SUSPICIOUS."
    )

    return decision, risk, conflict, reasoning


# ============================================================
# FIND MODEL DISAGREEMENT
# ============================================================

def find_model_disagreement():

    csv_files = glob.glob(
        os.path.join(
            "data",
            "*.csv"
        )
    )

    if not csv_files:
        return None

    path = csv_files[0]

    try:

        df = pd.read_csv(
            path,
            nrows=50000
        )

    except Exception:
        return None

    url_column = None

    for col in df.columns:

        if col.lower() in [
            "url",
            "urls",
            "urladdress",
            "link"
        ]:
            url_column = col
            break

    if url_column is None:
        return None

    disagreements = []

    for _, row in df.iterrows():

        current_url = str(
            row[url_column]
        )

        if not current_url.startswith(
            ("http://", "https://")
        ):
            continue

        try:

            result = predict_url(
                current_url
            )

            rf_probability = float(
                result.get(
                    "ml_phishing_probability",
                    0
                )
            )

            # We use the existing XGBoost model directly
            # if available.

            xgb_path = os.path.join(
                "models",
                "xgboost.pkl"
            )

            feature_path = os.path.join(
                "models",
                "feature_names.pkl"
            )

            if not os.path.exists(
                xgb_path
            ):
                continue

            import joblib

            xgb_model = joblib.load(
                xgb_path
            )

            feature_names = joblib.load(
                feature_path
            )

            vector = []

            valid = True

            for feature in feature_names:

                if feature not in row.index:
                    valid = False
                    break

                vector.append(
                    row[feature]
                )

            if not valid:
                continue

            vector = np.array(
                vector,
                dtype=float
            ).reshape(
                1,
                -1
            )

            xgb_probability = float(
                xgb_model.predict_proba(
                    vector
                )[0][1]
            )

            rf_class = (
                1
                if rf_probability >= 0.5
                else 0
            )

            xgb_class = (
                1
                if xgb_probability >= 0.5
                else 0
            )

            if rf_class != xgb_class:

                disagreements.append(
                    {
                        "url": current_url,
                        "rf": rf_probability,
                        "xgb": xgb_probability
                    }
                )

                if len(disagreements) >= 10:
                    break

        except Exception:
            continue

    if disagreements:
        return disagreements

    return None


# ============================================================
# SIDEBAR — AGENT CONTROLS
# ============================================================

with st.sidebar:

    st.header("⚙️ Agent Controls")

    st.write(
        "Select how the agent should choose a sample."
    )

    mode = st.radio(
        "Analysis mode",
        [
            "Analyze entered URL",
            "Select sample",
            "Find model disagreement"
        ]
    )

    st.divider()

    st.markdown(
        """
        ### Agent capabilities

        ✓ Dynamic evidence selection  
        ✓ RF + XGBoost comparison  
        ✓ Raw URL analysis  
        ✓ Rule-based analysis  
        ✓ SHAP investigation  
        ✓ Conflict detection  
        ✓ Additional evidence request  
        ✓ Evidence fusion  
        ✓ Final reasoning
        """
    )


# ============================================================
# SAMPLE / URL INPUT
# ============================================================

if mode == "Analyze entered URL":

    url = st.text_input(
        "Enter URL",
        placeholder="https://www.google.com"
    )

    analyze_button = st.button(
        "🔍 ANALYZE URL",
        type="primary",
        use_container_width=True
    )

elif mode == "Select sample":

    sample_urls = [
        "https://www.google.com",
        "https://www.microsoft.com",
        "http://login-securityexample.com/login?user=123",
        "http://secure-login-paypal-account-verif.com/login.php?user=12345"
    ]

    url = st.selectbox(
        "Select sample",
        sample_urls
    )

    analyze_button = st.button(
        "🔍 ANALYZE SAMPLE",
        type="primary",
        use_container_width=True
    )

else:

    url = None

    analyze_button = st.button(
        "🔎 FIND MODEL DISAGREEMENT",
        type="primary",
        use_container_width=True
    )


# ============================================================
# MODEL DISAGREEMENT MODE
# ============================================================

if mode == "Find model disagreement" and analyze_button:

    with st.spinner(
        "Agent is searching for RF/XGBoost disagreement..."
    ):

        disagreements = find_model_disagreement()

    if disagreements:

        st.subheader(
            "⚠️ Model Disagreement Samples"
        )

        table = pd.DataFrame(
            disagreements
        )

        table["rf"] = (
            table["rf"] * 100
        ).round(2)

        table["xgb"] = (
            table["xgb"] * 100
        ).round(2)

        table = table.rename(
            columns={
                "url": "URL",
                "rf": "RF Phishing %",
                "xgb": "XGBoost Phishing %"
            }
        )

        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True
        )

        selected = st.selectbox(
            "Select disagreement sample for additional investigation",
            table["URL"].tolist()
        )

        if st.button(
            "🧠 INVESTIGATE SELECTED SAMPLE",
            type="primary"
        ):

            url = selected
            analyze_button = True

    else:

        st.warning(
            "No RF/XGBoost disagreement was found in the scanned dataset samples."
        )


# ============================================================
# MAIN ANALYSIS
# ============================================================

if (
    analyze_button
    and url
):

    if not url.strip():

        st.warning(
            "Please enter a URL."
        )

        st.stop()

    # --------------------------------------------------------
    # STEP 1 — RAW URL
    # --------------------------------------------------------

    with st.spinner(
        "Agent is selecting evidence..."
    ):

        raw = analyze_raw_url(
            url
        )

    # --------------------------------------------------------
    # STEP 2 — DATASET ML
    # --------------------------------------------------------

    with st.spinner(
        "Consulting PhiUSIIL ML models..."
    ):

        model_data = get_model_predictions(
            url
        )

    ml_result = model_data["result"]

    ml_probability = model_data[
        "ml_probability"
    ]

    rule_probability = model_data[
        "rule_probability"
    ]

    features = model_data[
        "features"
    ]

    # --------------------------------------------------------
    # STEP 3 — ADDITIONAL XAI
    # --------------------------------------------------------

    # XAI is requested automatically when evidence is
    # uncertain or conflicting.

    needs_xai = (
        raw["decision"] !=
        ml_result.get(
            "decision",
            "UNKNOWN"
        )
        or raw["score"] >= 35
        or rule_probability >= 0.35
    )

    with st.spinner(
        "Agent is checking whether additional XAI is required..."
    ):

        xai_result = explain_url(
            url
        )

    xai_probability = float(
        xai_result.get(
            "probability",
            ml_probability
        )
    )

    shap_values = xai_result.get(
        "shap_values",
        []
    )

    # --------------------------------------------------------
    # STEP 4 — FUSION
    # --------------------------------------------------------

    decision, risk, conflict, reasoning = fuse_evidence(
        ml_probability,
        rule_probability,
        raw["score"],
        xai_probability
    )

    risk = max(
        0,
        min(
            100,
            risk
        )
    )

    # ========================================================
    # AGENT STATUS
    # ========================================================

    st.divider()

    st.subheader(
        "🤖 Agent Investigation"
    )

    steps = [
        ("URL received", True),
        ("Raw URL structure analyzed", True),
        ("PhiUSIIL ML consulted", True),
        ("Rule evidence analyzed", True),
        ("Evidence consistency checked", True),
        (
            "Additional XAI requested",
            needs_xai
        ),
        (
            "Conflict detection",
            conflict
        ),
        ("Evidence fusion", True),
        ("Final decision generated", True)
    ]

    for step, status in steps:

        if status:
            st.write(
                f"✓ {step}"
            )
        else:
            st.write(
                f"○ {step}"
            )

    # ========================================================
    # FINAL DECISION
    # ========================================================

    st.divider()

    st.subheader(
        "🎯 FINAL AGENT DECISION"
    )

    if decision == "PHISHING":

        st.error(
            "🚨 PHISHING"
        )

    elif decision == "SUSPICIOUS":

        st.warning(
            "⚠️ SUSPICIOUS"
        )

    else:

        st.success(
            "✅ LEGITIMATE"
        )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Agent Risk",
            f"{risk:.2f}%"
        )

    with c2:

        st.metric(
            "ML Phishing",
            f"{ml_probability * 100:.2f}%"
        )

    with c3:

        st.metric(
            "Raw URL Risk",
            f"{raw['score']:.2f}%"
        )

    with c4:

        st.metric(
            "SHAP Evidence",
            f"{xai_probability * 100:.2f}%"
        )

    # ========================================================
    # EVIDENCE TABLE
    # ========================================================

    st.divider()

    st.subheader(
        "🔬 Evidence Overview"
    )

    evidence_data = pd.DataFrame(
        [
            [
                "Link Structure",
                raw["score"],
                raw["decision"]
            ],
            [
                "Domain / URL Evidence",
                raw["score"],
                raw["decision"]
            ],
            [
                "Security Evidence",
                rule_probability * 100,
                (
                    "PHISHING"
                    if rule_probability >= 0.5
                    else "LEGITIMATE"
                )
            ],
            [
                "PhiUSIIL ML",
                ml_probability * 100,
                (
                    "PHISHING"
                    if ml_probability >= 0.5
                    else "LEGITIMATE"
                )
            ],
            [
                "SHAP Evidence",
                xai_probability * 100,
                (
                    "PHISHING"
                    if xai_probability >= 0.5
                    else "LEGITIMATE"
                )
            ]
        ],
        columns=[
            "Evidence Type",
            "Evidence Score",
            "Direction"
        ]
    )

    st.dataframe(
        evidence_data,
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    st.divider()

    st.subheader(
        "⚖️ Model Comparison"
    )

    model_cols = st.columns(2)

    with model_cols[0]:

        st.markdown(
            "### 🌲 Random Forest"
        )

        st.write(
            f"Phishing probability: "
            f"**{ml_probability * 100:.2f}%**"
        )

        st.write(
            "Decision: **"
            + (
                "PHISHING"
                if ml_probability >= 0.5
                else "LEGITIMATE"
            )
            + "**"
        )

    # Try XGBoost directly
    with model_cols[1]:

        try:

            import joblib

            xgb_path = os.path.join(
                "models",
                "xgboost.pkl"
            )

            feature_path = os.path.join(
                "models",
                "feature_names.pkl"
            )

            if (
                os.path.exists(xgb_path)
                and os.path.exists(feature_path)
            ):

                xgb_model = joblib.load(
                    xgb_path
                )

                feature_names = joblib.load(
                    feature_path
                )

                vector = []

                for feature in feature_names:

                    vector.append(
                        features.get(
                            feature,
                            0
                        )
                    )

                vector = np.array(
                    vector,
                    dtype=float
                ).reshape(
                    1,
                    -1
                )

                xgb_probability = float(
                    xgb_model.predict_proba(
                        vector
                    )[0][1]
                )

                st.markdown(
                    "### 🚀 XGBoost"
                )

                st.write(
                    f"Phishing probability: "
                    f"**{xgb_probability * 100:.2f}%**"
                )

                st.write(
                    "Decision: **"
                    + (
                        "PHISHING"
                        if xgb_probability >= 0.5
                        else "LEGITIMATE"
                    )
                    + "**"
                )

                if (
                    ml_probability >= 0.5
                    and xgb_probability < 0.5
                ) or (
                    ml_probability < 0.5
                    and xgb_probability >= 0.5
                ):

                    st.warning(
                        "⚠️ RF and XGBoost disagree."
                    )

                else:

                    st.success(
                        "✓ RF and XGBoost agree."
                    )

        except Exception as e:

            st.warning(
                "XGBoost comparison unavailable."
            )

    # ========================================================
    # RAW URL EVIDENCE
    # ========================================================

    st.divider()

    st.subheader(
        "🌐 Independent URL Analysis"
    )

    st.write(
        f"**Raw URL risk:** {raw['score']}/100"
    )

    st.write(
        f"**Raw URL decision:** {raw['decision']}"
    )

    for reason in raw["reasons"]:

        st.write(
            "•",
            reason
        )

    # ========================================================
    # EXISTING RULE EVIDENCE
    # ========================================================

    st.subheader(
        "📋 Rule-Based Evidence"
    )

    st.write(
        f"Rule probability: "
        f"**{rule_probability * 100:.2f}%**"
    )

    for reason in ml_result.get(
        "reasons",
        []
    ):

        st.write(
            "•",
            reason
        )

    # ========================================================
    # CONFLICT
    # ========================================================

    st.divider()

    st.subheader(
        "⚔️ Conflict Resolution"
    )

    ml_decision = (
        "PHISHING"
        if ml_probability >= 0.5
        else "LEGITIMATE"
    )

    if conflict:

        st.warning(
            f"""
            **CONFLICT DETECTED**

            Dataset ML → **{ml_decision}**

            Independent URL analysis →
            **{raw['decision']}**

            Rule evidence →
            **{"PHISHING" if rule_probability >= 0.5 else "LEGITIMATE"}**

            The agent therefore requested additional XAI
            evidence before producing the final decision.
            """
        )

    else:

        st.success(
            "Major evidence sources are broadly consistent."
        )

    # ========================================================
    # SHAP
    # ========================================================

    st.divider()

    st.subheader(
        "🔬 Additional XAI Evidence — SHAP"
    )

    if shap_values:

        shap_table = []

        for item in shap_values:

            feature = item.get(
                "feature",
                "Unknown"
            )

            value = item.get(
                "value",
                0
            )

            shap_value = float(
                item.get(
                    "shap",
                    0
                )
            )

            direction = (
                "PHISHING"
                if shap_value > 0
                else "LEGITIMATE"
            )

            shap_table.append(
                [
                    feature,
                    value,
                    shap_value,
                    direction
                ]
            )

        shap_df = pd.DataFrame(
            shap_table,
            columns=[
                "Feature",
                "Value",
                "SHAP",
                "Direction"
            ]
        )

        st.dataframe(
            shap_df,
            use_container_width=True,
            hide_index=True
        )

        # SHAP chart
        chart_df = shap_df.copy()

        chart_df = chart_df.sort_values(
            "SHAP"
        )

        st.bar_chart(
            chart_df.set_index(
                "Feature"
            )["SHAP"]
        )

    else:

        st.info(
            "No SHAP evidence available."
        )

    # ========================================================
    # DATASET FEATURES
    # ========================================================

    st.divider()

    st.subheader(
        "📊 PhiUSIIL Dataset Features"
    )

    if features:

        feature_df = pd.DataFrame(
            list(
                features.items()
            ),
            columns=[
                "Feature",
                "Value"
            ]
        )

        st.dataframe(
            feature_df,
            use_container_width=True,
            hide_index=True
        )

    # ========================================================
    # FINAL REASONING
    # ========================================================

    st.divider()

    st.subheader(
        "🧠 Agent Reasoning"
    )

    st.info(
        reasoning
    )

    # ========================================================
    # PIPELINE
    # ========================================================

    st.divider()

    st.subheader(
        "⚙️ Agentic Decision Pipeline"
    )

    pipeline = [
        "Receive URL",
        "Select relevant evidence",
        "Analyze raw URL structure",
        "Consult Random Forest",
        "Consult XGBoost",
        "Analyze rule-based evidence",
        "Compare model outputs",
        "Check evidence sufficiency",
        "Request additional SHAP evidence",
        "Detect conflicting evidence",
        "Fuse evidence",
        "Generate final decision",
        "Generate human-readable reasoning"
    ]

    for i, step in enumerate(
        pipeline,
        1
    ):

        st.write(
            f"**{i}.** {step} ✓"
        )