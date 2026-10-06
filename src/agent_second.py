import os
import joblib
import numpy as np
import pandas as pd
import shap

from scipy.io import arff


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

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
# LOAD DATA
# ============================================================

print("Loading dataset...")

data, meta = arff.loadarff(DATA_PATH)
df = pd.DataFrame(data)

for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].apply(
            lambda x: x.decode("utf-8")
            if isinstance(x, bytes)
            else x
        )

X = df.drop(columns=["Result"]).astype(float)

print("Dataset loaded:", X.shape)


# ============================================================
# LOAD MODELS
# ============================================================

print("Loading models...")

rf = joblib.load(RF_PATH)
xgb = joblib.load(XGB_PATH)
calibrator = joblib.load(CALIBRATOR_PATH)

print("Random Forest loaded.")
print("XGBoost loaded.")
print("Calibrator loaded.")


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

print("Creating SHAP explainer...")

explainer = shap.TreeExplainer(rf)

print("SHAP ready.")


# ============================================================
# SHAP FUNCTION
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
# EVIDENCE SELECTION USING SHAP
# ============================================================

def select_evidence(row, shap_values):

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
            if f in shap_df["Feature"].values
        ]

        if not available:
            scores[group] = 0
            continue

        group_values = shap_df[
            shap_df["Feature"].isin(available)
        ]["Absolute"]

        scores[group] = group_values.sum()

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked


# ============================================================
# FIND DISAGREEMENT SAMPLE
# ============================================================

def find_disagreement():

    rf_prob = rf.predict_proba(X)[:, 1]

    xgb_prob = xgb.predict_proba(X)[:, 1]

    rf_calibrated = calibrator.predict_proba(
        rf_prob.reshape(-1, 1)
    )[:, 1]

    rf_pred = rf_calibrated >= 0.5
    xgb_pred = xgb_prob >= 0.5

    disagreement = np.where(
        rf_pred != xgb_pred
    )[0]

    if len(disagreement) == 0:
        return None

    # Prefer largest disagreement
    differences = np.abs(
        rf_calibrated[disagreement]
        - xgb_prob[disagreement]
    )

    best = disagreement[
        np.argmax(differences)
    ]

    return int(best)


# ============================================================
# AGENT
# ============================================================

def analyze(row, index=None):

    print("\n")
    print("=" * 70)
    print("AGENTIC PHISHING DETECTION")
    print("=" * 70)

    if index is not None:
        print(f"\nAnalyzing sample #{index}")

    sample = row.to_frame().T

    # --------------------------------------------------------
    # MODEL EVIDENCE
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

    print("\nINITIAL MODEL EVIDENCE")
    print("-" * 70)

    print(
        f"Random Forest raw probability : "
        f"{rf_raw:.4f}"
    )

    print(
        f"Random Forest calibrated      : "
        f"{rf_calibrated:.4f}"
    )

    print(
        f"XGBoost probability           : "
        f"{xgb_probability:.4f}"
    )

    # --------------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------------

    rf_prediction = rf_calibrated >= 0.5
    xgb_prediction = xgb_probability >= 0.5

    disagreement = (
        rf_prediction != xgb_prediction
    )

    difference = abs(
        rf_calibrated - xgb_probability
    )

    average_probability = (
        rf_calibrated + xgb_probability
    ) / 2

    confidence = max(
        average_probability,
        1 - average_probability
    )

    print("\nMODEL COMPARISON")
    print("-" * 70)

    print(
        "RF prediction     :",
        "PHISHING"
        if rf_prediction
        else "LEGITIMATE"
    )

    print(
        "XGBoost prediction:",
        "PHISHING"
        if xgb_prediction
        else "LEGITIMATE"
    )

    print(
        f"Probability gap   : {difference:.4f}"
    )

    print(
        "Model agreement   :",
        "NO" if disagreement else "YES"
    )

    print("\nAGENT CONFIDENCE")
    print("-" * 70)

    print(
        f"Initial confidence: "
        f"{confidence:.4f}"
    )

    # --------------------------------------------------------
    # SHAP
    # --------------------------------------------------------

    shap_values = calculate_shap(row)

    # --------------------------------------------------------
    # DYNAMIC EVIDENCE SELECTION
    # --------------------------------------------------------

    ranked_evidence = select_evidence(
        row,
        shap_values
    )

    selected = ranked_evidence[:3]

    print("\nEVIDENCE SELECTED BY AGENT")
    print("-" * 70)

    for group, score in selected:

        print(
            f"✓ {group:<25}"
            f" SHAP evidence = {score:.4f}"
        )

    # --------------------------------------------------------
    # UNCERTAINTY / CONFLICT
    # --------------------------------------------------------

    uncertain = (
        disagreement
        or difference > 0.20
        or confidence < 0.70
    )

    if uncertain:

        print("\nAGENT ACTION")
        print("-" * 70)

        if disagreement:
            print(
                "⚠ Model conflict detected."
            )
        else:
            print(
                "⚠ Insufficient confidence detected."
            )

        print(
            "→ Triggering additional XAI analysis."
        )

        # ----------------------------------------------------
        # TOP SHAP FEATURES
        # ----------------------------------------------------

        shap_df = pd.DataFrame({
            "Feature": X.columns,
            "SHAP": shap_values
        })

        shap_df["Absolute"] = np.abs(
            shap_df["SHAP"]
        )

        shap_df = shap_df.sort_values(
            "Absolute",
            ascending=False
        )

        top_features = shap_df.head(5)

        print("\nADDITIONAL XAI EVIDENCE")
        print("-" * 70)

        for _, item in top_features.iterrows():

            if item["SHAP"] > 0:
                direction = "PHISHING"
            else:
                direction = "LEGITIMATE"

            print(
                f"{item['Feature']:<30}"
                f"{item['SHAP']:+.4f}"
                f" → {direction}"
            )

        # ----------------------------------------------------
        # XAI FUSION
        # ----------------------------------------------------

        positive = top_features[
            top_features["SHAP"] > 0
        ]["Absolute"].sum()

        negative = top_features[
            top_features["SHAP"] < 0
        ]["Absolute"].sum()

        if positive > negative:
            shap_signal = 1
        else:
            shap_signal = 0

        final_score = (
            0.70 * average_probability
            + 0.30 * shap_signal
        )

        reasoning = (
            "The agent detected uncertainty or "
            "model disagreement. It dynamically "
            "triggered SHAP analysis, treated the "
            "resulting explanations as additional "
            "evidence, and fused model and XAI evidence."
        )

    else:

        print("\nAGENT ACTION")
        print("-" * 70)

        print(
            "✓ Models agree and confidence is high."
        )

        print(
            "→ Additional analysis not required."
        )

        top_features = None

        final_score = average_probability

        reasoning = (
            "The agent found sufficient agreement "
            "between the predictive models and "
            "therefore avoided unnecessary additional analysis."
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

    print("\n")
    print("=" * 70)
    print("FINAL AGENT DECISION")
    print("=" * 70)

    print(
        f"\nDecision   : {decision}"
    )

    print(
        f"Risk score : {final_score * 100:.2f}%"
    )

    print(
        f"Confidence : {final_confidence * 100:.2f}%"
    )

    print("\nAgent reasoning:")
    print(reasoning)

    print("\nEvidence used:")

    for group, score in selected:

        print(
            f"  • {group}"
        )

    if top_features is not None:

        print("\nXAI evidence used:")

        for _, item in top_features.iterrows():

            print(
                f"  • {item['Feature']}: "
                f"{item['SHAP']:+.4f}"
            )

    print("\n" + "=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 70)
    print("INTELLIGENT AGENTIC PHISHING DETECTION")
    print("=" * 70)

    print("\n1. Analyze a specific sample")
    print("2. Automatically find a model disagreement")

    choice = input(
        "\nChoose option: "
    ).strip()

    if choice == "1":

        index = int(
            input(
                f"Enter row number (0-{len(X)-1}): "
            )
        )

        if index < 0 or index >= len(X):

            print("Invalid row.")
            exit()

        analyze(
            X.iloc[index],
            index
        )

    elif choice == "2":

        print(
            "\nSearching for model disagreement..."
        )

        index = find_disagreement()

        if index is None:

            print(
                "No disagreement found."
            )

        else:

            print(
                f"Disagreement found at row {index}"
            )

            analyze(
                X.iloc[index],
                index
            )

    else:

        print("Invalid choice.")