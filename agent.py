import pandas as pd
import joblib
import warnings

warnings.filterwarnings(
    "ignore",
    message="X has feature names, but DecisionTreeClassifier was fitted without feature names"
)

DATA_PATH = "data/PhiUSIIL_Phishing_URL_Dataset.csv"

df = pd.read_csv(DATA_PATH)


models = {
    "url": joblib.load("models/url_model.pkl"),
    "domain": joblib.load("models/domain_model.pkl"),
    "webpage": joblib.load("models/webpage_model.pkl"),
    "behavior": joblib.load("models/behavior_model.pkl"),
    "resources": joblib.load("models/resources_model.pkl"),
    "intent": joblib.load("models/intent_model.pkl")
}


groups = {
    "url": [
        "URLLength",
        "DomainLength",
        "IsDomainIP",
        "URLSimilarityIndex",
        "CharContinuationRate",
        "URLCharProb",
        "TLDLength",
        "NoOfSubDomain",
        "HasObfuscation",
        "NoOfObfuscatedChar",
        "ObfuscationRatio",
        "NoOfLettersInURL",
        "LetterRatioInURL",
        "NoOfDegitsInURL",
        "DegitRatioInURL",
        "NoOfEqualsInURL",
        "NoOfQMarkInURL",
        "NoOfAmpersandInURL",
        "NoOfOtherSpecialCharsInURL",
        "SpacialCharRatioInURL"
    ],

    "domain": [
        "TLDLegitimateProb",
        "TLDLength",
        "IsHTTPS",
        "DomainTitleMatchScore",
        "URLTitleMatchScore"
    ],

    "webpage": [
        "LineOfCode",
        "LargestLineLength",
        "HasTitle",
        "HasFavicon",
        "Robots",
        "IsResponsive",
        "HasDescription",
        "HasCopyrightInfo"
    ],

    "behavior": [
        "NoOfURLRedirect",
        "NoOfSelfRedirect",
        "NoOfPopup",
        "NoOfiFrame",
        "HasExternalFormSubmit",
        "HasSocialNet",
        "HasSubmitButton",
        "HasHiddenFields",
        "HasPasswordField"
    ],

    "resources": [
        "NoOfImage",
        "NoOfCSS",
        "NoOfJS",
        "NoOfSelfRef",
        "NoOfEmptyRef",
        "NoOfExternalRef"
    ],

    "intent": [
        "Bank",
        "Pay",
        "Crypto"
    ]
}


def get_phishing_score(model, row, features):

    X = pd.DataFrame(
        [[row[f] for f in features]],
        columns=features
    )

    probabilities = model.predict_proba(X)[0]

    # Dataset:
    # 0 = PHISHING
    # 1 = LEGITIMATE

    phishing_probability = probabilities[0]

    # Random Forest tree-vote smoothing
    n_trees = len(model.estimators_)

    phishing_votes = sum(
        tree.predict(X)[0] == 0
        for tree in model.estimators_
    )

    smoothed_score = (
        phishing_votes + 1
    ) / (
        n_trees + 2
    )

    return smoothed_score


def analyze(row):

    evidence = {}
    reasoning = []

    # ================================================
    # STEP 1: URL EVIDENCE
    # ================================================

    url_probability = get_phishing_score(
        models["url"],
        row,
        groups["url"]
    )

    evidence["url"] = url_probability

    print("\nURL EVIDENCE")
    print(
        "Phishing probability:",
        round(url_probability, 4)
    )

    if url_probability >= 0.80:

        reasoning.append(
            "URL evidence provided high-confidence phishing evidence."
        )

        return {
            "decision": "PHISHING",
            "probability": url_probability,
            "evidence": evidence,
            "reasoning": reasoning
        }

    if url_probability <= 0.20:

        reasoning.append(
            "URL evidence provided strong evidence that the URL is legitimate."
        )

        return {
            "decision": "LEGITIMATE",
            "probability": url_probability,
            "evidence": evidence,
            "reasoning": reasoning
        }


    # ================================================
    # STEP 2: DOMAIN EVIDENCE
    # ================================================

    reasoning.append(
        "URL evidence was uncertain, so the agent requested domain evidence."
    )

    domain_probability = get_phishing_score(
        models["domain"],
        row,
        groups["domain"]
    )

    evidence["domain"] = domain_probability

    print("\nDOMAIN EVIDENCE")
    print(
        "Phishing probability:",
        round(domain_probability, 4)
    )


    # ================================================
    # STEP 3: EVIDENCE FUSION
    # ================================================

    combined_probability = (
        url_probability * 0.55 +
        domain_probability * 0.45
    )

    print("\nCOMBINED EVIDENCE")
    print(
        "Phishing probability:",
        round(combined_probability, 4)
    )


    if combined_probability >= 0.80:

        reasoning.append(
            "URL and domain evidence jointly provided high-confidence phishing evidence."
        )

        return {
            "decision": "PHISHING",
            "probability": combined_probability,
            "evidence": evidence,
            "reasoning": reasoning
        }


    if combined_probability <= 0.20:

        reasoning.append(
            "URL and domain evidence jointly provided strong legitimate evidence."
        )

        return {
            "decision": "LEGITIMATE",
            "probability": combined_probability,
            "evidence": evidence,
            "reasoning": reasoning
        }


    # ================================================
    # STEP 4: ADDITIONAL WEBPAGE ANALYSIS
    # ================================================

    reasoning.append(
        "Combined evidence remained uncertain, so the agent triggered webpage analysis."
    )

    webpage_probability = get_phishing_score(
        models["webpage"],
        row,
        groups["webpage"]
    )

    evidence["webpage"] = webpage_probability

    print("\nWEBPAGE EVIDENCE")
    print(
        "Phishing probability:",
        round(webpage_probability, 4)
    )


    # ================================================
    # STEP 5: FINAL EVIDENCE FUSION
    # ================================================

    final_probability = (
        url_probability * 0.40 +
        domain_probability * 0.25 +
        webpage_probability * 0.35
    )

    print("\nFINAL FUSED PROBABILITY")
    print(
        round(final_probability, 4)
    )


    if final_probability >= 0.65:

        decision = "PHISHING"

    elif final_probability <= 0.35:

        decision = "LEGITIMATE"

    else:

        decision = "SUSPICIOUS"


    reasoning.append(
        "The agent fused URL, domain, and webpage evidence to reach the final decision."
    )

    return {
        "decision": decision,
        "probability": final_probability,
        "evidence": evidence,
        "reasoning": reasoning
    }


# ================================================
# DEMONSTRATION
# ================================================

print("\n" + "=" * 60)
print("AGENTIC PHISHING DETECTION")
print("=" * 60)


sample = df.sample(1, random_state=42).iloc[0]

print("\nURL:")
print(sample["URL"])


result = analyze(sample)


print("\n" + "=" * 60)
print("FINAL DECISION")
print("=" * 60)

print("Decision:", result["decision"])

print(
    "Risk Score:",
    round(result["probability"] * 100, 2),
    "%"
)


print("\nEvidence selected by agent:")

for key, value in result["evidence"].items():

    print(
        key,
        "->",
        round(value, 4)
    )


print("\nAgent reasoning:")

for reason in result["reasoning"]:

    print("-", reason)