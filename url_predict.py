
import joblib
import pandas as pd
import re

from src.url_analyzer import calculate_url_features

MODEL_PATH = "models/url_model.pkl"
FEATURE_PATH = "models/url_features.pkl"

model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURE_PATH)


SUSPICIOUS_KEYWORDS = [
    "login",
    "signin",
    "verify",
    "verification",
    "secure",
    "security",
    "account",
    "update",
    "confirm",
    "password",
    "credential",
    "wallet",
    "bank",
    "payment",
    "paypal",
    "microsoft",
    "appleid",
    "recover",
    "unlock"
]


def calculate_rule_score(url):

    score = 0
    reasons = []

    url_lower = url.lower()

    if not url_lower.startswith("https://"):
        score += 15
        reasons.append("Connection does not use HTTPS")

    keyword_count = sum(
        1 for keyword in SUSPICIOUS_KEYWORDS
        if keyword in url_lower
    )

    if keyword_count >= 2:
        score += 25
        reasons.append(
            "Multiple credential/security-related keywords"
        )

    elif keyword_count == 1:
        score += 10
        reasons.append(
            "Credential/security-related keyword detected"
        )

    hyphen_count = url.count("-")

    if hyphen_count >= 2:
        score += 15
        reasons.append("Multiple hyphens in URL")

    if url.count("@") > 0:
        score += 15
        reasons.append("@ symbol detected in URL")

    if url.count("=") >= 2:
        score += 10
        reasons.append("Multiple '=' parameters detected")

    if url.count("?") >= 2:
        score += 10
        reasons.append("Multiple query markers detected")

    if len(url) > 100:
        score += 10
        reasons.append("Unusually long URL")

    if re.search(
        r"https?://\d{1,3}(\.\d{1,3}){3}",
        url
    ):
        score += 25
        reasons.append(
            "URL uses an IP address instead of a domain"
        )

    return min(score, 100), reasons


def predict_url(url):

    url = url.strip()

    if not url:

        return {
            "decision": "INVALID",
            "risk": 0,
            "ml_phishing_probability": 0,
            "ml_legitimate_probability": 0,
            "rule_probability": 0,
            "rule_score": 0,
            "reasons": ["No URL entered"]
        }

    features = calculate_url_features(url)

    X = pd.DataFrame(
        [[features[name] for name in feature_names]],
        columns=feature_names
    )

    ml_probability = float(
        model.predict_proba(X)[0][1]
    )

    ml_legitimate = 1 - ml_probability

    rule_score, reasons = calculate_rule_score(url)

    rule_probability = rule_score / 100.0

    url_lower = url.lower()

    keyword_count = sum(
        1 for keyword in SUSPICIOUS_KEYWORDS
        if keyword in url_lower
    )

    hyphen_count = url.count("-")

    # -------------------------------------------------
    # AGENTIC EVIDENCE FUSION
    # -------------------------------------------------

    conflict = (
        ml_probability >= 0.80
        and rule_probability <= 0.15
    ) or (
        ml_probability <= 0.20
        and rule_probability >= 0.50
    )

    if conflict:

        final_risk = (
            0.25 * ml_probability
            + 0.75 * rule_probability
        )

        decision_basis = "MODEL-RULE CONFLICT RESOLVED"

    else:

        final_risk = (
            0.70 * ml_probability
            + 0.30 * rule_probability
        )

        decision_basis = "ML + RULE EVIDENCE AGREEMENT"

    # -------------------------------------------------
    # BENIGN STRUCTURAL EVIDENCE
    # -------------------------------------------------

    benign_structure = (
        features.get("IsDomainIP", 0) == 0
        and features.get("HasObfuscation", 0) == 0
        and features.get("NoOfObfuscatedChar", 0) == 0
        and features.get("NoOfQMarkInURL", 0) <= 1
        and features.get("NoOfAmpersandInURL", 0) <= 1
        and features.get("NoOfEqualsInURL", 0) <= 1
        and url_lower.startswith("https://")
        and rule_score <= 10
    )

    if conflict and benign_structure:

        final_risk *= 0.35

        decision_basis = "CONFLICT + BENIGN STRUCTURE"

    # -------------------------------------------------
    # STRONG PHISHING STRUCTURAL EVIDENCE
    # -------------------------------------------------

    strong_phishing_pattern = (
        rule_score >= 50
        and keyword_count >= 2
        and hyphen_count >= 2
        and not url_lower.startswith("https://")
    )

    if strong_phishing_pattern:

        final_risk = max(final_risk, 0.80)

        decision = "PHISHING"

        decision_basis = (
            "MULTIPLE INDEPENDENT PHISHING INDICATORS"
        )

    elif final_risk >= 0.65:

        decision = "PHISHING"

    elif final_risk >= 0.35:

        decision = "SUSPICIOUS"

    else:

        decision = "LEGITIMATE"

    if not reasons:

        reasons = [
            "No major suspicious URL patterns detected"
        ]

    return {
        "decision": decision,
        "risk": round(final_risk, 4),
        "ml_phishing_probability": round(
            ml_probability, 4
        ),
        "ml_legitimate_probability": round(
            ml_legitimate, 4
        ),
        "rule_probability": round(
            rule_probability, 4
        ),
        "rule_score": rule_score,
        "reasons": reasons,
        "features": features,
        "decision_basis": decision_basis
    }


def print_result(url, result):

    print()
    print("=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    print(f"URL                     : {url}")

    print(
        f"Decision                : "
        f"{result['decision']}"
    )

    print(
        f"Risk score              : "
        f"{result['risk'] * 100:.2f}%"
    )

    print(
        f"ML phishing probability : "
        f"{result['ml_phishing_probability'] * 100:.2f}%"
    )

    print(
        f"ML legitimate probability: "
        f"{result['ml_legitimate_probability'] * 100:.2f}%"
    )

    print(
        f"Rule probability        : "
        f"{result['rule_probability'] * 100:.2f}%"
    )

    print(
        f"Rule score              : "
        f"{result['rule_score']}"
    )

    print(
        f"Decision basis          : "
        f"{result['decision_basis']}"
    )

    print()
    print("REASONS")
    print("-" * 70)

    for reason in result["reasons"]:
        print(f"• {reason}")

    print()
    print("URL FEATURES")
    print("-" * 70)

    for name in feature_names:

        print(
            f"{name:<30}: "
            f"{result['features'][name]}"
        )

    print("=" * 70)


if __name__ == "__main__":

    print("=" * 70)
    print("URL PHISHING DETECTOR")
    print("=" * 70)

    while True:

        url = input(
            "\nEnter URL (or type 'exit'): "
        ).strip()

        if url.lower() == "exit":
            break

        print("\nAnalyzing URL...")

        result = predict_url(url)

        print_result(url, result)