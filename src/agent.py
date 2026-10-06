from .url_predict import predict_url
from .xai import explain_url


def get_value(data, *keys, default=0):
    for key in keys:
        if key in data:
            return data[key]
    return default


def calculate_xai_score(xai_result):
    direct_score = get_value(
        xai_result,
        "phishing_score",
        "xai_phishing_score",
        "risk_score",
        default=None
    )

    if direct_score is not None:
        score = float(direct_score)
        if score > 1:
            score /= 100
        return max(0.0, min(1.0, score))

    shap_values = xai_result.get("shap_values", [])

    if not shap_values:
        return 0.5

    positive = 0.0
    negative = 0.0

    for item in shap_values:
        value = float(item.get("shap", 0))

        if value > 0:
            positive += value
        else:
            negative += abs(value)

    total = positive + negative

    if total == 0:
        return 0.5

    return positive / total


def classify_evidence(ml, rule, xai):

    # Strong ML evidence
    if ml >= 0.80:
        return "PHISHING"

    # Strong agreement between independent sources
    if ml >= 0.60 and rule >= 0.50:
        return "PHISHING"

    if ml >= 0.60 and xai >= 0.50:
        return "PHISHING"

    # Strong rule evidence by itself
    if rule >= 0.80:
        return "PHISHING"

    # Clear legitimate case:
    # all major evidence sources agree
    if (
        ml <= 0.25
        and rule < 0.50
        and xai <= 0.25
    ):
        return "LEGITIMATE"

    # Conflicting evidence
    if (
        ml <= 0.25
        and rule >= 0.50
    ):
        return "SUSPICIOUS"

    if (
        xai <= 0.25
        and rule >= 0.50
    ):
        return "SUSPICIOUS"

    if (
        ml >= 0.50
        and xai <= 0.25
    ):
        return "SUSPICIOUS"

    # Moderate phishing evidence
    if (
        ml >= 0.50
        or rule >= 0.60
        or xai >= 0.50
    ):
        return "SUSPICIOUS"

    return "LEGITIMATE"


def calculate_risk(ml, rule, xai, decision):

    fused = (
        0.50 * ml
        + 0.30 * rule
        + 0.20 * xai
    )

    if decision == "PHISHING":
        risk = max(fused, 0.70)

    elif decision == "LEGITIMATE":
        risk = min(fused, 0.25)

    else:
        # Suspicious cases should remain in the uncertainty zone
        risk = max(0.35, min(0.65, fused))

    return risk


def get_xai_evidence(xai_result):
    shap_values = xai_result.get("shap_values", [])

    phishing_support = []
    legitimate_support = []

    for item in shap_values:
        feature = item.get("feature", "Unknown")
        value = item.get("value", 0)
        shap = float(item.get("shap", 0))

        evidence = {
            "feature": feature,
            "value": value,
            "shap": shap
        }

        if shap > 0:
            phishing_support.append(evidence)
        elif shap < 0:
            legitimate_support.append(evidence)

    phishing_support.sort(
        key=lambda x: abs(x["shap"]),
        reverse=True
    )

    legitimate_support.sort(
        key=lambda x: abs(x["shap"]),
        reverse=True
    )

    return phishing_support[:5], legitimate_support[:5]


def generate_reasoning(ml, rule, xai, decision, conflict):

    if decision == "PHISHING":

        if conflict:
            return (
                "The evidence sources initially disagreed. "
                "The agent performed additional XAI analysis and "
                "found sufficient phishing-supporting evidence. "
                "The final decision is PHISHING."
            )

        return (
            "Multiple independent evidence sources provide strong "
            "phishing support. The agent therefore classified the "
            "URL as PHISHING."
        )

    if decision == "LEGITIMATE":

        return (
            "The machine-learning, rule-based and XAI evidence "
            "consistently indicate low phishing risk. The agent "
            "therefore classified the URL as LEGITIMATE."
        )

    return (
        "The evidence sources are conflicting. Machine-learning "
        "and XAI evidence lean toward legitimate, while rule-based "
        "analysis identifies suspicious URL characteristics. "
        "Because the evidence is not sufficiently consistent, "
        "the agent classified the URL as SUSPICIOUS."
    )


def analyze_url(url):
    print("\n" + "=" * 70)
    print("AGENTIC PHISHING DETECTION")
    print("=" * 70)

    print("\n[AGENT] Receiving URL...")
    print("[AGENT] Selecting URL evidence...")

    prediction = predict_url(url)

    ml = float(
        get_value(
            prediction,
            "phishing_probability",
            "ml_phishing_probability",
            default=0
        )
    )

    if ml > 1:
        ml /= 100

    rule = float(
        get_value(
            prediction,
            "rule_probability",
            "rule_prob",
            default=0
        )
    )

    if rule > 1:
        rule /= 100

    print("[AGENT] URL evidence analyzed.")
    print(f"[AGENT] ML phishing probability: {ml * 100:.2f}%")
    print(f"[AGENT] Rule probability: {rule * 100:.2f}%")

    conflict = abs(ml - rule) >= 0.30

    print("\n[AGENT] Checking whether evidence is sufficient...")

    if conflict:
        print("[AGENT] Evidence conflict detected.")
        print("[AGENT] Additional investigation required.")
        print("[AGENT] Requesting XAI evidence...")
        xai_result = explain_url(url)
        print("[AGENT] XAI evidence generated.")
    else:
        print("[AGENT] Initial evidence is sufficiently consistent.")
        print("[AGENT] Requesting XAI evidence for explainability...")
        xai_result = explain_url(url)
        print("[AGENT] XAI evidence generated.")

    xai = calculate_xai_score(xai_result)

    print(f"[AGENT] XAI phishing score: {xai * 100:.2f}%")

    print("\n[AGENT] Performing multi-source evidence fusion...")

    decision = classify_evidence(
        ml,
        rule,
        xai
    )

    risk = calculate_risk(
        ml,
        rule,
        xai,
        decision
    )

    phishing_xai, legitimate_xai = get_xai_evidence(
        xai_result
    )

    reasoning = generate_reasoning(
        ml,
        rule,
        xai,
        decision,
        conflict
    )

    print("[AGENT] Evidence fusion completed.")

    print("\n" + "=" * 70)
    print("FINAL AGENT DECISION")
    print("=" * 70)

    print(f"URL                    : {url}")
    print(f"Decision               : {decision}")
    print(f"ML phishing probability: {ml * 100:.2f}%")
    print(f"Rule probability       : {rule * 100:.2f}%")
    print(f"XAI phishing score     : {xai * 100:.2f}%")
    print(f"Risk score             : {risk * 100:.2f}%")

    if conflict:
        print("Evidence status        : CONFLICT RESOLVED")
        print("Decision basis         : ML + RULE + XAI EVIDENCE FUSION")
    else:
        print("Evidence status        : CONSISTENT")
        print("Decision basis         : MULTI-SOURCE EVIDENCE AGREEMENT")

    print("\nEVIDENCE USED")
    print("-" * 70)

    print("✓ URL structural evidence")
    print("✓ Machine-learning prediction")
    print("✓ Rule-based evidence")
    print("✓ SHAP explainability evidence")

    if conflict:
        print("✓ Conflict detection")
        print("✓ Additional XAI investigation")

    if phishing_xai:
        print("\nPHISHING-SUPPORTING XAI EVIDENCE")
        print("-" * 70)

        for item in phishing_xai:
            print(
                f"• {item['feature']} "
                f"(value={item['value']}, "
                f"SHAP={item['shap']:.4f})"
            )

    if legitimate_xai:
        print("\nLEGITIMATE-SUPPORTING XAI EVIDENCE")
        print("-" * 70)

        for item in legitimate_xai:
            print(
                f"• {item['feature']} "
                f"(value={item['value']}, "
                f"SHAP={item['shap']:.4f})"
            )

    print("\nAGENT REASONING")
    print("-" * 70)
    print(reasoning)

    print("=" * 70)

    return {
        "url": url,
        "decision": decision,
        "ml_phishing_probability": ml,
        "rule_probability": rule,
        "xai_phishing_score": xai,
        "risk_score": risk,
        "evidence_conflict": conflict,
        "phishing_xai_evidence": phishing_xai,
        "legitimate_xai_evidence": legitimate_xai,
        "reasoning": reasoning
    }


def main():
    while True:
        url = input(
            "\nEnter URL (or type 'exit'): "
        ).strip()

        if url.lower() == "exit":
            break

        if not url:
            print("Please enter a URL.")
            continue

        try:
            print("\nAnalyzing URL...")
            analyze_url(url)

        except Exception as e:
            print("\n[AGENT ERROR]")
            print(str(e))


if __name__ == "__main__":
    main()