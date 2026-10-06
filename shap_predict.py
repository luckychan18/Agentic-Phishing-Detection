import joblib
import numpy as np
import pandas as pd
import shap

from src.url_analyzer import calculate_url_features


MODEL_PATH = "models/url_model.pkl"
FEATURES_PATH = "models/url_features.pkl"


model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURES_PATH)

explainer = shap.TreeExplainer(model)


def explain_url(url):

    features = calculate_url_features(url)

    row = pd.DataFrame([features])

    row = row[feature_names]

    prediction = model.predict(row)[0]

    probability = model.predict_proba(row)[0]

    phishing_probability = float(probability[1])

    shap_values = explainer.shap_values(row)

    if isinstance(shap_values, list):
        values = shap_values[1][0]
    else:
        values = shap_values[0]

        if len(values.shape) > 1:
            values = values[:, 1]

    explanation = pd.DataFrame({
        "feature": feature_names,
        "value": row.iloc[0].values,
        "shap": values
    })

    explanation["abs_shap"] = explanation["shap"].abs()

    explanation = explanation.sort_values(
        "abs_shap",
        ascending=False
    )

    print("\n" + "=" * 70)
    print("SHAP EXPLANATION")
    print("=" * 70)

    print("URL:", url)

    if prediction == 1:
        print("MODEL DECISION: PHISHING")
    else:
        print("MODEL DECISION: LEGITIMATE")

    print(
        f"ML phishing probability: "
        f"{phishing_probability * 100:.2f}%"
    )

    print("\nTOP CONTRIBUTING FEATURES")
    print("-" * 70)

    for _, item in explanation.head(10).iterrows():

        direction = (
            "→ PHISHING"
            if item["shap"] > 0
            else "→ LEGITIMATE"
        )

        print(
            f"{item['feature']:<30} "
            f"value={item['value']:<12} "
            f"SHAP={item['shap']:+.5f} "
            f"{direction}"
        )

    print("=" * 70)

    return explanation


while True:

    url = input(
        "\nEnter URL (or type 'exit'): "
    ).strip()

    if url.lower() == "exit":
        break

    if not url:
        print("Please enter a URL.")
        continue

    print("\nAnalyzing URL...")

    try:
        explain_url(url)

    except Exception as e:
        print("\nError:", e)