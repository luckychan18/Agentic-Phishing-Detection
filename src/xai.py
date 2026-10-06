import os
import sys
import joblib
import pandas as pd
import shap
import numpy as np

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

MODEL_PATH = "models/url_model.pkl"
FEATURE_PATH = "models/url_features.pkl"

model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURE_PATH)


def explain_url(url):

    from src.url_analyzer import calculate_url_features

    feature_dict = calculate_url_features(url)

    X = pd.DataFrame(
        [[feature_dict[f] for f in features]],
        columns=features
    )

    probability = model.predict_proba(X)[0][1]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 3:
        values = shap_values[0, :, 1]

    elif shap_values.ndim == 2:
        values = shap_values[0]

    elif shap_values.ndim == 1:
        values = shap_values

    else:
        raise ValueError(
            f"Unexpected SHAP output shape: {shap_values.shape}"
        )

    results = []

    for feature, value, shap_value in zip(
        features,
        X.iloc[0].values,
        values
    ):
        results.append({
            "feature": feature,
            "value": float(value),
            "shap": float(shap_value)
        })

    results.sort(
        key=lambda x: abs(x["shap"]),
        reverse=True
    )

    return {
        "url": url,
        "prediction": (
            "PHISHING"
            if probability >= 0.5
            else "LEGITIMATE"
        ),
        "probability": float(probability),
        "shap_values": results
    }