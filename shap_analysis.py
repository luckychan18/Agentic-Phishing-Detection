import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os

print("Loading model and data...")

model = joblib.load("models/url_model.pkl")
features = joblib.load("models/url_features.pkl")

df = pd.read_csv("data/PhiUSIIL_Phishing_URL_Dataset.csv")

X = df[features]

print("Features:", len(features))
print("Creating SHAP explanation...")

sample_size = min(1000, len(X))
X_sample = X.sample(sample_size, random_state=42)

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_sample)

if isinstance(shap_values, list):
    values = shap_values[1]
else:
    values = shap_values

if len(values.shape) == 3:
    values = values[:, :, 1]

importance = np.abs(values).mean(axis=0)

importance_df = pd.DataFrame({
    "feature": features,
    "mean_abs_shap": importance
})

importance_df = importance_df.sort_values(
    "mean_abs_shap",
    ascending=False
)

os.makedirs("models", exist_ok=True)

importance_df.to_csv(
    "models/url_shap_importance.csv",
    index=False
)

print("\nTOP SHAP FEATURES")
print("=" * 50)

for _, row in importance_df.head(10).iterrows():
    print(
        f"{row['feature']:<30} "
        f"{row['mean_abs_shap']:.6f}"
    )

plt.figure(figsize=(10, 7))

top = importance_df.head(15).sort_values(
    "mean_abs_shap"
)

plt.barh(
    top["feature"],
    top["mean_abs_shap"]
)

plt.xlabel("Mean Absolute SHAP Value")
plt.ylabel("Feature")
plt.title("Global SHAP Feature Importance - URL Model")

plt.tight_layout()

plt.savefig(
    "models/url_shap_summary.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()

print("\nSaved:")
print("models/url_shap_importance.csv")
print("models/url_shap_summary.png")