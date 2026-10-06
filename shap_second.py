import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt

from scipy.io import arff

DATA_PATH = "data/Training Dataset.arff"

print("Loading dataset...")

data, meta = arff.loadarff(DATA_PATH)
df = pd.DataFrame(data)

for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].apply(
            lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
        )

X = df.drop(columns=["Result"]).astype(float)

print("Dataset shape:", X.shape)

# Load our trained Random Forest
model = joblib.load(
    "models/second_dataset_random_forest.pkl"
)

print("Random Forest loaded.")

# ------------------------------------------------
# SHAP SAMPLE
# ------------------------------------------------

sample_size = min(1000, len(X))

X_sample = X.sample(
    n=sample_size,
    random_state=42
)

print("SHAP sample size:", len(X_sample))

# ------------------------------------------------
# SHAP EXPLAINER
# ------------------------------------------------

print("\nCreating SHAP explainer...")

explainer = shap.TreeExplainer(model)

print("Calculating SHAP values...")

shap_values = explainer.shap_values(X_sample)

# Handle SHAP versions
if isinstance(shap_values, list):
    values = shap_values[1]
else:
    values = shap_values

# Handle 3D output
if len(values.shape) == 3:
    values = values[:, :, 1]

print("SHAP calculation completed.")
print("SHAP shape:", values.shape)

# ------------------------------------------------
# GLOBAL FEATURE IMPORTANCE
# ------------------------------------------------

importance = np.abs(values).mean(axis=0)

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Mean_ABS_SHAP": importance
})

importance_df = importance_df.sort_values(
    "Mean_ABS_SHAP",
    ascending=False
)

print("\n" + "=" * 60)
print("GLOBAL SHAP FEATURE IMPORTANCE")
print("=" * 60)

print(
    importance_df.to_string(index=False)
)

importance_df.to_csv(
    "models/second_dataset_shap_importance.csv",
    index=False
)

# ------------------------------------------------
# SHAP SUMMARY
# ------------------------------------------------

print("\nGenerating SHAP summary plot...")

shap.summary_plot(
    values,
    X_sample,
    show=False
)

plt.tight_layout()

plt.savefig(
    "models/second_dataset_shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved:")
print("models/second_dataset_shap_summary.png")

# ------------------------------------------------
# SHAP BAR
# ------------------------------------------------

print("\nGenerating SHAP bar plot...")

shap.summary_plot(
    values,
    X_sample,
    plot_type="bar",
    show=False
)

plt.tight_layout()

plt.savefig(
    "models/second_dataset_shap_bar.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved:")
print("models/second_dataset_shap_bar.png")

print("\nSHAP analysis completed.")