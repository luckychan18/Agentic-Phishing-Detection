import pandas as pd
import joblib
from scipy.io import arff

print("Loading dataset...")

data, meta = arff.loadarff("data/Training Dataset.arff")
df = pd.DataFrame(data)

df["Result"] = (
    df["Result"]
    .astype(str)
    .str.replace("b'", "", regex=False)
    .str.replace("'", "", regex=False)
)

df["Result"] = df["Result"].map({
    "1": 0,
    "-1": 1
})

rf = joblib.load("models/second_dataset_random_forest.pkl")
feature_names = joblib.load("models/second_dataset_features.pkl")

X = df[feature_names]

print("Dataset loaded:", X.shape)
print("\nSearching for clearly legitimate samples using Random Forest...\n")

rf_prob = rf.predict_proba(X)[:, 1]

found = 0

for i in range(len(df)):

    if rf_prob[i] < 0.10:

        print("=" * 60)
        print(f"Row       : {i}")
        print(f"RF        : {rf_prob[i]:.4f}")

        if df.iloc[i]["Result"] == 0:
            print("Actual    : LEGITIMATE")
        else:
            print("Actual    : PHISHING")

        found += 1

        if found == 10:
            break

print("\nFound", found, "candidates.")