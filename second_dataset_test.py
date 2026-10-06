import pandas as pd
import numpy as np

from scipy.io import arff

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

DATA_PATH = "data/Training Dataset.arff"

print("Loading dataset...")

data, meta = arff.loadarff(DATA_PATH)
df = pd.DataFrame(data)

print("Original shape:", df.shape)

# Convert byte strings to normal strings
for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].apply(
            lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
        )

print("\nTarget distribution:")
print(df["Result"].value_counts())

# Convert features to numeric
feature_columns = [c for c in df.columns if c != "Result"]

X = df[feature_columns].astype(float)

# Result:
# -1 = phishing
#  1 = legitimate
#
# Convert to:
# 1 = phishing
# 0 = legitimate

y = df["Result"].astype(int)

y = y.map({
    -1: 1,
     1: 0
})

print("\nFinal target distribution:")
print(y.value_counts())

# --------------------------------------------------
# TRAIN / TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))

# --------------------------------------------------
# RANDOM FOREST
# --------------------------------------------------

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# --------------------------------------------------
# PREDICTIONS
# --------------------------------------------------

predictions = model.predict(X_test)

probabilities = model.predict_proba(X_test)[:, 1]

# --------------------------------------------------
# METRICS
# --------------------------------------------------

print("\n" + "=" * 60)
print("RANDOM FOREST RESULTS")
print("=" * 60)

print("Accuracy :", round(
    accuracy_score(y_test, predictions), 6
))

print("Precision:", round(
    precision_score(y_test, predictions), 6
))

print("Recall   :", round(
    recall_score(y_test, predictions), 6
))

print("F1 Score :", round(
    f1_score(y_test, predictions), 6
))

print("ROC-AUC  :", round(
    roc_auc_score(y_test, probabilities), 6
))

# --------------------------------------------------
# PROBABILITY DISTRIBUTION
# --------------------------------------------------

print("\n" + "=" * 60)
print("PROBABILITY DISTRIBUTION")
print("=" * 60)

print("Minimum :", round(probabilities.min(), 6))
print("Maximum :", round(probabilities.max(), 6))
print("Mean    :", round(probabilities.mean(), 6))
print("Median  :", round(np.median(probabilities), 6))

print("\nProbability bins:")

bins = [
    0,
    0.01,
    0.05,
    0.10,
    0.25,
    0.50,
    0.75,
    0.90,
    0.95,
    0.99,
    1.0
]

counts = pd.cut(
    probabilities,
    bins=bins,
    include_lowest=True
).value_counts().sort_index()

print(counts)

# --------------------------------------------------
# FEATURE IMPORTANCE
# --------------------------------------------------

importance = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print("\n" + "=" * 60)
print("TOP FEATURES")
print("=" * 60)

print(importance.head(15).to_string(index=False))

# --------------------------------------------------
# SAVE MODEL
# --------------------------------------------------

import joblib

joblib.dump(
    model,
    "models/second_dataset_random_forest.pkl"
)

joblib.dump(
    feature_columns,
    "models/second_dataset_features.pkl"
)

print("\nModel saved.")