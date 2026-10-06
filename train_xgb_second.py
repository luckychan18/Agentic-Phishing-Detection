import pandas as pd
import numpy as np
from scipy.io import arff

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from xgboost import XGBClassifier
import joblib


DATA_PATH = "data/Training Dataset.arff"


print("Loading dataset...")

data, meta = arff.loadarff(DATA_PATH)
df = pd.DataFrame(data)

# Convert byte strings
for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].apply(
            lambda x: x.decode("utf-8")
            if isinstance(x, bytes)
            else x
        )

# --------------------------------------------------
# FEATURES AND TARGET
# --------------------------------------------------

X = df.drop(columns=["Result"]).astype(float)

y = df["Result"].astype(int)

# Original dataset:
# -1 = phishing
#  1 = legitimate
#
# Our convention:
# 1 = phishing
# 0 = legitimate

y = y.map({
    -1: 1,
     1: 0
})

print("Dataset shape:", X.shape)

print("\nTarget distribution:")
print(y.value_counts())

# --------------------------------------------------
# SAME TRAIN / TEST SPLIT
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
# XGBOOST
# --------------------------------------------------

print("\nTraining XGBoost...")

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)

print("Training completed.")

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

predictions = model.predict(X_test)

probabilities = model.predict_proba(X_test)[:, 1]

# --------------------------------------------------
# METRICS
# --------------------------------------------------

print("\n" + "=" * 60)
print("XGBOOST RESULTS")
print("=" * 60)

print(
    "Accuracy :",
    round(accuracy_score(y_test, predictions), 6)
)

print(
    "Precision:",
    round(precision_score(y_test, predictions), 6)
)

print(
    "Recall   :",
    round(recall_score(y_test, predictions), 6)
)

print(
    "F1 Score :",
    round(f1_score(y_test, predictions), 6)
)

print(
    "ROC-AUC  :",
    round(roc_auc_score(y_test, probabilities), 6)
)

# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        predictions
    )
)

# --------------------------------------------------
# PROBABILITY DISTRIBUTION
# --------------------------------------------------

print("\n" + "=" * 60)
print("XGBOOST PROBABILITY DISTRIBUTION")
print("=" * 60)

print(
    "Minimum:",
    round(probabilities.min(), 6)
)

print(
    "Maximum:",
    round(probabilities.max(), 6)
)

print(
    "Mean   :",
    round(probabilities.mean(), 6)
)

print(
    "Median :",
    round(np.median(probabilities), 6)
)

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

print("\nProbability bins:")

print(
    pd.cut(
        probabilities,
        bins=bins,
        include_lowest=True
    ).value_counts().sort_index()
)

# --------------------------------------------------
# FEATURE IMPORTANCE
# --------------------------------------------------

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print("\n" + "=" * 60)
print("TOP XGBOOST FEATURES")
print("=" * 60)

print(
    importance.head(15).to_string(index=False)
)

importance.to_csv(
    "models/second_dataset_xgb_importance.csv",
    index=False
)

# --------------------------------------------------
# SAVE MODEL
# --------------------------------------------------

joblib.dump(
    model,
    "models/second_dataset_xgboost.pkl"
)

joblib.dump(
    list(X.columns),
    "models/second_dataset_xgboost_features.pkl"
)

print("\nXGBoost model saved:")
print("models/second_dataset_xgboost.pkl")

print("\nDone.")