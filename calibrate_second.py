import pandas as pd
import numpy as np
from scipy.io import arff

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score
)

import joblib

DATA_PATH = "data/Training Dataset.arff"

data, meta = arff.loadarff(DATA_PATH)
df = pd.DataFrame(data)

for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].apply(
            lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
        )

X = df.drop(columns=["Result"]).astype(float)

y = df["Result"].astype(int)

# -1 = phishing
#  1 = legitimate
#
# Convert:
# 1 = phishing
# 0 = legitimate

y = y.map({
    -1: 1,
     1: 0
})

# ------------------------------------------------
# TRAIN / CALIBRATION / TEST
# ------------------------------------------------

X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_train, X_cal, y_train, y_cal = train_test_split(
    X_temp,
    y_temp,
    test_size=0.25,
    random_state=42,
    stratify=y_temp
)

print("Training samples   :", len(X_train))
print("Calibration samples:", len(X_cal))
print("Testing samples    :", len(X_test))

# ------------------------------------------------
# RANDOM FOREST
# ------------------------------------------------

print("\nTraining Random Forest...")

rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# Raw probabilities
raw_cal_prob = rf.predict_proba(X_cal)[:, 1]
raw_test_prob = rf.predict_proba(X_test)[:, 1]

# ------------------------------------------------
# PLATT SCALING
# ------------------------------------------------

print("Calibrating probabilities...")

calibrator = LogisticRegression()

calibrator.fit(
    raw_cal_prob.reshape(-1, 1),
    y_cal
)

calibrated_prob = calibrator.predict_proba(
    raw_test_prob.reshape(-1, 1)
)[:, 1]

# ------------------------------------------------
# RESULTS
# ------------------------------------------------

print("\n" + "=" * 60)
print("CALIBRATION RESULTS")
print("=" * 60)

print("\nRAW MODEL")

print("Brier Score :", round(
    brier_score_loss(y_test, raw_test_prob), 6
))

print("Log Loss    :", round(
    log_loss(y_test, raw_test_prob), 6
))

print("ROC-AUC     :", round(
    roc_auc_score(y_test, raw_test_prob), 6
))

print("\nCALIBRATED MODEL")

print("Brier Score :", round(
    brier_score_loss(y_test, calibrated_prob), 6
))

print("Log Loss    :", round(
    log_loss(y_test, calibrated_prob), 6
))

print("ROC-AUC     :", round(
    roc_auc_score(y_test, calibrated_prob), 6
))

# ------------------------------------------------
# PROBABILITY DISTRIBUTION
# ------------------------------------------------

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

print("\n" + "=" * 60)
print("RAW PROBABILITY DISTRIBUTION")
print("=" * 60)

print(
    pd.cut(
        raw_test_prob,
        bins=bins,
        include_lowest=True
    ).value_counts().sort_index()
)

print("\n" + "=" * 60)
print("CALIBRATED PROBABILITY DISTRIBUTION")
print("=" * 60)

print(
    pd.cut(
        calibrated_prob,
        bins=bins,
        include_lowest=True
    ).value_counts().sort_index()
)

print("\nRaw median       :", round(np.median(raw_test_prob), 6))
print(
    "Calibrated median:",
    round(np.median(calibrated_prob), 6)
)

# ------------------------------------------------
# EXAMPLE PROBABILITIES
# ------------------------------------------------

print("\n" + "=" * 60)
print("EXAMPLE PROBABILITY CHANGES")
print("=" * 60)

for raw, calibrated in zip(
    raw_test_prob[:20],
    calibrated_prob[:20]
):
    print(
        f"{raw:.4f}  ->  {calibrated:.4f}"
    )

# ------------------------------------------------
# SAVE MODELS
# ------------------------------------------------

joblib.dump(
    rf,
    "models/second_dataset_random_forest.pkl"
)

joblib.dump(
    calibrator,
    "models/second_dataset_calibrator.pkl"
)

joblib.dump(
    list(X.columns),
    "models/second_dataset_features.pkl"
)

print("\nRandom Forest saved.")
print("Calibrator saved.")
print("Features saved.")