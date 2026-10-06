import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss


DATA_PATH = "data/PhiUSIIL_Phishing_URL_Dataset.csv"

df = pd.read_csv(DATA_PATH)

features = [
    "URLLength",
    "DomainLength",
    "IsDomainIP",
    "URLSimilarityIndex",
    "CharContinuationRate",
    "URLCharProb",
    "TLDLength",
    "NoOfSubDomain",
    "HasObfuscation",
    "NoOfObfuscatedChar",
    "ObfuscationRatio",
    "NoOfLettersInURL",
    "LetterRatioInURL",
    "NoOfDegitsInURL",
    "DegitRatioInURL",
    "NoOfEqualsInURL",
    "NoOfQMarkInURL",
    "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL",
    "SpacialCharRatioInURL"
]

X = df[features]
y = df["label"]


# ------------------------------------------------
# Create train / calibration / test sets
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
    test_size=0.1875,
    random_state=42,
    stratify=y_temp
)

print("Training samples:", len(X_train))
print("Calibration samples:", len(X_cal))
print("Test samples:", len(X_test))


# ------------------------------------------------
# Train Random Forest
# ------------------------------------------------

print("\nTraining Random Forest...")

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

rf.fit(X_train, y_train)


# ------------------------------------------------
# Raw probabilities
# ------------------------------------------------

raw_probability = rf.predict_proba(X_cal)[:, 0]


# ------------------------------------------------
# Calibrate probabilities
# ------------------------------------------------

print("\nCalibrating probabilities...")

try:
    from sklearn.frozen import FrozenEstimator

    calibrated = CalibratedClassifierCV(
        FrozenEstimator(rf),
        method="sigmoid"
    )

except ImportError:

    calibrated = CalibratedClassifierCV(
        rf,
        method="sigmoid",
        cv="prefit"
    )

calibrated.fit(X_cal, y_cal)


# ------------------------------------------------
# Compare raw vs calibrated
# ------------------------------------------------

calibrated_probability = calibrated.predict_proba(X_cal)[:, 0]

raw_brier = brier_score_loss(
    1 - y_cal,
    raw_probability
)

calibrated_brier = brier_score_loss(
    1 - y_cal,
    calibrated_probability
)

print("\nCALIBRATION RESULTS")

print(
    "Raw Brier Score       :",
    round(raw_brier, 5)
)

print(
    "Calibrated Brier Score:",
    round(calibrated_brier, 5)
)


# ------------------------------------------------
# Test a few examples
# ------------------------------------------------

print("\nEXAMPLE PROBABILITIES")

for i in range(10):

    print(
        "Raw:",
        round(raw_probability[i], 4),
        "→ Calibrated:",
        round(calibrated_probability[i], 4)
    )


# ------------------------------------------------
# Save calibrated model
# ------------------------------------------------

joblib.dump(
    calibrated,
    "models/url_calibrated.pkl"
)

joblib.dump(
    features,
    "models/url_features.pkl"
)

print("\nCalibrated URL model saved:")
print("models/url_calibrated.pkl")