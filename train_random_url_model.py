import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


DATA_PATH = "data/PhiUSIIL_Phishing_URL_Dataset.csv"

MODEL_PATH = "models/random_url_model.pkl"
FEATURE_PATH = "models/random_url_features.pkl"


# These features can be calculated directly from a URL.
# We intentionally exclude:
# URLSimilarityIndex
# URLCharProb

FEATURES = [
    "URLLength",
    "DomainLength",
    "IsDomainIP",
    "CharContinuationRate",
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


print("=" * 70)
print("TRAINING RANDOM URL MODEL")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)

X = df[FEATURES].copy()
y = df["label"].copy()

print("\nFeatures used:")
for feature in FEATURES:
    print(" -", feature)

print("\nLabel distribution:")
print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

model.fit(X_train, y_train)

print("\nEvaluating...")

pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 0]

accuracy = accuracy_score(y_test, pred)
precision = precision_score(y_test, pred, pos_label=0)
recall = recall_score(y_test, pred, pos_label=0)
f1 = f1_score(y_test, pred, pos_label=0)
auc = roc_auc_score(y_test, prob)

print("\n" + "=" * 70)
print("RANDOM URL MODEL RESULTS")
print("=" * 70)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {auc:.4f}")

print("\nSaving model...")

joblib.dump(model, MODEL_PATH)
joblib.dump(FEATURES, FEATURE_PATH)

print("\nSaved:")
print(MODEL_PATH)
print(FEATURE_PATH)

print("\nDone.")