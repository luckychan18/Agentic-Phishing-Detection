import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

DATA_PATH = "data/PhiUSIIL_Phishing_URL_Dataset.csv"

df = pd.read_csv(DATA_PATH)

target = "label"

# Numeric URL-related features
url_features = [
    "URLLength",
    "DomainLength",
    "IsDomainIP",
    "URLSimilarityIndex",
    "CharContinuationRate",
    "TLDLegitimateProb",
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
    "SpacialCharRatioInURL",
    "IsHTTPS"
]

# Check that all columns exist
missing = [f for f in url_features if f not in df.columns]

if missing:
    print("Missing features:")
    print(missing)
    exit()

X = df[url_features].copy()
y = df[target]

# Same split for every experiment
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Features identified by leakage/separation analysis
strong_features = [
    "IsDomainIP",
    "HasObfuscation",
    "NoOfObfuscatedChar",
    "ObfuscationRatio",
    "NoOfEqualsInURL",
    "NoOfQMarkInURL",
    "NoOfAmpersandInURL"
]

def evaluate_model(name, features):

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train[features], y_train)

    predictions = model.predict(X_test[features])
    probabilities = model.predict_proba(X_test[features])[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    auc = roc_auc_score(y_test, probabilities)

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print("Number of features:", len(features))
    print("Accuracy :", round(accuracy, 6))
    print("Precision:", round(precision, 6))
    print("Recall   :", round(recall, 6))
    print("F1 Score :", round(f1, 6))
    print("ROC-AUC  :", round(auc, 6))

    print("\nProbability distribution:")
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

    return model, probabilities


# ---------------------------------------------------
# EXPERIMENT 1: FULL URL MODEL
# ---------------------------------------------------

full_model, full_prob = evaluate_model(
    "EXPERIMENT 1: FULL URL FEATURES",
    url_features
)


# ---------------------------------------------------
# EXPERIMENT 2: REDUCED URL MODEL
# Remove strongest separating features
# ---------------------------------------------------

reduced_features = [
    f for f in url_features
    if f not in strong_features
]

reduced_model, reduced_prob = evaluate_model(
    "EXPERIMENT 2: REDUCED URL FEATURES",
    reduced_features
)


# ---------------------------------------------------
# EXPERIMENT 3: EXTREME REDUCTION
# Keep only basic URL structure
# ---------------------------------------------------

basic_features = [
    "URLLength",
    "DomainLength",
    "TLDLength",
    "NoOfSubDomain",
    "NoOfLettersInURL",
    "LetterRatioInURL",
    "NoOfDegitsInURL",
    "DegitRatioInURL",
    "SpacialCharRatioInURL",
    "IsHTTPS"
]

basic_model, basic_prob = evaluate_model(
    "EXPERIMENT 3: BASIC URL FEATURES",
    basic_features
)


# ---------------------------------------------------
# COMPARISON
# ---------------------------------------------------

results = pd.DataFrame({
    "Model": [
        "Full URL",
        "Reduced URL",
        "Basic URL"
    ],
    "Features": [
        len(url_features),
        len(reduced_features),
        len(basic_features)
    ],
    "Accuracy": [
        accuracy_score(y_test, full_model.predict(X_test[url_features])),
        accuracy_score(y_test, reduced_model.predict(X_test[reduced_features])),
        accuracy_score(y_test, basic_model.predict(X_test[basic_features]))
    ],
    "ROC_AUC": [
        roc_auc_score(
            y_test,
            full_model.predict_proba(X_test[url_features])[:, 1]
        ),
        roc_auc_score(
            y_test,
            reduced_model.predict_proba(X_test[reduced_features])[:, 1]
        ),
        roc_auc_score(
            y_test,
            basic_model.predict_proba(X_test[basic_features])[:, 1]
        )
    ]
})

print("\n\n")
print("=" * 70)
print("FINAL ABLATION COMPARISON")
print("=" * 70)
print(results.to_string(index=False))

results.to_csv(
    "models/ablation_results.csv",
    index=False
)

print("\nResults saved to models/ablation_results.csv")