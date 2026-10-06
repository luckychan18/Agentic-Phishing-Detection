import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


DATA_PATH = "data/PhiUSIIL_Phishing_URL_Dataset.csv"

df = pd.read_csv(DATA_PATH)

y = df["label"]

groups = {
    "url": [
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
    ],

    "domain": [
        "TLDLegitimateProb",
        "TLDLength",
        "IsHTTPS",
        "DomainTitleMatchScore",
        "URLTitleMatchScore"
    ],

    "webpage": [
        "LineOfCode",
        "LargestLineLength",
        "HasTitle",
        "Title",
        "HasFavicon",
        "Robots",
        "IsResponsive",
        "HasDescription",
        "HasCopyrightInfo"
    ],

    "behavior": [
        "NoOfURLRedirect",
        "NoOfSelfRedirect",
        "NoOfPopup",
        "NoOfiFrame",
        "HasExternalFormSubmit",
        "HasSocialNet",
        "HasSubmitButton",
        "HasHiddenFields",
        "HasPasswordField"
    ],

    "resources": [
        "NoOfImage",
        "NoOfCSS",
        "NoOfJS",
        "NoOfSelfRef",
        "NoOfEmptyRef",
        "NoOfExternalRef"
    ],

    "intent": [
        "Bank",
        "Pay",
        "Crypto"
    ]
}


# Title is text, so remove it
groups["webpage"].remove("Title")


X_train, X_test, y_train, y_test = train_test_split(
    df,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


for group_name, features in groups.items():

    print("\n" + "=" * 50)
    print("EVIDENCE GROUP:", group_name)
    print("=" * 50)

    Xtr = X_train[features]
    Xte = X_test[features]

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    model.fit(Xtr, y_train)

    pred = model.predict(Xte)
    prob = model.predict_proba(Xte)[:, 1]

    print("Features:", len(features))
    print("Accuracy :", accuracy_score(y_test, pred))
    print("Precision:", precision_score(y_test, pred))
    print("Recall   :", recall_score(y_test, pred))
    print("F1 Score :", f1_score(y_test, pred))

    joblib.dump(
        model,
        f"models/{group_name}_model.pkl"
    )

print("\nAll evidence models saved.")