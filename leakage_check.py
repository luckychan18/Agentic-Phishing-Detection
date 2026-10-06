import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


DATA_PATH = "data/PhiUSIIL_Phishing_URL_Dataset.csv"

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("DATA LEAKAGE CHECK")
print("=" * 60)

print("\nDataset shape:")
print(df.shape)

print("\nTarget distribution:")
print(df["label"].value_counts())


# ---------------------------------------------------------
# 1. Check whether any feature is exactly equal to label
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("1. FEATURES IDENTICAL TO LABEL")
print("=" * 60)

for col in df.columns:

    if col == "label":
        continue

    if pd.api.types.is_numeric_dtype(df[col]):

        match = (df[col] == df["label"]).mean()

        if match == 1.0:
            print("EXACT MATCH:", col)

        elif match > 0.99:
            print(
                "VERY SUSPICIOUS:",
                col,
                "match =",
                round(match, 5)
            )


# ---------------------------------------------------------
# 2. Check perfect / near-perfect separation
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("2. SINGLE FEATURE SEPARATION")
print("=" * 60)

numeric_cols = df.select_dtypes(
    include=np.number
).columns.tolist()

numeric_cols.remove("label")

for col in numeric_cols:

    grouped = df.groupby("label")[col]

    try:

        means = grouped.mean()

        if len(means) == 2:

            a = means.iloc[0]
            b = means.iloc[1]

            if a != b:

                separation = abs(a - b) / (
                    abs(a) + abs(b) + 1e-9
                )

                if separation > 0.95:

                    print(
                        col,
                        "→ very strong separation",
                        round(separation, 4)
                    )

    except:
        pass


# ---------------------------------------------------------
# 3. Check whether categorical columns reveal label
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("3. CATEGORICAL LABEL LEAKAGE")
print("=" * 60)

categorical_cols = df.select_dtypes(
    include=["object"]
).columns.tolist()

for col in categorical_cols:

    if col in ["URL", "Domain", "FILENAME"]:
        continue

    try:

        table = pd.crosstab(
            df[col],
            df["label"]
        )

        if len(table) > 0:

            purity = table.max(axis=1) / table.sum(axis=1)

            high_purity = (purity > 0.999).mean()

            if high_purity > 0.1:

                print(
                    col,
                    "→",
                    round(high_purity * 100, 2),
                    "% of values have >99.9% label purity"
                )

    except:
        pass


# ---------------------------------------------------------
# 4. Check repeated domains across labels
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("4. DOMAIN REUSE")
print("=" * 60)

domain_labels = df.groupby("Domain")["label"].nunique()

mixed_domains = (
    domain_labels > 1
).sum()

single_label_domains = (
    domain_labels == 1
).sum()

print(
    "Domains appearing with BOTH labels:",
    mixed_domains
)

print(
    "Domains appearing with only ONE label:",
    single_label_domains
)

print(
    "Total unique domains:",
    domain_labels.shape[0]
)


# ---------------------------------------------------------
# 5. Check URL duplicates across labels
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("5. URL DUPLICATES")
print("=" * 60)

url_labels = df.groupby("URL")["label"].nunique()

mixed_urls = (
    url_labels > 1
).sum()

print(
    "URLs appearing with both labels:",
    mixed_urls
)

print(
    "Repeated URLs:",
    (df["URL"].duplicated()).sum()
)


# ---------------------------------------------------------
# 6. More realistic domain-group split
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("6. DOMAIN-BASED GENERALIZATION TEST")
print("=" * 60)

from sklearn.model_selection import GroupShuffleSplit

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
groups = df["Domain"].astype(str)

gss = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    gss.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

train_domains = set(
    groups.iloc[train_idx]
)

test_domains = set(
    groups.iloc[test_idx]
)

overlap = train_domains.intersection(
    test_domains
)

print(
    "Training rows:",
    len(X_train)
)

print(
    "Testing rows:",
    len(X_test)
)

print(
    "Training domains:",
    len(train_domains)
)

print(
    "Testing domains:",
    len(test_domains)
)

print(
    "Domain overlap:",
    len(overlap)
)


rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

print("\nTraining domain-split Random Forest...")

rf.fit(X_train, y_train)

pred = rf.predict(X_test)

accuracy = accuracy_score(
    y_test,
    pred
)

print(
    "\nDOMAIN-SPLIT ACCURACY:",
    round(accuracy, 5)
)