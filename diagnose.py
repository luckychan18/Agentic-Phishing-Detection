import pandas as pd
import numpy as np

df = pd.read_csv("data/PhiUSIIL_Phishing_URL_Dataset.csv")

print("DATASET SHAPE:", df.shape)

print("\n1. DUPLICATE ROWS")
print("Duplicate rows:", df.duplicated().sum())

print("\n2. DUPLICATE FEATURE ROWS")
X = df.drop(columns=["label"])
print("Duplicate feature rows:", X.duplicated().sum())

print("\n3. UNIQUE VALUES PER COLUMN")
for col in df.columns:
    print(f"{col}: {df[col].nunique()}")

print("\n4. NUMERIC CORRELATION WITH LABEL")

numeric_cols = df.select_dtypes(include=np.number).columns

correlations = (
    df[numeric_cols]
    .corr()["label"]
    .drop("label")
    .abs()
    .sort_values(ascending=False)
)

print(correlations.head(15))

print("\n5. FEATURES WITH PERFECT LABEL SEPARATION")

for col in numeric_cols:
    if col == "label":
        continue

    grouped = df.groupby(col)["label"].nunique()

    if grouped.max() == 1:
        print(col, "-> perfectly separates classes")