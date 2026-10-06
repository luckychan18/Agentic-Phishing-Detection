import pandas as pd

df = pd.read_csv("data/PhiUSIIL_Phishing_URL_Dataset.csv")

print("SHAPE:", df.shape)

print("\nALL COLUMNS:")
for i, col in enumerate(df.columns):
    print(f"{i}: {col}")

print("\nTARGET DISTRIBUTION:")
print(df["label"].value_counts())

print("\nDATA TYPES:")
print(df.dtypes)