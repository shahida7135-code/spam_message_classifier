import pandas as pd

# Load the SMS Spam Collection dataset
df = pd.read_csv(
    "data/SMSSpamCollection.txt",
    sep="\t",
    header=None,
    names=["label", "message"]
)

print("\n========== DATASET INFORMATION ==========\n")

print("Number of rows:", len(df))
print("Number of columns:", len(df.columns))

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 10 messages:")
print(df.head(10))

print("\nLabel distribution:")
print(df["label"].value_counts())

print("\nMissing values:")
print(df.isnull().sum())