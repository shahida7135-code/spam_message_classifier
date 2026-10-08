import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


print("\n========== LOADING PROCESSED DATA ==========\n")

# Load processed dataset
df = pd.read_csv(
    "data/processed_spam.csv",
    keep_default_na=False
)

# Remove empty cleaned messages
df = df[df["cleaned_message"].str.strip() != ""]

# Remove rows with missing labels
df = df.dropna(subset=["label"])

print("Total messages:", len(df))


# Input and target
X = df["cleaned_message"]
y = df["label"]


# ==================================================
# TRAIN TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training messages:", len(X_train))
print("Testing messages:", len(X_test))


# ==================================================
# TF-IDF VECTORIZATION
# ==================================================

print("\n========== TF-IDF VECTORIZATION ==========\n")

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("Training TF-IDF shape:", X_train_tfidf.shape)
print("Testing TF-IDF shape:", X_test_tfidf.shape)


# ==================================================
# MODEL TRAINING
# ==================================================

print("\n========== TRAINING MODEL ==========\n")

model = LogisticRegression(
    max_iter=1000
)

model.fit(X_train_tfidf, y_train)

print("Model training completed!")


# ==================================================
# SAVE MODEL AND VECTORIZER
# ==================================================

joblib.dump(
    model,
    "models/spam_classifier.pkl"
)

joblib.dump(
    vectorizer,
    "models/tfidf_vectorizer.pkl"
)

print("\nModel saved to:")
print("models/spam_classifier.pkl")

print("\nTF-IDF vectorizer saved to:")
print("models/tfidf_vectorizer.pkl")


# ==================================================
# QUICK TEST
# ==================================================

print("\n========== QUICK TEST ==========\n")

sample_messages = [
    "Congratulations! You won a free prize. Call now!",
    "Hey, are we meeting today?"
]

sample_tfidf = vectorizer.transform(sample_messages)

predictions = model.predict(sample_tfidf)

for message, prediction in zip(sample_messages, predictions):

    result = "SPAM" if prediction == 1 else "NOT SPAM"

    print("Message:", message)
    print("Prediction:", result)
    print("-" * 60)