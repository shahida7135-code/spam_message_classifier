import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


print("\n========== LOADING DATA ==========\n")

# Load processed dataset
df = pd.read_csv(
    "data/processed_spam.csv",
    keep_default_na=False
)

# Remove empty messages
df = df[df["cleaned_message"].str.strip() != ""]

# Separate input and labels
X = df["cleaned_message"]
y = df["label"]


# ==========================================
# TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# LOAD MODEL AND TF-IDF
# ==========================================

model = joblib.load(
    "models/spam_classifier.pkl"
)

vectorizer = joblib.load(
    "models/tfidf_vectorizer.pkl"
)


# ==========================================
# TRANSFORM TEST DATA
# ==========================================

X_test_tfidf = vectorizer.transform(X_test)


# ==========================================
# MAKE PREDICTIONS
# ==========================================

y_pred = model.predict(X_test_tfidf)


# ==========================================
# CALCULATE METRICS
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)


print("\n========== MODEL PERFORMANCE ==========\n")

print(f"Accuracy :  {accuracy:.4f}")
print(f"Precision:  {precision:.4f}")
print(f"Recall   :  {recall:.4f}")
print(f"F1-Score :  {f1:.4f}")


# ==========================================
# CLASSIFICATION REPORT
# ==========================================

print("\n========== CLASSIFICATION REPORT ==========\n")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Not Spam", "Spam"]
    )
)


# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n========== CONFUSION MATRIX ==========\n")

print(cm)


# ==========================================
# VISUALIZE CONFUSION MATRIX
# ==========================================

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=["Not Spam", "Spam"],
    yticklabels=["Not Spam", "Spam"]
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Spam Message Classifier - Confusion Matrix")

plt.tight_layout()

plt.show()