import re
import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


# Download NLTK resources
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")


# Initialize NLTK tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    """
    Clean and preprocess a single SMS message.
    """

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove numbers and special characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stop words
    tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    # Lemmatize
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
    ]

    return " ".join(tokens)


# ==========================================
# MAIN PROGRAM
# ==========================================

if __name__ == "__main__":

    print("\n========== LOADING DATASET ==========\n")

    # Load dataset
    df = pd.read_csv(
        "data/SMSSpamCollection.txt",
        sep="\t",
        header=None,
        names=["label", "message"]
    )

    print("Original dataset size:", len(df))

    # Remove missing values
    df = df.dropna()

    # Remove duplicate messages
    df = df.drop_duplicates()

    print("After cleaning:", len(df))

    print("\n========== PREPROCESSING TEXT ==========\n")

    # Apply preprocessing
    df["cleaned_message"] = df["message"].apply(clean_text)

    # Convert labels to numbers
    # ham = 0
    # spam = 1
    df["label"] = df["label"].map({
        "ham": 0,
        "spam": 1
    })

    # Save processed dataset
    df.to_csv(
        "data/processed_spam.csv",
        index=False
    )

    print("Preprocessing completed!")

    print("\nFinal dataset:")
    print(df.head())

    print("\nLabel distribution:")
    print(df["label"].value_counts())

    print("\nProcessed dataset saved to:")
    print("data/processed_spam.csv")