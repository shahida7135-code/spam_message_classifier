import streamlit as st
import joblib
import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


# ==================================================
# NLTK SETUP
# ==================================================

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)


# ==================================================
# LOAD MODEL
# ==================================================

model = joblib.load("models/spam_classifier.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")


# ==================================================
# TEXT PREPROCESSING
# ==================================================

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text):

    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove HTML
    text = re.sub(r"<.*?>", "", text)

    # Keep only letters and spaces
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Tokenization
    tokens = word_tokenize(text)

    # Remove stopwords and lemmatize
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words
    ]

    return " ".join(tokens)


# ==================================================
# STREAMLIT PAGE
# ==================================================

st.set_page_config(
    page_title="Spam Message Classifier",
    page_icon="📩",
    layout="centered"
)


# ==================================================
# HEADER
# ==================================================

st.title("📩 Spam Message Classifier")

st.write(
    "Enter an SMS or message below to check whether it is "
    "Spam or Not Spam."
)

st.divider()


# ==================================================
# MESSAGE INPUT
# ==================================================

message = st.text_area(
    "Enter your message:",
    height=180,
    placeholder="Example: Congratulations! You have won a free prize!"
)


# ==================================================
# PREDICTION
# ==================================================

if st.button("🔍 Predict", use_container_width=True):

    if message.strip() == "":
        st.warning("⚠️ Please enter a message first.")

    else:

        cleaned_message = clean_text(message)

        # Convert text into TF-IDF features
        message_tfidf = vectorizer.transform([cleaned_message])

        # Prediction
        prediction = model.predict(message_tfidf)[0]

        # Probability
        probabilities = model.predict_proba(message_tfidf)[0]

        if prediction == 1:

            confidence = probabilities[1] * 100

            st.error("🚨 SPAM MESSAGE")

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

        else:

            confidence = probabilities[0] * 100

            st.success("✅ NOT SPAM")

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )


st.divider()

st.caption(
    "Spam Message Classifier | TF-IDF + Logistic Regression"
)