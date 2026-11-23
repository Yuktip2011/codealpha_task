"""
preprocess.py
-------------
Handles all NLP text preprocessing for the Smart FAQ Chatbot.
Includes tokenization, lowercasing, punctuation removal, and stopword filtering.
"""

import re
import string
import nltk

# Download required NLTK data (runs only if not already downloaded)
def download_nltk_data():
    """Download necessary NLTK datasets silently."""
    packages = ["punkt", "stopwords", "wordnet", "punkt_tab"]
    for pkg in packages:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass

download_nltk_data()

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Initialize lemmatizer and stopwords once (expensive to reload)
_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words("english"))

# Keep question words — they carry meaning in FAQ matching
_KEEP_WORDS = {"what", "when", "where", "who", "why", "how", "which", "whom", "whose"}
_FILTERED_STOPS = _stop_words - _KEEP_WORDS


def clean_text(text: str) -> str:
    """
    Full NLP preprocessing pipeline.

    Steps:
    1. Convert to lowercase.
    2. Remove URLs, emails, and special characters.
    3. Remove punctuation.
    4. Tokenize into words.
    5. Remove stopwords (keeping question-words).
    6. Lemmatize each token.
    7. Rejoin into a clean string.

    Args:
        text (str): Raw input text.

    Returns:
        str: Cleaned, preprocessed text string.
    """
    if not text or not isinstance(text, str):
        return ""

    # Step 1: Lowercase
    text = text.lower().strip()

    # Step 2: Remove URLs and emails
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)

    # Step 3: Remove digits and special characters (keep spaces and letters)
    text = re.sub(r"[^a-z\s]", " ", text)

    # Step 4: Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Step 5: Tokenize
    try:
        tokens = word_tokenize(text)
    except Exception:
        tokens = text.split()

    # Step 6: Remove stopwords + Step 7: Lemmatize
    cleaned_tokens = [
        _lemmatizer.lemmatize(token)
        for token in tokens
        if token not in _FILTERED_STOPS and len(token) > 1
    ]

    return " ".join(cleaned_tokens)


def tokenize_only(text: str) -> list:
    """
    Tokenize text without stopword removal (for analysis purposes).

    Args:
        text (str): Raw input text.

    Returns:
        list: List of lowercase tokens.
    """
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    try:
        return word_tokenize(text)
    except Exception:
        return text.split()


def get_preprocessing_steps(text: str) -> dict:
    """
    Returns a step-by-step breakdown of the preprocessing for visualization.

    Args:
        text (str): Original user input.

    Returns:
        dict: Dictionary showing each transformation step.
    """
    steps = {}
    steps["original"] = text

    lowered = text.lower().strip()
    steps["lowercase"] = lowered

    no_special = re.sub(r"[^a-z\s]", " ", re.sub(r"http\S+|\S+@\S+", "", lowered))
    no_special = re.sub(r"\s+", " ", no_special).strip()
    steps["cleaned"] = no_special

    try:
        tokens = word_tokenize(no_special)
    except Exception:
        tokens = no_special.split()
    steps["tokenized"] = tokens

    no_stop = [t for t in tokens if t not in _FILTERED_STOPS and len(t) > 1]
    steps["stopwords_removed"] = no_stop

    lemmatized = [_lemmatizer.lemmatize(t) for t in no_stop]
    steps["lemmatized"] = lemmatized

    steps["final"] = " ".join(lemmatized)
    return steps

