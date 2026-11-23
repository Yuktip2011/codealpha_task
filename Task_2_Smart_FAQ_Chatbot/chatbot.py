"""
chatbot.py
----------
Core NLP engine for the Smart FAQ Chatbot.
Handles FAQ loading, TF-IDF vectorization, and Cosine Similarity matching.
"""

import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocess import clean_text

# ── Constants ──────────────────────────────────────────────────────────────────
FAQ_CSV_PATH = os.path.join(os.path.dirname(__file__), "faqs.csv")

# Similarity threshold: answers below this score are considered "no match"
SIMILARITY_THRESHOLD = 0.20

# Confidence label thresholds
HIGH_CONFIDENCE = 0.65
MEDIUM_CONFIDENCE = 0.40


class FAQChatbot:
    """
    TF-IDF + Cosine Similarity FAQ Chatbot engine.

    Workflow:
    1. Load FAQs from CSV.
    2. Preprocess all FAQ questions.
    3. Build a TF-IDF matrix over preprocessed questions.
    4. For each user query, preprocess it and compute cosine similarity.
    5. Return the best-matching answer with metadata.
    """

    def __init__(self, csv_path: str = FAQ_CSV_PATH):
        """
        Initialize the chatbot by loading FAQs and fitting the TF-IDF model.

        Args:
            csv_path (str): Path to the FAQ CSV file.
        """
        self.csv_path = csv_path
        self.df = None                  # Raw FAQ DataFrame
        self.cleaned_questions = []     # Preprocessed question strings
        self.vectorizer = None          # Fitted TfidfVectorizer
        self.tfidf_matrix = None        # TF-IDF matrix of FAQ questions
        self.is_ready = False           # Flag: model is loaded and fitted

        self._load_and_fit()

    # ── Private helpers ────────────────────────────────────────────────────────

    def _load_and_fit(self):
        """Load the CSV, preprocess questions, and fit TF-IDF vectorizer."""
        try:
            self.df = pd.read_csv(self.csv_path)
            self._validate_columns()

            # Clean each FAQ question using our NLP pipeline
            self.cleaned_questions = [
                clean_text(q) for q in self.df["question"].tolist()
            ]

            # Build TF-IDF model
            # ngram_range=(1,2) captures both single words and two-word phrases
            self.vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                analyzer="word",
                min_df=1,
                sublinear_tf=True,  # Apply log normalization to term frequency
            )
            self.tfidf_matrix = self.vectorizer.fit_transform(self.cleaned_questions)
            self.is_ready = True

        except FileNotFoundError:
            raise FileNotFoundError(
                f"FAQ file not found at: {self.csv_path}\n"
                "Please ensure faqs.csv is in the same directory as chatbot.py."
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize chatbot: {exc}") from exc

    def _validate_columns(self):
        """Ensure the CSV has required columns."""
        required = {"category", "question", "answer"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(
                f"faqs.csv is missing required columns: {missing}. "
                f"Found columns: {list(self.df.columns)}"
            )

    def _confidence_label(self, score: float) -> str:
        """Return a human-readable confidence label for a similarity score."""
        if score >= HIGH_CONFIDENCE:
            return "High"
        elif score >= MEDIUM_CONFIDENCE:
            return "Medium"
        else:
            return "Low"

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_response(self, user_query: str) -> dict:
        """
        Find the best-matching FAQ answer for a user query.

        Args:
            user_query (str): Raw text entered by the user.

        Returns:
            dict with keys:
                - answer       (str)  : The matched answer or fallback message.
                - matched_question (str): The FAQ question that was matched.
                - category     (str)  : Category of the matched FAQ.
                - score        (float): Cosine similarity score (0–1).
                - confidence   (str)  : "High" / "Medium" / "Low" / "No Match".
                - found        (bool) : Whether a match above threshold was found.
        """
        if not self.is_ready:
            return self._no_match_response(0.0, error=True)

        # Handle empty / whitespace-only input
        query_stripped = user_query.strip()
        if not query_stripped:
            return {
                "answer": "Please type a question so I can help you! 😊",
                "matched_question": "",
                "category": "",
                "score": 0.0,
                "confidence": "N/A",
                "found": False,
            }

        # Preprocess the user query the same way we preprocessed FAQ questions
        cleaned_query = clean_text(query_stripped)

        # Edge-case: preprocessing strips everything (e.g., user typed only "??")
        if not cleaned_query:
            return {
                "answer": (
                    "Your question seems to contain only special characters. "
                    "Please rephrase and try again."
                ),
                "matched_question": "",
                "category": "",
                "score": 0.0,
                "confidence": "N/A",
                "found": False,
            }

        # Transform user query into TF-IDF vector space
        try:
            query_vector = self.vectorizer.transform([cleaned_query])
        except Exception:
            return self._no_match_response(0.0, error=True)

        # Compute cosine similarity between query and all FAQ questions
        similarity_scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

        # Find the best-matching FAQ index
        best_idx = int(np.argmax(similarity_scores))
        best_score = float(similarity_scores[best_idx])

        if best_score < SIMILARITY_THRESHOLD:
            return self._no_match_response(best_score)

        # Retrieve matched FAQ details
        matched_row = self.df.iloc[best_idx]
        return {
            "answer": matched_row["answer"],
            "matched_question": matched_row["question"],
            "category": matched_row["category"],
            "score": round(best_score, 4),
            "confidence": self._confidence_label(best_score),
            "found": True,
        }

    def _no_match_response(self, score: float, error: bool = False) -> dict:
        """Return a standardized no-match response."""
        if error:
            answer = (
                "⚠️ An internal error occurred. "
                "Please refresh the page and try again."
            )
        else:
            answer = (
                "Sorry, I couldn't find a suitable answer to your question. "
                "Please try asking differently, or browse the suggested questions below."
            )
        return {
            "answer": answer,
            "matched_question": "",
            "category": "",
            "score": round(score, 4),
            "confidence": "No Match",
            "found": False,
        }

    # ── Dataset utilities ──────────────────────────────────────────────────────

    def get_categories(self) -> list:
        """Return a sorted list of unique FAQ categories."""
        if self.df is None:
            return []
        return sorted(self.df["category"].unique().tolist())

    def get_total_faqs(self) -> int:
        """Return the total number of FAQ entries."""
        return len(self.df) if self.df is not None else 0

    def get_faqs_by_category(self, category: str) -> pd.DataFrame:
        """Return all FAQs for a given category."""
        if self.df is None:
            return pd.DataFrame()
        return self.df[self.df["category"] == category][["question", "answer"]]

    def get_sample_questions(self, n: int = 6) -> list:
        """
        Return n sample questions spread across categories for the
        'Suggested Questions' feature.

        Args:
            n (int): Number of sample questions to return.

        Returns:
            list of str: Sample FAQ question strings.
        """
        if self.df is None:
            return []
        categories = self.get_categories()
        samples = []
        per_cat = max(1, n // len(categories))
        for cat in categories:
            cat_df = self.df[self.df["category"] == cat]
            picked = cat_df["question"].head(per_cat).tolist()
            samples.extend(picked)
        return samples[:n]
