from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os
from src.config import MAX_FEATURES, MODELS_DIR

def build_tfidf_vectorizer(texts, max_features=MAX_FEATURES):
    """Build and fit a TF-IDF vectorizer on given texts."""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        stop_words='english',
        sublinear_tf=True
    )
    matrix = vectorizer.fit_transform(texts)
    print(f"✅ TF-IDF matrix shape: {matrix.shape}")
    return vectorizer, matrix

def transform_text(vectorizer, texts):
    """Transform new texts using fitted vectorizer."""
    return vectorizer.transform(texts)

def save_vectorizer(vectorizer, path=None):
    """Save vectorizer to disk."""
    if path is None:
        path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
    joblib.dump(vectorizer, path)
    print(f"✅ Vectorizer saved to {path}")

def load_vectorizer(path=None):
    """Load vectorizer from disk."""
    if path is None:
        path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
    return joblib.load(path)