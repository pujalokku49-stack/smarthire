import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data paths
RAW_DATA_DIR    = os.path.join(BASE_DIR, "data", "raw")
INTERIM_DIR     = os.path.join(BASE_DIR, "data", "interim")
PROCESSED_DIR   = os.path.join(BASE_DIR, "data", "processed")

# Model paths
MODELS_DIR      = os.path.join(BASE_DIR, "models")
CLASSIFIER_PATH = os.path.join(MODELS_DIR, "classifier.pkl")
TFIDF_PATH      = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")

# Model params
TOP_N_JOBS   = 10
N_CLUSTERS   = 10
MAX_FEATURES = 5000