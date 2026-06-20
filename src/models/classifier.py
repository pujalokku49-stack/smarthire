from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
from src.config import CLASSIFIER_PATH, MAX_FEATURES

def train_classifier(df, text_col="clean_resume", label_col="Category"):
    X_train, X_test, y_train, y_test = train_test_split(
        df[text_col], df[label_col],
        test_size=0.2, random_state=42, stratify=df[label_col]
    )
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=(1, 2))),
        ("clf",   LogisticRegression(max_iter=1000, C=5))
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred))
    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    os.makedirs(os.path.dirname(CLASSIFIER_PATH), exist_ok=True)
    joblib.dump(pipe, CLASSIFIER_PATH)
    print(f"Model saved to {CLASSIFIER_PATH}")
    return pipe

def predict_category(text):
    pipe = joblib.load(CLASSIFIER_PATH)
    return pipe.predict([text])[0]