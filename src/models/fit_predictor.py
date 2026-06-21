import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
from src.config import MODELS_DIR
from src.features.match_features import (
    compute_skill_overlap,
    compute_experience_match,
    compute_education_match
)
from src.data.preprocess import clean_text

FIT_PREDICTOR_PATH = os.path.join(MODELS_DIR, "fit_predictor.pkl")

def build_features(resumes, jobs_df, sample_size=500):
    rows = []
    sample_jobs = jobs_df.sample(sample_size, random_state=42)

    for _, resume_row in resumes.iterrows():
        resume_text = clean_text(resume_row['Resume'])
        resume_cat  = resume_row['Category']

        for _, job_row in sample_jobs.iterrows():
            skill_overlap    = compute_skill_overlap(
                resume_text, job_row['skills']
            )
            exp_match        = compute_experience_match(
                resume_text, job_row['experience']
            )
            edu_match        = compute_education_match(
                resume_text, job_row['description']
            )
            label = 1 if any(
                word.lower() in str(job_row['title']).lower()
                for word in resume_cat.split()
            ) else 0

            rows.append({
                'skill_overlap'  : skill_overlap,
                'exp_match'      : exp_match,
                'edu_match'      : edu_match,
                'label'          : label
            })

    return pd.DataFrame(rows)

def train_fit_predictor(resumes, jobs_df):
    print("⏳ Building features...")
    df = build_features(resumes, jobs_df)

    X = df[['skill_overlap', 'exp_match', 'edu_match']]
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Logistic Regression
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_prob = lr.predict_proba(X_test)[:, 1]
    lr_auc  = roc_auc_score(y_test, lr_prob)

    # XGBoost
    xgb = XGBClassifier(random_state=42, eval_metric='logloss')
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    xgb_prob = xgb.predict_proba(X_test)[:, 1]
    xgb_auc  = roc_auc_score(y_test, xgb_prob)

    # Comparison table
    print("\n╔══════════════════════════════════════════╗")
    print("║      Fit Predictor - Model Comparison    ║")
    print("╠══════════════════════════════════════════╣")
    print(f"║ {'Model':<20} {'ROC-AUC':>10} {'F1':>8} ║")
    print("╠══════════════════════════════════════════╣")
    from sklearn.metrics import f1_score
    print(f"║ {'Logistic Regression':<20} {lr_auc:>10.4f} {f1_score(y_test, lr_pred):>8.4f} ║")
    print(f"║ {'XGBoost':<20} {xgb_auc:>10.4f} {f1_score(y_test, xgb_pred):>8.4f} ║")
    print("╚══════════════════════════════════════════╝")

    # Save best model
    best_model = xgb if xgb_auc >= lr_auc else lr
    joblib.dump(best_model, FIT_PREDICTOR_PATH)
    print(f"\n✅ Best model saved!")
    return best_model

def predict_fit(resume_text, resume_skills, job_row):
    try:
        model         = joblib.load(FIT_PREDICTOR_PATH)
        skill_overlap = compute_skill_overlap(
            resume_text, job_row.get('skills', '')
        )
        exp_match     = compute_experience_match(
            resume_text, job_row.get('experience', '')
        )
        edu_match     = compute_education_match(
            resume_text, job_row.get('description', '')
        )
        if skill_overlap == 0:
            desc = str(job_row.get('description', ''))[:300]
            skill_overlap = compute_skill_overlap(resume_text, desc)

        features = pd.DataFrame([{
            'skill_overlap': skill_overlap,
            'exp_match'    : exp_match,
            'edu_match'    : edu_match
        }])
        prob  = model.predict_proba(features)[0][1]
        score = min(prob * 100 * 1.5, 99.0)
        return round(score, 2)
    except:
        return 0.0