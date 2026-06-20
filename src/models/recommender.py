import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.config import MAX_FEATURES, TOP_N_JOBS

def build_recommender(jobs_df, text_col="clean_text"):
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=(1, 2))
    job_matrix = vectorizer.fit_transform(jobs_df[text_col])
    return vectorizer, job_matrix

def recommend_jobs(resume_text, jobs_df, vectorizer, job_matrix, top_n=TOP_N_JOBS):
    resume_vec = vectorizer.transform([resume_text])
    scores     = cosine_similarity(resume_vec, job_matrix).flatten()
    top_idx    = np.argsort(scores)[::-1][:top_n]
    results    = jobs_df.iloc[top_idx].copy()
    results["match_score"] = (scores[top_idx] * 100).round(2)
    return results[["title", "company", "location", "skills", "match_score"]]