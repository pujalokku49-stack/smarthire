import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def compute_skill_overlap(resume_skills, job_skills):
    """
    Compute skill overlap between resume and job.
    Returns overlap score between 0 and 1.
    """
    if not resume_skills or not job_skills:
        return 0.0
    resume_set = set(str(resume_skills).lower().split())
    job_set    = set(str(job_skills).lower().replace("|", " ").split())
    if not job_set:
        return 0.0
    overlap = resume_set.intersection(job_set)
    return len(overlap) / len(job_set)

def compute_experience_match(resume_text, job_experience):
    """
    Simple experience match based on keywords.
    Returns 1 if match found, 0 otherwise.
    """
    if not job_experience:
        return 0
    resume_lower = str(resume_text).lower()
    exp_lower    = str(job_experience).lower()

    # Check experience level keywords
    levels = {
        "entry"      : ["fresher", "entry", "0-1", "junior", "intern"],
        "mid"        : ["2-5", "mid", "intermediate", "3-5"],
        "senior"     : ["senior", "5-10", "lead", "principal", "expert"]
    }
    for level, keywords in levels.items():
        if any(k in exp_lower for k in keywords):
            if any(k in resume_lower for k in keywords):
                return 1
    return 0

def build_match_features(resume_text, resume_skills, jobs_df):
    """
    Build feature matrix for fit prediction.
    Features: skill_overlap, experience_match, text_similarity
    """
    features = []
    for _, job in jobs_df.iterrows():
        skill_overlap    = compute_skill_overlap(resume_skills, job.get("skills", ""))
        experience_match = compute_experience_match(resume_text, job.get("experience", ""))
        features.append({
            "skill_overlap"    : skill_overlap,
            "experience_match" : experience_match,
        })
    return pd.DataFrame(features)