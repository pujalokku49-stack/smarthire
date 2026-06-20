from sklearn.cluster import KMeans
from collections import Counter
import matplotlib.pyplot as plt
import os
from src.config import N_CLUSTERS

def train_clustering(job_matrix, n_clusters=N_CLUSTERS):
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km.fit(job_matrix)
    return km

def find_best_k(job_matrix, k_range=range(3, 15)):
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(job_matrix)
        inertias.append(km.inertia_)

    plt.figure(figsize=(8, 4))
    plt.plot(list(k_range), inertias, marker="o")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for Optimal k")
    plt.tight_layout()
    os.makedirs("../reports/figures", exist_ok=True)
    plt.savefig("../reports/figures/elbow_plot.png")
    plt.show()

def get_skill_gap(resume_text, jobs_df, top_n=10):
    # Handle both | and , separators
    all_skills = []
    for skill_str in jobs_df["skills"].dropna():
        skill_str = str(skill_str).lower()
        # Split by both | and ,
        if "|" in skill_str:
            skills = [s.strip() for s in skill_str.split("|")]
        else:
            skills = [s.strip() for s in skill_str.split(",")]
        all_skills.extend([s for s in skills if len(s) > 2])

    # Get top skills
    common_skills = [s for s, _ in Counter(all_skills).most_common(top_n)]

    # Compare with resume
    resume_words = set(resume_text.lower().split())
    resume_text_lower = resume_text.lower()

    present = [s for s in common_skills if s in resume_text_lower]
    missing = [s for s in common_skills if s not in resume_text_lower]

    return {"skills_you_have": present, "skills_to_learn": missing}