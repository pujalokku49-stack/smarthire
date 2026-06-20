from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
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

    # Plot elbow curve
    plt.figure(figsize=(8, 4))
    plt.plot(list(k_range), inertias, marker="o")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for Optimal k")
    plt.tight_layout()

    # Fix path
    os.makedirs("../reports/figures", exist_ok=True)
    plt.savefig("../reports/figures/elbow_plot.png")
    plt.show()

def get_skill_gap(resume_text, jobs_df, top_n=10):
    cluster_skills = " ".join(jobs_df["skills"].dropna()).lower()
    all_skills     = [s.strip() for s in cluster_skills.split(",") if s.strip()]
    common_skills  = [s for s, _ in Counter(all_skills).most_common(top_n)]
    resume_words   = set(resume_text.lower().split())
    missing = [s for s in common_skills if s not in resume_words]
    present = [s for s in common_skills if s in resume_words]
    return {"skills_you_have": present, "skills_to_learn": missing}