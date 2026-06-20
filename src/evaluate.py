from sklearn.metrics import (
    classification_report, confusion_matrix,
    silhouette_score, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_confusion_matrix(y_test, y_pred, labels, save_path="reports/figures/confusion_matrix.png"):
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"Saved to {save_path}")

def evaluate_clustering(job_matrix, labels):
    score = silhouette_score(job_matrix, labels, sample_size=1000)
    print(f"Silhouette Score: {score:.4f}")
    return score

def precision_at_k(recommended_titles, relevant_titles, k=10):
    top_k    = recommended_titles[:k]
    hits     = sum(1 for t in top_k if t in relevant_titles)
    score    = hits / k
    print(f"Precision@{k}: {score:.4f}")
    return score