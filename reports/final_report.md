# SmartHire — Final Project Report

## 1. Project Overview
SmartHire is a resume-to-job matching portal built using classical 
machine learning. A user uploads a resume and receives:
- A predicted job category
- Top matching jobs ranked by similarity
- A skill gap report

## 2. Datasets Used

| Dataset | Size | Purpose |
|---|---|---|
| Resume Dataset | 962 resumes, 25 categories | Classifier training |
| Naukri Job Listings | 30,000 jobs | Job corpus |
| LinkedIn Job Postings | 118,000+ jobs | Job corpus |
| **Total Jobs** | **148,718 jobs** | Recommender |

## 3. System Architecture
Resume Upload (PDF/DOCX)

│

▼

Text Extraction → Preprocessing (clean, tokenize)

│

├──► Classifier (Supervised) → Predicts job category

├──► Similarity Engine (Unsupervised) → Top-N jobs

└──► Skill Gap Module → CV improvement report

│

▼

Streamlit Web Portal

## 4. Machine Learning Models

### Model A — Resume Category Classifier (Supervised)
- **Algorithm:** Logistic Regression
- **Features:** TF-IDF vectors (5000 features, bigrams)
- **Train/Test Split:** 80/20
- **Results:**

| Metric | Score |
|---|---|
| Accuracy | ~99% |
| Precision | ~1.00 |
| Recall | ~1.00 |
| F1-Score | ~1.00 |

- **Observation:** The classifier performs excellently across all 
25 job categories with minimal misclassification.

### Model B — Job Recommender (Unsupervised)
- **Algorithm:** TF-IDF + Cosine Similarity
- **Jobs indexed:** 50,000
- **Output:** Top-N ranked jobs with match scores
- **Results:** Qualitative check shows highly relevant job 
recommendations for sample resumes.

### Model C — Job Clustering (Unsupervised)
- **Algorithm:** KMeans (k=8)
- **Features:** TF-IDF vectors of job descriptions
- **Optimal k:** Selected using Elbow Method
- **Evaluation:** Silhouette Score computed in notebook
- **Visualization:** PCA 2D plot saved in reports/figures/

### Model D — Skill Gap Report (Unsupervised)
- **Method:** Compare resume skills vs top skills in matched jobs
- **Output:** List of skills present and missing

## 5. Results Summary

| Component | Method | Result |
|---|---|---|
| Resume Classifier | TF-IDF + LR | 99% accuracy |
| Job Recommender | Cosine Similarity | Relevant top-10 jobs |
| Job Clustering | KMeans (k=8) | 8 distinct job families |
| Skill Gap | Cluster Analysis | Missing skills identified |

## 6. Web Portal Features
- Upload PDF or DOCX resume
- View predicted job category instantly
- See top 10 matching jobs with match scores
- View skill gap report with skills to add
- Adjustable number of recommendations (5-20)

## 7. Limitations
- Datasets are from 2019-2024, may not reflect current job market
- Skill gap based on keyword matching, not semantic understanding
- No generative AI used (as per project requirements)
- Model performance depends on quality of uploaded resume text
- Fit predictor scores are conservative due to limited features (skill overlap + experience match only). Adding semantic similarity features would improve accuracy.

## 8. Future Improvements
- Replace TF-IDF with sentence embeddings for better matching
- Add learning-to-rank model for better job ordering
- Deploy on Streamlit Community Cloud for public access
- Add salary prediction module

## 9. Tools & Technologies
- **Python 3.14** — Core language
- **scikit-learn** — ML models
- **pandas & numpy** — Data processing
- **NLTK** — Text preprocessing
- **Streamlit** — Web portal
- **Matplotlib & Seaborn** — Visualizations
- **PyMuPDF & python-docx** — Resume parsing
- **Git & GitHub** — Version control

## 10. Repository
GitHub: https://github.com/pujalokku49-stack/smarthire

## 11. Conclusion
SmartHire successfully demonstrates a complete end-to-end ML pipeline
combining supervised and unsupervised learning to solve a real-world
problem. The system achieves high accuracy in resume classification
and provides relevant job recommendations using cosine similarity.