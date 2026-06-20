# 🎯 SmartHire — Resume-to-Job Matching & Career Guidance Engine

A complete, end-to-end machine learning project combining supervised and
unsupervised techniques to recommend jobs, predict shortlisting, and generate
CV improvement reports.

## 🚀 Live Demo
Run locally using Streamlit — see setup instructions below.

## 📌 Project Overview
SmartHire is a portal where a student uploads a resume (CV) and the system returns:
- ✅ A ranked list of matching jobs from a job database
- ✅ A predicted job category for the resume
- ✅ A skill-gap report telling the candidate what to improve

## 🧠 Machine Learning Components

| Component | Type | Algorithm |
|---|---|---|
| Resume Category Classifier | Supervised | TF-IDF + Logistic Regression |
| Job Recommender | Unsupervised | TF-IDF + Cosine Similarity |
| Job Clustering | Unsupervised | KMeans |
| Skill Gap Report | Unsupervised | Cluster Analysis |

## 📁 Project Structure
smarthire/

├── data/

│   ├── raw/              # Original datasets

│   ├── interim/          # Partially cleaned data

│   └── processed/        # Final model-ready data

├── notebooks/

│   ├── 01_eda.ipynb

│   ├── 02_resume_classifier.ipynb

│   ├── 03_recommender.ipynb

│   └── 04_clustering_topics.ipynb

├── src/

│   ├── data/             # Data loading & preprocessing

│   ├── features/         # Feature engineering

│   ├── models/           # ML models

│   └── parsing/          # Resume parser

├── models/               # Saved .pkl model files

├── app/

│   └── streamlit_app.py  # Web portal

└── reports/

└── figures/          # Plots and visualizations

## 📦 Datasets Used
- **Resume Dataset** — 962 resumes across 25 job categories (Kaggle)
- **Naukri Job Listings** — 30,000 Indian job postings (Kaggle)
- **LinkedIn Job Postings 2023-2024** — Large set of job descriptions (Kaggle)

## ⚙️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/smarthire.git
cd smarthire
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 4. Add datasets
Place these files in `data/raw/`:
- `resume_dataset.csv`
- `naukri_jobs.csv`
- `linkedin_jobs.csv`

### 5. Run notebooks in order
notebooks/01_eda.ipynb

notebooks/02_resume_classifier.ipynb

notebooks/03_recommender.ipynb

notebooks/04_clustering_topics.ipynb

### 6. Launch the web app
```bash
streamlit run app/streamlit_app.py
```

## 📊 Results

| Model | Metric | Score |
|---|---|---|
| Resume Classifier | Accuracy | ~99% |
| Resume Classifier | F1-Score | ~1.00 |
| Job Recommender | Top-10 Match | ✅ Relevant |
| Clustering | Silhouette Score | Computed in notebook |
## 📦 Datasets Used
- **Resume Dataset** — 962 resumes across 25 job categories (Kaggle)
- **Naukri Job Listings** — 30,000 Indian job postings (Kaggle)
- **LinkedIn Job Postings 2023-2024** — Large set of job descriptions (Kaggle)

## ⚙️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/smarthire.git
cd smarthire
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 4. Add datasets
Place these files in `data/raw/`:
- `resume_dataset.csv`
- `naukri_jobs.csv`
- `linkedin_jobs.csv`

### 5. Run notebooks in order
notebooks/01_eda.ipynb

notebooks/02_resume_classifier.ipynb

notebooks/03_recommender.ipynb

notebooks/04_clustering_topics.ipynb
### 6. Launch the web app
```bash
streamlit run app/streamlit_app.py
```

## 📊 Results

| Model | Metric | Score |
|---|---|---|
| Resume Classifier | Accuracy | ~99% |
| Resume Classifier | F1-Score | ~1.00 |
| Job Recommender | Top-10 Match | ✅ Relevant |
| Clustering | Silhouette Score | Computed in notebook |

## 🛠️ Tech Stack
- **Python 3.14**
- **scikit-learn** — ML models
- **pandas & numpy** — Data processing
- **NLTK** — Text preprocessing
- **Streamlit** — Web portal
- **Matplotlib & Seaborn** — Visualizations
- **PyMuPDF & python-docx** — Resume parsing

## 👩‍💻 Team
- Built as part of Industrial Machine Learning Project
- Duration: 3 weeks

## 📄 License
MIT License

