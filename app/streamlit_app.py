import streamlit as st
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import tempfile
from src.parsing.resume_parser import parse_resume
from src.data.preprocess import clean_text
from src.models.classifier import predict_category
from src.models.recommender import recommend_jobs
from src.models.clustering import get_skill_gap
from src.models.fit_predictor import predict_fit

# ── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="SmartHire",
    page_icon="🎯",
    layout="wide"
)

# ── Header ────────────────────────────────────────────────
st.title("🎯 SmartHire — Resume to Job Matcher")
st.markdown("Upload your resume and get **matched jobs + fit scores + skill gap report** instantly!")
st.divider()

# ── Load Models ───────────────────────────────────────────
@st.cache_resource
def load_models():
    vectorizer  = joblib.load("models/tfidf_vectorizer.pkl")
    job_matrix  = joblib.load("models/job_matrix.pkl")
    jobs_df     = joblib.load("models/jobs_df.pkl")
    return vectorizer, job_matrix, jobs_df

with st.spinner("⏳ Loading models..."):
    vectorizer, job_matrix, jobs_df = load_models()
st.success("✅ Models loaded!")

# ── Sidebar ───────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")
top_n = st.sidebar.slider("Number of job recommendations", 5, 20, 10)
st.sidebar.divider()
st.sidebar.markdown("### 📌 How to use:")
st.sidebar.markdown("1. Upload your resume (PDF or DOCX)")
st.sidebar.markdown("2. View your predicted job category")
st.sidebar.markdown("3. See top matching jobs with fit scores")
st.sidebar.markdown("4. Check your skill gap report")

# ── File Upload ───────────────────────────────────────────
st.subheader("📄 Upload Your Resume")
uploaded = st.file_uploader(
    "Supported formats: PDF, DOCX",
    type=["pdf", "docx"]
)

if uploaded:
    # Save temp file
    suffix = os.path.splitext(uploaded.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(uploaded.read())
        tmp_path = f.name

    # Parse resume
    with st.spinner("📖 Parsing resume..."):
        raw_text = parse_resume(tmp_path)
        clean    = clean_text(raw_text)
    st.success("✅ Resume parsed successfully!")

    st.divider()

    # ── Row 1: Category + Resume Preview ─────────────────
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📂 Predicted Job Category")
        try:
            category = predict_category(clean)
            st.info(f"### 🏷️ {category}")
        except Exception as e:
            st.warning(f"Classifier error: {e}")
            category = "Unknown"

    with col2:
        st.subheader("👁️ Resume Preview")
        with st.expander("Click to view extracted text"):
            st.text(raw_text[:1500] + "...")

    st.divider()

    # ── Row 2: Job Recommendations + Fit Scores ──────────
    st.subheader("🔝 Top Matching Jobs with Fit Scores")
    with st.spinner("🔍 Finding best jobs and computing fit scores..."):
        results = recommend_jobs(
            clean, jobs_df, vectorizer, job_matrix, top_n=top_n
        )

        # Add fit score for each job
        fit_scores = []
        for _, job_row in results.iterrows():
            try:
                score = predict_fit(clean, clean, job_row)
            except:
                score = 0.0
            fit_scores.append(score)
        results['fit_score_%'] = fit_scores

    # Color match score
    def color_score(val):
        if val >= 30:
            color = "green"
        elif val >= 15:
            color = "orange"
        else:
            color = "red"
        return f"color: {color}; font-weight: bold"

    st.dataframe(
        results[['title', 'company', 'location', 
                 'skills', 'match_score', 'fit_score_%']].style.map(
            color_score, subset=["match_score", "fit_score_%"]
        ),
        use_container_width=True
    )

    st.divider()

    # ── Row 3: Fit Score Summary ──────────────────────────
    st.subheader("🎯 Fit Score Summary")
    col3, col4, col5 = st.columns(3)

    avg_fit  = sum(fit_scores) / len(fit_scores) if fit_scores else 0
    max_fit  = max(fit_scores) if fit_scores else 0
    best_job = results.iloc[fit_scores.index(max_fit)]['title'] if fit_scores else "N/A"

    with col3:
        st.metric("Average Fit Score", f"{avg_fit:.1f}%")
    with col4:
        st.metric("Best Fit Score", f"{max_fit:.1f}%")
    with col5:
        st.metric("Best Matching Job", best_job[:30])

    st.divider()

    # ── Row 4: Skill Gap Report ───────────────────────────
    st.subheader("📊 Skill Gap Report")
    gap = get_skill_gap(clean, results)

    col6, col7 = st.columns(2)
    with col6:
        st.markdown("### ✅ Skills You Have")
        if gap["skills_you_have"]:
            for skill in gap["skills_you_have"]:
                st.success(f"✔ {skill}")
        else:
            st.info("No matching skills found")

    with col7:
        st.markdown("### ⚠️ Skills to Add")
        if gap["skills_to_learn"]:
            for skill in gap["skills_to_learn"]:
                st.warning(f"➕ {skill}")
        else:
            st.info("Great! No missing skills found")

    st.divider()

    # ── Footer ────────────────────────────────────────────
    st.markdown(
        "**SmartHire** — Built with ❤️ using Python, scikit-learn & Streamlit"
    )

    # Cleanup temp file
    os.unlink(tmp_path)

else:
    st.info("👆 Please upload your resume to get started!")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📄 Resumes Trained On", "962")
    with col2:
        st.metric("💼 Jobs in Database", "50,000+")
    with col3:
        st.metric("🏷️ Job Categories", "25")