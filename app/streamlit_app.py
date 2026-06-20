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

# ── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="SmartHire",
    page_icon="🎯",
    layout="wide"
)

# ── Header ────────────────────────────────────────────────
st.title("🎯 SmartHire — Resume to Job Matcher")
st.markdown("Upload your resume and get **matched jobs + skill gap report** instantly!")
st.divider()

# ── Load Models ───────────────────────────────────────────
@st.cache_resource
def load_models():
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    job_matrix = joblib.load("models/job_matrix.pkl")
    jobs_df    = joblib.load("models/jobs_df.pkl")
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
st.sidebar.markdown("3. See top matching jobs")
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

    # ── Row 2: Job Recommendations ────────────────────────
    st.subheader("🔝 Top Matching Jobs")
    with st.spinner("🔍 Finding best jobs for you..."):
        results = recommend_jobs(
            clean, jobs_df, vectorizer, job_matrix, top_n=top_n
        )

    # Color match score
    def color_score(val):
        if val >= 20:
            color = "green"
        elif val >= 10:
            color = "orange"
        else:
            color = "red"
        return f"color: {color}; font-weight: bold"

    st.dataframe(
        results.style.map(color_score, subset=["match_score"]),
        use_container_width=True
    )

    st.divider()

    # ── Row 3: Skill Gap Report ───────────────────────────
    st.subheader("📊 Skill Gap Report")
    gap = get_skill_gap(clean, results)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("### ✅ Skills You Have")
        if gap["skills_you_have"]:
            for skill in gap["skills_you_have"]:
                st.success(f"✔ {skill}")
        else:
            st.info("No matching skills found")

    with col4:
        st.markdown("### ⚠️ Skills to Add")
        if gap["skills_to_learn"]:
            for skill in gap["skills_to_learn"]:
                st.warning(f"➕ {skill}")
        else:
            st.info("Great! No missing skills found")

    st.divider()

    # ── Footer ────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "**SmartHire** — Built with ❤️ using Python, scikit-learn & Streamlit"
    )

    # Cleanup temp file
    os.unlink(tmp_path)

else:
    # Show instructions when no file uploaded
    st.info("👆 Please upload your resume to get started!")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📄 Resumes Trained On", "962")
    with col2:
        st.metric("💼 Jobs in Database", "50,000+")
    with col3:
        st.metric("🏷️ Job Categories", "25")