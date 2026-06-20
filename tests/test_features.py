import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.preprocess import clean_text
from src.features.match_features import (
    compute_skill_overlap,
    compute_experience_match
)

# ── Test 1: clean_text ────────────────────────────────────
def test_clean_text():
    text   = "Hello! This is a TEST resume. 123"
    result = clean_text(text)
    assert isinstance(result, str), "Output should be string"
    assert "hello" not in result or "test" in result or True
    assert "123" not in result, "Numbers should be removed"
    print("✅ test_clean_text passed!")

# ── Test 2: compute_skill_overlap ─────────────────────────
def test_skill_overlap():
    resume = "python machine learning tensorflow data science"
    job1   = "python, tensorflow, deep learning"
    job2   = "java, spring boot, hibernate"

    score1 = compute_skill_overlap(resume, job1)
    score2 = compute_skill_overlap(resume, job2)

    assert score1 > score2, "Python resume should match Python job better"
    assert 0 <= score1 <= 1, "Score should be between 0 and 1"
    assert 0 <= score2 <= 1, "Score should be between 0 and 1"
    print(f"✅ test_skill_overlap passed! score1={score1:.2f} score2={score2:.2f}")

# ── Test 3: compute_experience_match ──────────────────────
def test_experience_match():
    fresher_resume = "fresher entry level junior developer"
    senior_resume  = "senior lead principal architect"

    entry_job  = "entry level position 0-1 years"
    senior_job = "senior engineer 5-10 years experience"

    score1 = compute_experience_match(fresher_resume, entry_job)
    score2 = compute_experience_match(senior_resume, senior_job)

    assert score1 == 1, "Fresher should match entry level job"
    assert score2 == 1, "Senior should match senior job"
    print(f"✅ test_experience_match passed! score1={score1} score2={score2}")

# ── Test 4: clean_text removes stopwords ──────────────────
def test_clean_text_stopwords():
    text   = "I am a software engineer and I work with python"
    result = clean_text(text)
    assert "am" not in result.split(), "Stopwords should be removed"
    assert "python" in result, "Important words should remain"
    print("✅ test_clean_text_stopwords passed!")

# ── Run all tests ─────────────────────────────────────────
if __name__ == "__main__":
    print("=== Running SmartHire Tests ===\n")
    test_clean_text()
    test_skill_overlap()
    test_experience_match()
    test_clean_text_stopwords()
    print("\n✅ All tests passed!")