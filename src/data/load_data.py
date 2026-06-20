import pandas as pd
import os
from src.config import RAW_DATA_DIR

def load_resumes():
    path = os.path.join(RAW_DATA_DIR, "resume_dataset.csv")
    df = pd.read_csv(path)
    return df[["Resume", "Category"]]

def load_jobs():
    naukri_path   = os.path.join(RAW_DATA_DIR, "naukri_jobs.csv")
    linkedin_path = os.path.join(RAW_DATA_DIR, "linkedin_jobs.csv")

    frames = []

    # === NAUKRI ===
    # Actual columns: Job Title, Key Skills, Location, Role Category,
    # Job Experience Required
    if os.path.exists(naukri_path):
        naukri = pd.read_csv(naukri_path)
        naukri = naukri.rename(columns={
            "Job Title"              : "title",
            "Key Skills"             : "skills",
            "Location"               : "location",
            "Role Category"          : "company",
            "Job Experience Required": "experience"
        })
        naukri["description"] = (
            naukri["skills"].fillna("") + " " +
            naukri.get("Functional Area", pd.Series([""] * len(naukri))).fillna("")
        )
        frames.append(naukri[["title", "company", "location",
                               "skills", "description", "experience"]])

    # === LINKEDIN ===
    # Actual columns: title, company_name, location, skills_desc,
    # description, formatted_experience_level
    if os.path.exists(linkedin_path):
        linkedin = pd.read_csv(linkedin_path)
        linkedin = linkedin.rename(columns={
            "title"                    : "title",
            "company_name"             : "company",
            "location"                 : "location",
            "skills_desc"              : "skills",
            "description"              : "description",
            "formatted_experience_level": "experience"
        })
        frames.append(linkedin[["title", "company", "location",
                                 "skills", "description", "experience"]])

    # === MERGE both datasets ===
    df = pd.concat(frames, ignore_index=True)
    df.drop_duplicates(inplace=True)
    df.dropna(subset=["title", "description"], inplace=True)
    df.fillna("", inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"✅ Total jobs in merged corpus: {len(df)}")
    print(f"✅ Columns: {df.columns.tolist()}")
    return df