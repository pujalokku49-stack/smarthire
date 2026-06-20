import fitz  # PyMuPDF
from docx import Document

def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    return " ".join(page.get_text() for page in doc)

def extract_text_from_docx(file_path):
    doc = Document(file_path)
    return " ".join(para.text for para in doc.paragraphs)

def parse_resume(file_path):
    if file_path.endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith(".docx"):
        return extract_text_from_docx(file_path)
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()