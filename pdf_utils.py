# utils/pdf_utils.py
from PyPDF2 import PdfReader

def extract_text_from_pdf(file_stream):
    reader = PdfReader(file_stream)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text.strip()