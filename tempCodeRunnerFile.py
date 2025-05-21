import os
import joblib
import fitz  # PyMuPDF
import re
import numpy as np
from flask import Flask, render_template, request, flash, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key = "your-secret-key-123"

# Configure folders
UPLOAD_FOLDER = 'uploads'
GROUP_FOLDER = 'group_uploads'
HIGHLIGHTED_FOLDER = 'highlighted_pdfs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GROUP_FOLDER, exist_ok=True)
os.makedirs(HIGHLIGHTED_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['GROUP_FOLDER'] = GROUP_FOLDER
app.config['HIGHLIGHTED_FOLDER'] = HIGHLIGHTED_FOLDER

# Load models
vectorizer = joblib.load("model/vectorizer.pkl")
source_vectors = joblib.load("model/source_vectors.pkl")
source_texts = joblib.load("model/source_texts.pkl")

def preprocess_text(text):
    """Enhanced text cleaning"""
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)     # Normalize whitespace
    return text.lower().strip()

def is_generic(text):
    """Check for common phrases that shouldn't trigger plagiarism"""
    generic_phrases = [
        "the", "and", "of", "in", "to", "a", "is", "that", "it",
        "with", "as", "for", "was", "on", "are", "this", "be", "by"
    ]
    words = text.lower().split()
    generic_count = sum(1 for word in words if word in generic_phrases)
    return generic_count > len(words) * 0.5  # More than 50% generic words

def extract_text_from_pdf(filepath):
    """Robust PDF text extraction"""
    try:
        doc = fitz.open(filepath)
        text = ""
        for page in doc:
            text += page.get_text()
        return preprocess_text(text) if text.strip() else None
    except:
        return None

def find_plagiarized_sentences(input_text, source_texts, vectorizer, source_vectors, threshold=0.7):
    """Find which sentences in the input text are plagiarized"""
    # Split input text into sentences
    sentences = [s.strip() for s in re.split(r'[.!?]', input_text) if s.strip()]
    
    plagiarized_sentences = []
    
    for sentence in sentences:
        if not sentence or len(sentence.split()) < 5:  # Skip very short sentences
            continue
            
        # Vectorize the sentence
        sentence_vec = vectorizer.transform([sentence])
        
        # Calculate similarity with all source texts
        scores = cosine_similarity(sentence_vec, source_vectors)[0]
        
        if np.max(scores) > threshold:
            # Get the most similar source text
            most_similar_idx = np.argmax(scores)
            source_text = source_texts[most_similar_idx]
            
            # Store both the sentence and the matching source text
            plagiarized_sentences.append({
                'sentence': sentence,
                'source': source_text,
                'score': np.max(scores)
            })
    
    return plagiarized_sentences

def highlight_pdf(input_pdf_path, output_pdf_path, plagiarized_sentences):
    """Highlight plagiarized sentences in the PDF"""
    doc = fitz.open(input_pdf_path)
    
    for page in doc:
        # Get the text blocks and their bounding boxes
        blocks = page.get_text("dict")["blocks"]
        
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        span_text = span["text"].lower()
                        span_text_clean = re.sub(r'[^\w\s]', '', span_text)
                        
                        # Check against each plagiarized sentence
                        for item in plagiarized_sentences:
                            sentence_clean = re.sub(r'[^\w\s]', '', item['sentence'].lower())
                            
                            # If we find a match in this span
                            if sentence_clean in span_text_clean:
                                # Highlight the entire span (could refine to just the matching words)
                                rect = fitz.Rect(span["bbox"])
                                highlight = page.add_highlight_annot(rect)
                                highlight.set_colors(stroke=(1, 1, 0))  # Yellow highlight
                                highlight.update()
    
    doc.save(output_pdf_path)
    doc.close()

def calculate_similarity(text):
    """Calculate adjusted similarity score with special handling for research papers"""
    input_vector = vectorizer.transform([text])
    raw_scores = cosine_similarity(input_vector, source_vectors)[0]
    max_raw = np.max(raw_scores) * 100
    
    # Check if text appears to be a research paper
    research_keywords = [
        "abstract", "introduction", "methodology", "results", "discussion",
        "conclusion", "references", "literature review", "hypothesis"
    ]
    is_research_paper = any(keyword in text.lower() for keyword in research_keywords)
    
    if is_research_paper:
        # Special handling for research papers (cap at 10%)
        adjusted = min(max_raw * 0.1, 10)  # Scale down to 10% max
    else:
        # Normal handling for other documents
        if max_raw > 80:
            adjusted = 60 + (max_raw - 80) * 0.5
        elif max_raw > 60:
            adjusted = 40 + (max_raw - 60) * 1.0
        elif max_raw > 40:
            adjusted = 20 + (max_raw - 40) * 1.0
        else:
            adjusted = max_raw * 1.0
    
    return round(min(adjusted, 100), 2)

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_type = request.form.get("input_type")
        
        # Handle text input
        if input_type == "text":
            input_text = preprocess_text(request.form.get("text_input", ""))
            if not input_text or len(input_text.split()) < 10:
                return render_template("result.html", 
                                    similarity=0.0,
                                    status="No Plagiarism Detected",
                                    is_pdf=False)

        # Handle file input
        elif input_type == "file":
            file = request.files.get("file_input")
            if not file or not file.filename.lower().endswith(".pdf"):
                flash("Please upload a valid PDF file")
                return redirect(url_for("index"))
            
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            input_text = extract_text_from_pdf(filepath)
            if not input_text:
                flash("Could not extract text from PDF")
                return redirect(url_for("index"))

            # Find plagiarized sentences
            plagiarized_sentences = find_plagiarized_sentences(
                input_text, source_texts, vectorizer, source_vectors
            )
            
            # Create highlighted PDF
            highlighted_filename = f"highlighted_{filename}"
            highlighted_path = os.path.join(app.config['HIGHLIGHTED_FOLDER'], highlighted_filename)
            
            highlight_pdf(filepath, highlighted_path, plagiarized_sentences)

        # Handle group comparison
        elif input_type == "group":
            files = request.files.getlist("group_files")
            if len(files) < 2:
                flash("Please upload at least 2 PDFs for comparison")
                return redirect(url_for("index"))
            
            results = []
            texts = []
            for file in files:
                if file and file.filename.lower().endswith(".pdf"):
                    filepath = os.path.join(app.config['GROUP_FOLDER'], secure_filename(file.filename))
                    file.save(filepath)
                    text = extract_text_from_pdf(filepath)
                    if text:
                        texts.append((file.filename, text))
            
            if len(texts) < 2:
                flash("Need at least 2 valid PDFs for comparison")
                return redirect(url_for("index"))
            
            # Compare all pairs
            for i in range(len(texts)):
                for j in range(i+1, len(texts)):
                    vec1 = vectorizer.transform([texts[i][1]])
                    vec2 = vectorizer.transform([texts[j][1]])
                    score = cosine_similarity(vec1, vec2)[0][0] * 100
                    results.append({
                        "file1": texts[i][0],
                        "file2": texts[j][0],
                        "score": round(score, 2)
                    })
            
            return render_template("result.html",
                                group_results=results,
                                is_pdf=True)

        # Skip generic content
        if is_generic(input_text):
            return render_template("result.html",
                                similarity=0.0,
                                status="No Plagiarism Detected",
                                is_pdf=(input_type == "file"))

        # Calculate similarity
        similarity_score = calculate_similarity(input_text)
        status = "Plagiarism Detected" if similarity_score > 25 else "No Plagiarism Detected"
        
        # For file input, include the download link
        if input_type == "file":
            return render_template("result.html",
                                similarity=similarity_score,
                                status=status,
                                is_pdf=True,
                                filename=filename,
                                highlighted_filename=highlighted_filename)
        else:
            return render_template("result.html",
                                similarity=similarity_score,
                                status=status,
                                is_pdf=False)
    
    return render_template("index.html")

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['HIGHLIGHTED_FOLDER'], filename, as_attachment=True)

if __name__ == "_main_":
    app.run(debug=True)