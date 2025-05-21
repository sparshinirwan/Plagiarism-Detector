import os
import joblib
import fitz  # PyMuPDF
import re
import numpy as np
from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key = "your-secret-key-123"

# Configure folders
UPLOAD_FOLDER = 'uploads'
GROUP_FOLDER = 'group_uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GROUP_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['GROUP_FOLDER'] = GROUP_FOLDER

# Load models
vectorizer = joblib.load("model/vectorizer.pkl")
source_vectors = joblib.load("model/source_vectors.pkl")
source_texts = joblib.load("model/source_texts.pkl")
feedback_model = joblib.load("model/feedback_model.pkl")

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

def generate_feedback(text, similarity_score):
    """Generate AI feedback based on similarity score"""
    if similarity_score < 20:
        return "Risk Level: Low\n- 👍 You're good to go!"
    
    # More specific prompt to prevent irrelevant responses
    prompt = f"""
    You are an academic integrity assistant analyzing a text with {similarity_score}% similarity to known sources.
    The problematic text is: "{text[:500]}" [...] (truncated for brevity)

    Provide specific, constructive feedback about:
    1. Why this might be considered plagiarism
    2. How to properly paraphrase this content
    3. How to cite sources appropriately
    4. Academic writing best practices

    Do NOT include unrelated information, links, or examples.
    Keep the feedback focused and professional.
    """
    
    try:
        feedback = feedback_model(
            prompt,
            max_length=30,
            num_return_sequences=1,
            temperature=0.5,  # Lower temperature for more focused responses
            truncation=True
        )[0]['generated_text']
        
        # Clean up the feedback
        feedback = feedback.replace(prompt, "").strip()
        if not validate_feedback(feedback):
            raise ValueError("Invalid feedback generated")
        return feedback
    except:
        # Fallback feedback if model fails
        return f"""⚠️ *Plagiarism Detected ({similarity_score}%)*
- This content appears too similar to existing sources
- Recommendation: Rewrite in your own words and cite properly
- Consider using paraphrasing techniques and adding original analysis"""

def validate_feedback(feedback):
    """Ensure feedback is relevant and appropriate"""
    blacklist = ["http://", "https://", "www.", ".com", ".org", "use this link", "example.com"]
    
    # Check for blacklisted phrases
    if any(phrase in feedback.lower() for phrase in blacklist):
        return False
    
    # Check minimum length and relevance
    if len(feedback.split()) < 10:
        return False    
    return True

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
                                    feedback="Text too short for analysis",
                                    is_pdf=False)
        # Handle file input
        elif input_type == "file":
            file = request.files.get("file_input")
            if not file or not file.filename.lower().endswith(".pdf"):
                flash("Please upload a valid PDF file")
                return redirect(url_for("index"))
            
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
            file.save(filepath)
            input_text = extract_text_from_pdf(filepath)
            if not input_text:
                flash("Could not extract text from PDF")
                return redirect(url_for("index"))

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
                                feedback="Content appears to be generic phrases",
                                is_pdf=(input_type == "file"))

        similarity_score = calculate_similarity(input_text)
        status = "Plagiarism Detected" if similarity_score > 0 else "No Plagiarism Detected"
        feedback = generate_feedback(input_text, similarity_score)

        return render_template("result.html",
                            similarity=similarity_score,
                            status=status,
                            feedback=feedback,
                            is_pdf=(input_type == "file"))
    
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)