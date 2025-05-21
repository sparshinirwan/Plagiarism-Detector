import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import os
import logging
from transformers import pipeline

# Set up logging
logging.basicConfig(filename='training.log', level=logging.INFO)

def train_models():
    try:
        # Create model directory if it doesn't exist
        os.makedirs("model", exist_ok=True)
        
        # Load dataset
        if not os.path.exists("dataset.csv"):
            raise FileNotFoundError("dataset.csv not found")
            
        df = pd.read_csv("dataset.csv")
        
        # Filter only plagiarized entries (label = 1)
        df = df[df["label"] == 1]
        
        # Get source texts
        source_texts = df["source_text"].astype(str).tolist()
        
        if not source_texts:
            raise ValueError("No source texts found in dataset")
        
        # Traditional TF-IDF Vectorizer
        vectorizer = TfidfVectorizer().fit(source_texts)
        X = vectorizer.transform(source_texts)
        
        # Paraphrase detection model
        logging.info("Loading paraphrase model...")
        paraphrase_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        source_embeddings = paraphrase_model.encode(source_texts)
        
        # Feedback model
        logging.info("Loading feedback model...")
        feedback_model = pipeline("text-generation", model="gpt2")
        
        # Save all models and data
        joblib.dump(vectorizer, "model/vectorizer.pkl")
        joblib.dump(X, "model/source_vectors.pkl")
        joblib.dump(source_texts, "model/source_texts.pkl")
        joblib.dump(paraphrase_model, "model/paraphrase_model.pkl")
        joblib.dump(source_embeddings, "model/source_embeddings.pkl")
        joblib.dump(feedback_model, "model/feedback_model.pkl")
        
        logging.info("Models trained and saved successfully")
        print("Models trained and saved.")
        
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        raise

if __name__ == '__main__':
    train_models()
