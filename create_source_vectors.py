from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

# Load vectorizer
vectorizer = joblib.load("model/vectorizer.pkl")

# Example source documents (replace with your real text sources)
source_documents = [
    "This is the first source document.",
    "Here is another source document.",
    "This one is different from the others."
]

# Vectorize the source documents
source_vectors = vectorizer.transform(source_documents)

# Save vectors
os.makedirs("model", exist_ok=True)
joblib.dump(source_vectors, "model/source_vectors.pkl")
print("Source vectors saved successfully!")
