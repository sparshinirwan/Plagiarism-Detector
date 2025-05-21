from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

# Example training data (replace this with your real training data)
training_data = [
    "This is a document.",
    "This document is another example.",
    "Let's try some more text data."
]

# Initialize and train the vectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit(training_data)

# Make sure 'model' folder exists
os.makedirs("model", exist_ok=True)

# Save the trained vectorizer
joblib.dump(vectorizer, "model/vectorizer.pkl")
print("Vectorizer saved successfully!")
