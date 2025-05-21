import joblib
import os

# Same source documents you used for vectorizing
source_texts = [
    "This is the first source document.",
    "Here is another source document.",
    "This one is different from the others."
]

# Save the source texts
os.makedirs("model", exist_ok=True)
joblib.dump(source_texts, "model/source_texts.pkl")
print("Source texts saved successfully!")
import joblib

texts = joblib.load("model/source_texts.pkl")
for i, t in enumerate(texts):
    print(f"Text {i+1} words:", len(t.split()))
