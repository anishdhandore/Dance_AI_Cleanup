from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# Load from Hugging Face Hub
MODEL_NAME = "anishdhandore/RoBERTa_text_classification"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()  # Evaluation mode

# Define emotion categories (replace with your exact order)
EMOTION_CATEGORIES = [
    'Positive_Affect_Joy',
    'Sadness_Low_Arousal_Negative',
    'Anger_High_Arousal_Negative',
    'Fear_Anxiety',
    'Surprise_Epistemic',
    'Neutral_Cat'
]

# Test input
text = "I feel so sad and angry about the situation."

# Tokenize
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)

# Predict
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.sigmoid(logits)[0].numpy()  # For multi-label classification

# Thresholding
threshold = 0.3
predicted_emotions = [EMOTION_CATEGORIES[i] for i, p in enumerate(probs) if p > threshold]

# If none above threshold, pick the max
if not predicted_emotions:
    predicted_emotions = [EMOTION_CATEGORIES[np.argmax(probs)]]

# Output
print("Text:", text)
print("Predicted emotions:", predicted_emotions)
print("Probabilities:", dict(zip(EMOTION_CATEGORIES, probs.round(3))))
