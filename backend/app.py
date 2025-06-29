from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
import numpy as np
import torch
import requests
from io import BytesIO
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)
CORS(app)

# HuggingFace model configurations
ROBERTA_MODEL_NAME = "anishdhandore/RoBERTa_text_classification"
SVR_MODEL_URL = "https://huggingface.co/anishdhandore/SVR_text_intensity/resolve/main/final_svr_model.joblib"

EMOTION_CATEGORIES = [
    'Positive_Affect_Joy',
    'Sadness_Low_Arousal_Negative',
    'Anger_High_Arousal_Negative',
    'Fear_Anxiety',
    'Surprise_Epistemic',
    'Neutral_Cat'
]

# Load models
try:
    # Load RoBERTa model and tokenizer from HuggingFace
    roberta_tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL_NAME)
    roberta_model = AutoModelForSequenceClassification.from_pretrained(ROBERTA_MODEL_NAME)
    roberta_model.eval()  # Set to evaluation mode
    
    # Load SVR model from HuggingFace
    svr_model = joblib.load(BytesIO(requests.get(SVR_MODEL_URL).content))
    print("Models loaded successfully from HuggingFace")
except Exception as e:
    print(f"Error loading models: {e}")
    roberta_tokenizer = None
    roberta_model = None
    svr_model = None

def predict_emotions_roberta(text, threshold=0.3):
    """Predict emotions using the RoBERTa model from HuggingFace"""
    if roberta_tokenizer is None or roberta_model is None:
        return []
    
    try:
        # Tokenize the input text
        inputs = roberta_tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        )
        
        # Get predictions
        with torch.no_grad():
            outputs = roberta_model(**inputs)
            logits = outputs.logits
            probabilities = torch.sigmoid(logits)
        
        # Convert to numpy
        probs = probabilities.numpy()[0]
        
        # Get emotions above threshold
        above_threshold = (probs > threshold).astype(int)
        emotions_above_threshold = [EMOTION_CATEGORIES[i] for i, pred in enumerate(above_threshold) if pred == 1]
        
        # If no emotions above threshold, get the highest probability emotion
        if not emotions_above_threshold:
            max_idx = np.argmax(probs)
            emotions_above_threshold = [EMOTION_CATEGORIES[max_idx]]
            print(f"No emotions above threshold. Using highest probability: {EMOTION_CATEGORIES[max_idx]} ({probs[max_idx]:.3f})")
        else:
            print(f"Emotions above threshold: {emotions_above_threshold}")
            print(f"Probabilities: {dict(zip(EMOTION_CATEGORIES, probs))}")
        
        return emotions_above_threshold
    except Exception as e:
        print(f"Error in RoBERTa prediction: {e}")
        return ["Neutral_Cat"]  # Default fallback

@app.route('/predict', methods=['POST'])
def predict():
    if not request.json or 'text' not in request.json:
        return jsonify({'error': 'No text provided'}), 400

    text = request.json['text']
    emotions = []
    intensity = None

    # Get emotion predictions using RoBERTa model
    emotions = predict_emotions_roberta(text)

    # Get intensity prediction using SVR model
    if svr_model:
        try:
            intensity = float(svr_model.predict([text])[0])
        except Exception as e:
            print(f"Error in SVR prediction: {e}")
            intensity = "Error in prediction"

    return jsonify({
        'emotions': emotions,
        'intensity': intensity
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    app.run(host='0.0.0.0', port=port)