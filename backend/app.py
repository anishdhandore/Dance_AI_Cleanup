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

# Lazy-load models
roberta_tokenizer = None
roberta_model = None
svr_model = None

def load_models():
    global roberta_tokenizer, roberta_model, svr_model
    if roberta_tokenizer is None or roberta_model is None:
        roberta_tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL_NAME)
        roberta_model = AutoModelForSequenceClassification.from_pretrained(ROBERTA_MODEL_NAME)
        roberta_model.eval()

    if svr_model is None:
        response = requests.get(SVR_MODEL_URL)
        svr_model = joblib.load(BytesIO(response.content))
        print("SVR model loaded")

def predict_emotions_roberta(text, threshold=0.3):
    load_models()

    try:
        inputs = roberta_tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        )
        with torch.no_grad():
            outputs = roberta_model(**inputs)
            probabilities = torch.sigmoid(outputs.logits)

        probs = probabilities.numpy()[0]
        above_threshold = (probs > threshold).astype(int)
        emotions_above_threshold = [EMOTION_CATEGORIES[i] for i, pred in enumerate(above_threshold) if pred == 1]

        if not emotions_above_threshold:
            max_idx = np.argmax(probs)
            emotions_above_threshold = [EMOTION_CATEGORIES[max_idx]]

        return emotions_above_threshold
    except Exception as e:
        print(f"Error in RoBERTa prediction: {e}")
        return ["Neutral_Cat"]

@app.route('/predict', methods=['POST'])
def predict():
    if not request.json or 'text' not in request.json:
        return jsonify({'error': 'No text provided'}), 400

    text = request.json['text']
    emotions = predict_emotions_roberta(text)

    intensity = None
    try:
        load_models()
        intensity = float(svr_model.predict([text])[0])
    except Exception as e:
        print(f"Error in SVR prediction: {e}")
        intensity = "Error"

    return jsonify({
        'emotions': emotions,
        'intensity': intensity
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    app.run(host='0.0.0.0', port=port)
