import joblib
import os
import numpy as np

# Define paths relative to this script's location (backend/testing/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "models")) # Up two levels to workspace root, then into models

GOEMOTIONS_MODEL_PATH = os.path.join(MODELS_DIR, "goemotions_multilabel_classifier.joblib")
SVR_MODEL_PATH = os.path.join(MODELS_DIR, "final_svr_model.joblib")

EMOTION_CATEGORIES = [
    'Positive_Affect_Joy',
    'Sadness_Low_Arousal_Negative',
    'Anger_High_Arousal_Negative',
    'Fear_Anxiety',
    'Surprise_Epistemic',
    'Neutral_Cat'
]

def load_models():
    """Loads the GoEmotions and SVR models."""
    try:
        goemotions_model = joblib.load(GOEMOTIONS_MODEL_PATH)
        print(f"GoEmotions model loaded successfully from {GOEMOTIONS_MODEL_PATH}")
    except FileNotFoundError:
        print(f"Error: GoEmotions model not found at {GOEMOTIONS_MODEL_PATH}")
        goemotions_model = None
    except Exception as e:
        print(f"Error loading GoEmotions model: {e}")
        goemotions_model = None

    try:
        svr_model = joblib.load(SVR_MODEL_PATH)
        print(f"SVR model loaded successfully from {SVR_MODEL_PATH}")
    except FileNotFoundError:
        print(f"Error: SVR model not found at {SVR_MODEL_PATH}")
        svr_model = None
    except Exception as e:
        print(f"Error loading SVR model: {e}")
        svr_model = None
        
    return goemotions_model, svr_model

def get_predictions(text, goemotions_model, svr_model):
    """Gets predictions from both models for the given text."""
    predicted_emotions = []
    valence_intensity = None

    if goemotions_model:
        try:
            # The GoEmotions model expects a list or array of texts
            goemotion_preds_proba = goemotions_model.predict_proba([text])[0] # Get probabilities for the first (and only) text
            
            # Assuming a threshold or just taking top probability if needed,
            # for now, let's just get the direct binary predictions
            goemotion_binary_preds = goemotions_model.predict([text])[0]

            predicted_emotions = [EMOTION_CATEGORIES[i] for i, pred in enumerate(goemotion_binary_preds) if pred == 1]
            if not predicted_emotions:
                 predicted_emotions = ["No specific category above threshold, consider Neutral or check probabilities"]

        except Exception as e:
            print(f"Error during GoEmotions prediction: {e}")
            predicted_emotions = ["Error in prediction"]
    else:
        predicted_emotions = ["GoEmotions model not loaded"]

    if svr_model:
        try:
            # The SVR model also expects a list or array of texts
            valence_intensity = svr_model.predict([text])[0]
        except Exception as e:
            print(f"Error during SVR prediction: {e}")
            valence_intensity = "Error in prediction"
    else:
        valence_intensity = "SVR model not loaded"
        
    return predicted_emotions, valence_intensity

if __name__ == "__main__":
    print("Attempting to load models...")
    goemotions_model, svr_model = load_models()

    if not goemotions_model and not svr_model:
        print("Neither model could be loaded. Exiting.")
    elif not goemotions_model:
        print("GoEmotions model failed to load. Predictions for emotion categories will not be available.")
    elif not svr_model:
        print("SVR model failed to load. Valence intensity prediction will not be available.")
    else:
        print("Both models loaded.")

    print("\nEnter text to get dance cues (or type 'quit' to exit):")
    while True:
        input_text = input("> ")
        if input_text.lower() == 'quit':
            break
        if not input_text.strip():
            print("Please enter some text.")
            continue

        emotions, intensity = get_predictions(input_text, goemotions_model, svr_model)
        
        print("--- Dance Cues ---")
        print(f"Input Text: {input_text}")
        print(f"Predicted Emotion(s): {emotions}")
        if isinstance(intensity, (float, np.float64)):
             print(f"Predicted Valence Intensity: {intensity:.4f}")
        else:
            print(f"Predicted Valence Intensity: {intensity}")
        print("--------------------\n") 