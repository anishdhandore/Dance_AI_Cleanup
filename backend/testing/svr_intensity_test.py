# import joblib
# import requests
# from io import BytesIO

# url = "https://huggingface.co/anishdhandore/SVR_text_intensity/resolve/main/final_svr_model.joblib"

# response = requests.get(url)
# if response.status_code == 200:
#     svr_model = joblib.load(BytesIO(response.content))
#     print("Model loaded successfully!")
# else:
#     print(f"Failed to download model. Status code: {response.status_code}")
import requests
import joblib
from io import BytesIO

# Download the model
model_url = "https://huggingface.co/anishdhandore/SVR_text_intensity/resolve/main/final_svr_model.joblib"
vectorizer_url = "https://huggingface.co/anishdhandore/SVR_text_intensity/resolve/main/tfidf_vectorizer.joblib"

# Load the SVR model
svr_model = joblib.load(BytesIO(requests.get(model_url).content))

# Load the TF-IDF vectorizer
# vectorizer = joblib.load(BytesIO(requests.get(vectorizer_url).content))

# Input text
text = ["heyyy"]

# Transform to TF-IDF vector
# X = vectorizer.transform(text)

# Predict intensity
intensity = svr_model.predict(text)
print(f"Predicted intensity: {intensity[0]}")
