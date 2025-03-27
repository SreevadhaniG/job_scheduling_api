from flask import Flask, request, jsonify
import os
import json
import firebase_admin
from firebase_admin import credentials, firestore
import joblib
import base64
import numpy as np
import tempfile  # Required to handle model loading correctly

app = Flask(__name__)

# Load Firebase credentials from environment variable
firebase_credentials = json.loads(os.getenv("FIREBASE_CREDENTIALS"))

# Initialize Firebase
cred = credentials.Certificate(firebase_credentials)
firebase_admin.initialize_app(cred)
db = firestore.client()

# Load model from Firestore
def load_model():
    """Fetches the trained model from Firestore and decodes it."""
    doc_ref = db.collection("models").document("job_scheduler")
    doc = doc_ref.get()

    if doc.exists:
        encoded_model = doc.to_dict().get("model_data")
        model_bytes = base64.b64decode(encoded_model)

        # Save the model bytes to a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_model:
            temp_model.write(model_bytes)
            temp_model_path = temp_model.name  # Store the file path
        
        # Load the model from the temp file
        model = joblib.load(temp_model_path)
        return model
    else:
        raise ValueError("❌ No model found in Firestore!")

# Load the model once
model = load_model()

@app.route("/")
def home():
    return jsonify({"message": "✅ Job Scheduling API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    """
    Predicts the priority based on input JSON data.
    Expected JSON format:
    {
        "days_left": 3,
        "quantity": 10,
        "workforce": 5
    }
    """
    try:
        data = request.json
        features = np.array([[data["days_left"], data["quantity"], data["workforce"]]])
        prediction = model.predict(features)[0]

        return jsonify({"priority": int(prediction)})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
