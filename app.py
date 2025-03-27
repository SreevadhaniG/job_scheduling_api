from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, firestore
import joblib
import base64
import numpy as np

app = Flask(__name__)

# Initialize Firebase
cred = credentials.Certificate("firebase_credentials.json")  # Ensure this file is in your project directory
firebase_admin.initialize_app(cred)
db = firestore.client()

# Load model from Firestore
def load_model():
    """Fetches the trained model from Firestore and decodes it."""
    doc_ref = db.collection("models").document("job_scheduler")
    doc = doc_ref.get()
    
    if doc.exists:
        encoded_model = doc.to_dict().get("model_base64")
        model_bytes = base64.b64decode(encoded_model)
        model = joblib.load(model_bytes)
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
