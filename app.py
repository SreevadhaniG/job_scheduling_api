from flask import Flask, request, jsonify
import os
import json
import firebase_admin
from firebase_admin import credentials, firestore
import joblib
import base64
import numpy as np
import tempfile  # Required to handle model loading correctly
import pandas as pd

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

def fetch_employees():
    """Fetches employees from Firestore and sorts them by highest rating."""
    employees_ref = db.collection("employees")
    employees = employees_ref.stream()

    employee_list = []
    for emp in employees:
        emp_data = emp.to_dict()
        employee_list.append({
            "id": emp.id,
            "name": emp_data.get("name"),
            "ratings": emp_data.get("ratings", 0)  # Default rating = 0
        })

    return sorted(employee_list, key=lambda x: x["ratings"], reverse=True)  # Sort by rating

def fetch_orders():
    """Fetches and preprocesses orders from Firestore."""
    orders_ref = db.collection("orders")
    orders = orders_ref.stream()

    data = []
    order_ids = []

    for order in orders:
        order_data = order.to_dict()
        customer_details = order_data.get("customerDetails", {})
        delivery_date_str = customer_details.get("deliveryDate")
        quantity = order_data.get("quantity", 1)  # Default: 1
        workforce = order_data.get("workforce", 2)  # Default: 2

        if delivery_date_str:
            delivery_date = pd.to_datetime(delivery_date_str)
            days_left = (delivery_date - pd.Timestamp.today()).days

            data.append([days_left, quantity, workforce])
            order_ids.append(order.id)
        else:
            print(f"⚠️ Warning: Order {order.id} missing 'deliveryDate'. Skipping...")

    return (pd.DataFrame(data, columns=["days_left", "quantity", "workforce"]), order_ids) if data else (None, None)

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

@app.route("/schedule_jobs", methods=["GET"])
def schedule_jobs():
    """Schedules jobs and assigns employees to them based on priority."""
    global model

    if not model:
        return jsonify({"error": "❌ Model not found!"}), 500

    df, order_ids = fetch_orders()
    employees = fetch_employees()

    if df is not None and employees:
        df["priority"] = df["days_left"].apply(lambda x: 3 if x <= 1 else (2 if x <= 3 else 1))

        # Predict priorities
        X = df[["days_left", "quantity", "workforce"]]
        predicted_priorities = model.predict(X)

        assignments = []

        for doc_id, priority, workforce_needed in zip(order_ids, predicted_priorities, df["workforce"]):
            available_employees = employees[:workforce_needed]
            employee_ids = [emp["id"] for emp in available_employees]

            assignments.append({"order_id": doc_id, "employees": employee_ids})
            employees = employees[workforce_needed:]  # Remove assigned employees

        # Store assignments in Firestore
        for assignment in assignments:
            db.collection("orders").document(assignment["order_id"]).update({
                "assignedEmployees": assignment["employees"]
            })

        return jsonify({"message": "✅ Job scheduling complete!", "assignments": assignments}), 200
    else:
        return jsonify({"error": "❌ No valid orders or employees found!"}), 400

if __name__ == "__main__":
    app.run(debug=True)
