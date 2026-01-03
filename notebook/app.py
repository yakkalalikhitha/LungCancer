from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# -----------------------------------
# Load trained objects
# -----------------------------------
model = joblib.load("lung_cancer_model.pkl")
scaler = joblib.load("scaler.pkl")
gender_encoder = joblib.load("gender_encoder.pkl")
target_encoder = joblib.load("target_encoder.pkl")

# -----------------------------------
# Home Page
# -----------------------------------
@app.route("/")
def home():
    return render_template("index.html")

# -----------------------------------
# Prediction Route
# -----------------------------------
@app.route("/predict", methods=["POST"])
def predict():

    # Read inputs from form
    gender = request.form["gender"]
    age = float(request.form["age"])
    smoking = int(request.form["smoking"])
    yellow_fingers = int(request.form["yellow_fingers"])
    anxiety = int(request.form["anxiety"])
    peer_pressure = int(request.form["peer_pressure"])
    chronic_disease = int(request.form["chronic_disease"])
    fatigue = int(request.form["fatigue"])
    allergy = int(request.form["allergy"])
    wheezing = int(request.form["wheezing"])
    alcohol = int(request.form["alcohol"])
    coughing = int(request.form["coughing"])
    shortness_of_breath = int(request.form["shortness_of_breath"])
    swallowing_difficulty = int(request.form["swallowing_difficulty"])
    chest_pain = int(request.form["chest_pain"])

    # -----------------------------------
    # Encode gender (FIXED)
    # -----------------------------------
    gender_encoded = gender_encoder.transform([gender])[0]

    # Create input array
    input_data = np.array([[
        gender_encoded,
        age,
        smoking,
        yellow_fingers,
        anxiety,
        peer_pressure,
        chronic_disease,
        fatigue,
        allergy,
        wheezing,
        alcohol,
        coughing,
        shortness_of_breath,
        swallowing_difficulty,
        chest_pain
    ]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)[0]

    # Decode result
    result = target_encoder.inverse_transform([prediction])[0]

    return render_template(
        "index.html",
        prediction=result
    )

# -----------------------------------
# Run App
# -----------------------------------
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

