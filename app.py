from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# =========================
# Load models
# =========================
heart_model = joblib.load("artifacts/heart_pipeline.pkl")

diabetes_model = joblib.load("artifacts/diabetes_pipeline.pkl")


print(type(heart_model))
print(type(diabetes_model))


# =========================
# Feature order (VERY IMPORTANT)
# =========================
FEATURE_ORDER = [
    'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
    'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump',
    'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 'MentHlth',
    'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income',
    'Vaccinated', 'Had_COVID'
]

# =========================
# Age Encoding (BRFSS standard)
# =========================
def encode_age(age):
    if age < 25: return 1
    elif age < 30: return 2
    elif age < 35: return 3
    elif age < 40: return 4
    elif age < 45: return 5
    elif age < 50: return 6
    elif age < 55: return 7
    elif age < 60: return 8
    elif age < 65: return 9
    elif age < 70: return 10
    elif age < 75: return 11
    elif age < 80: return 12
    else: return 13

# =========================
# Routes
# =========================
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = {}

        for feature in FEATURE_ORDER:
            if feature == "Age":
                age_val = int(request.form["Age"])
                data["Age"] = encode_age(age_val)
            else:
                data[feature] = float(request.form[feature])

        input_df = pd.DataFrame([data])[FEATURE_ORDER]

        heart_pred = heart_model.predict(input_df)[0]
        heart_prob = heart_model.predict_proba(input_df)[0][1] * 100

        diab_pred = diabetes_model.predict(input_df)[0]
        diab_prob = diabetes_model.predict_proba(input_df)[0][1] * 100

        return render_template(
            "index.html",
            heart_result="High Risk" if heart_pred == 1 else "Low Risk",
            heart_prob=round(heart_prob, 2),
            diabetes_result="High Risk" if diab_pred == 1 else "Low Risk",
            diabetes_prob=round(diab_prob, 2)
        )

    except Exception as e:
        return render_template("index.html", error=str(e))

if __name__ == "__main__":
    app.run(debug=True)