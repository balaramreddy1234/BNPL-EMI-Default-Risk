import joblib
import json
import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Load artifacts
model = joblib.load(os.path.join(MODEL_DIR, "random_forest_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

with open(os.path.join(MODEL_DIR, "feature_columns.json")) as f:
    FEATURE_COLUMNS = json.load(f)


def predict_risk(input_dict):
    """
    input_dict = {
        income, loan, emi, tenure, ontime, delays, credit
    }
    """

    df = pd.DataFrame([input_dict])

    # Feature engineering (MUST match training)
    df["emi_income_ratio"] = df["emi"] / (df["income"] + 1)
    df["delay_ratio"] = df["delays"] / (df["ontime"] + df["delays"] + 1)

    df = df[FEATURE_COLUMNS]

    X_scaled = scaler.transform(df)

    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0][1]

    risk = "High Risk" if prediction == 1 else "Low Risk"

    return risk, float(probability)
