import joblib
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

model = joblib.load(os.path.join(MODEL_DIR, "random_forest_model.pkl"))

with open(os.path.join(MODEL_DIR, "feature_columns.json")) as f:
    features = json.load(f)

importance = model.feature_importances_

print("🔍 Feature Importance:")
for f, i in sorted(zip(features, importance), key=lambda x: x[1], reverse=True):
    print(f"{f}: {round(i, 3)}")
