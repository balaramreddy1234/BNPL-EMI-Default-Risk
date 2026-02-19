import joblib
import os
from sklearn.metrics import accuracy_score, confusion_matrix

from data_preprocessing import preprocess_data
from feature_engineering import engineer_features

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

df = preprocess_data()
df = engineer_features(df)

X = df.drop("default", axis=1)
y = df["default"]

scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
model = joblib.load(os.path.join(MODEL_DIR, "random_forest_model.pkl"))

X_scaled = scaler.transform(X)
preds = model.predict(X_scaled)

print("Accuracy:", accuracy_score(y, preds))
print("Confusion Matrix:\n", confusion_matrix(y, preds))
