import joblib
import json
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_preprocessing import preprocess_data
from feature_engineering import engineer_features

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Load & prepare data
df = preprocess_data()
df = engineer_features(df)

X = df.drop("default", axis=1)
y = df["default"]

feature_columns = list(X.columns)

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train Random Forest
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

# Save model artifacts
joblib.dump(model, os.path.join(MODEL_DIR, "random_forest_model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

with open(os.path.join(MODEL_DIR, "feature_columns.json"), "w") as f:
    json.dump(feature_columns, f)

print("✅ Model training completed successfully")
