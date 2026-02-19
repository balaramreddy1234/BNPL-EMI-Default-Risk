import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "..", "dataset", "feature_engineered.csv")

def preprocess_data():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    df = df.drop_duplicates()
    df = df.fillna(0)
    return df
