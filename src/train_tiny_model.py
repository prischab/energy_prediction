import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestRegressor


def train_demo_model():
    DATA_PATH = "data/feature_engineered.csv"
    MODEL_DIR = "models"
    MODEL_PATH = os.path.join(MODEL_DIR, "demo_model.pkl")

    # Load data
    df = pd.read_csv(DATA_PATH)

    # Select only numeric columns
    num_df = df.select_dtypes(include="number").fillna(0)

    # Define target and features
    if "global_active_power" not in num_df.columns:
        raise ValueError("global_active_power column missing from feature_engineered.csv")

    X = num_df.drop(columns=["global_active_power"])
    y = num_df["global_active_power"]

    # Train RandomForest model
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"✅ Model trained and saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_demo_model()
