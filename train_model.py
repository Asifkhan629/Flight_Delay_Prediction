"""
train_model.py
----------------
Flight Delay Prediction System - Training Script

This script performs the following steps:
1. Generates a realistic synthetic flight dataset (flights.csv) if it does not
   already exist.
2. Cleans and encodes the data.
3. Splits the data into training and testing sets.
4. Trains a Random Forest Classifier to predict flight delays.
5. Evaluates the model (accuracy, confusion matrix, classification report).
6. Saves visualizations (feature importance, confusion matrix) as PNG files.
7. Saves the trained model, encoders, and metrics to model.pkl for use in app.py.

Run this script BEFORE launching the Streamlit app:
    python train_model.py
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

# ------------------------------------------------------------------
# Configuration / Constants
# ------------------------------------------------------------------
RANDOM_SEED = 42
DATA_PATH = "flights.csv"
MODEL_PATH = "model.pkl"
N_ROWS = 6000

AIRLINES = ["SkyJet", "AeroLine", "BlueWing", "SunAir", "NorthStar", "PacificFly"]
CITIES = ["New York", "Los Angeles", "Chicago", "Houston", "Miami",
          "Seattle", "Denver", "Boston", "Atlanta", "Dallas"]
WEATHER_OPTIONS = ["Clear", "Cloudy", "Rain", "Fog", "Storm", "Snow"]
TRAFFIC_LEVELS = ["Low", "Medium", "High"]

np.random.seed(RANDOM_SEED)


# ------------------------------------------------------------------
# STEP 1: Dataset Generation
# ------------------------------------------------------------------
def generate_dataset(n_rows: int = N_ROWS) -> pd.DataFrame:
    """
    Generates a realistic synthetic flight dataset with logical
    relationships between weather, traffic, distance and delays.
    """
    print("Generating synthetic flight dataset ...")

    records = []
    for _ in range(n_rows):
        airline = np.random.choice(AIRLINES)

        source, destination = np.random.choice(CITIES, size=2, replace=False)

        distance = int(np.random.normal(1200, 500))
        distance = max(150, min(distance, 3000))

        dep_hour = np.random.randint(0, 24)
        flight_duration = max(1, distance // 500)
        arr_hour = (dep_hour + flight_duration) % 24

        day_of_week = np.random.randint(1, 8)   # 1 = Monday ... 7 = Sunday
        month = np.random.randint(1, 13)

        weather = np.random.choice(
            WEATHER_OPTIONS,
            p=[0.40, 0.20, 0.15, 0.10, 0.08, 0.07]
        )
        traffic = np.random.choice(TRAFFIC_LEVELS, p=[0.45, 0.35, 0.20])

        # ------------------------------------------------------------
        # Build a "delay probability" using realistic rules so the
        # model has genuine signal to learn from.
        # ------------------------------------------------------------
        delay_prob = 0.10  # base probability of delay

        # Weather impact
        weather_impact = {
            "Clear": 0.00, "Cloudy": 0.05, "Rain": 0.20,
            "Fog": 0.30, "Storm": 0.45, "Snow": 0.35,
        }
        delay_prob += weather_impact[weather]

        # Traffic impact
        traffic_impact = {"Low": 0.00, "Medium": 0.15, "High": 0.30}
        delay_prob += traffic_impact[traffic]

        # Rush hour departures (early morning / evening) are more delayed
        if dep_hour in [6, 7, 8, 17, 18, 19, 20]:
            delay_prob += 0.10

        # Weekends have slightly less business traffic
        if day_of_week in [6, 7]:
            delay_prob -= 0.05

        # Winter holiday months have more congestion
        if month in [12, 1, 7]:
            delay_prob += 0.08

        # Long-distance flights have more chance of accumulating delay
        if distance > 2000:
            delay_prob += 0.05

        delay_prob = np.clip(delay_prob, 0.02, 0.95)
        delay = np.random.binomial(1, delay_prob)

        records.append({
            "Airline": airline,
            "Source": source,
            "Destination": destination,
            "Distance": distance,
            "Scheduled_Departure": dep_hour,
            "Scheduled_Arrival": arr_hour,
            "Day_of_Week": day_of_week,
            "Month": month,
            "Weather": weather,
            "Traffic_Level": traffic,
            "Delay": delay,
        })

    df = pd.DataFrame(records)
    df.to_csv(DATA_PATH, index=False)
    print(f"Dataset saved to '{DATA_PATH}' with {len(df)} rows.")
    return df


def load_or_create_dataset() -> pd.DataFrame:
    """Loads flights.csv if it exists, otherwise generates it."""
    if os.path.exists(DATA_PATH):
        print(f"Found existing dataset at '{DATA_PATH}'. Loading ...")
        df = pd.read_csv(DATA_PATH)
    else:
        df = generate_dataset()
    return df


# ------------------------------------------------------------------
# STEP 2: Data Cleaning
# ------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic data cleaning: drop duplicates and handle missing values."""
    df = df.drop_duplicates().reset_index(drop=True)
    df = df.dropna().reset_index(drop=True)
    return df


# ------------------------------------------------------------------
# STEP 3: Encoding
# ------------------------------------------------------------------
def encode_data(df: pd.DataFrame):
    """
    Encodes categorical columns using LabelEncoder.
    Returns the encoded dataframe and a dictionary of fitted encoders
    so the same transformations can be reused in the Streamlit app.
    """
    categorical_cols = ["Airline", "Source", "Destination", "Weather", "Traffic_Level"]
    encoders = {}

    df_encoded = df.copy()
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        encoders[col] = le

    return df_encoded, encoders


# ------------------------------------------------------------------
# STEP 4 & 5: Train / Test Split + Model Training
# ------------------------------------------------------------------
def train_model(df_encoded: pd.DataFrame):
    """Splits data, trains a Random Forest Classifier, and returns results."""
    feature_cols = [
        "Airline", "Source", "Destination", "Distance",
        "Scheduled_Departure", "Scheduled_Arrival",
        "Day_of_Week", "Month", "Weather", "Traffic_Level",
    ]
    X = df_encoded[feature_cols]
    y = df_encoded["Delay"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    report_text = classification_report(y_test, y_pred)

    print(f"\nModel Accuracy: {accuracy:.4f}\n")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report_text)

    return {
        "model": model,
        "feature_cols": feature_cols,
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "classification_report": report,
        "classification_report_text": report_text,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


# ------------------------------------------------------------------
# STEP 6: Visualizations (saved as PNG for README / reference)
# ------------------------------------------------------------------
def plot_feature_importance(model, feature_cols):
    """Plots and saves a feature importance bar chart."""
    importances = model.feature_importances_
    imp_df = pd.DataFrame({
        "Feature": feature_cols, "Importance": importances
    }).sort_values("Importance", ascending=False)

    plt.figure(figsize=(9, 6))
    sns.barplot(data=imp_df, x="Importance", y="Feature", hue="Feature",
                palette="Blues_r", legend=False)
    plt.title("Feature Importance - Random Forest", fontsize=14, weight="bold")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=120)
    plt.close()
    print("Saved feature_importance.png")

    return imp_df


def plot_confusion_matrix(cm):
    """Plots and saves the confusion matrix as a heatmap."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["On Time", "Delayed"],
        yticklabels=["On Time", "Delayed"]
    )
    plt.title("Confusion Matrix", fontsize=14, weight="bold")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=120)
    plt.close()
    print("Saved confusion_matrix.png")


# ------------------------------------------------------------------
# STEP 7: Save Model Bundle
# ------------------------------------------------------------------
def save_model_bundle(results, encoders):
    """
    Saves the trained model, encoders, feature columns, and evaluation
    metrics into a single model.pkl file for the Streamlit app to load.
    """
    bundle = {
        "model": results["model"],
        "encoders": encoders,
        "feature_cols": results["feature_cols"],
        "accuracy": results["accuracy"],
        "confusion_matrix": results["confusion_matrix"],
        "classification_report": results["classification_report"],
        "classification_report_text": results["classification_report_text"],
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\nModel bundle saved to '{MODEL_PATH}'")


# ------------------------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------------------------
def main():
    print("=" * 60)
    print("FLIGHT DELAY PREDICTION - MODEL TRAINING PIPELINE")
    print("=" * 60)

    df = load_or_create_dataset()
    df = clean_data(df)
    df_encoded, encoders = encode_data(df)

    results = train_model(df_encoded)
    plot_feature_importance(results["model"], results["feature_cols"])
    plot_confusion_matrix(results["confusion_matrix"])

    save_model_bundle(results, encoders)

    print("\nTraining pipeline complete. You can now run:  streamlit run app.py")


if __name__ == "__main__":
    main()
