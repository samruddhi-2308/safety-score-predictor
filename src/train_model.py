import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from time import time
from modeling import build_model_pipeline, get_feature_columns

# Paths
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
DATA_FILE = PROCESSED_DIR / "crime_population_labeled.csv"

def train_random_forest():
    # Load dataset
    df = pd.read_csv(DATA_FILE)

    # Select features
    feature_columns = get_feature_columns(df)
    X = df[feature_columns]
    y = df["safety_label"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = build_model_pipeline(feature_columns=feature_columns, random_state=42)

    start_time = time()
    model.fit(X_train, y_train)
    end_time = time()

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Weighted Precision:", precision_score(y_test, y_pred, average="weighted"))
    print("Macro Precision:", precision_score(y_test, y_pred, average="macro"))
    print("Weighted Recall:", recall_score(y_test, y_pred, average="weighted"))
    print("Weighted F1:", f1_score(y_test, y_pred, average="weighted"))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print(f"\nTraining time: {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    train_random_forest()
