import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score
from time import time

# Paths
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
DATA_FILE = PROCESSED_DIR / "crime_population_labeled.csv"

def train_random_forest():
    # Load dataset
    df = pd.read_csv(DATA_FILE)

    # Select features
    feature_cols = [
        "violent_crime_rate",
        "property_crime_rate",
        "homicide_rate",
        "population",
        "latitude",
        "longitude"
    ]
    X = df[feature_cols]
    y = df["safety_label"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Apply PCA only on correlated crime features
    crime_features = ["violent_crime_rate", "property_crime_rate", "homicide_rate"]
    pca = PCA(n_components=2)
    X_train_crime_pca = pca.fit_transform(X_train[crime_features])
    X_test_crime_pca = pca.transform(X_test[crime_features])

    # Keep uncorrelated features
    X_train_final = pd.concat([
        pd.DataFrame(X_train_crime_pca, index=X_train.index, columns=["crime_pc1","crime_pc2"]),
        X_train[["population","latitude","longitude"]]
    ], axis=1)

    X_test_final = pd.concat([
        pd.DataFrame(X_test_crime_pca, index=X_test.index, columns=["crime_pc1","crime_pc2"]),
        X_test[["population","latitude","longitude"]]
    ], axis=1)

    # Random Forest Classifier
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    start_time = time()
    model.fit(X_train_final, y_train)
    end_time = time()

    # Predictions
    y_pred = model.predict(X_test_final)

    # Metrics
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print(f"\nTraining time: {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    train_random_forest()
