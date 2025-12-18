import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import joblib
from sklearn.preprocessing import StandardScaler


# Paths
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_DIR.mkdir(exist_ok=True)

DATA_PATH = PROCESSED_DIR / "crime_population_labeled.csv"
MODEL_PATH = MODEL_DIR / "rf_safety_model.pkl"
PCA_PATH = MODEL_DIR / "pca_transformer.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"

# Load data
df = pd.read_csv(DATA_PATH)

# Features & target
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
# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)


# Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_pca, y)

# Save model + PCA

joblib.dump(rf, MODEL_PATH)
joblib.dump(pca, PCA_PATH)
joblib.dump(scaler, SCALER_PATH)

print(f"Model saved to: {MODEL_PATH}")
print(f"PCA saved to: {PCA_PATH}")
print(f"Scaler saved to: {SCALER_PATH}")
