import joblib
import pandas as pd
from pathlib import Path

# Paths
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_PATH = MODEL_DIR / "rf_safety_model.pkl"
PCA_PATH = MODEL_DIR / "pca_transformer.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"


# Load model + PCA + Scaler
rf = joblib.load(MODEL_PATH)
pca = joblib.load(PCA_PATH)
scaler = joblib.load(SCALER_PATH)

# Example input data
example_data = {
    "violent_crime_rate": [1500],
    "property_crime_rate": [4000],
    "homicide_rate": [30],
    "population": [150000],
    "latitude": [40.7128],
    "longitude": [-74.0060]
}

df_input = pd.DataFrame(example_data)

# Scale features
X_scaled = scaler.transform(df_input)

# Apply PCA
X_pca = pca.transform(X_scaled)

# Predict
pred = rf.predict(X_pca)
print(f"Predicted Safety Label: {pred[0]}")
