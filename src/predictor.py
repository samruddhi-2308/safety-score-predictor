import joblib
import pandas as pd
from pathlib import Path

# Paths
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_PATH = MODEL_DIR / "safety_pipeline.pkl"


# Load model artifact
artifact = joblib.load(MODEL_PATH)
pipeline = artifact["pipeline"]
feature_columns = artifact["feature_columns"]

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

for column in feature_columns:
    if column not in df_input.columns:
        df_input[column] = 0.0

df_input = df_input[feature_columns]

# Predict
pred = pipeline.predict(df_input)
print(f"Predicted Safety Label: {pred[0]}")
