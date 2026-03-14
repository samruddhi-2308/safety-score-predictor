import pandas as pd
from pathlib import Path
import joblib
from modeling import build_model_pipeline, get_feature_columns


# Paths
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_DIR.mkdir(exist_ok=True)

DATA_PATH = PROCESSED_DIR / "crime_population_labeled.csv"
MODEL_PATH = MODEL_DIR / "safety_pipeline.pkl"

# Load data
df = pd.read_csv(DATA_PATH)

# Features & target
feature_columns = get_feature_columns(df)
X = df[feature_columns]
y = df["safety_label"]
pipeline = build_model_pipeline(feature_columns=feature_columns, random_state=42)
pipeline.fit(X, y)

artifact = {
	"pipeline": pipeline,
	"feature_columns": feature_columns,
}
joblib.dump(artifact, MODEL_PATH)

print(f"Model saved to: {MODEL_PATH}")
