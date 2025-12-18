import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def add_crime_rate_features():
    input_path = PROCESSED_DIR / "crime_population_clean.csv"
    output_path = PROCESSED_DIR / "crime_population_features.csv"

    if not input_path.exists():
        raise FileNotFoundError("Clean dataset not found. Run preprocess.py first.")

    df = pd.read_csv(input_path)

    # Rate per 100k population
    RATE_BASE = 100_000

    df["violent_crime_rate"] = (df["violent_crime_total"] / df["population"]) * RATE_BASE
    df["property_crime_rate"] = (df["property_crime_total"] / df["population"]) * RATE_BASE
    df["homicide_rate"] = (df["homicide_total"] / df["population"]) * RATE_BASE

    # Drop raw totals (no longer needed for modeling)
    df = df.drop(columns=[
        "violent_crime_total",
        "property_crime_total",
        "homicide_total"
    ])

    df.to_csv(output_path, index=False)

    print("Feature engineering complete.")
    print("Final shape:", df.shape)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    add_crime_rate_features()
