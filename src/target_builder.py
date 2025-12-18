import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def build_safety_score():
    input_path = PROCESSED_DIR / "crime_population_features.csv"
    output_path = PROCESSED_DIR / "crime_population_labeled.csv"

    if not input_path.exists():
        raise FileNotFoundError("Feature dataset not found. Run feature_engineering.py first.")

    df = pd.read_csv(input_path)

    crime_features = [
        "violent_crime_rate",
        "property_crime_rate",
        "homicide_rate"
    ]

    # Normalize crime rates
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(df[crime_features])

    norm_df = pd.DataFrame(
        normalized,
        columns=[f"{c}_norm" for c in crime_features]
    )

    # Safety score = inverse weighted sum
    df["safety_score"] = (
        1
        - (
            0.5 * norm_df["violent_crime_rate_norm"]
            + 0.3 * norm_df["property_crime_rate_norm"]
            + 0.2 * norm_df["homicide_rate_norm"]
        )
    )

    # Convert score to labels
    df["safety_label"] = pd.qcut(
        df["safety_score"],
        q=3,
        labels=["Low", "Medium", "High"]
    )

    df.to_csv(output_path, index=False)

    print("Target construction complete.")
    print(df["safety_label"].value_counts())
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    build_safety_score()
