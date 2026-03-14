import pandas as pd
from pathlib import Path
import numpy as np

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    if "year" not in df.columns:
        return df

    if df["year"].dropna().nunique() < 2:
        return df

    group_cols = ["agency", "state"]
    required_group_cols = all(column in df.columns for column in group_cols)
    if not required_group_cols:
        return df

    rate_cols = ["violent_crime_rate", "property_crime_rate", "homicide_rate"]

    df = df.sort_values(group_cols + ["year"]).copy()

    for column in rate_cols:
        df[f"{column}_lag1"] = df.groupby(group_cols)[column].shift(1)
        df[f"{column}_rolling3"] = (
            df.groupby(group_cols)[column]
            .transform(lambda series: series.shift(1).rolling(window=3, min_periods=1).mean())
        )
        df[f"{column}_trend"] = df[column] - df[f"{column}_lag1"]

    trend_cols = [f"{column}_trend" for column in rate_cols]
    lag_cols = [f"{column}_lag1" for column in rate_cols]
    rolling_cols = [f"{column}_rolling3" for column in rate_cols]

    fill_cols = lag_cols + rolling_cols + trend_cols
    df[fill_cols] = df[fill_cols].replace([np.inf, -np.inf], np.nan)
    df[fill_cols] = df[fill_cols].fillna(0.0)
    return df


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

    if "months_reported" in df.columns:
        df["months_reported"] = pd.to_numeric(df["months_reported"], errors="coerce")
        df["coverage_ratio"] = (df["months_reported"] / 12.0).clip(lower=0.0, upper=1.0)
    else:
        df["coverage_ratio"] = 1.0

    df = _add_temporal_features(df)

    # Drop raw totals (no longer needed for modeling)
    df = df.drop(columns=[
        "violent_crime_total",
        "property_crime_total",
        "homicide_total"
    ], errors="ignore")

    df.to_csv(output_path, index=False)

    print("Feature engineering complete.")
    print("Final shape:", df.shape)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    add_crime_rate_features()
