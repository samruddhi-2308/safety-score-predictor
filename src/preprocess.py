import pandas as pd
from pathlib import Path
import re

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def _infer_year_from_filename(file_path: Path):
    match = re.search(r"(19|20)\d{2}", file_path.stem)
    return int(match.group(0)) if match else None


def _load_raw_files() -> pd.DataFrame:
    files = sorted(RAW_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError("No CSV files found in data/raw/")

    frames = []
    for file_path in files:
        df = pd.read_csv(file_path)
        inferred_year = _infer_year_from_filename(file_path)

        if "Year" not in df.columns:
            df["Year"] = inferred_year

        df["source_file"] = file_path.name
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def preprocess_crime_population_data():
    output_path = PROCESSED_DIR / "crime_population_clean.csv"
    df = _load_raw_files()

    # Select only columns we will actually use
    selected_columns = [
        "Agency",
        "State",
        "Year",
        "Months",
        "Population",
        "Violent_crime_total",
        "Property_crime_total",
        "Murder_and_Manslaughter",
        "lat",
        "long",
        "source_file",
    ]

    missing_columns = [column for column in selected_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing expected columns after loading raw files: {missing_columns}")

    df = df[selected_columns]

    # Standardize column names
    df = df.rename(columns={
        "Agency": "agency",
        "State": "state",
        "Year": "year",
        "Months": "months_reported",
        "Population": "population",
        "Violent_crime_total": "violent_crime_total",
        "Property_crime_total": "property_crime_total",
        "Murder_and_Manslaughter": "homicide_total",
        "lat": "latitude",
        "long": "longitude"
    })

    # Basic sanity filters
    df = df[df["population"] > 0]
    df["months_reported"] = pd.to_numeric(df["months_reported"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print("Preprocessing complete.")
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    preprocess_crime_population_data()
