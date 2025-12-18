import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def preprocess_crime_population_data():
    input_path = RAW_DIR / "crime_population.csv"
    output_path = PROCESSED_DIR / "crime_population_clean.csv"

    if not input_path.exists():
        raise FileNotFoundError("crime_population.csv not found in data/raw/")

    df = pd.read_csv(input_path)

    # Select only columns we will actually use
    selected_columns = [
        "Agency",
        "State",
        "Population",
        "Violent_crime_total",
        "Property_crime_total",
        "Murder_and_Manslaughter",
        "lat",
        "long"
    ]

    df = df[selected_columns]

    # Standardize column names
    df = df.rename(columns={
        "Agency": "agency",
        "State": "state",
        "Population": "population",
        "Violent_crime_total": "violent_crime_total",
        "Property_crime_total": "property_crime_total",
        "Murder_and_Manslaughter": "homicide_total",
        "lat": "latitude",
        "long": "longitude"
    })

    # Basic sanity filters
    df = df[df["population"] > 0]

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print("Preprocessing complete.")
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    preprocess_crime_population_data()
