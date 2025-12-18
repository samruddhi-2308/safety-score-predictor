import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

def load_crime_population_data() -> pd.DataFrame:
    """
    Load crime + population dataset.
    """
    file_path = DATA_DIR / "crime_population.csv"

    if not file_path.exists():
        raise FileNotFoundError("crime_population.csv not found in data/raw/")

    df = pd.read_csv(file_path)
    return df


if __name__ == "__main__":
    df = load_crime_population_data()

    print("SHAPE:")
    print(df.shape)

    print("\nCOLUMNS:")
    print(df.columns.tolist())

    print("\nFIRST 5 ROWS:")
    print(df.head())

    print("\nINFO:")
    print(df.info())
