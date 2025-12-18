import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def compute_feature_importance():
    data_path = PROCESSED_DIR / "crime_population_labeled.csv"

    df = pd.read_csv(data_path)

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

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, y)

    importance_df = (
        pd.DataFrame({
            "feature": feature_cols,
            "importance": model.feature_importances_
        })
        .sort_values(by="importance", ascending=False)
        .reset_index(drop=True)
    )

    print("\nFeature Importances:\n")
    print(importance_df)


if __name__ == "__main__":
    compute_feature_importance()
