from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_FEATURE_COLUMNS = [
    "violent_crime_rate",
    "property_crime_rate",
    "homicide_rate",
    "population",
    "latitude",
    "longitude",
]

CRIME_FEATURES = [
    "violent_crime_rate",
    "property_crime_rate",
    "homicide_rate",
]

OTHER_FEATURES = ["population", "latitude", "longitude"]

TEMPORAL_OPTIONAL_COLUMNS = [
    "coverage_ratio",
    "violent_crime_rate_lag1",
    "property_crime_rate_lag1",
    "homicide_rate_lag1",
    "violent_crime_rate_rolling3",
    "property_crime_rate_rolling3",
    "homicide_rate_rolling3",
    "violent_crime_rate_trend",
    "property_crime_rate_trend",
    "homicide_rate_trend",
]


def get_feature_columns(df) -> list:
    features = list(BASE_FEATURE_COLUMNS)
    temporal = [column for column in TEMPORAL_OPTIONAL_COLUMNS if column in df.columns]
    features.extend(temporal)
    return features


def build_model_pipeline(feature_columns: list, random_state: int = 42) -> Pipeline:
    other_features = [column for column in feature_columns if column not in CRIME_FEATURES]

    preprocess = ColumnTransformer(
        transformers=[
            (
                "crime_pca",
                Pipeline(
                    steps=[
                        ("scale", StandardScaler()),
                        ("pca", PCA(n_components=2)),
                    ]
                ),
                CRIME_FEATURES,
            ),
            ("other", "passthrough", other_features),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=random_state,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", model),
        ]
    )