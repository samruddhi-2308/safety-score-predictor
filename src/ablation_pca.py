from pathlib import Path
from time import perf_counter
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from modeling import CRIME_FEATURES, get_feature_columns


PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
DATA_FILE = PROCESSED_DIR / "crime_population_labeled.csv"


def _build_no_pca_pipeline(feature_columns, random_state=42):
    model = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
    return Pipeline(
        steps=[
            (
                "preprocess",
                ColumnTransformer(
                    transformers=[("features", "passthrough", feature_columns)]
                ),
            ),
            ("model", model),
        ]
    )


def _build_partial_pca_pipeline(feature_columns, random_state=42):
    other_features = [column for column in feature_columns if column not in CRIME_FEATURES]
    model = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
    return Pipeline(
        steps=[
            (
                "preprocess",
                ColumnTransformer(
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
                ),
            ),
            ("model", model),
        ]
    )


def _build_full_pca_pipeline(feature_columns, random_state=42):
    model = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
    return Pipeline(
        steps=[
            (
                "preprocess",
                ColumnTransformer(
                    transformers=[
                        (
                            "all_pca",
                            Pipeline(
                                steps=[
                                    ("scale", StandardScaler()),
                                    ("pca", PCA(n_components=0.95)),
                                ]
                            ),
                            feature_columns,
                        )
                    ]
                ),
            ),
            ("model", model),
        ]
    )


def run_ablation(repeats=10, random_state=42):
    df = pd.read_csv(DATA_FILE)
    feature_columns = get_feature_columns(df)
    X = df[feature_columns]
    y = df["safety_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y
    )

    configs = {
        "no_pca": _build_no_pca_pipeline,
        "partial_pca": _build_partial_pca_pipeline,
        "full_pca": _build_full_pca_pipeline,
    }

    rows = []
    for config_name, builder in configs.items():
        fit_times = []
        pred_times = []
        acc = []
        precision = []
        recall = []
        f1 = []

        for _ in range(repeats):
            model = builder(feature_columns=feature_columns, random_state=random_state)

            t0 = perf_counter()
            model.fit(X_train, y_train)
            t1 = perf_counter()
            y_pred = model.predict(X_test)
            t2 = perf_counter()

            fit_times.append(t1 - t0)
            pred_times.append(t2 - t1)
            acc.append(accuracy_score(y_test, y_pred))
            precision.append(precision_score(y_test, y_pred, average="weighted"))
            recall.append(recall_score(y_test, y_pred, average="weighted"))
            f1.append(f1_score(y_test, y_pred, average="weighted"))

        rows.append(
            {
                "config": config_name,
                "repeats": repeats,
                "fit_time_mean_s": np.mean(fit_times),
                "fit_time_std_s": np.std(fit_times),
                "pred_time_mean_s": np.mean(pred_times),
                "accuracy_mean": np.mean(acc),
                "precision_weighted_mean": np.mean(precision),
                "recall_weighted_mean": np.mean(recall),
                "f1_weighted_mean": np.mean(f1),
            }
        )

    result_df = pd.DataFrame(rows).sort_values(by="fit_time_mean_s")
    print("=== PCA Ablation (fixed split/seed, repeated runs) ===")
    print(result_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    fastest = result_df.iloc[0]
    print("\nFastest config:", fastest["config"])


if __name__ == "__main__":
    run_ablation(repeats=10, random_state=42)