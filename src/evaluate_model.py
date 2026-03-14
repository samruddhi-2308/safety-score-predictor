from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split

from modeling import build_model_pipeline, get_feature_columns


PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
DATA_FILE = PROCESSED_DIR / "crime_population_labeled.csv"


def _mean_ci(values, z=1.96):
    arr = np.array(values, dtype=float)
    mean = arr.mean()
    if len(arr) <= 1:
        return mean, mean, mean
    std = arr.std(ddof=1)
    margin = z * (std / np.sqrt(len(arr)))
    return mean, mean - margin, mean + margin


def evaluate_model(n_splits=5, random_state=42):
    df = pd.read_csv(DATA_FILE)
    feature_columns = get_feature_columns(df)
    X = df[feature_columns]
    y = df["safety_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y
    )

    pipeline = build_model_pipeline(feature_columns=feature_columns, random_state=random_state)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print("=== Holdout Metrics (single split) ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Weighted Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"Macro Precision: {precision_score(y_test, y_pred, average='macro'):.4f}")
    print(f"Weighted Recall: {recall_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"Weighted F1: {f1_score(y_test, y_pred, average='weighted'):.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = {"accuracy": [], "precision_macro": [], "recall_macro": [], "f1_macro": []}

    for train_idx, test_idx in cv.split(X, y):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        model = build_model_pipeline(feature_columns=feature_columns, random_state=random_state)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)

        scores["accuracy"].append(accuracy_score(y_te, preds))
        scores["precision_macro"].append(precision_score(y_te, preds, average="macro"))
        scores["recall_macro"].append(recall_score(y_te, preds, average="macro"))
        scores["f1_macro"].append(f1_score(y_te, preds, average="macro"))

    print(f"\n=== Stratified {n_splits}-Fold CV (mean ± 95% CI) ===")
    for metric_name, values in scores.items():
        mean, low, high = _mean_ci(values)
        print(f"{metric_name}: {mean:.4f} ({low:.4f}, {high:.4f})")

    print("\nLeakage Note:")
    print(
        "Current safety labels are derived from crime-rate features used in training, "
        "so metrics likely overestimate generalization to externally defined safety labels."
    )


if __name__ == "__main__":
    evaluate_model()