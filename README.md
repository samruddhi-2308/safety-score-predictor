# Safety Score Predictor

Predicts relative regional safety labels (`Low`, `Medium`, `High`) from crime and population data using a reproducible scikit-learn pipeline with Random Forest and PCA experiments.

## Project Summary

- End-to-end workflow: preprocessing, feature engineering, target construction, training, evaluation, ablation, model persistence, and inference.
- Uses per-100k crime rates (`violent`, `property`, `homicide`) plus population and location features.
- Uses a unified scikit-learn pipeline so train/save/predict transformations stay consistent.
- Supports multi-file raw ingestion for larger multi-year datasets.

## Current Verified Results

From the current dataset in `data/processed/crime_population_labeled.csv`:

- Dataset size: 213 records
- Holdout split accuracy (single run, seed 42): ~0.944
- Weighted precision (single run, seed 42): ~0.947

These numbers can change slightly by seed/split. For stable reporting, use cross-validation metrics from `src/evaluate_model.py`.

## PCA Ablation (No PCA vs Partial PCA vs Full PCA)

Run:

```bash
python src/ablation_pca.py
```

What it does:

- Uses the same train/test split and random seed across all variants.
- Repeats each variant multiple times (`repeats=10` by default).
- Compares:
	- `no_pca`: Random Forest on full feature set.
	- `partial_pca`: PCA on crime-rate block only + passthrough of other features.
	- `full_pca`: PCA on all features (95% explained variance).
- Reports mean fit time and weighted metrics so PCA benefit can be proven empirically.

## Important Limitation

`safety_label` is generated from the same crime-rate features used for training. This introduces target leakage risk and can inflate performance versus a real externally sourced safety label.

## Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn (Pipeline, ColumnTransformer, PCA, RandomForest)

## Workflow

Raw Data
→ Preprocessing (single or multi-file ingestion)
→ Feature Engineering (crime rates per 100k + temporal features)
→ Safety Score + Label Construction
→ Model Training (Random Forest + PCA on crime features)
→ Evaluation (holdout + stratified CV)
→ Ablation Benchmark
→ Model Saving
→ Inference

## Multi-Year Data Scaling and Temporal Features

To scale data and improve quality:

1. Add multiple yearly CSV files into `data/raw/` (for example: `crime_population_2019.csv`, `crime_population_2020.csv`, ...).
2. Run preprocessing and feature engineering scripts.

The preprocessing step automatically:

- Loads all CSVs in `data/raw/`
- Adds `source_file`
- Infers `year` from filename when possible (if no `Year` column exists)

Temporal features are generated when at least 2 years of history exist per agency/state:

- Lag 1 (`*_lag1`)
- Rolling mean over previous 3 periods (`*_rolling3`)
- Year-over-year trend (`*_trend`)

If history is insufficient, temporal features are safely skipped or defaulted.
With the current single raw file, temporal columns are not fully activated yet; add additional yearly files to enable them.

## How to Run

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Run full data pipeline (if needed)

```bash
python src/preprocess.py
python src/feature_engineering.py
python src/target_builder.py
```

### 3) Train model and view quick metrics

```bash
python src/train_model.py
```

### 4) Run robust evaluation (recommended)

```bash
python src/evaluate_model.py
```

### 5) Run PCA ablation benchmark

```bash
python src/ablation_pca.py
```

### 6) Save trained pipeline

```bash
python src/save_model.py
```

### 7) Predict on new input

```bash
python src/predictor.py
```

## Files

- `src/modeling.py`: shared feature definitions and model pipeline
- `src/train_model.py`: holdout training metrics
- `src/evaluate_model.py`: holdout + stratified cross-validation metrics
- `src/ablation_pca.py`: fixed-split repeated-run PCA ablation benchmark
- `src/save_model.py`: saves `models/safety_pipeline.pkl`
- `src/predictor.py`: inference using saved pipeline
