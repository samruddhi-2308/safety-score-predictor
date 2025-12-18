# Safety Score Predictor

An end-to-end machine learning project that predicts regional safety levels using crime and population data.

## Overview
This project builds a full ML pipeline including data preprocessing, feature engineering, target construction, model training, dimensionality reduction (PCA), model persistence, and inference.

Safety levels are classified as **Low**, **Medium**, or **High**, relative to crime rates within the dataset.

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Random Forest
- PCA

## 📌 Machine Learning Workflow

This project implements a complete, production-style machine learning pipeline:

Raw Data
 → 
Data Preprocessing
 → 
Feature Engineering (Per-Capita Crime Rates)
 → 
Safety Label Construction (Low / Medium / High)
 → 
Model Training (Random Forest)
 → 
Dimensionality Reduction (PCA)
 → 
Model Persistence (Saved Model + Transformer)
 → 
Inference on New, Unseen Data


## How to Run

### 1. Install dependencies
``bash
pip install -r requirements.txt


###2. Predict safety for new data
``bash
src/predictor.py

##Notes

- Safety labels are relative (quantile-based), not absolute crime thresholds.
- Model and PCA transformer are pre-trained and stored for reuse.
