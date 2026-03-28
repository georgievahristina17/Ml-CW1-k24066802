# ML Regression Pipeline — Predicting Outcome from Tabular Features

A reproducible end-to-end regression pipeline built with Scikit-learn, 
comparing multiple model families and optimising for out-of-sample R².

## Overview
- Dataset: 10,000 observations, 30 features (numerical + categorical)
- Task: Predict a continuous target variable (`outcome`)
- Best model: Histogram Gradient Boosting (CV R² = 0.4741)

## Approach
1. **EDA & Preprocessing** — outlier removal, StandardScaler, one-hot encoding, 
   ColumnTransformer pipeline
2. **Model Selection** — compared Ridge, Random Forest, and HistGradientBoosting 
   via 5-fold cross-validation
3. **Hyperparameter Tuning** — RandomizedSearchCV (20 candidates, 5-fold CV)

## Structure
- `notebooks/train_model.ipynb` — full workflow
- `requirements.txt` — dependencies

## How to Run
1. Create and activate a virtual environment
2. Install dependencies: `pip install -r requirements.txt`
3. Run the notebook top to bottom
