# Medical Cost (Insurance) — Regression on Kaggle with a Clean ML Pipeline

Predict **insurance charges** using the Kaggle dataset  
**`mirichoi0218/insurance`** with a reproducible scikit-learn pipeline.

## What this project shows
- **Direct Kaggle download** using API token
- **Sanity checks**: shape, head, missing values, duplicates
- **Preprocessing pipeline**: numeric (median+scale), categorical (mode+one-hot)
- **Baseline model**: Linear Regression (RMSE/MAE/R² + residuals plot)
- **Stronger model**: RandomForestRegressor + side-by-side comparison
- **Artifact saved**: one `.joblib` file with model + preprocessing + feature schema

## Dataset
- Kaggle: `mirichoi0218/insurance`
- Target: `charges`
- Features include: `age`, `sex`, `bmi`, `children`, `smoker`, `region`

## Quickstart
1. **Kaggle API Token**
   - Go to Kaggle → Account → **Create New API Token** (downloads `kaggle.json`)
2. **Notebook Step 1**
   - Fill `KAGGLE_USERNAME` and `KAGGLE_KEY` at the top of the cell
   - Run to download and unzip into `./data/insurance_regression/`
3. **Notebook Step 2**
   - Load `insurance.csv`, sanity checks, split into train/validation
4. **Notebook Step 3**
   - Preprocess (numeric+categorical), train **Linear Regression**, view RMSE/MAE/R² and residuals
5. **Notebook Step 4**
   - Train **RandomForestRegressor**, compare metrics, plot predicted vs actual, permutation importance
6. **Notebook Step 5**
   - Save artifact → `./artifacts/insurance_regression_best_model.joblib`
   - Reload and run inference demo (validation rows + one manual row)

## Inference in any script
```python
import joblib, pandas as pd, numpy as np

art = joblib.load("./artifacts/insurance_regression_best_model.joblib")
pipe = art["model"]
cols = art["features"]

# Build a single example row with all required features
# (Fill these with appropriate values for your use-case)
row = {c: None for c in cols}
# e.g., typical values
row.update({
    "age": 35,         # numeric
    "sex": "male",     # categorical
    "bmi": 28.0,       # numeric
    "children": 1,     # numeric
    "smoker": "no",    # categorical
    "region": "southeast"  # categorical
})
X_one = pd.DataFrame([row], columns=cols)

pred = pipe.predict(X_one)[0]
print("Predicted charges:", f"{pred:,.2f}")
