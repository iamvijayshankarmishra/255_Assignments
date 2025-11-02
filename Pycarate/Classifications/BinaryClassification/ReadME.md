# Pima Diabetes — Binary Classification (Kaggle) with Clean ML Pipeline

This project trains a **diabetes risk classifier** using the Kaggle dataset  
**`uciml/pima-indians-diabetes-database`** and a clean, production-ready pipeline.

## What’s inside
- **Step 1:** Download dataset via Kaggle API (datasets, not competitions)
- **Step 2:** Load CSV, sanity checks, stratified split
- **Step 3:** Minimal preprocessing + **Logistic Regression** baseline
- **Step 4:** Stronger model (**Random Forest**) + side-by-side metrics
- **Step 5:** **Threshold tuning (F1)** and **save** a production artifact  
  → `./artifacts/pima_diabetes_best_model.joblib`

## Why this is solid
- Fixes Pima “impossible zeros” (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`) by treating `0` as missing and imputing
- Uses a **single pipeline** for preprocessing (impute/scale) + model, so train/test processing is identical
- Saves **one file** (model + tuned decision threshold + feature order), ready for reuse

## Quickstart
1. **Create Kaggle token**: *Kaggle → Account → Create New API Token* (downloads `kaggle.json`)
2. Put token at `~/.kaggle/kaggle.json` (Unix/mac) or set `ALT_KAGGLE_JSON` to its path inside the notebook
3. Run notebooks/cells in order:
   - Step 1: Kaggle download (creates `./data/pima_diabetes/diabetes.csv`)
   - Step 2: Load/split
   - Step 3: Baseline Logistic Regression
   - Step 4: Random Forest + comparison
   - Step 5: Threshold tuning + save artifact

## Using the saved model (inference)
```python
import joblib, pandas as pd

art = joblib.load("./artifacts/pima_diabetes_best_model.joblib")
pipe = art["model"]
thr  = art["threshold"]
cols = art["features"]

# Example: single row (must include all features)
row = {c: 0 for c in cols}
row.update({"Pregnancies":1,"Glucose":130,"BloodPressure":72,"SkinThickness":20,"Insulin":85,"BMI":30.5,"DiabetesPedigreeFunction":0.5,"Age":45})
X_new = pd.DataFrame([row], columns=cols)

proba = pipe.predict_proba(X_new)[:,1]
pred  = (proba >= thr).astype(int)
print("proba:", float(proba[0]), "pred:", int(pred[0]))
