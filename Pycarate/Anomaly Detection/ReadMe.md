# Credit Card Fraud Detection — Kaggle → Production (Calibrated Logistic + Isolation Forest)

End-to-end anomaly/fraud detection using Kaggle’s **`mlg-ulb/creditcardfraud`** dataset.
We build supervised and unsupervised baselines, calibrate probabilities, pick a **cost-sensitive threshold**, and save a single **production artifact**.

## What this project covers
- ✅ Direct Kaggle download (token-based)
- ✅ Robust scaling for `Time` and `Amount` (PCA features already standardized)
- ✅ Stratified train/validation split (preserves rare class)
- ✅ Two baselines:
  - **Logistic Regression (supervised)** with `class_weight`
  - **Isolation Forest (unsupervised)** for anomaly scoring
- ✅ Metrics: **ROC-AUC**, **PR-AUC**, ROC/PR curves
- ✅ **Isotonic calibration** of probabilities (keeps the better PR-AUC)
- ✅ **Cost-sensitive thresholding** (minimize FN/FP cost)
- ✅ Single **artifact** with model + features + chosen threshold

## Dataset
- Kaggle: `mlg-ulb/creditcardfraud`  
- Target: `Class` (`1` = fraud, `0` = normal) — **heavily imbalanced**
- Features: `Time`, `Amount`, and PCA components `V1..V28`

## Quickstart

### 1) Kaggle setup & data download (Notebook Step 1)
- From Kaggle Account, create an API token → download `kaggle.json`.
- Paste your `KAGGLE_USERNAME` and `KAGGLE_KEY` into the cell.
- Run the cell to download & unzip into: `./data/creditcard_anomaly/`
- You should see: `creditcard.csv`

### 2) Load & prepare (Step 2)
- Load CSV, check shape/head/missing/duplicates.
- Define `TARGET="Class"`.
- Scale only `Time` and `Amount` with **RobustScaler**.
- Stratified split into train/valid.
- Compute a suggested `class_weight` for supervised training.

### 3) Baselines (Step 3)
- **Logistic Regression** with `class_weight` → compute **ROC-AUC**, **PR-AUC**.
- Sweep thresholds on validation to find **best F1**.
- **Isolation Forest** (unsupervised) → ROC-AUC, PR-AUC from anomaly scores.
- Plot **ROC** and **PR** curves for both.

### 4) Calibration + cost-sensitive threshold + artifact (Step 4)
- **Isotonic calibration** on validation; keep version with **better PR-AUC**.
- Choose threshold to **minimize expected cost**, given:
  - `COST_FN` (missed fraud) >> `COST_FP` (false alarm)
- Save artifact → `./artifacts/creditcard_fraud_detector.joblib`  
  Contents:
  - `model`: calibrated (CalibratedClassifierCV) or raw logistic pipeline
  - `threshold`: selected operating point
  - `features`: list of columns expected at inference
  - `costs`, `class_weight` (for documentation)

## Inference on new data
```python
import joblib, pandas as pd

art = joblib.load("./artifacts/creditcard_fraud_detector.joblib")
model     = art["model"]
threshold = art["threshold"]
features  = art["features"]

# df_new must contain the same columns as in training
# (including scaled 'Time' and 'Amount' if your pipeline doesn't re-fit scaling)
proba = model.predict_proba(df_new[features])[:, 1]
pred  = (proba >= threshold).astype(int)
