# Iris — Multiclass Classification (Kaggle) with a Clean ML Pipeline

Train a **3-class flower classifier** on Kaggle’s **`uciml/iris`** dataset using a clean, reproducible scikit-learn pipeline.  
We compare a transparent **Logistic Regression** (softmax) against a stronger **Random Forest**, then save a **production-ready artifact** for reuse.

## What this project shows
- **Reliable data access**: Download directly from Kaggle with API token
- **Multiclass setup**: Label-encode Species and stratified train/valid split
- **Baselines first**: Impute → Scale → Logistic Regression (softmax)
- **Non-linear model**: Random Forest + side-by-side comparison
- **Artifacts matter**: Save one `.joblib` with model + features + class names

## Dataset
- Kaggle: **`uciml/iris`**
- Target: `Species` with 3 classes: `setosa`, `versicolor`, `virginica`
- Features: sepal length/width, petal length/width

## Quickstart
1. **Kaggle token**: Get `kaggle.json` (Kaggle → Account → Create New API Token).
2. **Notebook Step 1**: Fill `KAGGLE_USERNAME` and `KAGGLE_KEY`, run to download and unzip into `./data/iris_kaggle/`.
3. **Notebook Step 2**: Load CSV, sanity checks, label encoding, stratified split.
4. **Notebook Step 3**: Train **Logistic Regression** baseline; print accuracy, macro-F1, confusion matrix.
5. **Notebook Step 4**: Train **Random Forest**; compare metrics and show feature importance.
6. **Notebook Step 5**: Save artifact → `./artifacts/iris_multiclass_best_model.joblib` and run a quick inference demo.

## Inference (any Python script)
```python
import joblib, pandas as pd, numpy as np

art = joblib.load("./artifacts/iris_multiclass_best_model.joblib")
pipe, feats, classes = art["model"], art["features"], art["classes"]

# Example: single row with all features
row = {c: 0.0 for c in feats}
row.update({"sepal_length": 5.8, "sepal_width": 2.8, "petal_length": 4.5, "petal_width": 1.3})
X_one = pd.DataFrame([row], columns=feats)

proba = pipe.predict_proba(X_one)[0]
pred  = classes[int(np.argmax(proba))]
print("pred:", pred, "| proba:", np.round(proba, 3))
