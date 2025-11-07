# Colab Setup Instructions

## Option 1: Direct Execution (Recommended)

1. Click "Open in Colab" badge in README
2. File → Save a copy in Drive
3. Upload `kaggle.json` when prompted (cell 3)
4. Runtime → Run all (⌘/Ctrl+F9)
5. Wait ~20 minutes for completion

## Option 2: Manual Upload

If Kaggle API not working:

1. Download `train.csv`, `test.csv`, `store.csv` from [Kaggle Rossmann](https://www.kaggle.com/competitions/rossmann-store-sales/data)
2. Upload to Colab: `files.upload()`
3. Skip data download cell
4. Continue from Phase 1

## Saving Results

### Google Drive (Persistent)
```python
from google.colab import drive
drive.mount('/content/drive')

# Save model
import joblib
joblib.dump(lgbm_model, '/content/drive/MyDrive/rossmann_model.joblib')

# Save critiques
!cp -r prompts/executed /content/drive/MyDrive/crisp_dm_critiques/
```

### Local Download
```python
from google.colab import files
files.download('deployment/model.joblib')
files.download('deployment/model_metadata.json')
```

## Troubleshooting

### Out of Memory
- Runtime → Change runtime type → GPU/High-RAM
- Reduce `sample_size` in SHAP analysis (line ~600)

### Kaggle API Error
```
OSError: Could not find kaggle.json
```
**Fix**: Re-upload `kaggle.json`, ensure permissions:
```bash
!chmod 600 ~/.kaggle/kaggle.json
```

### Import Errors
```
ModuleNotFoundError: No module named 'lightgbm'
```
**Fix**: Rerun first cell with `!pip install ...`

## Notes

- **Runtime**: ~20 min on free Colab (CPU)
- **Data Size**: 2GB download + 1.5GB in memory
- **GPU**: Not needed (XGBoost/LightGBM use CPU)
