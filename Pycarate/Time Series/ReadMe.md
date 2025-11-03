# Delhi Climate Forecasting — Auto-ARIMA vs SARIMAX (with Exogenous)

Forecast daily **mean temperature** using Kaggle’s **Delhi Climate** dataset:
- **Univariate** model: Auto-ARIMA (no exogenous)
- **Univariate with exogenous**: SARIMAX using `humidity`, `wind_speed`, `meanpressure`

The notebook produces: clean plots, validation metrics, residual diagnostics, and test-period forecasts saved to `./artifacts/`.

---

## 🗂️ Dataset

- **Kaggle**: `sumanthvrao/daily-climate-time-series-data`  
- Files used:
  - `DailyDelhiClimateTrain.csv`
  - `DailyDelhiClimateTest.csv`
- Target: `meantemp` (daily)
- Exogenous (for SARIMAX): `humidity`, `wind_speed`, `meanpressure`

---

## ⚙️ Environment

- **Python**: 3.8 recommended
- Install (done inside Step 1 cell):
  - `numpy==1.23.5`, `pandas==1.5.3`, `matplotlib==3.7.3`
  - `scikit-learn==1.3.2`
  - `pmdarima==2.0.4` (Auto-ARIMA)
  - `statsmodels==0.13.5` (SARIMAX)
  - `kaggle` (CLI for dataset download)

> Make sure you have a Kaggle API token at `~/.kaggle/kaggle.json` with file permissions `600`.

---

## 🚦 How to Run (Notebook Steps)

### **Step 1 — Install + Download**
- Installs all libs and downloads/unzips the dataset to:  
  `./data/delhi_climate/`

### **Step 2 — Load & Prepare**
- Reads train/test CSVs
- Parses `date`, sets daily frequency, clips outliers, fills tiny gaps
- Sets:
  - `TARGET_COL = "meantemp"`
  - `EXOG_COLS  = ["humidity","wind_speed","meanpressure"]`
- Makes a **validation window** (last *H* days of the train file)

### **Step 3 — Auto-ARIMA (no exogenous)**
- Auto-detects orders with **weekly seasonality (`m=7`)**
- Forecasts the validation window
- Computes **MAE, RMSE, MAPE**
- Saves model → `./artifacts/auto_arima_univariate.joblib`

### **Step 4 — SARIMAX (with exogenous)**
- Tiny AIC search over `(p,d,q)(P,D,Q, m=7)`
- Forecasts validation using exogenous variables
- **Refits on full training range** and forecasts the **test period**
- Saves:
  - Model results → `./artifacts/sarimax_exog_results.pkl`
  - Test forecast CSV → `./artifacts/delhi_test_forecast_exog.csv`

### **Step 5 — Diagnostics & Comparison**
- Side-by-side validation **metrics table**: Auto-ARIMA vs SARIMAX
- Residual checks for SARIMAX:
  - Residual plot
  - **ACF/PACF**
  - **Ljung–Box** test (autocorrelation)
  - **QQ plot** + **Jarque–Bera** (normality)
- Saves:
  - `./artifacts/validation_metrics_comparison.csv`
  - `./artifacts/sarimax_residuals.csv`

### **Step 6 — Final Packaging**
- Refit Auto-ARIMA on **full** training data and forecast **test** range
- Combine **ARIMA** and **SARIMAX** test forecasts:
  - CSV → `./artifacts/test_forecasts_combined.csv`
  - Plot → `./artifacts/final_test_forecasts.png`
  - Summary → `./artifacts/summary.txt`

---

## 📈 What to Look At

- **Validation metrics**: lower MAE/RMSE/MAPE indicates better fit
- **Residual diagnostics**:
  - Ljung–Box p-values **not too small** → little remaining autocorrelation
  - QQ plot roughly straight → residuals ~ normal (nice to have)
- **Final plot**: Compare ARIMA vs SARIMAX forecasts into the test period

---

## 🛠️ Troubleshooting

- **Kaggle token error**  
  Create token at Kaggle → Settings → *Create New API Token*.  
  Save to `~/.kaggle/kaggle.json` and set permissions:
  ```bash
  chmod 600 ~/.kaggle/kaggle.json

  ### Here is the video explanation : https://drive.google.com/drive/folders/1NPPObpYYQWUkHfSnbolgNfHGKAnx0yBJ
