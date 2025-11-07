# CRISP-DM Master Prompt: Rossmann Store Sales Forecasting

**Generated**: 2025-11-06  
**Methodology**: CRISP-DM (Cross-Industry Standard Process for Data Mining)  
**Dataset**: Rossmann Store Sales (Kaggle Competition)  
**Problem Type**: Time-Series Forecasting

---

## Objective

Build a production-quality time-series forecasting system following CRISP-DM methodology to predict daily sales for 1,115 Rossmann drug stores up to 6 weeks in advance.

---

## CRISP-DM Phases

### 1. Business Understanding (30 min)

**Deliverables**:
- `reports/business_understanding.md` documenting:
  - Business objectives (inventory optimization, staffing, procurement)
  - Success criteria (target sMAPE < 13%, beat naive baselines)
  - Constraints (6-week horizon, daily granularity, per-store predictions)
  - Risks (holiday/promo leakage, store closures, concept drift)

**Key Questions**:
- What are the costs of over-forecasting vs under-forecasting?
- Which stores/departments are most critical?
- How will predictions be consumed (API, batch, dashboard)?

**Baseline Models**:
- Naive Last Week (lag-7)
- Naive Last Year Same Week (lag-364)
- Simple Moving Average (7-day, 28-day)

---

### 2. Data Understanding (45 min)

**Deliverables**:
- `reports/data_dictionary.md` with field definitions, types, cardinalities
- EDA notebook cells with:
  - Missing value analysis
  - Temporal patterns (weekday, monthly, yearly trends)
  - Store-level heterogeneity (clustering by sales patterns)
  - Promotion/holiday effects
  - Outlier detection (closed stores = Sales=0, Open=0)

**Key Checks**:
- Train period: 2013-01-01 to 2015-07-31 (942 days)
- Test period: 2015-08-01 to 2015-09-17 (48 days, ~6 weeks)
- No. stores: 1,115
- Features: Store, DayOfWeek, Date, Sales, Customers, Open, Promo, StateHoliday, SchoolHoliday + store metadata (StoreType, Assortment, CompetitionDistance, Promo2)

---

### 3. Data Preparation (60 min)

**Deliverables**:
- `data/processed/train_features.csv`, `test_features.csv`
- `src/feature_engineering.py` module
- Leakage tests in `tests/test_leakage.py`

**Feature Engineering**:
1. **Temporal Features**:
   - DayOfWeek, Day, Month, Year, WeekOfYear
   - IsWeekend, IsMonthStart, IsMonthEnd, IsQuarterEnd
   - DaysSincePromo, DaysUntilPromo

2. **Lag Features** (per Store):
   - Sales lags: [1, 2, 7, 14, 28, 364]
   - Customer lags: [1, 7, 28]

3. **Rolling Statistics** (per Store, 7/14/28-day windows):
   - mean, median, std, min, max of Sales & Customers

4. **Store Metadata**:
   - CompetitionOpen (months since competition opened)
   - Promo2Active (is store in long-term promo campaign?)

5. **Holiday Encoding**:
   - StateHoliday (one-hot)
   - SchoolHoliday (binary)

**Critical**:
- Use `TimeSeriesSplit` (5 folds) to respect temporal order
- No future information in lag/rolling features (shift by +1)
- Handle store closures (Open=0) → predict 0 sales

**Pipeline**:
```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

pipeline = Pipeline([
    ('features', feature_engineering_transformer),
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', LGBMRegressor())
])
```

---

### 4. Modeling (75 min)

**Models to Train**:
1. Linear: Ridge, Lasso (with polynomial features)
2. Tree-based: Random Forest, XGBoost, LightGBM
3. Ensemble: Stacked ensemble of top 3

**Hyperparameter Tuning**:
- Use `TimeSeriesSplit` with 5 folds
- Optimize for RMSPE (competition metric) or sMAPE
- Log experiments to MLflow

**Evaluation Metrics**:
- **Primary**: sMAPE (Symmetric Mean Absolute Percentage Error)
- **Secondary**: MAE, RMSE, WAPE (Weighted Absolute Percentage Error)
- **Business**: Sales-weighted MAE (critical stores weighted more)

**SHAP Analysis**:
- Global feature importance (bar plot, beeswarm)
- Local explanations for 3 sample stores (high/medium/low sales)
- Dependence plots for top features (DayOfWeek, Promo, lags)

---

### 5. Evaluation (30 min)

**Deliverables**:
- `reports/evaluation.md` with:
  - Holdout performance vs baselines
  - Per-store error distribution (identify struggling stores)
  - Stability analysis (is model consistent across weeks?)
  - Sensitivity analysis (holiday weeks, promo weeks)
  - Business impact (translate MAE to $ cost/benefit)

**Questions**:
- Does model beat naive baselines by >10%?
- Are errors acceptable for business use (< €500/store/day)?
- Is model robust to holidays/promotions?

---

### 6. Deployment (45 min)

**Deliverables**:
- `deployment/app.py` - FastAPI service
- `deployment/model.joblib` - Serialized pipeline
- `reports/monitoring_plan.md` - Drift detection strategy

**FastAPI Endpoints**:
```python
POST /predict
{
  "store_id": 1,
  "date": "2015-09-18",
  "day_of_week": 5,
  "promo": 1,
  ...
}
→ {"predicted_sales": 5432.1, "confidence_interval": [5100, 5800]}

GET /health
→ {"status": "ok", "model_version": "1.0.0"}
```

**Monitoring**:
- Evidently data drift report (weekly)
- Track prediction distribution shifts
- Alert if MAE > threshold for 3 consecutive days

---

## Critic Checkpoints

After each phase, pause and invoke the **Dr. Foster Provost** persona:

> "You've completed [Phase]. As a veteran data scientist who's seen projects fail from poor business alignment and data leakage, I need to stress-test your work. Specifically:
> - [Phase-specific question 1]
> - [Phase-specific question 2]
> - [Phase-specific question 3]
> Address these concerns before proceeding."

Document the critique + your response in the notebook and save to `prompts/executed/<timestamp>_<phase>.md`.

---

## Success Criteria

- ✅ Notebook runs top-to-bottom without errors
- ✅ All tests pass (`pytest tests/`)
- ✅ sMAPE < 13% on holdout (beat naive baseline by ≥10%)
- ✅ FastAPI service responds in <200ms
- ✅ All reports generated and actionable
- ✅ No data leakage (tests confirm)
- ✅ SHAP plots interpretable and align with domain knowledge

---

## Time Budget: 4-5 hours (single sitting or spread across sessions)
