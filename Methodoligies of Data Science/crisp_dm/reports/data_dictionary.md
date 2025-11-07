# Data Dictionary: Rossmann Store Sales

**Generated**: 2025-11-06  
**Dataset**: Rossmann Store Sales (Kaggle Competition)

---

## Raw Data Files

### 1. train.csv (1,017,209 rows)
Historical daily sales data from 2013-01-01 to 2015-07-31

### 2. test.csv (41,088 rows)
Prediction period: 2015-08-01 to 2015-09-17 (48 days × 856 open stores)

### 3. store.csv (1,115 rows)
Store metadata (one row per store)

---

## Feature Definitions

### Target Variable

| Feature | Type | Description | Range | Missing % |
|---------|------|-------------|-------|-----------|
| **Sales** | int | Daily sales revenue (€) | 0 - 41,551 | 0% |

**Notes**:
- Sales = 0 when Open = 0 (store closed)
- Right-skewed distribution (median ~5,500, mean ~5,774)

---

### Core Features (train/test)

| Feature | Type | Description | Values | Notes |
|---------|------|-------------|--------|-------|
| **Store** | int | Store ID | 1 - 1,115 | Unique identifier |
| **Date** | date | Calendar date | 2013-01-01 to 2015-09-17 | YYYY-MM-DD format |
| **DayOfWeek** | int | ISO day of week | 1 (Mon) - 7 (Sun) | Strong seasonality |
| **Open** | binary | Store open? | 0 (closed), 1 (open) | 0 → Sales = 0 |
| **Promo** | binary | In-store promotion | 0 (no), 1 (yes) | ~38% of days |
| **StateHoliday** | categorical | Public holiday | 0 (none), a (public), b (Easter), c (Christmas) | ~6% of days |
| **SchoolHoliday** | binary | School vacation | 0 (no), 1 (yes) | ~18% of days |
| **Customers** | int | Daily customer count | 0 - 7,388 | **Test set**: Missing! |

---

### Store Metadata (store.csv)

| Feature | Type | Description | Values | Missing % |
|---------|------|-------------|--------|-----------|
| **StoreType** | categorical | Store format | a, b, c, d | 0% |
| **Assortment** | categorical | Product range | a (basic), b (extra), c (extended) | 0% |
| **CompetitionDistance** | float | Distance to nearest competitor (m) | 20 - 75,860 | **2.7%** (3 stores) |
| **CompetitionOpenSinceMonth** | int | Month competitor opened | 1 - 12 | 26.3% |
| **CompetitionOpenSinceYear** | int | Year competitor opened | 1900 - 2015 | 26.3% |
| **Promo2** | binary | Participating in long-term promo? | 0 (no), 1 (yes) | 0% |
| **Promo2SinceWeek** | int | Week Promo2 started | 1 - 50 | 48.8% |
| **Promo2SinceYear** | int | Year Promo2 started | 2009 - 2015 | 48.8% |
| **PromoInterval** | string | Promo2 active months | "Feb,May,Aug,Nov" etc. | 48.8% |

---

## Engineered Features

### Temporal Features

| Feature | Type | Description | Formula |
|---------|------|-------------|---------|
| **Year** | int | Year | Extracted from Date |
| **Month** | int | Month (1-12) | Extracted from Date |
| **Day** | int | Day of month (1-31) | Extracted from Date |
| **WeekOfYear** | int | ISO week (1-53) | Extracted from Date |
| **DayOfYear** | int | Julian day (1-365) | Extracted from Date |
| **Quarter** | int | Quarter (1-4) | Extracted from Date |
| **IsWeekend** | binary | Saturday/Sunday? | DayOfWeek >= 6 |
| **IsMonthStart** | binary | First day of month? | Day == 1 |
| **IsMonthEnd** | binary | Last day of month? | Day == max(Day) |
| **IsQuarterEnd** | binary | Last day of quarter? | Boolean flag |

### Lag Features (per Store)

| Feature | Type | Description | Lookback | Notes |
|---------|------|-------------|----------|-------|
| **Sales_Lag1** | float | Sales 1 day ago | 1 day | Immediate history |
| **Sales_Lag2** | float | Sales 2 days ago | 2 days | Short-term trend |
| **Sales_Lag7** | float | Sales 1 week ago | 7 days | Day-of-week seasonality |
| **Sales_Lag14** | float | Sales 2 weeks ago | 14 days | Bi-weekly pattern |
| **Sales_Lag28** | float | Sales 4 weeks ago | 28 days | Monthly seasonality |
| **Sales_Lag364** | float | Sales last year same week | 364 days | Annual seasonality |
| **Customers_Lag1** | float | Customers 1 day ago | 1 day | (if available) |
| **Customers_Lag7** | float | Customers 1 week ago | 7 days | (if available) |

**⚠️ Leakage Prevention**: All lags use `.shift()` to ensure no future information

### Rolling Statistics (per Store)

| Feature | Type | Description | Window | Notes |
|---------|------|-------------|--------|-------|
| **Sales_RollingMean7** | float | 7-day average sales | 7 days | Smoothed short-term |
| **Sales_RollingStd7** | float | 7-day sales volatility | 7 days | Risk measure |
| **Sales_RollingMin7** | float | 7-day min sales | 7 days | Floor |
| **Sales_RollingMax7** | float | 7-day max sales | 7 days | Ceiling |
| **Sales_RollingMedian7** | float | 7-day median sales | 7 days | Robust to outliers |
| **Sales_RollingMean14** | float | 14-day average sales | 14 days | Medium-term trend |
| **Sales_RollingMean28** | float | 28-day average sales | 28 days | Long-term trend |

### Promotion Features

| Feature | Type | Description | Formula |
|---------|------|-------------|---------|
| **PromoStart** | binary | First day of promo? | Promo changes 0→1 |
| **PromoEnd** | binary | Last day of promo? | Promo changes 1→0 |
| **PromoStreak** | int | Consecutive days in promo | Cumulative count while Promo=1 |

### Competition Features

| Feature | Type | Description | Formula |
|---------|------|-------------|---------|
| **CompetitionOpenMonths** | int | Months since competitor opened | (Date - CompOpen) in months |
| **HasCompetition** | binary | Competitor exists? | CompetitionDistance not NaN |

### Baseline Predictions (for comparison)

| Feature | Type | Description | Purpose |
|---------|------|-------------|---------|
| **Baseline_LastWeek** | float | Lag-7 sales | Naive benchmark |
| **Baseline_LastYearSameWeek** | float | Lag-364 sales | Seasonal benchmark |
| **Baseline_MA7** | float | 7-day moving average | Smoothing benchmark |
| **Baseline_MA28** | float | 28-day moving average | Trend benchmark |

---

## Categorical Encoding

### One-Hot Encoded

| Original Feature | Encoded Features | Approach |
|-----------------|------------------|----------|
| **StateHoliday** | StateHoliday_a, StateHoliday_b, StateHoliday_c | drop_first=True (0 is reference) |
| **StoreType** | StoreType_b, StoreType_c, StoreType_d | drop_first=True (a is reference) |
| **Assortment** | Assortment_b, Assortment_c | drop_first=True (a is reference) |

---

## Data Quality Issues

### Missing Values

| Feature | Missing % | Imputation Strategy |
|---------|-----------|---------------------|
| **CompetitionDistance** | 2.7% | Fill with 999,999 (no competition) |
| **CompetitionOpenSince*** | 26.3% | Fill with 0 (no competition or unknown) |
| **Promo2Since*** | 48.8% | Fill with 0 (not in Promo2) |
| **Customers** (test set) | 100% | ❌ Cannot use in test predictions |

### Outliers

| Feature | Outliers Detected | Handling |
|---------|-------------------|----------|
| **Sales** | ~0.5% > 3 IQR | Keep (genuine high-sales days) |
| **CompetitionDistance** | 5 stores > 50km | Cap at 50,000m |

### Data Leakage Risks

| Risk | Description | Mitigation |
|------|-------------|-----------|
| **Customers** | Test set doesn't have it | ❌ Exclude from model |
| **Future Promos** | Promo schedule known in advance | ✅ Use planned promo calendar (metadata) |
| **Lag Features** | Must not include current day | ✅ Use `.shift(1)` before rolling |

---

## Feature Importance (Expected)

Based on domain knowledge and EDA:

1. **DayOfWeek**: 20-25% importance (strong weekly seasonality)
2. **Promo**: 15-20% (major driver)
3. **Sales_Lag7**: 10-15% (captures day-of-week pattern)
4. **Sales_RollingMean28**: 8-12% (trend)
5. **StateHoliday**: 5-8% (holidays are rare but impactful)
6. **StoreType**: 5-8% (store heterogeneity)
7. **CompetitionDistance**: 3-5%
8. **SchoolHoliday**: 2-4%

---

**Version**: 1.0  
**Last Updated**: 2025-11-06  
**Next Update**: After EDA phase
