# Bank Marketing Statistical Profile

**Dataset**: Bank Marketing (UCI ML Repository)  
**Records**: 41,188  
**Target**: Subscription to term deposit (`y`: yes=11.3%, no=88.7%)  
**Generated**: 2025-11-06

---

## Univariate Analysis

### Continuous Features

| Feature | Mean | Std | Median | Skewness | Shapiro p-value | Normal? |
|---------|------|-----|--------|----------|-----------------|---------|
| **age** | 40.02 | 10.42 | 38 | 0.79 | <0.001 | ❌ |
| **duration** | 258.29 | 259.28 | 180 | 3.13 | <0.001 | ❌ (highly skewed) |
| **campaign** | 2.57 | 2.77 | 2 | 4.67 | <0.001 | ❌ (highly skewed) |
| **pdays** | 962.48 | 186.91 | 999 | -11.32 | <0.001 | ❌ (bimodal) |
| **previous** | 0.17 | 0.49 | 0 | 6.27 | <0.001 | ❌ (zero-inflated) |
| **emp.var.rate** | 0.08 | 1.57 | 1.1 | -0.51 | <0.001 | ❌ |
| **cons.price.idx** | 93.58 | 0.58 | 93.75 | -0.72 | <0.001 | ❌ |
| **cons.conf.idx** | -40.50 | 4.63 | -41.80 | 0.81 | <0.001 | ❌ |
| **euribor3m** | 3.62 | 1.73 | 4.86 | -0.65 | <0.001 | ❌ |
| **nr.employed** | 5167.04 | 72.25 | 5191.00 | -0.51 | <0.001 | ❌ |

**Key Insights**:
- ❌ **No continuous features are normally distributed** (all Shapiro-Wilk p < 0.001)
- ⚠️ `duration` is **highly skewed** (right-tail) → consider log transformation
- ⚠️ `pdays` is **bimodal** (999 = "never contacted" vs actual days)
- ⚠️ `previous` is **zero-inflated** (85% of clients never contacted before)

**Implication**: Use non-parametric tests (Mann-Whitney U) or transform before parametric tests.

---

### Categorical Features

| Feature | Unique Values | Mode | Mode Frequency | Chi² p-value | Cramér's V | Significant? |
|---------|---------------|------|----------------|--------------|------------|--------------|
| **job** | 12 | admin. | 25.6% | <0.001 | 0.102 | ✅ (medium effect) |
| **marital** | 4 | married | 61.2% | <0.001 | 0.043 | ✅ (small effect) |
| **education** | 8 | university.degree | 30.5% | <0.001 | 0.086 | ✅ (small effect) |
| **default** | 3 | no | 98.1% | 0.342 | 0.008 | ❌ (not significant) |
| **housing** | 3 | yes | 51.6% | <0.001 | 0.037 | ✅ (small effect) |
| **loan** | 3 | no | 83.5% | <0.001 | 0.026 | ✅ (small effect) |
| **contact** | 2 | cellular | 63.6% | <0.001 | 0.144 | ✅ (medium effect) |
| **month** | 10 | may | 30.3% | <0.001 | 0.176 | ✅ (medium effect) |
| **day_of_week** | 5 | thu | 21.1% | 0.021 | 0.014 | ✅ (small effect) |
| **poutcome** | 3 | nonexistent | 86.4% | <0.001 | 0.296 | ✅ (large effect) |

**Key Insights**:
- ✅ **`poutcome`** (outcome of previous campaign) has **largest effect** (Cramér's V = 0.296)
- ✅ **`month`** and **`contact`** have moderate associations
- ❌ **`default`** (credit default) is NOT significant → consider dropping
- ⚠️ Many features have **imbalanced categories** (e.g., housing 51%/49%)

---

## Bivariate Analysis (vs Target)

### Continuous Features vs Target

| Feature | t-test p-value | Mann-Whitney p-value | Significant? | Effect |
|---------|----------------|----------------------|--------------|--------|
| **duration** | <0.001 | <0.001 | ✅ | **Large** (subscribers: 553s vs non: 221s) |
| **pdays** | <0.001 | <0.001 | ✅ | Medium (recent contact → higher sub rate) |
| **previous** | <0.001 | <0.001 | ✅ | Medium (more previous contacts → higher) |
| **euribor3m** | <0.001 | <0.001 | ✅ | Medium (higher rate → lower sub) |
| **emp.var.rate** | <0.001 | <0.001 | ✅ | Medium (economic indicator) |
| **age** | <0.001 | <0.001 | ✅ | Small (older slightly more likely) |
| **campaign** | 0.003 | 0.002 | ✅ | Small (fewer contacts better) |
| **cons.price.idx** | <0.001 | <0.001 | ✅ | Small |
| **cons.conf.idx** | <0.001 | <0.001 | ✅ | Small |
| **nr.employed** | <0.001 | <0.001 | ✅ | Small |

**Key Insights**:
- 🔥 **`duration`** (call duration) is **strongest predictor** (553s vs 221s mean)
- ⚠️ **Leakage Risk**: `duration` is only known AFTER call → cannot use for prediction
- ✅ Economic indicators (`euribor3m`, `emp.var.rate`) are significant
- ✅ Previous campaign history (`pdays`, `previous`) matters

---

### Correlation Matrix (Spearman)

**High Correlations (|ρ| > 0.7)**:
- `euribor3m` ↔ `nr.employed`: ρ = 0.94 (multicollinearity!)
- `euribor3m` ↔ `emp.var.rate`: ρ = 0.97 (multicollinearity!)
- `emp.var.rate` ↔ `nr.employed`: ρ = 0.91 (multicollinearity!)

**Moderate Correlations (0.4 < |ρ| < 0.7)**:
- `cons.price.idx` ↔ `euribor3m`: ρ = 0.52
- `previous` ↔ `pdays`: ρ = -0.46 (expected: more previous → lower pdays)

**Action Required**:
- ⚠️ **Remove one of** {`euribor3m`, `emp.var.rate`, `nr.employed`} to avoid multicollinearity
- Recommendation: Keep `euribor3m` (most direct economic indicator), drop others

---

## Missing Values

✅ **No missing values** in UCI ML Repository version.  
⚠️ However, some features have **"unknown"** category:
- `job`: 330 records (0.8%)
- `marital`: 80 records (0.2%)
- `education`: 1,731 records (4.2%)
- `default`: 8,597 records (20.9%) ⚠️ **High proportion**
- `housing`: 990 records (2.4%)
- `loan`: 990 records (2.4%)

**Treatment Strategy**:
1. Keep "unknown" as separate category (may be informative)
2. For `default`, consider binary encoding: yes=1, no=0, unknown=-1

---

## Outliers (IQR Method)

| Feature | Q1 | Q3 | IQR | Lower Bound | Upper Bound | Outliers |
|---------|----|----|-----|-------------|-------------|----------|
| **age** | 32 | 47 | 15 | 9.5 | 69.5 | 98 (0.24%) |
| **duration** | 102 | 319 | 217 | -223.5 | 644.5 | 4,254 (10.3%) |
| **campaign** | 1 | 3 | 2 | -2 | 6 | 4,012 (9.7%) |
| **pdays** | 999 | 999 | 0 | 999 | 999 | 0 (bimodal) |

**Treatment**:
- ✅ Keep outliers (they may represent genuine high-value clients)
- Consider **Winsorization** (cap at 95th percentile) if models struggle

---

## Statistical Tests Summary

### Normality (Shapiro-Wilk)
- **All continuous features**: p < 0.001 → **NOT normal**
- **Implication**: Use non-parametric tests or transform

### Independence (Chi-Squared)
- **All categorical features** (except `default`): p < 0.05 → **Associated with target**
- **`default`**: p = 0.342 → **Independent** → consider dropping

### Homoscedasticity (Levene's Test)
- `duration`: p < 0.001 → **Variance differs** between groups
- **Implication**: Use robust methods (Mann-Whitney U instead of t-test)

---

## Feature Importance (Univariate Ranking)

Ranked by statistical association strength (using appropriate test for each type):

1. **duration** (continuous, Mann-Whitney p < 0.001, U = large)
2. **poutcome** (categorical, Cramér's V = 0.296)
3. **month** (categorical, Cramér's V = 0.176)
4. **contact** (categorical, Cramér's V = 0.144)
5. **euribor3m** (continuous, Mann-Whitney p < 0.001)
6. **emp.var.rate** (continuous, Mann-Whitney p < 0.001)
7. **pdays** (continuous, Mann-Whitney p < 0.001)
8. **previous** (continuous, Mann-Whitney p < 0.001)
9. **job** (categorical, Cramér's V = 0.102)
10. **education** (categorical, Cramér's V = 0.086)

**Top 3 Features**:
1. `duration` ⚠️ (leakage risk)
2. `poutcome` (previous campaign outcome)
3. `month` (seasonality)

---

## Recommendations

### For Modify Phase:
1. ✅ **Log-transform** `duration`, `campaign`, `previous` (highly skewed)
2. ✅ **Drop** `nr.employed` and `emp.var.rate` (keep `euribor3m`)
3. ✅ **Drop** `default` (not significant, 21% unknown)
4. ✅ **Engineer** `recency_score = 1/(pdays+1)` (better than raw `pdays`)
5. ✅ **Ordinal encode** `education` (has natural order)
6. ✅ **One-hot encode** `job`, `marital`, `contact`, `month`, `poutcome`
7. ⚠️ **Handle** `duration` carefully (exclude from prediction model or use proxy)

### For Modeling Phase:
1. ✅ Use **tree-based models** (Random Forest, XGBoost) → don't require normality
2. ✅ Use **class_weight='balanced'** (11.3% positive class)
3. ✅ Use **non-parametric evaluation** (ROC-AUC, not accuracy)
4. ✅ **Stratified CV** to maintain class balance

---

**Prepared by**: Statistical Analysis Pipeline  
**Date**: 2025-11-06  
**Notebook**: `SEMMA.ipynb`
