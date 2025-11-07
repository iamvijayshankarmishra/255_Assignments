# SEMMA Methodology: Bank Marketing Campaign Prediction

## Overview

**Dataset**: Bank Marketing (UCI ML Repository)  
**Problem**: Binary Classification - Predict if client subscribes to term deposit  
**Framework**: SEMMA (Sample → Explore → Modify → Model → Assess)  
**Origin**: SAS Institute (statistical roots)

---

## SEMMA: A Statistically-Grounded Approach

SEMMA emphasizes rigorous statistical analysis before jumping to modeling. Each phase builds on SAS's strengths in statistical profiling.

### Phase 1: Sample (15 min)
**Goal**: Create representative training/validation/test sets.

**Tasks**:
1. Load Bank Marketing dataset (45,211 records, 17 features)
2. Stratified sampling (preserve class balance: ~11.3% positive)
3. Create train (60%), validation (20%), test (20%) splits
4. Document sampling strategy

**Key Deliverables**:
- train.csv, val.csv, test.csv
- reports/sampling_strategy.md
- Test: stratification maintained (χ² test)

**Critic Checkpoint**: Dr. Raymond Hettinger reviews sampling bias.

---

### Phase 2: Explore (30 min)
**Goal**: Statistical profiling to understand distributions and relationships.

**Tasks**:
1. **Univariate Analysis**:
   - Continuous: histograms, box plots, normality tests (Shapiro-Wilk)
   - Categorical: frequency tables, chi-squared tests
2. **Bivariate Analysis**:
   - Continuous vs Target: t-tests, Mann-Whitney U
   - Categorical vs Target: chi-squared, Cramér's V
   - Correlation matrix (Pearson + Spearman)
3. **Missing Value Analysis**: No missing values, but check for "unknown" categories
4. **Outlier Detection**: IQR, Z-scores for age, duration, campaign

**Key Deliverables**:
- reports/statistical_profile.md (distributions, tests, p-values)
- Visualizations: pair plots, correlation heatmap, box plots
- Feature importance (univariate association with target)

**Critic Checkpoint**: Did you test for non-linear relationships?

---

### Phase 3: Modify (25 min)
**Goal**: Transform features to improve model performance.

**Tasks**:
1. **Encoding**:
   - Ordinal: education (basic.4y → basic.6y → basic.9y → high.school → university.degree → professional.course)
   - One-hot: job, marital, contact, month, day_of_week, poutcome
   - Binary: default, housing, loan
2. **Scaling**:
   - Standardization (Z-score) for age, duration, campaign, pdays
   - No normalization (tree models don't need it)
3. **Feature Engineering**:
   - `contact_frequency` = campaign + previous
   - `recency_score` = 1 / (pdays + 1) if pdays != 999 else 0
   - `economic_confidence` = emp.var.rate + cons.conf.idx
4. **Binning** (optional):
   - age → age_groups: <30, 30-40, 40-50, 50+
   - duration → duration_bins: <100s, 100-300s, 300+s

**Key Deliverables**:
- src/modification.py (transformers)
- modified_train.csv, modified_val.csv, modified_test.csv
- Feature correlation check (remove multicollinearity >0.9)

**Critic Checkpoint**: Did you validate transformations don't leak information?

---

### Phase 4: Model (35 min)
**Goal**: Train multiple classification models, tune hyperparameters.

**Models to Compare**:
1. **Logistic Regression** (baseline, interpretable)
2. **Decision Tree** (CART, easy to explain)
3. **Random Forest** (ensemble, handles non-linearity)
4. **XGBoost** (gradient boosting, high performance)

**Evaluation Metrics** (imbalanced class):
- **Primary**: ROC-AUC (overall discriminative power)
- **Secondary**: Precision-Recall AUC (handles class imbalance)
- **Business**: Lift @ 20% (top 20% predicted positives)
- **Calibration**: Brier score (are probabilities accurate?)

**Hyperparameter Tuning**:
- 5-fold stratified CV on training set
- GridSearchCV for Logistic/Tree
- RandomizedSearchCV for RF/XGBoost
- Cost-sensitive learning: `class_weight='balanced'`

**Key Deliverables**:
- 4 trained models (joblib)
- Model comparison table (ROC-AUC, PR-AUC, Lift)
- Feature importance plots
- ROC curves (all models on same plot)
- Confusion matrices

**Critic Checkpoint**: Did you calibrate probabilities? (Platt scaling)

---

### Phase 5: Assess (30 min)
**Goal**: Validate best model on test set, compute business impact.

**Tasks**:
1. **Holdout Performance**:
   - Test set ROC-AUC, PR-AUC, Lift
   - Compare to validation (check overfitting)
2. **Lift Chart Analysis**:
   - Top deciles: % of positive class captured
   - Business question: "If we call top 20%, what's capture rate?"
3. **Calibration Plot**:
   - Are predicted probabilities reliable?
   - Isotonic regression if needed
4. **Business Impact**:
   - Cost per call: €5
   - Revenue per subscription: €200
   - Expected profit: `(TP * 200 - FP * 5 - FN * 0) / total_test`
   - Compare to random calling strategy
5. **Model Card**: Intended use, limitations, fairness (check age/marital bias)

**Success Criteria**:
- ROC-AUC > 0.80 (industry benchmark)
- Lift @ 20% > 2.5x (2.5x better than random)
- Positive ROI vs random calling

**Critic Checkpoint**: Did you assess fairness across demographic groups?

---

## Execution Checklist

- [ ] Phase 1: Sample - Stratified splits created
- [ ] Phase 2: Explore - Statistical tests run, distributions visualized
- [ ] Phase 3: Modify - Features encoded, scaled, engineered
- [ ] Phase 4: Model - 4 models trained, hyperparameters tuned
- [ ] Phase 5: Assess - Test set evaluated, lift chart, business ROI

---

## SEMMA vs CRISP-DM

| Aspect | SEMMA | CRISP-DM |
|--------|-------|----------|
| **Origin** | SAS Institute (1990s) | Industry Consortium (1996) |
| **Focus** | Statistical modeling | Business problem-solving |
| **Phases** | 5 (Sample → Assess) | 6 (Business → Deployment) |
| **Strengths** | Rigorous statistical testing | Stakeholder alignment |
| **Tools** | SAS Enterprise Miner | Tool-agnostic |
| **Best For** | Academic, research | Enterprise, consulting |

**When to Use SEMMA**:
- Dataset is already collected (no "Business Understanding" phase)
- Statistical rigor is paramount (hypothesis testing, p-values)
- Using SAS stack
- Classification/regression with clear target variable

---

## Expected Runtime

- **Total**: ~2 hours (including SAS parallel implementation)
- **Python Notebook**: ~45 min
- **SAS Code**: ~30 min (requires SAS license)

---

## Critic Persona: Dr. Raymond Hettinger

**Background**: SAS expert, statistical purist, author of "Applied Logistic Regression".

**Critique Style**: Questions assumptions about distributions, tests for violations (e.g., "Did you check for heteroscedasticity?"), demands p-values for every claim.

**Signature Phrases**:
- "Show me the p-value."
- "Did you test for normality before applying that transformation?"
- "Lift charts don't lie - but you need calibration too."

---

## Key Success Metrics

| Metric | Target | Rationale |
|--------|--------|-----------|
| **ROC-AUC** | >0.80 | Industry standard for balanced performance |
| **Lift @ 20%** | >2.5x | Campaign efficiency (call 20%, get 50%+ positives) |
| **Precision @ 50% Recall** | >0.30 | Business constraint (max 50% call capacity) |
| **Brier Score** | <0.10 | Probability calibration |
| **ROI vs Random** | +150% | Cost-benefit (€5 call, €200 revenue) |

---

**Prepared by**: Data Science Portfolio Team  
**Date**: November 6, 2025  
**Version**: 1.0
