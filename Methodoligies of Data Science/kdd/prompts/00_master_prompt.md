# KDD Methodology: Credit Card Fraud Detection

## Overview

**Dataset**: Credit Card Fraud Detection (Kaggle)  
**Problem**: Binary Classification - Detect fraudulent transactions  
**Framework**: KDD (Knowledge Discovery in Databases)  
**Challenge**: Extreme imbalance (0.172% fraud rate)

---

## KDD: Knowledge Discovery from Imbalanced Data

KDD emphasizes the full pipeline from raw data selection to actionable insights, with explicit focus on data transformation and pattern mining.

### Phase 1: Selection (15 min)
**Goal**: Select relevant data and features from raw database.

**Tasks**:
1. Load Credit Card Fraud dataset (284,807 transactions)
2. Data profiling (understand PCA anonymization)
3. Initial quality assessment (missing values, duplicates)
4. Feature selection (all 28 PCA components + Time + Amount)

**Key Deliverables**:
- selected_data.csv
- reports/data_selection_rationale.md
- Feature type documentation (PCA vs raw)

**Critic Checkpoint**: Dr. Nitesh Chawla reviews selection criteria.

---

### Phase 2: Preprocessing (20 min)
**Goal**: Clean and prepare data for transformation.

**Tasks**:
1. **Handle PCA Features**: Already scaled (V1-V28)
2. **Scale Raw Features**: Standardize `Time` and `Amount`
3. **Missing Values**: Verify none exist
4. **Outliers**: Detect but keep (may be fraud indicators)
5. **Data Types**: Ensure numeric consistency

**Key Deliverables**:
- preprocessed_train.csv, preprocessed_test.csv
- Outlier analysis report
- Feature distribution visualizations

**Critic Checkpoint**: Did you validate PCA feature integrity?

---

### Phase 3: Transformation (30 min)
**Goal**: Handle class imbalance and engineer features.

**Challenge**: 492 frauds (0.172%) vs 284,315 legitimate (99.828%)

**Imbalance Techniques**:
1. **SMOTE** (Synthetic Minority Over-sampling Technique)
   - Generate synthetic fraud examples
   - Use k=5 nearest neighbors
2. **ADASYN** (Adaptive Synthetic Sampling)
   - Focus on hard-to-learn examples
3. **Random Under-sampling**
   - Reduce majority class to balance
4. **Combination**: SMOTE + Under-sampling (50/50 split)

**Feature Engineering**:
- Time-based: hour_of_day, day_of_week (from `Time`)
- Amount-based: log(Amount+1), Amount_bin

**Key Deliverables**:
- transformed_train.csv (balanced)
- Comparison of balancing strategies
- src/transformation.py

**Critic Checkpoint**: Did you validate SMOTE doesn't create unrealistic samples?

---

### Phase 4: Data Mining (40 min)
**Goal**: Apply multiple algorithms to discover fraud patterns.

**Algorithms**:
1. **Isolation Forest** (anomaly detection, unsupervised)
2. **Random Forest** (with class_weight='balanced')
3. **XGBoost** (with scale_pos_weight)
4. **LightGBM** (fast, handles imbalance)

**Evaluation Metrics** (NOT accuracy - useless for imbalanced data):
- **Primary**: PR-AUC (Precision-Recall Area Under Curve)
- **Secondary**: ROC-AUC
- **Business**: Precision@90% Recall (catch 90% fraud, minimize false alarms)
- **Cost-Sensitive**: Weighted F1 (fraud FP cost = €100, fraud FN cost = €1000)

**Hyperparameter Tuning**:
- Stratified 5-fold CV
- Optimize for PR-AUC (not ROC-AUC)
- Threshold tuning (default 0.5 is bad for imbalanced data)

**Key Deliverables**:
- 4 trained models
- Confusion matrices at optimal thresholds
- Precision-Recall curves
- Cost-sensitive analysis

**Critic Checkpoint**: Did you use proper metrics for imbalanced data?

---

### Phase 5: Interpretation/Evaluation (35 min)
**Goal**: Validate models, interpret patterns, assess business impact.

**Tasks**:
1. **Test Set Evaluation**:
   - PR-AUC, Precision@90% Recall
   - Confusion matrix at optimal threshold
   - Compare to baseline (always predict majority class)
2. **Pattern Interpretation**:
   - Feature importance (which PCA components matter?)
   - SHAP values for fraud examples
   - Common fraud patterns discovered
3. **Business Impact**:
   - Cost per false positive: €100 (manual review)
   - Cost per false negative: €1000 (fraud loss)
   - Expected savings vs no-fraud-detection system
4. **Model Card**:
   - Intended use (real-time fraud detection)
   - Limitations (PCA interpretation, concept drift)
   - Fairness (can't audit - features anonymized)

**Success Criteria**:
- PR-AUC > 0.70 (industry benchmark for fraud)
- Precision@90% Recall > 0.05 (5% of flagged transactions are fraud)
- Positive ROI vs baseline

**Critic Checkpoint**: Did you assess cost-sensitive performance?

---

## Execution Checklist

- [ ] Phase 1: Selection - Data loaded, features documented
- [ ] Phase 2: Preprocessing - Scaling, outlier analysis
- [ ] Phase 3: Transformation - SMOTE/ADASYN applied, features engineered
- [ ] Phase 4: Data Mining - 4 models trained, PR-AUC optimized
- [ ] Phase 5: Interpretation - Patterns discovered, business ROI calculated

---

## KDD vs CRISP-DM vs SEMMA

| Aspect | KDD | CRISP-DM | SEMMA |
|--------|-----|----------|-------|
| **Origin** | Academia (1996) | Industry (1996) | SAS (1990s) |
| **Phases** | 5 (Selection → Interpretation) | 6 (Business → Deployment) | 5 (Sample → Assess) |
| **Focus** | Pattern discovery | Business problem-solving | Statistical modeling |
| **Strengths** | Transformation emphasis | Stakeholder alignment | Hypothesis testing |
| **Best For** | Research, anomaly detection | Enterprise projects | Marketing, classification |

**When to Use KDD**:
- Pattern discovery in large databases (not just ML)
- Data transformation is critical (imbalance, anonymization)
- Anomaly detection, clustering, association rules
- Research/academic projects

---

## Expected Runtime

- **Total**: ~2-3 hours (including hyperparameter tuning)
- **Python Notebook**: ~1.5 hours
- **SMOTE transformation**: ~20 min (computationally intensive)
- **Model training**: ~40 min

---

## Critic Persona: Dr. Nitesh Chawla

**Background**: Creator of SMOTE algorithm, expert in imbalanced learning, Professor at Notre Dame.

**Critique Style**: Questions imbalance handling, demands cost-sensitive evaluation, checks for unrealistic synthetic samples.

**Signature Phrases**:
- "Accuracy is a useless metric for imbalanced data. Show me PR-AUC."
- "Did you validate your synthetic samples don't violate domain constraints?"
- "Threshold tuning is not optional - it's essential."

---

## Key Success Metrics

| Metric | Target | Rationale |
|--------|--------|-----------|
| **PR-AUC** | >0.70 | Handles imbalance (0.172% fraud) |
| **Precision@90% Recall** | >0.05 | Business constraint (review capacity) |
| **Cost-Sensitive F1** | Maximized | €1000 FN cost vs €100 FP cost |
| **ROI vs Baseline** | +500% | Justify fraud detection system |

---

**Prepared by**: Data Science Portfolio Team  
**Date**: November 6, 2025  
**Version**: 1.0
