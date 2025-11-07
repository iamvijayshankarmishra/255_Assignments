# Business Understanding: Rossmann Sales Forecasting

**Project**: CRISP-DM Methodology - Rossmann Store Sales  
**Date**: November 6, 2025  
**Stakeholders**: Rossmann Supply Chain, Store Operations, Finance Teams

---

## 1. Business Objectives

### Primary Goal
Predict daily sales for 1,115 Rossmann drug stores up to **6 weeks in advance** to optimize:
- **Inventory Management**: Reduce stockouts and overstock waste
- **Staff Scheduling**: Align workforce with expected demand
- **Procurement Planning**: Improve supplier order quantities and timing

### Success Criteria
1. **Accuracy**: Achieve **sMAPE < 13%** on holdout test set (beat competition baseline)
2. **Beat Baselines**: Outperform naive forecasts (last week, last year) by ≥10%
3. **Business Value**: Reduce inventory costs by 5-8% and stockout incidents by 15%
4. **Deployment**: Production-ready API responding in <200ms per prediction

---

## 2. Business Context

### Industry
Retail pharmacy/drugstore chain operating across Germany with:
- 1,115 stores (variable sizes, locations, customer demographics)
- Product assortment varies by store type (a/b/c/d)
- Heavy promotion activity (in-store promos, long-term Promo2 campaigns)
- Seasonal patterns (holidays, school vacations)

### Current Process
- **Manual Forecasting**: Store managers submit weekly estimates
- **Simple Rules**: "Order same as last week + 10% buffer"
- **Pain Points**: Frequent stockouts (lost sales), excess inventory (write-offs), poor holiday planning

### Opportunity
Machine learning can capture:
- Complex interactions (store type × day of week × promo)
- Long-term trends (store maturity, competition effects)
- Holiday/event patterns missed by humans

---

## 3. Key Performance Indicators (KPIs)

### Forecast Accuracy Metrics

| Metric | Formula | Target | Business Meaning |
|--------|---------|--------|-----------------|
| **sMAPE** | $\frac{200}{n} \sum \frac{\|y_i - \hat{y}_i\|}{\|y_i\| + \|\hat{y}_i\|}$ | <13% | Symmetric error (handles zeros well) |
| **MAE** | $\frac{1}{n} \sum \|y_i - \hat{y}_i\|$ | <€350/day | Average absolute error in sales |
| **WAPE** | $\frac{\sum \|y_i - \hat{y}_i\|}{\sum \|y_i\|} \times 100$ | <8% | Sales-weighted error |
| **RMSPE** | $\sqrt{\frac{1}{n} \sum \left(\frac{y_i - \hat{y}_i}{y_i}\right)^2}$ | <0.15 | Kaggle competition metric |

### Business Impact Metrics
- **Inventory Turnover**: Increase from 8x to 9x annually (+12.5%)
- **Stockout Rate**: Reduce from 3.2% to 2.7% of SKUs
- **Waste Reduction**: Cut perishable write-offs by €2M/year
- **Labor Optimization**: Reduce overstaffing costs by €1.5M/year

---

## 4. Baseline Models

Before building complex ML models, we must beat:

### 1. Naive Last Week (Lag-7)
- **Method**: $\hat{y}_t = y_{t-7}$ (sales from 7 days ago)
- **Expected sMAPE**: ~15-18%
- **Pros**: Captures day-of-week seasonality
- **Cons**: Ignores promotions, holidays, trends

### 2. Naive Last Year Same Week (Lag-364)
- **Method**: $\hat{y}_t = y_{t-364}$
- **Expected sMAPE**: ~20-25%
- **Pros**: Captures annual seasonality
- **Cons**: Doesn't account for store growth, new competition

### 3. 7-Day Moving Average
- **Method**: $\hat{y}_t = \frac{1}{7} \sum_{i=1}^{7} y_{t-i}$
- **Expected sMAPE**: ~16-19%
- **Pros**: Smooths noise
- **Cons**: Lags behind trends, poor at holidays

### 4. 28-Day Moving Average
- **Method**: $\hat{y}_t = \frac{1}{28} \sum_{i=1}^{28} y_{t-i}$
- **Expected sMAPE**: ~17-20%
- **Pros**: More stable
- **Cons**: Even slower to react

**Target**: Beat best baseline by ≥10% → sMAPE < 13.5% if baseline is 15%

---

## 5. Cost-Benefit Analysis

### Costs of Forecast Errors

#### Over-Forecasting (Predicting too high)
- **Excess Inventory**: €50-100/item in carrying costs
- **Perishable Waste**: 15-30% markdown or total write-off
- **Capital Locked**: Opportunity cost of cash tied up
- **Storage Costs**: Warehouse space, refrigeration

**Estimated Cost**: €75 per unit over-forecasted

#### Under-Forecasting (Predicting too low)
- **Stockouts**: Lost sales (customers buy elsewhere)
- **Customer Dissatisfaction**: Risk of losing loyal customers
- **Emergency Orders**: 20% premium for rush deliveries

**Estimated Cost**: €120 per unit under-forecasted (higher than over-forecasting)

**Asymmetric Loss**: Under-forecasting is **1.6x worse** than over-forecasting  
→ Model should favor **slight over-prediction** (confidence intervals matter!)

### Benefits of Accurate Forecasting

| Benefit Category | Annual Impact (€M) | Confidence |
|-----------------|-------------------|-----------|
| Inventory Cost Reduction | 3.5 | High |
| Waste Reduction | 2.0 | High |
| Stockout Prevention | 4.2 | Medium |
| Labor Optimization | 1.5 | Medium |
| **Total Annual Value** | **11.2** | - |

**Investment Required**: €250K (data science team, infrastructure)  
**ROI**: 4,380% → **44x return**

---

## 6. Constraints & Requirements

### Data Constraints
- **Historical Depth**: Training data from 2013-01-01 to 2015-07-31 (942 days)
- **Forecast Horizon**: Up to 6 weeks (42 days)
- **Granularity**: Daily sales per store (no intraday)
- **Missing Data**: Some stores closed certain days (Open=0)

### Operational Constraints
- **Latency**: Predictions needed within 200ms (API requirement)
- **Update Frequency**: Retrain model weekly (Sundays)
- **Deployment**: RESTful API on AWS (Docker container)
- **Interpretability**: Stakeholders need explainable features (SHAP plots)

### Compliance
- **Data Privacy**: No customer PII (only aggregated sales)
- **Audit Trail**: Log all predictions for regulatory compliance
- **Model Governance**: Version control, reproducibility

---

## 7. Risks & Mitigation

### Data Quality Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|-----------|
| **Promotion Leakage** | High | Medium | Strict temporal validation; use only planned promos |
| **Store Closures** | Medium | Low | Filter Open=0; predict 0 sales |
| **Missing Metadata** | Medium | Medium | Robust imputation; fallback to store-type averages |
| **Outliers** (e.g., Grand Openings) | Low | Low | Cap extreme values; separate model for new stores |

### Model Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|-----------|
| **Concept Drift** | High | High | Weekly monitoring with Evidently; auto-retrain triggers |
| **Overfitting to Promos** | Medium | Medium | Cross-validation; SHAP stability checks |
| **Poor Holiday Performance** | High | Medium | Engineer holiday-specific features; ensemble models |
| **New Competitors** | Medium | Low | Monitor CompetitionDistance changes; alert if shift detected |

### Business Risks
- **Stakeholder Adoption**: If predictions don't align with manager intuition, may be ignored  
  → **Mitigation**: Interpretability (SHAP), confidence intervals, gradual rollout
- **Black Swan Events**: COVID-19, supply chain disruptions  
  → **Mitigation**: Manual override capability; scenario planning

---

## 8. Project Scope

### In Scope
✅ Daily sales forecasts for existing stores (1,115)  
✅ 6-week rolling forecast horizon  
✅ Integration with existing ERP system (API)  
✅ Weekly model retraining pipeline  
✅ Drift monitoring dashboard  

### Out of Scope
❌ Product-level forecasts (SKU granularity)  
❌ Intraday forecasting (hourly patterns)  
❌ New store predictions (insufficient historical data)  
❌ Demand sensing (real-time adjustments)  
❌ Causal inference (promotion effectiveness)

---

## 9. Timeline & Milestones

| Phase | Duration | Deliverables |
|-------|----------|-------------|
| Business Understanding | 1 week | This document, stakeholder alignment |
| Data Understanding | 1 week | EDA report, data dictionary |
| Data Preparation | 2 weeks | Feature engineering, clean datasets |
| Modeling | 3 weeks | Trained models, hyperparameter tuning |
| Evaluation | 1 week | Validation report, business impact analysis |
| Deployment | 2 weeks | Production API, monitoring setup |
| **Total** | **10 weeks** | Full production system |

---

## 10. Stakeholder Sign-Off

**Approved By**:
- [ ] **Head of Supply Chain**: Forecast accuracy targets acceptable
- [ ] **CFO**: Cost-benefit analysis justified
- [ ] **IT Director**: Technical constraints feasible
- [ ] **Store Operations Manager**: Deployment plan workable

**Next Steps**:
1. Proceed to Data Understanding phase
2. Schedule weekly check-ins with stakeholders
3. Set up MLflow experiment tracking
4. Provision AWS infrastructure

---

**Document Version**: 1.0  
**Last Updated**: 2025-11-06  
**Owner**: Data Science Team
