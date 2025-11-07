# Monitoring Plan: Rossmann Sales Forecasting

**Model**: LightGBM Sales Forecaster v1.0  
**Deployment Date**: 2025-11-06  
**Review Frequency**: Weekly (automated), Monthly (manual deep-dive)

---

## 1. Performance Monitoring

### Real-Time Metrics (Daily)

| Metric | Threshold | Alert Level | Action |
|--------|-----------|-------------|--------|
| **sMAPE** | > 15% | ⚠️ Warning | Investigate within 24h |
| **sMAPE** | > 18% | 🚨 Critical | Rollback to previous model |
| **MAE (€)** | > €450 | ⚠️ Warning | Check for outliers |
| **API Latency (p95)** | > 200ms | ⚠️ Warning | Scale infrastructure |
| **API Error Rate** | > 1% | 🚨 Critical | Check logs, rollback if needed |

### Weekly Aggregation Report

**Generated Every Monday at 6 AM**

- **Overall Performance**: sMAPE, MAE, RMSE for past 7 days
- **Per-Store Breakdown**: Top 10 worst-performing stores
- **Per-DayOfWeek**: Identify systematic errors (e.g., Saturdays always over-predicted)
- **Promo vs Non-Promo**: Separate metrics
- **Holiday Impact**: Flag if StateHoliday days present

**Recipients**: Data Science Team, Supply Chain Manager

---

## 2. Data Drift Detection

### Feature Drift (Evidently AI)

**Run Weekly (Sundays)**

Monitor distribution shifts in key features:

| Feature | Drift Test | Threshold | Notes |
|---------|-----------|-----------|-------|
| **Sales_Lag7** | KS Test | p < 0.05 | Sales pattern changing? |
| **Promo** | Chi-Squared | p < 0.05 | Promo frequency shift? |
| **DayOfWeek** | Chi-Squared | p < 0.05 | (Shouldn't drift - sanity check) |
| **StoreType** | Chi-Squared | p < 0.05 | Store closures/openings? |
| **CompetitionDistance** | KS Test | p < 0.05 | New competitors? |

**Dashboard**: Evidently HTML report saved to `reports/drift/YYYY-MM-DD_drift_report.html`

**Alerts**:
- If ≥3 features show drift → **Trigger retraining**
- If `Sales_Lag7` drifts significantly → **Investigate business change** (new product line? pricing shift?)

### Prediction Drift

Monitor if **predicted sales distribution** shifts from training:

- **Mean Sales**: Expected ~€5,800/day ± 10%
- **Std Sales**: Expected ~€2,100 ± 15%
- **Skewness**: Expected ~0.8 (right-skewed)

**Alert**: If mean shifts >15% for 5 consecutive days → Investigate

---

## 3. Model Retraining

### Scheduled Retraining

**Frequency**: **Every Sunday at 2 AM**

**Process**:
1. Fetch last 18 months of sales data (rolling window)
2. Re-run feature engineering pipeline
3. Retrain LightGBM with same hyperparameters
4. Validate on last 4 weeks (holdout)
5. If sMAPE < 14% on validation → Deploy new model
6. If sMAPE ≥ 14% → Keep current model, alert team

**Automation**: GitHub Actions workflow / Airflow DAG

### Triggered Retraining

**Immediate retraining if**:
1. sMAPE > 18% for 3 consecutive days
2. ≥3 features show significant drift
3. New competitor detected (CompetitionDistance changes for >5% of stores)
4. Major business event (e.g., merger, new store format)

**Manual Override**: Data scientist can trigger via `/retrain` API endpoint (requires auth)

---

## 4. Alerting System

### Alert Channels

- **Slack**: #ml-monitoring channel (real-time alerts)
- **Email**: data-science-team@rossmann.de (daily summaries)
- **PagerDuty**: Critical alerts (sMAPE > 18%, API down)

### Alert Severity

| Level | Description | Response Time | Escalation |
|-------|-------------|---------------|-----------|
| **Info** | Weekly report, normal drift | N/A | None |
| **Warning** | sMAPE 15-18%, minor drift | 24 hours | Data scientist review |
| **Critical** | sMAPE > 18%, API error > 1% | 2 hours | Rollback + senior engineer |

---

## 5. Explainability Monitoring

### SHAP Consistency Check (Monthly)

**Goal**: Ensure model logic remains stable

1. Compute SHAP values on 1,000 random predictions
2. Compare top 5 features with baseline (deployment month)
3. **Alert if**:
   - DayOfWeek drops below 3rd place (should always be top 3)
   - New feature suddenly jumps to top 5 (investigate why)

**Expected Stable Ranking**:
1. DayOfWeek
2. Promo
3. Sales_Lag7
4. Sales_RollingMean28
5. StoreType

### Prediction Auditing

**Random Sample Review** (10 predictions/day):
- Save input features + prediction + actual (next day)
- Data scientist reviews for anomalies (e.g., predicted €10K but actual €500)
- Document failure patterns

---

## 6. Business KPI Tracking

### Inventory Metrics (Monthly)

Track actual business impact:

| KPI | Baseline (Pre-ML) | Target (With ML) | Actual | Status |
|-----|------------------|------------------|--------|--------|
| **Stockout Rate** | 3.2% | <2.7% | TBD | 🟡 Monitoring |
| **Excess Inventory** | €15M | <€12M | TBD | 🟡 Monitoring |
| **Waste (Perishables)** | €8.2M/year | <€5M/year | TBD | 🟡 Monitoring |
| **Forecast Accuracy** | 84% (manual) | >90% | TBD | 🟡 Monitoring |

**Review Cadence**: Monthly with Supply Chain team

### Customer Satisfaction

- **Survey Question**: "Products I wanted were available" (1-5 scale)
- **Target**: Increase from 4.1 to 4.3 within 6 months
- **Track**: Correlation between stockout rate and NPS

---

## 7. Model Versioning & Rollback

### Version Control

All models stored in **MLflow Model Registry**:

```
models/
  rossmann-sales-forecaster/
    v1.0/  # Initial deployment
      model.joblib
      metadata.json (hyperparams, metrics, training date)
      requirements.txt
    v1.1/  # First retrain
      ...
```

**Metadata Tracked**:
- Training date, data range
- Hyperparameters
- Cross-validation sMAPE
- Holdout test sMAPE
- Training time, model size
- Git commit hash

### Rollback Procedure

**When to Rollback**:
1. New model performs worse (sMAPE > prev_model + 1pp)
2. API errors spike after deployment
3. Unexplained SHAP shifts

**How**:
```bash
# Via API
curl -X POST https://api.rossmann.ml/rollback \
  -d '{"version": "v1.0"}' \
  -H "Authorization: Bearer $TOKEN"

# Via MLflow
mlflow deployments update --name rossmann-sales-forecaster --model-uri models:/rossmann-sales-forecaster/v1.0
```

**Automatic Rollback**: If sMAPE > 18% for 6 consecutive hours → Auto-rollback to previous version

---

## 8. Compliance & Auditability

### Prediction Logging

**All predictions saved to database**:

```sql
CREATE TABLE predictions (
  prediction_id UUID PRIMARY KEY,
  timestamp TIMESTAMP,
  store_id INT,
  date DATE,
  predicted_sales FLOAT,
  actual_sales FLOAT (filled next day),
  model_version VARCHAR,
  input_features JSONB,
  shap_values JSONB
);
```

**Retention**: 2 years (regulatory requirement)

### Model Audit Trail

**Quarterly Audit (every 3 months)**:
1. Review top 20 worst predictions (highest sMAPE)
2. Check for bias (e.g., consistently over/under for certain StoreTypes)
3. Document any manual overrides used by store managers
4. Update model card with findings

**Compliance**:
- ✅ No PII (only aggregated sales data)
- ✅ GDPR-compliant (no customer identifiers)
- ✅ Explainable (SHAP values on demand)

---

## 9. Infrastructure Monitoring

### API Health

**Metrics (via Prometheus + Grafana)**:

- **Request Rate**: Expect ~100 req/min during planning hours (Mon-Fri 9-11 AM)
- **Latency**: p50 <50ms, p95 <150ms, p99 <200ms
- **Error Rate**: <0.1% (4xx/5xx responses)
- **Model Load Time**: <2s on container restart

**Alerts**:
- Latency p95 > 200ms for 5 min → Scale horizontally
- Error rate > 1% for 2 min → Investigate + alert

### Resource Utilization

| Resource | Normal | Alert Threshold | Action |
|----------|--------|----------------|--------|
| **CPU** | 20-40% | >80% for 10 min | Scale up |
| **Memory** | 2-4 GB | >6 GB | Check for memory leak |
| **Disk (model artifacts)** | 500 MB | >2 GB | Clean old versions |

---

## 10. Continuous Improvement

### Monthly Model Review Meeting

**Attendees**: Data Science, Supply Chain, Store Ops

**Agenda**:
1. Review performance metrics (vs targets)
2. Discuss business feedback (store manager complaints?)
3. Prioritize failure cases for next iteration
4. Plan A/B tests (e.g., new features, alternative models)

### Experiment Tracking

**Ongoing Experiments** (via MLflow):

| Experiment | Status | Goal | Expected Impact |
|-----------|--------|------|----------------|
| **Holiday Ensemble** | 🟡 In Progress | sMAPE < 14% on holidays | -3pp holiday error |
| **Weather Features** | 📋 Planned | Add temperature, rain | -0.5pp overall |
| **Promo Lift Model** | 💡 Idea | Causal inference on promo | Better promo planning |

---

## 11. Incident Response

### Runbook: High sMAPE Alert

**Symptoms**: sMAPE > 15% for 3+ days

**Steps**:
1. **Check Data Pipeline**: Any missing data? (Sales_Lag7 all NaN?)
2. **Check Recent Events**: New competitor opened? Holiday?
3. **Compare Predictions vs Baselines**: Is naive model also failing? (External shock)
4. **Review Top Errors**: Which stores/days are worst? Pattern?
5. **Manual Override**: If systematic, apply correction factor temporarily
6. **Trigger Retrain**: If drift detected

**Resolution Time**: <24 hours

### Runbook: API Down

**Symptoms**: Health check fails, 503 errors

**Steps**:
1. **Check Container**: `docker ps`, restart if crashed
2. **Check Model File**: Is `model.joblib` accessible?
3. **Check Logs**: Any Python exceptions?
4. **Rollback**: If recent deployment, rollback to previous version
5. **Scale**: If traffic spike, add replicas

**Resolution Time**: <30 minutes (critical)

---

## 12. Success Metrics (6-Month Review)

**Green Light** (Model Working Well):
- ✅ sMAPE consistently <13.5%
- ✅ Inventory costs reduced by ≥5%
- ✅ Stockout rate <2.7%
- ✅ No critical incidents (rollbacks)

**Yellow Light** (Needs Improvement):
- ⚠️ sMAPE 13.5-15%
- ⚠️ 1-2 rollbacks due to drift
- ⚠️ Holiday predictions still weak

**Red Light** (Re-evaluate Approach):
- 🚨 sMAPE >15% consistently
- 🚨 Business KPIs not improving
- 🚨 Frequent rollbacks/incidents
- 🚨 Stakeholder distrust (manual overrides common)

---

**Version**: 1.0  
**Next Review**: 2025-12-06 (1 month post-deployment)  
**Owner**: Data Science Team  
**On-Call**: [Your Name] (Week 1-2), [Colleague] (Week 3-4)
