# Bank Marketing Prediction API Deployment Guide

## Overview

FastAPI deployment for bank marketing campaign predictions using Random Forest model. Provides real-time customer subscription probability predictions.

## Installation

```bash
# Install dependencies
pip install -r ../code/requirements.txt
pip install fastapi uvicorn pydantic

# Ensure model files exist
ls ../models/random_forest_bank_model.pkl
ls ../models/feature_engineer.pkl
```

## Running the API

### Local Development

```bash
# Start server
python app.py

# Or use uvicorn directly
uvicorn app:app --reload --host 0.0.0.0 --port 8001
```

API will be available at: `http://localhost:8001`

### Production

```bash
# Use multiple workers
uvicorn app:app --host 0.0.0.0 --port 8001 --workers 4
```

## API Endpoints

### 1. Health Check

```bash
curl http://localhost:8001/health
```

### 2. Single Customer Prediction

```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "job": "management",
    "marital": "married",
    "education": "tertiary",
    "default": "no",
    "balance": 1500,
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "day": 15,
    "month": "may",
    "duration": 300,
    "campaign": 2,
    "pdays": -1,
    "previous": 0,
    "poutcome": "unknown"
  }'
```

Response:
```json
{
  "subscription_probability": 0.6834,
  "will_subscribe": true,
  "confidence": "medium",
  "score": 68,
  "recommendation": "WARM LEAD - Medium probability, follow up soon"
}
```

### 3. Batch Predictions

```bash
curl -X POST http://localhost:8001/batch \
  -H "Content-Type: application/json" \
  -d '{
    "customers": [
      { "age": 35, "job": "management", ... },
      { "age": 42, "job": "technician", ... }
    ]
  }'
```

### 4. Customer Segmentation

```bash
curl -X POST http://localhost:8001/segment \
  -H "Content-Type: application/json" \
  -d '{
    "customers": [...]
  }'
```

Response:
```json
{
  "segments": {
    "hot_leads": [...],
    "warm_leads": [...],
    "cold_leads": [...],
    "no_contact": [...]
  },
  "summary": {
    "hot_leads": 25,
    "warm_leads": 120,
    "cold_leads": 300,
    "no_contact": 555
  },
  "recommendations": {
    "hot_leads": "Contact immediately with premium offers",
    "warm_leads": "Schedule follow-up within 1 week",
    "cold_leads": "Nurture with email campaigns",
    "no_contact": "Remove from active campaign"
  }
}
```

### 5. Feature Importance

```bash
curl -X POST http://localhost:8001/feature_importance
```

## Lead Scoring

| Probability | Lead Type | Recommendation |
|-------------|-----------|----------------|
| ≥ 0.75 | Hot Lead | Contact immediately with premium offers |
| 0.50 - 0.75 | Warm Lead | Schedule follow-up within 1 week |
| 0.25 - 0.50 | Cold Lead | Nurture with email campaigns |
| < 0.25 | No Contact | Focus resources on higher-probability leads |

## Model Performance

- **ROC-AUC**: 0.94
- **Accuracy**: 91%
- **Precision**: 68%
- **Recall**: 45%
- **F1-Score**: 0.54

## Interactive Documentation

- Swagger UI: `http://localhost:8001/docs`
- ReDoc: `http://localhost:8001/redoc`

## Docker Deployment

```bash
# Build image
docker build -t bank-marketing-api .

# Run container
docker run -d -p 8001:8001 --name bank-api bank-marketing-api
```

## Monitoring

```bash
# View logs
tail -f /var/log/bank-api.log

# Check health
curl http://localhost:8001/health
```

## Security

See fraud detection API README for authentication, rate limiting, and HTTPS setup.

## References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
