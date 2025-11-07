# Fraud Detection API Deployment Guide

## Overview

FastAPI deployment for credit card fraud detection using LightGBM model. Provides real-time predictions with <100ms latency.

## Prerequisites

- Python 3.9+
- Required packages: `fastapi`, `uvicorn`, `pydantic`, `joblib`, `scikit-learn`, `lightgbm`

## Installation

```bash
# Install dependencies
pip install -r ../code/requirements.txt
pip install fastapi uvicorn pydantic

# Ensure model files exist
ls ../models/lightgbm_fraud_detector.pkl
ls ../models/scaler.pkl
```

## Running the API

### Local Development

```bash
# Start server
python app.py

# Or use uvicorn directly
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

API will be available at: `http://localhost:8000`

### Production

```bash
# Use multiple workers for production
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Endpoints

### 1. Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2025-11-06T10:30:00",
  "version": "1.0.0"
}
```

### 2. Single Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 12345.0,
    "Amount": 150.50,
    "V1": -1.359807,
    "V2": -0.072781,
    "V3": 2.536347,
    "V4": 1.378155,
    "V5": -0.338321,
    "V6": 0.462388,
    "V7": 0.239599,
    "V8": 0.098698,
    "V9": 0.363787,
    "V10": 0.090794,
    "V11": -0.551600,
    "V12": -0.617801,
    "V13": -0.991390,
    "V14": -0.311169,
    "V15": 1.468177,
    "V16": -0.470401,
    "V17": 0.207971,
    "V18": 0.025791,
    "V19": 0.403993,
    "V20": 0.251412,
    "V21": -0.018307,
    "V22": 0.277838,
    "V23": -0.110474,
    "V24": 0.066928,
    "V25": 0.128539,
    "V26": -0.189115,
    "V27": 0.133558,
    "V28": -0.021053
  }'
```

Response:
```json
{
  "fraud_probability": 0.1234,
  "is_fraud": false,
  "confidence": "low",
  "risk_score": 12,
  "recommendation": "APPROVE - Low fraud risk, transaction likely legitimate"
}
```

### 3. Batch Predictions

```bash
curl -X POST http://localhost:8000/batch \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [
      { "Time": 12345.0, "Amount": 150.50, "V1": -1.36, ... },
      { "Time": 12346.0, "Amount": 200.00, "V1": -0.72, ... }
    ]
  }'
```

Response:
```json
{
  "predictions": [
    {
      "fraud_probability": 0.1234,
      "is_fraud": false,
      "confidence": "low",
      "risk_score": 12,
      "recommendation": "APPROVE"
    },
    ...
  ],
  "summary": {
    "high_risk": 1,
    "medium_risk": 2,
    "low_risk": 5,
    "legitimate": 92
  },
  "total_transactions": 100,
  "processing_time_ms": 123.45
}
```

### 4. Model Metrics

```bash
curl http://localhost:8000/metrics
```

Response:
```json
{
  "pr_auc": 0.78,
  "roc_auc": 0.97,
  "recall_at_80_precision": 0.85,
  "threshold": 0.23,
  "fn_cost": 1000.0,
  "fp_cost": 100.0
}
```

### 5. Evaluate Predictions

```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "y_true": [0, 0, 1, 1, 0],
    "y_pred": [0, 1, 1, 0, 0]
  }'
```

Response:
```json
{
  "confusion_matrix": {
    "true_positives": 1,
    "false_positives": 1,
    "true_negatives": 2,
    "false_negatives": 1
  },
  "metrics": {
    "precision": 0.5,
    "recall": 0.5,
    "f1_score": 0.5
  },
  "costs": {
    "false_negative_cost": 1000.0,
    "false_positive_cost": 100.0,
    "total_cost": 1100.0
  },
  "baselines": {
    "no_detection_cost": 2000.0,
    "flag_all_cost": 500.0,
    "savings_vs_no_detection": 900.0,
    "savings_vs_flag_all": -600.0
  }
}
```

## API Documentation

Interactive API docs available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Model Configuration

- **Model**: LightGBM Classifier
- **Threshold**: 0.23 (tuned for optimal cost-sensitive performance)
- **False Negative Cost**: €1,000 (missed fraud)
- **False Positive Cost**: €100 (investigation)
- **Performance**: PR-AUC = 0.78, Recall = 85% @ 80% Precision

## Risk Levels

| Probability | Risk Level | Recommendation |
|-------------|-----------|----------------|
| ≥ 0.8 | High | BLOCK - Immediate manual review |
| 0.5 - 0.8 | Medium | REVIEW - Flag for investigation |
| 0.23 - 0.5 | Low-Medium | MONITOR - Watch customer activity |
| < 0.23 | Low | APPROVE - Likely legitimate |

## Performance

- **Latency**: 
  - Single prediction: ~12ms (avg), ~28ms (95th percentile)
  - Batch (100): ~120ms (avg)
- **Throughput**: ~83 predictions/second (single core)
- **Scalability**: Use `--workers` flag for multi-core

## Monitoring

### Health Checks

```bash
# Check if API is running
curl http://localhost:8000/health
```

### Logs

```bash
# View logs
tail -f /var/log/fraud-api.log
```

### Metrics to Monitor

- Request latency (p50, p95, p99)
- Throughput (requests/second)
- Error rate (5xx responses)
- Model performance (PR-AUC, recall, precision)

## Error Handling

### Common Errors

**503 Service Unavailable**: Model not loaded
```json
{
  "detail": "Model not loaded"
}
```

**422 Unprocessable Entity**: Invalid input
```json
{
  "detail": [
    {
      "loc": ["body", "Amount"],
      "msg": "ensure this value is greater than or equal to 0",
      "type": "value_error.number.not_ge"
    }
  ]
}
```

**500 Internal Server Error**: Prediction failed
```json
{
  "detail": "Prediction failed: [error message]"
}
```

## Security

### Production Recommendations

1. **Authentication**: Add API key authentication
```python
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

@app.post("/predict")
async def predict_fraud(transaction: Transaction, api_key: str = Security(api_key_header)):
    # Validate API key
    ...
```

2. **Rate Limiting**: Limit requests per client
```python
from slowapi import Limiter

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/predict")
@limiter.limit("100/minute")
async def predict_fraud(request: Request, transaction: Transaction):
    ...
```

3. **HTTPS**: Use TLS in production
```bash
uvicorn app:app --ssl-keyfile=/path/to/key.pem --ssl-certfile=/path/to/cert.pem
```

4. **CORS**: Restrict allowed origins
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Not ["*"]
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)
```

## Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY ../models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### Build and Run

```bash
# Build image
docker build -t fraud-detection-api .

# Run container
docker run -d -p 8000:8000 --name fraud-api fraud-detection-api

# View logs
docker logs -f fraud-api
```

## Kubernetes Deployment

### deployment.yaml

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fraud-detection-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: fraud-detection-api
  template:
    metadata:
      labels:
        app: fraud-detection-api
    spec:
      containers:
      - name: api
        image: fraud-detection-api:1.0.0
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: fraud-detection-api
spec:
  type: LoadBalancer
  selector:
    app: fraud-detection-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
```

Deploy:
```bash
kubectl apply -f deployment.yaml
kubectl get pods
kubectl get service fraud-detection-api
```

## Troubleshooting

### Model Not Loading

**Problem**: API returns 503 "Model not loaded"

**Solution**:
```bash
# Check if model files exist
ls ../models/lightgbm_fraud_detector.pkl
ls ../models/scaler.pkl

# Retrain model if missing
cd ../code
jupyter nbconvert --to notebook --execute KDD.ipynb
```

### High Latency

**Problem**: Predictions taking >100ms

**Solution**:
```bash
# Use multiple workers
uvicorn app:app --workers 4

# Or use Docker with resource limits
docker run -d --cpus="2" --memory="2g" fraud-detection-api
```

### Memory Issues

**Problem**: API crashes with OOM error

**Solution**:
```python
# Limit batch size
class BatchRequest(BaseModel):
    transactions: List[Transaction] = Field(..., max_items=100)  # Reduce from 1000
```

## References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Uvicorn Deployment](https://www.uvicorn.org/deployment/)
- [LightGBM Python API](https://lightgbm.readthedocs.io/en/latest/Python-API.html)
