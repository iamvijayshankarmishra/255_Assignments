"""
Fraud Detection API
===================
FastAPI deployment for credit card fraud detection model.

Endpoints:
- POST /predict: Single transaction prediction
- POST /batch: Batch transaction predictions
- GET /health: Health check
- GET /metrics: Model performance metrics
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time credit card fraud detection using LightGBM",
    version="1.0.0"
)

# CORS middleware (adjust origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and scaler at startup
MODEL_PATH = "../models/lightgbm_fraud_detector.pkl"
SCALER_PATH = "../models/scaler.pkl"

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    logger.info("Model and scaler loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None
    scaler = None

# Model configuration
THRESHOLD = 0.23  # Optimal threshold from evaluation
FN_COST = 1000.0  # False negative cost (€)
FP_COST = 100.0   # False positive cost (€)


class Transaction(BaseModel):
    """Single transaction request schema."""
    Time: float = Field(..., description="Seconds since first transaction")
    Amount: float = Field(..., ge=0, description="Transaction amount (€)")
    V1: float = Field(..., description="PCA feature 1")
    V2: float = Field(..., description="PCA feature 2")
    V3: float = Field(..., description="PCA feature 3")
    V4: float = Field(..., description="PCA feature 4")
    V5: float = Field(..., description="PCA feature 5")
    V6: float = Field(..., description="PCA feature 6")
    V7: float = Field(..., description="PCA feature 7")
    V8: float = Field(..., description="PCA feature 8")
    V9: float = Field(..., description="PCA feature 9")
    V10: float = Field(..., description="PCA feature 10")
    V11: float = Field(..., description="PCA feature 11")
    V12: float = Field(..., description="PCA feature 12")
    V13: float = Field(..., description="PCA feature 13")
    V14: float = Field(..., description="PCA feature 14")
    V15: float = Field(..., description="PCA feature 15")
    V16: float = Field(..., description="PCA feature 16")
    V17: float = Field(..., description="PCA feature 17")
    V18: float = Field(..., description="PCA feature 18")
    V19: float = Field(..., description="PCA feature 19")
    V20: float = Field(..., description="PCA feature 20")
    V21: float = Field(..., description="PCA feature 21")
    V22: float = Field(..., description="PCA feature 22")
    V23: float = Field(..., description="PCA feature 23")
    V24: float = Field(..., description="PCA feature 24")
    V25: float = Field(..., description="PCA feature 25")
    V26: float = Field(..., description="PCA feature 26")
    V27: float = Field(..., description="PCA feature 27")
    V28: float = Field(..., description="PCA feature 28")
    
    @validator('Time')
    def validate_time(cls, v):
        if v < 0 or v > 200000:  # ~48 hours
            raise ValueError("Time must be between 0 and 200000 seconds")
        return v
    
    @validator('Amount')
    def validate_amount(cls, v):
        if v > 30000:  # €30K max reasonable
            raise ValueError("Amount must be <= €30,000")
        return v


class BatchRequest(BaseModel):
    """Batch transaction request schema."""
    transactions: List[Transaction] = Field(..., max_items=1000)


class PredictionResponse(BaseModel):
    """Single prediction response schema."""
    fraud_probability: float = Field(..., description="Probability of fraud [0, 1]")
    is_fraud: bool = Field(..., description="Binary prediction (threshold=0.23)")
    confidence: str = Field(..., description="high/medium/low")
    risk_score: int = Field(..., description="Risk score 0-100")
    recommendation: str = Field(..., description="Action recommendation")


class BatchResponse(BaseModel):
    """Batch prediction response schema."""
    predictions: List[PredictionResponse]
    summary: Dict[str, int]
    total_transactions: int
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str
    model_loaded: bool
    timestamp: str
    version: str


class MetricsResponse(BaseModel):
    """Model metrics response schema."""
    pr_auc: float
    roc_auc: float
    recall_at_80_precision: float
    threshold: float
    fn_cost: float
    fp_cost: float


def preprocess_transaction(transaction: Transaction) -> np.ndarray:
    """
    Preprocess transaction for model input.
    
    Args:
        transaction: Transaction object
    
    Returns:
        Preprocessed feature array (1, 30)
    """
    # Convert to DataFrame
    df = pd.DataFrame([transaction.dict()])
    
    # Ensure correct column order (V1-V28, Time, Amount)
    feature_order = [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount']
    df = df[feature_order]
    
    # Scale features (Time and Amount only - V1-V28 already scaled)
    df_scaled = df.copy()
    df_scaled[['Time', 'Amount']] = scaler.transform(df[['Time', 'Amount']])
    
    return df_scaled.values


def get_confidence_level(probability: float) -> str:
    """Get confidence level from probability."""
    if probability >= 0.8:
        return "high"
    elif probability >= 0.5:
        return "medium"
    else:
        return "low"


def get_recommendation(probability: float, amount: float) -> str:
    """Get action recommendation based on probability and amount."""
    if probability >= 0.8:
        return "BLOCK - High fraud risk, immediate manual review required"
    elif probability >= 0.5:
        return "REVIEW - Medium fraud risk, flag for investigation"
    elif probability >= 0.23:  # Threshold
        if amount > 500:
            return "REVIEW - Low-medium risk but high amount, recommend review"
        else:
            return "MONITOR - Low-medium risk, monitor customer activity"
    else:
        return "APPROVE - Low fraud risk, transaction likely legitimate"


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "Fraud Detection API",
        "version": "1.0.0",
        "endpoints": "/docs for API documentation"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get model performance metrics."""
    return MetricsResponse(
        pr_auc=0.78,
        roc_auc=0.97,
        recall_at_80_precision=0.85,
        threshold=THRESHOLD,
        fn_cost=FN_COST,
        fp_cost=FP_COST
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(transaction: Transaction):
    """
    Predict fraud probability for a single transaction.
    
    Args:
        transaction: Transaction with features V1-V28, Time, Amount
    
    Returns:
        Prediction with fraud probability, binary decision, confidence, and recommendation
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Preprocess transaction
        X = preprocess_transaction(transaction)
        
        # Predict probability
        fraud_prob = model.predict_proba(X)[0, 1]
        
        # Binary prediction
        is_fraud = fraud_prob >= THRESHOLD
        
        # Confidence level
        confidence = get_confidence_level(fraud_prob)
        
        # Risk score (0-100)
        risk_score = int(fraud_prob * 100)
        
        # Recommendation
        recommendation = get_recommendation(fraud_prob, transaction.Amount)
        
        logger.info(f"Prediction: prob={fraud_prob:.3f}, is_fraud={is_fraud}, amount=€{transaction.Amount:.2f}")
        
        return PredictionResponse(
            fraud_probability=round(fraud_prob, 4),
            is_fraud=is_fraud,
            confidence=confidence,
            risk_score=risk_score,
            recommendation=recommendation
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/batch", response_model=BatchResponse)
async def predict_batch(request: BatchRequest):
    """
    Predict fraud probabilities for batch of transactions.
    
    Args:
        request: BatchRequest with list of transactions (max 1000)
    
    Returns:
        Batch predictions with summary statistics
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = datetime.now()
    
    try:
        predictions = []
        fraud_counts = {"high_risk": 0, "medium_risk": 0, "low_risk": 0, "legitimate": 0}
        
        for transaction in request.transactions:
            # Preprocess and predict
            X = preprocess_transaction(transaction)
            fraud_prob = model.predict_proba(X)[0, 1]
            is_fraud = fraud_prob >= THRESHOLD
            confidence = get_confidence_level(fraud_prob)
            risk_score = int(fraud_prob * 100)
            recommendation = get_recommendation(fraud_prob, transaction.Amount)
            
            predictions.append(PredictionResponse(
                fraud_probability=round(fraud_prob, 4),
                is_fraud=is_fraud,
                confidence=confidence,
                risk_score=risk_score,
                recommendation=recommendation
            ))
            
            # Update counts
            if fraud_prob >= 0.8:
                fraud_counts["high_risk"] += 1
            elif fraud_prob >= 0.5:
                fraud_counts["medium_risk"] += 1
            elif fraud_prob >= THRESHOLD:
                fraud_counts["low_risk"] += 1
            else:
                fraud_counts["legitimate"] += 1
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        logger.info(f"Batch prediction: {len(predictions)} transactions, {processing_time:.1f}ms")
        
        return BatchResponse(
            predictions=predictions,
            summary=fraud_counts,
            total_transactions=len(predictions),
            processing_time_ms=round(processing_time, 2)
        )
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.post("/evaluate")
async def evaluate_predictions(y_true: List[int], y_pred: List[int]):
    """
    Evaluate model predictions with cost-sensitive metrics.
    
    Args:
        y_true: True labels (0=legitimate, 1=fraud)
        y_pred: Predicted labels (0=legitimate, 1=fraud)
    
    Returns:
        Cost-sensitive evaluation metrics
    """
    if len(y_true) != len(y_pred):
        raise HTTPException(status_code=400, detail="y_true and y_pred must have same length")
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Confusion matrix
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    # Cost calculation
    fn_cost_total = fn * FN_COST
    fp_cost_total = fp * FP_COST
    total_cost = fn_cost_total + fp_cost_total
    
    # Baseline costs
    no_detection_cost = np.sum(y_true) * FN_COST
    flag_all_cost = len(y_true) * FP_COST
    
    return {
        "confusion_matrix": {
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn)
        },
        "metrics": {
            "precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
            "recall": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
            "f1_score": float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
        },
        "costs": {
            "false_negative_cost": float(fn_cost_total),
            "false_positive_cost": float(fp_cost_total),
            "total_cost": float(total_cost)
        },
        "baselines": {
            "no_detection_cost": float(no_detection_cost),
            "flag_all_cost": float(flag_all_cost),
            "savings_vs_no_detection": float(no_detection_cost - total_cost),
            "savings_vs_flag_all": float(flag_all_cost - total_cost)
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
