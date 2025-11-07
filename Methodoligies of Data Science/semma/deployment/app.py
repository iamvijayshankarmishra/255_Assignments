"""
Bank Marketing Prediction API
==============================
FastAPI deployment for bank marketing campaign predictions.

Endpoints:
- POST /predict: Single customer prediction
- POST /batch: Batch customer predictions
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
    title="Bank Marketing Prediction API",
    description="Predict customer subscription to term deposit using Random Forest",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and feature engineer at startup
MODEL_PATH = "../models/random_forest_bank_model.pkl"
ENGINEER_PATH = "../models/feature_engineer.pkl"

try:
    model = joblib.load(MODEL_PATH)
    feature_engineer = joblib.load(ENGINEER_PATH)
    logger.info("Model and feature engineer loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None
    feature_engineer = None

# Model configuration
THRESHOLD = 0.5  # Classification threshold


class Customer(BaseModel):
    """Single customer request schema."""
    age: int = Field(..., ge=18, le=100, description="Customer age")
    job: str = Field(..., description="Job type (admin, technician, services, etc.)")
    marital: str = Field(..., description="Marital status (married, single, divorced)")
    education: str = Field(..., description="Education level (primary, secondary, tertiary, unknown)")
    default: str = Field(..., description="Has credit in default? (yes, no)")
    balance: float = Field(..., description="Average yearly balance (€)")
    housing: str = Field(..., description="Has housing loan? (yes, no)")
    loan: str = Field(..., description="Has personal loan? (yes, no)")
    contact: str = Field(..., description="Contact type (cellular, telephone, unknown)")
    day: int = Field(..., ge=1, le=31, description="Last contact day")
    month: str = Field(..., description="Last contact month (jan, feb, mar, etc.)")
    duration: int = Field(..., ge=0, description="Last contact duration (seconds)")
    campaign: int = Field(..., ge=1, description="Number of contacts this campaign")
    pdays: int = Field(..., ge=-1, description="Days since last contact (-1 = never)")
    previous: int = Field(..., ge=0, description="Number of contacts before this campaign")
    poutcome: str = Field(..., description="Previous campaign outcome (success, failure, unknown, other)")
    
    @validator('job')
    def validate_job(cls, v):
        valid_jobs = ['admin', 'technician', 'services', 'management', 'retired', 
                      'blue-collar', 'unemployed', 'entrepreneur', 'housemaid', 
                      'unknown', 'self-employed', 'student']
        if v not in valid_jobs:
            raise ValueError(f"Job must be one of {valid_jobs}")
        return v
    
    @validator('marital')
    def validate_marital(cls, v):
        if v not in ['married', 'single', 'divorced']:
            raise ValueError("Marital must be married, single, or divorced")
        return v
    
    @validator('education')
    def validate_education(cls, v):
        if v not in ['primary', 'secondary', 'tertiary', 'unknown']:
            raise ValueError("Education must be primary, secondary, tertiary, or unknown")
        return v
    
    @validator('default', 'housing', 'loan')
    def validate_yes_no(cls, v):
        if v not in ['yes', 'no']:
            raise ValueError("Must be yes or no")
        return v
    
    @validator('month')
    def validate_month(cls, v):
        valid_months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 
                       'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        if v not in valid_months:
            raise ValueError(f"Month must be one of {valid_months}")
        return v


class BatchRequest(BaseModel):
    """Batch customer request schema."""
    customers: List[Customer] = Field(..., max_items=1000)


class PredictionResponse(BaseModel):
    """Single prediction response schema."""
    subscription_probability: float = Field(..., description="Probability of subscription [0, 1]")
    will_subscribe: bool = Field(..., description="Binary prediction")
    confidence: str = Field(..., description="high/medium/low")
    score: int = Field(..., description="Lead score 0-100")
    recommendation: str = Field(..., description="Action recommendation")


class BatchResponse(BaseModel):
    """Batch prediction response schema."""
    predictions: List[PredictionResponse]
    summary: Dict[str, int]
    total_customers: int
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str
    model_loaded: bool
    timestamp: str
    version: str


class MetricsResponse(BaseModel):
    """Model metrics response schema."""
    roc_auc: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float


def preprocess_customer(customer: Customer) -> np.ndarray:
    """
    Preprocess customer for model input.
    
    Args:
        customer: Customer object
    
    Returns:
        Preprocessed feature array
    """
    # Convert to DataFrame
    df = pd.DataFrame([customer.dict()])
    
    # Apply feature engineering
    df = feature_engineer.transform(df)
    
    return df.values


def get_confidence_level(probability: float) -> str:
    """Get confidence level from probability."""
    if probability >= 0.75 or probability <= 0.25:
        return "high"
    elif probability >= 0.6 or probability <= 0.4:
        return "medium"
    else:
        return "low"


def get_recommendation(probability: float, duration: int, previous: int) -> str:
    """Get action recommendation based on prediction."""
    if probability >= 0.75:
        return "HOT LEAD - High probability, prioritize contact"
    elif probability >= 0.5:
        if duration < 180:  # < 3 minutes last call
            return "WARM LEAD - Medium probability, extend call duration"
        else:
            return "WARM LEAD - Medium probability, follow up soon"
    elif probability >= 0.25:
        if previous == 0:
            return "COLD LEAD - Low probability, consider one more contact"
        else:
            return "COLD LEAD - Low probability, deprioritize"
    else:
        return "NO CONTACT - Very low probability, focus on other leads"


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "Bank Marketing Prediction API",
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
        roc_auc=0.94,
        accuracy=0.91,
        precision=0.68,
        recall=0.45,
        f1_score=0.54
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_subscription(customer: Customer):
    """
    Predict subscription probability for a single customer.
    
    Args:
        customer: Customer with demographics and contact history
    
    Returns:
        Prediction with probability, binary decision, confidence, and recommendation
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Preprocess customer
        X = preprocess_customer(customer)
        
        # Predict probability
        sub_prob = model.predict_proba(X)[0, 1]
        
        # Binary prediction
        will_subscribe = sub_prob >= THRESHOLD
        
        # Confidence level
        confidence = get_confidence_level(sub_prob)
        
        # Lead score (0-100)
        score = int(sub_prob * 100)
        
        # Recommendation
        recommendation = get_recommendation(sub_prob, customer.duration, customer.previous)
        
        logger.info(f"Prediction: prob={sub_prob:.3f}, subscribe={will_subscribe}, age={customer.age}")
        
        return PredictionResponse(
            subscription_probability=round(sub_prob, 4),
            will_subscribe=will_subscribe,
            confidence=confidence,
            score=score,
            recommendation=recommendation
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/batch", response_model=BatchResponse)
async def predict_batch(request: BatchRequest):
    """
    Predict subscription probabilities for batch of customers.
    
    Args:
        request: BatchRequest with list of customers (max 1000)
    
    Returns:
        Batch predictions with summary statistics
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = datetime.now()
    
    try:
        predictions = []
        lead_counts = {"hot": 0, "warm": 0, "cold": 0, "no_contact": 0}
        
        for customer in request.customers:
            # Preprocess and predict
            X = preprocess_customer(customer)
            sub_prob = model.predict_proba(X)[0, 1]
            will_subscribe = sub_prob >= THRESHOLD
            confidence = get_confidence_level(sub_prob)
            score = int(sub_prob * 100)
            recommendation = get_recommendation(sub_prob, customer.duration, customer.previous)
            
            predictions.append(PredictionResponse(
                subscription_probability=round(sub_prob, 4),
                will_subscribe=will_subscribe,
                confidence=confidence,
                score=score,
                recommendation=recommendation
            ))
            
            # Update counts
            if sub_prob >= 0.75:
                lead_counts["hot"] += 1
            elif sub_prob >= 0.5:
                lead_counts["warm"] += 1
            elif sub_prob >= 0.25:
                lead_counts["cold"] += 1
            else:
                lead_counts["no_contact"] += 1
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        logger.info(f"Batch prediction: {len(predictions)} customers, {processing_time:.1f}ms")
        
        return BatchResponse(
            predictions=predictions,
            summary=lead_counts,
            total_customers=len(predictions),
            processing_time_ms=round(processing_time, 2)
        )
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.post("/segment")
async def segment_customers(request: BatchRequest):
    """
    Segment customers by subscription probability.
    
    Args:
        request: BatchRequest with list of customers
    
    Returns:
        Customer segments with statistics
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        segments = {
            "hot_leads": [],
            "warm_leads": [],
            "cold_leads": [],
            "no_contact": []
        }
        
        for i, customer in enumerate(request.customers):
            X = preprocess_customer(customer)
            sub_prob = model.predict_proba(X)[0, 1]
            
            customer_data = customer.dict()
            customer_data['subscription_probability'] = round(sub_prob, 4)
            customer_data['customer_id'] = i
            
            if sub_prob >= 0.75:
                segments["hot_leads"].append(customer_data)
            elif sub_prob >= 0.5:
                segments["warm_leads"].append(customer_data)
            elif sub_prob >= 0.25:
                segments["cold_leads"].append(customer_data)
            else:
                segments["no_contact"].append(customer_data)
        
        return {
            "segments": segments,
            "summary": {
                "hot_leads": len(segments["hot_leads"]),
                "warm_leads": len(segments["warm_leads"]),
                "cold_leads": len(segments["cold_leads"]),
                "no_contact": len(segments["no_contact"])
            },
            "recommendations": {
                "hot_leads": "Contact immediately with premium offers",
                "warm_leads": "Schedule follow-up within 1 week",
                "cold_leads": "Nurture with email campaigns",
                "no_contact": "Remove from active campaign"
            }
        }
    
    except Exception as e:
        logger.error(f"Segmentation error: {e}")
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")


@app.post("/feature_importance")
async def get_feature_importance():
    """Get feature importance from Random Forest model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Get feature importances
        importances = model.feature_importances_
        feature_names = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else \
                        [f'feature_{i}' for i in range(len(importances))]
        
        # Sort by importance
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return {
            "top_10_features": importance_df.head(10).to_dict(orient='records'),
            "all_features": importance_df.to_dict(orient='records')
        }
    
    except Exception as e:
        logger.error(f"Feature importance error: {e}")
        raise HTTPException(status_code=500, detail=f"Feature importance failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
