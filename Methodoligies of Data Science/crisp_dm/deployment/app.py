"""
FastAPI Deployment for Rossmann Sales Forecasting
CRISP-DM: Deployment Phase
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Rossmann Sales Forecasting API",
    description="Production API for predicting daily sales across Rossmann stores",
    version="1.0.0"
)

# Load model (in production, use proper model registry)
MODEL_PATH = "deployment/model.joblib"
try:
    model_pipeline = joblib.load(MODEL_PATH)
    logger.info(f"✓ Model loaded from {MODEL_PATH}")
except FileNotFoundError:
    logger.warning(f"❌ Model not found at {MODEL_PATH}. Train and save model first.")
    model_pipeline = None


# Request schema
class PredictionRequest(BaseModel):
    store_id: int = Field(..., description="Store ID (1-1115)", ge=1, le=1115)
    date: str = Field(..., description="Prediction date (YYYY-MM-DD)")
    day_of_week: int = Field(..., description="Day of week (1=Monday, 7=Sunday)", ge=1, le=7)
    open: int = Field(1, description="Is store open? (1=Yes, 0=No)", ge=0, le=1)
    promo: int = Field(0, description="Is promo active? (1=Yes, 0=No)", ge=0, le=1)
    state_holiday: str = Field("0", description="State holiday (0/a/b/c)")
    school_holiday: int = Field(0, description="School holiday (1=Yes, 0=No)", ge=0, le=1)
    
    # Store metadata (typically fetched from database in production)
    store_type: Optional[str] = Field("a", description="Store type (a/b/c/d)")
    assortment: Optional[str] = Field("a", description="Assortment level (a/b/c)")
    competition_distance: Optional[float] = Field(None, description="Distance to nearest competitor (meters)")
    promo2: Optional[int] = Field(0, description="Long-term promo participation", ge=0, le=1)
    
    class Config:
        json_schema_extra = {
            "example": {
                "store_id": 1,
                "date": "2015-09-18",
                "day_of_week": 5,
                "open": 1,
                "promo": 1,
                "state_holiday": "0",
                "school_holiday": 0,
                "store_type": "c",
                "assortment": "a",
                "competition_distance": 1270.0,
                "promo2": 1
            }
        }


class PredictionResponse(BaseModel):
    store_id: int
    date: str
    predicted_sales: float
    confidence_interval_lower: Optional[float] = None
    confidence_interval_upper: Optional[float] = None
    is_open: bool
    model_version: str = "1.0.0"
    timestamp: str


class BatchPredictionRequest(BaseModel):
    predictions: List[PredictionRequest]


@app.get("/")
def root():
    """Root endpoint."""
    return {
        "message": "Rossmann Sales Forecasting API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "Single store-date prediction",
            "/predict/batch": "Batch predictions",
            "/health": "Health check"
        }
    }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_sales(request: PredictionRequest):
    """
    Predict sales for a single store-date combination.
    
    Args:
        request: Prediction request with store ID, date, and features
    
    Returns:
        Prediction response with forecasted sales
    """
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert request to dataframe
        input_data = pd.DataFrame([{
            'Store': request.store_id,
            'Date': request.date,
            'DayOfWeek': request.day_of_week,
            'Open': request.open,
            'Promo': request.promo,
            'StateHoliday': request.state_holiday,
            'SchoolHoliday': request.school_holiday,
            'StoreType': request.store_type,
            'Assortment': request.assortment,
            'CompetitionDistance': request.competition_distance,
            'Promo2': request.promo2
        }])
        
        # If store is closed, return 0 sales immediately
        if request.open == 0:
            return PredictionResponse(
                store_id=request.store_id,
                date=request.date,
                predicted_sales=0.0,
                is_open=False,
                timestamp=datetime.now().isoformat()
            )
        
        # Make prediction
        prediction = model_pipeline.predict(input_data)[0]
        prediction = max(0, prediction)  # Ensure non-negative
        
        # Compute confidence interval (simple approach: ±15%)
        # In production, use quantile regression or prediction intervals
        ci_lower = prediction * 0.85
        ci_upper = prediction * 1.15
        
        logger.info(f"Prediction for Store {request.store_id} on {request.date}: {prediction:.2f}")
        
        return PredictionResponse(
            store_id=request.store_id,
            date=request.date,
            predicted_sales=round(prediction, 2),
            confidence_interval_lower=round(ci_lower, 2),
            confidence_interval_upper=round(ci_upper, 2),
            is_open=True,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch")
def predict_sales_batch(request: BatchPredictionRequest):
    """
    Batch prediction for multiple store-date combinations.
    
    Args:
        request: List of prediction requests
    
    Returns:
        List of prediction responses
    """
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        results = []
        for pred_request in request.predictions:
            result = predict_sales(pred_request)
            results.append(result)
        
        return {
            "predictions": results,
            "count": len(results),
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/stores/{store_id}/forecast")
def forecast_store_sales(store_id: int, days_ahead: int = 7):
    """
    Generate multi-day forecast for a single store.
    
    Args:
        store_id: Store ID
        days_ahead: Number of days to forecast (default: 7)
    
    Returns:
        Forecast for next N days
    """
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if days_ahead > 42:
        raise HTTPException(status_code=400, detail="Maximum forecast horizon is 42 days")
    
    # In production, fetch store metadata from database
    # For now, return placeholder
    return {
        "store_id": store_id,
        "forecast_horizon_days": days_ahead,
        "message": "Multi-day forecasting requires iterative prediction with lag features. Implement in production.",
        "recommendation": "Use /predict/batch endpoint with pre-computed dates"
    }


# Run with: uvicorn app:app --host 0.0.0.0 --port 8000 --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
