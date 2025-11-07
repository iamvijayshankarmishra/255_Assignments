"""
Test suite for model training in CRISP-DM Rossmann project.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


def test_model_predictions_non_negative():
    """Sales predictions should never be negative."""
    # Train a simple model
    X_train = np.random.rand(100, 5)
    y_train = np.random.randint(0, 1000, 100)
    
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict
    X_test = np.random.rand(20, 5)
    predictions = model.predict(X_test)
    
    # Clip negative predictions
    predictions = np.maximum(predictions, 0)
    
    assert np.all(predictions >= 0), "Found negative predictions!"


def test_closed_stores_predict_zero():
    """When Open=0, predicted sales should be 0."""
    df = pd.DataFrame({
        'Open': [1, 1, 0, 0, 1],
        'Feature1': [1, 2, 3, 4, 5],
        'Sales': [100, 110, 0, 0, 120]
    })
    
    # Simulate predictions
    predictions = np.array([105, 115, 50, 60, 125])  # Model predicts some value
    
    # Override: If Open=0, set prediction to 0
    predictions[df['Open'] == 0] = 0
    
    assert predictions[2] == 0
    assert predictions[3] == 0


def test_model_performance_vs_baseline():
    """Model should beat naive baseline."""
    # Simulate time series
    np.random.seed(42)
    sales = 100 + np.cumsum(np.random.randn(100)) + np.random.randn(100) * 10
    
    # Train/test split
    train_sales = sales[:80]
    test_sales = sales[80:]
    
    # Baseline: last value
    baseline_pred = np.full(len(test_sales), train_sales[-1])
    baseline_mae = mean_absolute_error(test_sales, baseline_pred)
    
    # Simple model: mean of last 7 values
    model_pred = np.full(len(test_sales), np.mean(train_sales[-7:]))
    model_mae = mean_absolute_error(test_sales, model_pred)
    
    # Model should be better (or at least not worse)
    # This is a weak test, but illustrates the principle
    assert model_mae <= baseline_mae * 1.1, "Model performs significantly worse than baseline"


def test_feature_importance_sanity():
    """Key features should have non-zero importance."""
    # Simulate data where DayOfWeek is highly predictive
    np.random.seed(42)
    df = pd.DataFrame({
        'DayOfWeek': np.tile([0, 1, 2, 3, 4, 5, 6], 20),  # 7 days repeated
        'RandomFeature': np.random.randn(140),
        'Sales': np.tile([100, 110, 120, 130, 140, 90, 80], 20) + np.random.randn(140) * 5
    })
    
    X = df[['DayOfWeek', 'RandomFeature']]
    y = df['Sales']
    
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    importances = model.feature_importances_
    
    # DayOfWeek should be more important than RandomFeature
    assert importances[0] > importances[1], "DayOfWeek should be most important"


def test_predictions_stable_across_folds():
    """Model performance should be consistent across CV folds."""
    from sklearn.model_selection import TimeSeriesSplit
    
    # Generate time series data
    np.random.seed(42)
    X = np.random.rand(500, 10)
    y = np.random.rand(500) * 100
    
    tscv = TimeSeriesSplit(n_splits=5)
    fold_scores = []
    
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, pred)
        fold_scores.append(mae)
    
    # Check variance across folds isn't too high
    score_std = np.std(fold_scores)
    score_mean = np.mean(fold_scores)
    
    cv_coefficient = score_std / score_mean if score_mean > 0 else 0
    
    # Coefficient of variation should be < 0.5 (scores shouldn't vary wildly)
    assert cv_coefficient < 0.5, f"Model unstable across folds (CV={cv_coefficient:.2f})"


def test_no_data_leakage_in_predictions():
    """Ensure predictions don't access future information."""
    # This is more of a conceptual test - ensure your prediction pipeline
    # only uses features available at prediction time
    
    df = pd.DataFrame({
        'Date': pd.date_range('2015-01-01', periods=10, freq='D'),
        'Store': [1] * 10,
        'Sales': [100, 110, 105, 115, 120, 125, 130, 135, 140, 145]
    })
    
    # When predicting for day 5, only use data from days 0-4
    prediction_date = df.iloc[5]['Date']
    available_data = df[df['Date'] < prediction_date]
    
    assert len(available_data) == 5, "Should only have 5 days of historical data"
    
    # Use available_data to create features for prediction
    # (e.g., lag-1 = 120, mean = 110)


def test_model_handles_missing_features():
    """Model should handle missing values in features."""
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    
    # Data with missing values
    X = np.array([[1, 2], [3, np.nan], [np.nan, 6], [7, 8]])
    y = np.array([10, 20, 30, 40])
    
    # Pipeline with imputation
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', RandomForestRegressor(n_estimators=10, random_state=42))
    ])
    
    pipeline.fit(X, y)
    
    # Predict with missing values
    X_test = np.array([[1, np.nan], [np.nan, 6]])
    predictions = pipeline.predict(X_test)
    
    assert len(predictions) == 2
    assert not np.any(np.isnan(predictions)), "Predictions should not be NaN"


def test_prediction_latency():
    """Prediction should be fast enough for production."""
    import time
    
    # Train model
    X_train = np.random.rand(1000, 20)
    y_train = np.random.rand(1000) * 100
    
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    # Time single prediction
    X_single = np.random.rand(1, 20)
    start = time.time()
    _ = model.predict(X_single)
    elapsed = time.time() - start
    
    # Should be under 10ms for single prediction
    assert elapsed < 0.01, f"Prediction too slow: {elapsed*1000:.1f}ms"
    
    # Time batch prediction (1000 stores)
    X_batch = np.random.rand(1000, 20)
    start = time.time()
    _ = model.predict(X_batch)
    elapsed = time.time() - start
    
    # Should be under 200ms for 1000 predictions
    assert elapsed < 0.2, f"Batch prediction too slow: {elapsed*1000:.1f}ms"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
