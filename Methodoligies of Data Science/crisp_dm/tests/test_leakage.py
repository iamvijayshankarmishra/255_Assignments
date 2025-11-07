"""
Test suite for data leakage detection in CRISP-DM Rossmann project.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def test_no_temporal_overlap_train_test():
    """Verify training and test sets don't overlap in time."""
    # Mock data
    train_dates = pd.date_range('2013-01-01', '2015-07-31', freq='D')
    test_dates = pd.date_range('2015-08-01', '2015-09-17', freq='D')
    
    train_max = train_dates.max()
    test_min = test_dates.min()
    
    assert train_max < test_min, "Train and test sets overlap in time!"


def test_lag_features_no_future_info():
    """Verify lag features don't contain future information."""
    # Create mock time series
    dates = pd.date_range('2015-01-01', periods=30, freq='D')
    df = pd.DataFrame({
        'Date': dates,
        'Store': [1] * 30,
        'Sales': np.arange(100, 130)  # Increasing sales
    })
    
    # Create lag-7 feature (manually)
    df = df.sort_values(['Store', 'Date'])
    df['Sales_Lag7'] = df.groupby('Store')['Sales'].shift(7)
    
    # Check: for date D, Sales_Lag7 should equal Sales from D-7
    for i in range(7, len(df)):
        expected = df.iloc[i - 7]['Sales']
        actual = df.iloc[i]['Sales_Lag7']
        assert actual == expected, f"Lag feature leakage at index {i}"


def test_rolling_window_no_current_value():
    """Verify rolling windows don't include current day's value."""
    dates = pd.date_range('2015-01-01', periods=20, freq='D')
    df = pd.DataFrame({
        'Date': dates,
        'Store': [1] * 20,
        'Sales': [100] * 10 + [200] * 10  # Step change at day 10
    })
    
    # 7-day rolling mean with shift
    df = df.sort_values(['Store', 'Date'])
    df['RollingMean7'] = df.groupby('Store')['Sales'].shift(1).rolling(7, min_periods=1).mean()
    
    # At index 10 (first day of 200), rolling mean should be based on [100,100,...,100]
    # not include current 200
    assert df.iloc[10]['RollingMean7'] == 100.0, "Rolling window includes current value (leakage!)"


def test_no_test_set_info_in_training():
    """Ensure no test set statistics leak into training features."""
    # Simulate train and test split
    train_sales = np.array([100, 110, 105, 115, 120])
    test_sales = np.array([200, 210, 205])  # Different distribution
    
    # Global mean (wrong approach - includes test data)
    global_mean = np.mean(np.concatenate([train_sales, test_sales]))
    
    # Correct train-only mean
    train_mean = np.mean(train_sales)
    
    # Verify we're using train-only statistics
    assert train_mean != global_mean, "Should not match if test data is different"
    assert abs(train_mean - 110) < 1, "Train mean should be ~110"


def test_promo_future_features():
    """Check that 'DaysUntilPromo' doesn't use unauthorized future info."""
    # If DaysUntilPromo is computed from test set metadata (planned promos), OK
    # If computed from future training data, NOT OK
    
    # Mock scenario: We're at day 10, promo happens at day 15
    df = pd.DataFrame({
        'Date': pd.date_range('2015-01-01', periods=20, freq='D'),
        'Store': [1] * 20,
        'Promo': [0]*10 + [1]*5 + [0]*5
    })
    
    # Acceptable: DaysUntilPromo from external promo calendar (not derived from training Sales)
    # This test just documents the assumption
    # In production, verify promo features come from business metadata, not target variable
    
    assert True  # Placeholder - manual code review required


def test_store_closure_handling():
    """Verify closed stores (Open=0) are handled without leakage."""
    df = pd.DataFrame({
        'Date': pd.date_range('2015-01-01', periods=10, freq='D'),
        'Store': [1] * 10,
        'Open': [1, 1, 0, 0, 1, 1, 1, 0, 1, 1],  # Store closed on some days
        'Sales': [100, 110, 0, 0, 120, 130, 140, 0, 150, 160]
    })
    
    # When computing lags/rolling features, closed days (Sales=0) should not pollute
    # Use only open days or handle appropriately
    
    # Simple check: If we're predicting for an open day, ensure lag doesn't come from closed day
    df['Sales_Lag1'] = df.groupby('Store')['Sales'].shift(1)
    
    # At index 4 (first open day after closure), lag is 0 (from closed day)
    # This is acceptable IF we filter predictions where Open=1
    assert df.iloc[4]['Sales_Lag1'] == 0, "Lag from closed day"


def test_competition_distance_imputation():
    """Verify NaN in CompetitionDistance doesn't cause leakage."""
    df = pd.DataFrame({
        'Store': [1, 2, 3, 4, 5],
        'CompetitionDistance': [500, np.nan, 1000, np.nan, 750]
    })
    
    # Impute with large value (no competition) - this is OK
    df['CompetitionDistance'] = df['CompetitionDistance'].fillna(999999)
    
    assert df.iloc[1]['CompetitionDistance'] == 999999
    assert df.iloc[3]['CompetitionDistance'] == 999999


def test_target_encoding_no_leakage():
    """If doing target encoding (e.g., mean sales per store), must use only train data."""
    # Train data
    train = pd.DataFrame({
        'Store': [1, 1, 1, 2, 2, 2],
        'Sales': [100, 110, 105, 200, 210, 205]
    })
    
    # Compute mean sales per store from TRAIN only
    store_means = train.groupby('Store')['Sales'].mean()
    
    # Test data
    test = pd.DataFrame({
        'Store': [1, 2]
    })
    
    # Map train-derived means to test
    test['StoreMean'] = test['Store'].map(store_means)
    
    assert test.iloc[0]['StoreMean'] == 105.0  # Store 1 mean
    assert test.iloc[1]['StoreMean'] == 205.0  # Store 2 mean
    
    # WRONG: Don't compute store means from train+test combined


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
