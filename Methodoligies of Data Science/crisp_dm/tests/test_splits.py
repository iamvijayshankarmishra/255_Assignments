"""
Test suite for time series splitting in CRISP-DM Rossmann project.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit


def test_time_series_split_order():
    """Verify TimeSeriesSplit respects temporal order."""
    dates = pd.date_range('2015-01-01', periods=100, freq='D')
    X = pd.DataFrame({
        'Date': dates,
        'Feature': np.random.randn(100)
    })
    y = np.random.randn(100)
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        # Validation dates must be after all training dates
        train_dates = X.iloc[train_idx]['Date']
        val_dates = X.iloc[val_idx]['Date']
        
        assert train_dates.max() < val_dates.min(), \
            f"Fold {fold}: Validation dates overlap with training"


def test_no_shuffle_in_time_series():
    """Ensure time series data is never shuffled."""
    dates = pd.date_range('2015-01-01', periods=50, freq='D')
    df = pd.DataFrame({
        'Date': dates,
        'Sales': np.arange(50)  # Sequential
    })
    
    # Sort by date
    df_sorted = df.sort_values('Date')
    
    # Verify order maintained
    assert (df_sorted['Sales'].values == np.arange(50)).all(), \
        "Data was shuffled - temporal order broken!"


def test_per_store_splits():
    """Verify splits work correctly for multiple stores."""
    dates = pd.date_range('2015-01-01', periods=30, freq='D')
    stores = [1, 2, 3]
    
    # Create data for 3 stores
    df_list = []
    for store in stores:
        df_list.append(pd.DataFrame({
            'Store': store,
            'Date': dates,
            'Sales': np.random.randint(100, 200, 30)
        }))
    df = pd.concat(df_list, ignore_index=True)
    
    # Split: last 7 days for test
    split_date = dates.max() - pd.Timedelta(days=7)
    train = df[df['Date'] <= split_date]
    test = df[df['Date'] > split_date]
    
    # Each store should have data in both train and test
    assert set(train['Store'].unique()) == set(stores)
    assert set(test['Store'].unique()) == set(stores)
    
    # Test set should have exactly 7 days * 3 stores = 21 rows
    assert len(test) == 7 * 3


def test_gap_between_train_val():
    """Test handling of gap between train and validation (optional but good practice)."""
    dates = pd.date_range('2015-01-01', periods=100, freq='D')
    
    # Split with 7-day gap
    train_end = dates[70]
    val_start = dates[78]  # 7-day gap
    
    train_dates = dates[:71]
    val_dates = dates[78:]
    
    gap = (val_dates.min() - train_dates.max()).days
    assert gap == 7, f"Expected 7-day gap, got {gap}"


def test_minimum_train_size():
    """Ensure training set has enough data for lag features."""
    # If using lag-364 (last year), need at least 365+ days of training
    dates = pd.date_range('2013-01-01', periods=365, freq='D')
    
    # Need at least 364 days to compute lag-364 for day 365
    assert len(dates) >= 364, "Insufficient data for lag-364 features"


def test_hold_out_set_separate():
    """Verify final holdout set is untouched during CV."""
    # Full data: 2013-01-01 to 2015-09-17
    all_dates = pd.date_range('2013-01-01', '2015-09-17', freq='D')
    
    # Hold out last 48 days as final test
    split_date = all_dates.max() - pd.Timedelta(days=48)
    
    train_val_dates = all_dates[all_dates <= split_date]
    holdout_dates = all_dates[all_dates > split_date]
    
    # No overlap
    assert train_val_dates.max() < holdout_dates.min()
    assert len(holdout_dates) == 48


def test_store_level_validation():
    """Test per-store performance validation."""
    # Some stores may have unique patterns - need per-store metrics
    df = pd.DataFrame({
        'Store': [1]*20 + [2]*20 + [3]*20,
        'Sales': ([100]*20 +  # Store 1: stable
                  list(range(100, 120)) +  # Store 2: trending
                  [50]*10 + [150]*10)  # Store 3: step change
    })
    
    # Compute per-store statistics
    store_stats = df.groupby('Store')['Sales'].agg(['mean', 'std'])
    
    # Verify heterogeneity
    assert store_stats.loc[1, 'std'] < 1  # Store 1 has low variance
    assert store_stats.loc[2, 'std'] > 5  # Store 2 has trend variance
    assert store_stats.loc[3, 'std'] > 40  # Store 3 has step change variance


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
