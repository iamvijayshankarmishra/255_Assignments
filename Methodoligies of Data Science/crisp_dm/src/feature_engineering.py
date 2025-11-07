"""
Feature Engineering Module for Rossmann Sales Forecasting
CRISP-DM: Data Preparation Phase
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Optional


class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract temporal features from date column."""
    
    def __init__(self, date_col: str = 'Date'):
        self.date_col = date_col
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X[self.date_col] = pd.to_datetime(X[self.date_col])
        
        # Basic temporal features
        X['Year'] = X[self.date_col].dt.year
        X['Month'] = X[self.date_col].dt.month
        X['Day'] = X[self.date_col].dt.day
        X['WeekOfYear'] = X[self.date_col].dt.isocalendar().week
        X['DayOfYear'] = X[self.date_col].dt.dayofyear
        X['Quarter'] = X[self.date_col].dt.quarter
        
        # Boolean temporal features
        X['IsWeekend'] = (X['DayOfWeek'] >= 6).astype(int)
        X['IsMonthStart'] = X[self.date_col].dt.is_month_start.astype(int)
        X['IsMonthEnd'] = X[self.date_col].dt.is_month_end.astype(int)
        X['IsQuarterStart'] = X[self.date_col].dt.is_quarter_start.astype(int)
        X['IsQuarterEnd'] = X[self.date_col].dt.is_quarter_end.astype(int)
        X['IsYearStart'] = X[self.date_col].dt.is_year_start.astype(int)
        X['IsYearEnd'] = X[self.date_col].dt.is_year_end.astype(int)
        
        return X


class LagFeatureCreator(BaseEstimator, TransformerMixin):
    """Create lag features per store (avoiding data leakage)."""
    
    def __init__(self, 
                 lags: List[int] = [1, 2, 7, 14, 28, 364],
                 group_col: str = 'Store',
                 target_col: str = 'Sales',
                 customers_col: str = 'Customers'):
        self.lags = lags
        self.group_col = group_col
        self.target_col = target_col
        self.customers_col = customers_col
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X = X.sort_values([self.group_col, 'Date'])
        
        for lag in self.lags:
            # Sales lags
            if self.target_col in X.columns:
                X[f'Sales_Lag{lag}'] = X.groupby(self.group_col)[self.target_col].shift(lag)
            
            # Customer lags (only for shorter lags)
            if self.customers_col in X.columns and lag in [1, 7, 28]:
                X[f'Customers_Lag{lag}'] = X.groupby(self.group_col)[self.customers_col].shift(lag)
        
        return X


class RollingFeatureCreator(BaseEstimator, TransformerMixin):
    """Create rolling window statistics per store."""
    
    def __init__(self,
                 windows: List[int] = [7, 14, 28],
                 group_col: str = 'Store',
                 target_col: str = 'Sales',
                 customers_col: str = 'Customers'):
        self.windows = windows
        self.group_col = group_col
        self.target_col = target_col
        self.customers_col = customers_col
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X = X.sort_values([self.group_col, 'Date'])
        
        for window in self.windows:
            # Sales rolling statistics
            if self.target_col in X.columns:
                grouped = X.groupby(self.group_col)[self.target_col]
                X[f'Sales_RollingMean{window}'] = grouped.shift(1).rolling(window, min_periods=1).mean()
                X[f'Sales_RollingStd{window}'] = grouped.shift(1).rolling(window, min_periods=1).std()
                X[f'Sales_RollingMin{window}'] = grouped.shift(1).rolling(window, min_periods=1).min()
                X[f'Sales_RollingMax{window}'] = grouped.shift(1).rolling(window, min_periods=1).max()
                X[f'Sales_RollingMedian{window}'] = grouped.shift(1).rolling(window, min_periods=1).median()
            
            # Customer rolling statistics (7-day window only)
            if self.customers_col in X.columns and window == 7:
                grouped_cust = X.groupby(self.group_col)[self.customers_col]
                X[f'Customers_RollingMean{window}'] = grouped_cust.shift(1).rolling(window, min_periods=1).mean()
        
        return X


class PromoFeatureEngineer(BaseEstimator, TransformerMixin):
    """Engineer promotion-related features."""
    
    def __init__(self, 
                 promo_col: str = 'Promo',
                 group_col: str = 'Store',
                 date_col: str = 'Date'):
        self.promo_col = promo_col
        self.group_col = group_col
        self.date_col = date_col
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X = X.sort_values([self.group_col, self.date_col])
        
        # Days since last promo
        X['PromoChange'] = X.groupby(self.group_col)[self.promo_col].diff()
        X['PromoStart'] = (X['PromoChange'] == 1).astype(int)
        X['PromoEnd'] = (X['PromoChange'] == -1).astype(int)
        
        # Promo streak length
        X['PromoStreak'] = (X.groupby([self.group_col, 
                                        (X[self.promo_col] != X.groupby(self.group_col)[self.promo_col].shift()).cumsum()])
                            .cumcount() + 1)
        X.loc[X[self.promo_col] == 0, 'PromoStreak'] = 0
        
        # Drop temporary columns
        X = X.drop(['PromoChange'], axis=1, errors='ignore')
        
        return X


class CompetitionFeatureEngineer(BaseEstimator, TransformerMixin):
    """Engineer competition-related features."""
    
    def __init__(self, date_col: str = 'Date'):
        self.date_col = date_col
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Competition open months
        if 'CompetitionOpenSinceYear' in X.columns and 'CompetitionOpenSinceMonth' in X.columns:
            X[self.date_col] = pd.to_datetime(X[self.date_col])
            
            # Create competition open date
            X['CompetitionOpenDate'] = pd.to_datetime(
                X['CompetitionOpenSinceYear'].astype(str) + '-' + 
                X['CompetitionOpenSinceMonth'].astype(str) + '-01',
                errors='coerce'
            )
            
            # Months since competition opened
            X['CompetitionOpenMonths'] = (
                (X[self.date_col].dt.year - X['CompetitionOpenDate'].dt.year) * 12 +
                (X[self.date_col].dt.month - X['CompetitionOpenDate'].dt.month)
            )
            X['CompetitionOpenMonths'] = X['CompetitionOpenMonths'].fillna(0).clip(lower=0)
            
            # Has competition
            X['HasCompetition'] = (X['CompetitionDistance'].notna()).astype(int)
            
            # Drop temporary column
            X = X.drop(['CompetitionOpenDate'], axis=1, errors='ignore')
        
        # Fill CompetitionDistance NaN with large value (no competition)
        if 'CompetitionDistance' in X.columns:
            X['CompetitionDistance'] = X['CompetitionDistance'].fillna(999999)
        
        return X


def create_baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create simple baseline prediction features."""
    df = df.copy()
    df = df.sort_values(['Store', 'Date'])
    
    # Naive last week
    df['Baseline_LastWeek'] = df.groupby('Store')['Sales'].shift(7)
    
    # Naive last year same week
    df['Baseline_LastYearSameWeek'] = df.groupby('Store')['Sales'].shift(364)
    
    # 7-day moving average
    df['Baseline_MA7'] = df.groupby('Store')['Sales'].shift(1).rolling(7, min_periods=1).mean()
    
    # 28-day moving average
    df['Baseline_MA28'] = df.groupby('Store')['Sales'].shift(1).rolling(28, min_periods=1).mean()
    
    return df


def prepare_data(train_df: pd.DataFrame, 
                 store_df: pd.DataFrame,
                 is_train: bool = True) -> pd.DataFrame:
    """
    Complete data preparation pipeline.
    
    Args:
        train_df: Raw training/test data
        store_df: Store metadata
        is_train: Whether this is training data (has Sales/Customers)
    
    Returns:
        Processed dataframe with all engineered features
    """
    # Merge store metadata
    df = train_df.merge(store_df, on='Store', how='left')
    
    # Handle store closures
    df['Open'] = df['Open'].fillna(0)
    
    # Apply transformers
    temporal = TemporalFeatureExtractor()
    df = temporal.transform(df)
    
    if is_train:
        # Only create lag/rolling features for training data
        lag_creator = LagFeatureCreator()
        df = lag_creator.transform(df)
        
        rolling_creator = RollingFeatureCreator()
        df = rolling_creator.transform(df)
        
        df = create_baseline_features(df)
    
    promo_engineer = PromoFeatureEngineer()
    df = promo_engineer.transform(df)
    
    comp_engineer = CompetitionFeatureEngineer()
    df = comp_engineer.transform(df)
    
    # Encode categoricals
    df = pd.get_dummies(df, columns=['StateHoliday', 'StoreType', 'Assortment'], 
                        drop_first=True, dtype=int)
    
    return df
