"""
SEMMA Modification Module
==========================
Feature transformations, encoding, and engineering for Bank Marketing data.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from typing import List


class BankFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom feature engineering for Bank Marketing dataset.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.ohe = None
        self.ordinal_encoder = None
        
    def fit(self, X, y=None):
        # Fit scaler on continuous features
        continuous_cols = ['age', 'duration', 'campaign', 'pdays', 
                          'emp.var.rate', 'cons.price.idx', 'cons.conf.idx',
                          'euribor3m', 'nr.employed']
        
        if all(col in X.columns for col in continuous_cols):
            self.scaler.fit(X[continuous_cols])
        
        # Fit ordinal encoder for education
        if 'education' in X.columns:
            education_order = [
                'illiterate', 'basic.4y', 'basic.6y', 'basic.9y',
                'high.school', 'professional.course', 'university.degree'
            ]
            self.ordinal_encoder = OrdinalEncoder(
                categories=[education_order],
                handle_unknown='use_encoded_value',
                unknown_value=-1
            )
            self.ordinal_encoder.fit(X[['education']])
        
        # Fit one-hot encoder for nominal categoricals
        nominal_cols = ['job', 'marital', 'contact', 'month', 'day_of_week', 'poutcome']
        existing_nominal = [col for col in nominal_cols if col in X.columns]
        
        if existing_nominal:
            self.ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
            self.ohe.fit(X[existing_nominal])
        
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # 1. Engineer new features
        X['contact_frequency'] = X['campaign'] + X['previous']
        
        # Recency score (inverse of days since last contact)
        X['recency_score'] = np.where(
            X['pdays'] == 999,
            0,
            1 / (X['pdays'] + 1)
        )
        
        # Economic confidence composite
        X['economic_confidence'] = X['emp.var.rate'] + X['cons.conf.idx']
        
        # Age groups
        X['age_group'] = pd.cut(
            X['age'],
            bins=[0, 30, 40, 50, 100],
            labels=['<30', '30-40', '40-50', '50+']
        ).astype(str)
        
        # Duration bins (in seconds)
        X['duration_bin'] = pd.cut(
            X['duration'],
            bins=[0, 100, 300, float('inf')],
            labels=['<100s', '100-300s', '300+s']
        ).astype(str)
        
        # 2. Scale continuous features
        continuous_cols = ['age', 'duration', 'campaign', 'pdays',
                          'emp.var.rate', 'cons.price.idx', 'cons.conf.idx',
                          'euribor3m', 'nr.employed',
                          'contact_frequency', 'recency_score', 'economic_confidence']
        existing_continuous = [col for col in continuous_cols if col in X.columns]
        
        X[existing_continuous] = self.scaler.transform(X[existing_continuous])
        
        # 3. Encode education (ordinal)
        if 'education' in X.columns and self.ordinal_encoder is not None:
            X['education_ord'] = self.ordinal_encoder.transform(X[['education']])
            X.drop('education', axis=1, inplace=True)
        
        # 4. Binary encode
        binary_cols = ['default', 'housing', 'loan']
        for col in binary_cols:
            if col in X.columns:
                X[col] = X[col].map({'yes': 1, 'no': 0, 'unknown': -1})
        
        # 5. One-hot encode nominal
        nominal_cols = ['job', 'marital', 'contact', 'month', 'day_of_week', 'poutcome']
        existing_nominal = [col for col in nominal_cols if col in X.columns]
        
        if existing_nominal and self.ohe is not None:
            ohe_features = self.ohe.transform(X[existing_nominal])
            ohe_col_names = self.ohe.get_feature_names_out(existing_nominal)
            ohe_df = pd.DataFrame(ohe_features, columns=ohe_col_names, index=X.index)
            
            X = X.drop(existing_nominal, axis=1)
            X = pd.concat([X, ohe_df], axis=1)
        
        # Also encode age_group and duration_bin
        if 'age_group' in X.columns:
            age_dummies = pd.get_dummies(X['age_group'], prefix='age_group', drop_first=True)
            X = pd.concat([X.drop('age_group', axis=1), age_dummies], axis=1)
        
        if 'duration_bin' in X.columns:
            dur_dummies = pd.get_dummies(X['duration_bin'], prefix='duration_bin', drop_first=True)
            X = pd.concat([X.drop('duration_bin', axis=1), dur_dummies], axis=1)
        
        return X


def remove_high_correlation(df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    """
    Remove features with correlation > threshold.
    
    Args:
        df: Feature dataframe
        threshold: Correlation threshold (default 0.9)
        
    Returns:
        DataFrame with high-correlation features removed
    """
    corr_matrix = df.corr().abs()
    
    # Upper triangle
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find columns with correlation > threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    print(f"Dropping {len(to_drop)} high-correlation features: {to_drop}")
    
    return df.drop(to_drop, axis=1)


def calculate_vif(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Variance Inflation Factor for multicollinearity detection.
    
    Args:
        df: Feature dataframe (numeric only)
        
    Returns:
        DataFrame with VIF scores
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    vif_data = pd.DataFrame()
    vif_data["Feature"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(len(df.columns))]
    
    return vif_data.sort_values('VIF', ascending=False)
