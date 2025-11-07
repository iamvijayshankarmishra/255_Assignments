"""
KDD Phase 2: Preprocessing
===========================
Data cleaning, scaling, and outlier handling for fraud detection.

Key Operations:
- StandardScaler for Time and Amount (V1-V28 already scaled by PCA)
- Outlier detection stratified by class (frauds often ARE outliers)
- PCA integrity verification
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


class FraudPreprocessor:
    """
    Preprocessing pipeline for fraud detection dataset.
    
    Handles:
    - Time/Amount scaling (PCA features already standardized)
    - Outlier detection (but don't remove frauds!)
    - Missing value check
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False
    
    def fit(self, df: pd.DataFrame) -> 'FraudPreprocessor':
        """
        Fit scaler on Time and Amount features.
        
        Args:
            df: Training data with Time, Amount columns
        
        Returns:
            self (fitted preprocessor)
        """
        # Only scale Time and Amount (V1-V28 already PCA-scaled)
        self.scaler.fit(df[['Time', 'Amount']])
        self.fitted = True
        
        print("✅ Scaler fitted on Time and Amount")
        print(f"   Time: μ={self.scaler.mean_[0]:.2f}, σ={np.sqrt(self.scaler.var_[0]):.2f}")
        print(f"   Amount: μ={self.scaler.mean_[1]:.2f}, σ={np.sqrt(self.scaler.var_[1]):.2f}")
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply scaling to Time and Amount.
        
        Args:
            df: Data to transform
        
        Returns:
            Transformed DataFrame (copy)
        """
        if not self.fitted:
            raise ValueError("Preprocessor not fitted. Call fit() first.")
        
        df_scaled = df.copy()
        df_scaled[['Time', 'Amount']] = self.scaler.transform(df[['Time', 'Amount']])
        
        return df_scaled
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            df: Training data
        
        Returns:
            Transformed DataFrame
        """
        self.fit(df)
        return self.transform(df)


def detect_outliers(df: pd.DataFrame, 
                   feature: str,
                   method: str = 'iqr',
                   threshold: float = 3.0) -> pd.Series:
    """
    Detect outliers using IQR or Z-score method.
    
    Args:
        df: Dataset
        feature: Column name to check
        method: 'iqr' or 'zscore'
        threshold: IQR multiplier (default 3.0) or Z-score threshold
    
    Returns:
        Boolean Series (True = outlier)
    """
    if method == 'iqr':
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        outliers = (df[feature] < lower_bound) | (df[feature] > upper_bound)
    
    elif method == 'zscore':
        z_scores = np.abs((df[feature] - df[feature].mean()) / df[feature].std())
        outliers = z_scores > threshold
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'iqr' or 'zscore'.")
    
    return outliers


def analyze_outliers_by_class(df: pd.DataFrame, 
                              feature: str,
                              method: str = 'iqr') -> Dict:
    """
    Analyze outliers separately for fraud and legitimate transactions.
    
    Critical insight: Frauds ARE often outliers, so we can't just remove them!
    
    Args:
        df: Dataset with Class column
        feature: Feature to analyze
        method: 'iqr' or 'zscore'
    
    Returns:
        Dict with outlier statistics by class
    """
    fraud = df[df['Class'] == 1]
    legit = df[df['Class'] == 0]
    
    fraud_outliers = detect_outliers(fraud, feature, method)
    legit_outliers = detect_outliers(legit, feature, method)
    
    results = {
        'feature': feature,
        'method': method,
        'fraud': {
            'total': len(fraud),
            'outliers': fraud_outliers.sum(),
            'outlier_rate': fraud_outliers.mean(),
        },
        'legitimate': {
            'total': len(legit),
            'outliers': legit_outliers.sum(),
            'outlier_rate': legit_outliers.mean(),
        }
    }
    
    print(f"\n📊 Outlier Analysis: {feature} ({method.upper()})")
    print(f"  Fraud: {fraud_outliers.sum():,}/{len(fraud):,} "
          f"({fraud_outliers.mean()*100:.1f}%) are outliers")
    print(f"  Legitimate: {legit_outliers.sum():,}/{len(legit):,} "
          f"({legit_outliers.mean()*100:.1f}%) are outliers")
    
    return results


def verify_pca_integrity(df: pd.DataFrame, 
                        tolerance: float = 1e-6) -> bool:
    """
    Verify PCA features (V1-V28) are properly standardized.
    
    PCA features should have:
    - Mean ≈ 0
    - Std ≈ 1 (or at least consistent scaling)
    
    Args:
        df: Dataset with V1-V28 columns
        tolerance: Acceptable deviation from 0 mean
    
    Returns:
        True if PCA integrity verified
    """
    pca_cols = [f'V{i}' for i in range(1, 29)]
    
    print("\n🔍 PCA Integrity Check:")
    
    issues = []
    for col in pca_cols:
        mean = df[col].mean()
        std = df[col].std()
        
        if abs(mean) > tolerance:
            issues.append(f"{col}: mean={mean:.6f} (expected ≈0)")
        
        # Check for zero variance (would break models)
        if std < tolerance:
            issues.append(f"{col}: std={std:.6f} (zero variance!)")
    
    if issues:
        print("  ⚠️ PCA integrity issues found:")
        for issue in issues:
            print(f"    - {issue}")
        return False
    else:
        print("  ✅ All PCA features have mean≈0 and std>0")
        return True


def plot_outliers(df: pd.DataFrame, 
                 features: List[str],
                 figsize: Tuple[int, int] = (15, 5)) -> None:
    """
    Visualize outliers in features, colored by class.
    
    Args:
        df: Dataset with Class column
        features: List of features to plot
        figsize: Figure size
    """
    n_features = len(features)
    fig, axes = plt.subplots(1, n_features, figsize=figsize)
    
    if n_features == 1:
        axes = [axes]
    
    for ax, feature in zip(axes, features):
        # Scatter plot with class coloring
        fraud = df[df['Class'] == 1]
        legit = df[df['Class'] == 0]
        
        ax.scatter(legit.index, legit[feature], 
                  alpha=0.3, s=1, c='#2ecc71', label='Legitimate')
        ax.scatter(fraud.index, fraud[feature], 
                  alpha=0.8, s=10, c='#e74c3c', label='Fraud')
        
        # IQR bounds
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 3 * IQR
        upper = Q3 + 3 * IQR
        
        ax.axhline(lower, color='orange', linestyle='--', linewidth=1, alpha=0.7)
        ax.axhline(upper, color='orange', linestyle='--', linewidth=1, alpha=0.7)
        
        ax.set_title(f'{feature}')
        ax.set_xlabel('Transaction Index')
        ax.set_ylabel('Value')
        ax.legend(loc='best', markerscale=3)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check for missing values (should be none for this dataset).
    
    Args:
        df: Dataset
    
    Returns:
        DataFrame with missing value counts
    """
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    
    result = pd.DataFrame({
        'Missing_Count': missing,
        'Missing_Percent': missing_pct
    })
    result = result[result['Missing_Count'] > 0].sort_values('Missing_Count', 
                                                              ascending=False)
    
    if len(result) == 0:
        print("✅ No missing values found")
    else:
        print(f"⚠️ {len(result)} features have missing values:")
        print(result)
    
    return result


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer time-based features from Time column.
    
    Assumes Time is in seconds from start of data collection.
    
    Args:
        df: Dataset with Time column (in seconds)
    
    Returns:
        DataFrame with added time features
    """
    df = df.copy()
    
    # Convert seconds to hours
    df['Time_Hours'] = df['Time'] / 3600
    
    # Hour of day (assuming Time=0 is midnight)
    df['Hour_of_Day'] = (df['Time'] // 3600) % 24
    
    # Day of week (assuming Time=0 is start of week)
    df['Day_of_Week'] = (df['Time'] // 86400) % 7
    
    print("✅ Created time features:")
    print("   - Time_Hours: continuous hours since start")
    print("   - Hour_of_Day: 0-23 (circadian patterns)")
    print("   - Day_of_Week: 0-6 (weekly patterns)")
    
    return df


def plot_temporal_patterns(df: pd.DataFrame, 
                          figsize: Tuple[int, int] = (15, 5)) -> None:
    """
    Visualize fraud patterns over time.
    
    Args:
        df: Dataset with Time and Class columns
        figsize: Figure size
    """
    df = create_time_features(df)
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # 1. Fraud rate over time (hours)
    df['Time_Bin'] = pd.cut(df['Time_Hours'], bins=20)
    fraud_rate_time = df.groupby('Time_Bin')['Class'].mean()
    
    axes[0].plot(range(len(fraud_rate_time)), fraud_rate_time.values, 
                marker='o', color='#e74c3c')
    axes[0].set_title('Fraud Rate Over Time')
    axes[0].set_xlabel('Time Period')
    axes[0].set_ylabel('Fraud Rate')
    axes[0].grid(True, alpha=0.3)
    
    # 2. Fraud rate by hour of day
    fraud_rate_hour = df.groupby('Hour_of_Day')['Class'].mean()
    
    axes[1].bar(fraud_rate_hour.index, fraud_rate_hour.values, color='#e74c3c')
    axes[1].set_title('Fraud Rate by Hour of Day')
    axes[1].set_xlabel('Hour (0-23)')
    axes[1].set_ylabel('Fraud Rate')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # 3. Fraud rate by day of week
    fraud_rate_day = df.groupby('Day_of_Week')['Class'].mean()
    
    axes[2].bar(fraud_rate_day.index, fraud_rate_day.values, color='#e74c3c')
    axes[2].set_title('Fraud Rate by Day of Week')
    axes[2].set_xlabel('Day (0=Mon, 6=Sun)')
    axes[2].set_ylabel('Fraud Rate')
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
