"""
KDD Phase 1: Selection
======================
Functions for data loading, feature profiling, and initial understanding
of the Credit Card Fraud Detection dataset.

Key Challenges:
- PCA anonymization (V1-V28 features are uninterpretable)
- Extreme class imbalance (0.172% fraud rate)
- Temporal data (Time feature in seconds)
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


def download_fraud_data(url: str = None, local_path: str = None) -> pd.DataFrame:
    """
    Download or load Credit Card Fraud Detection dataset.
    
    Dataset: 284,807 transactions, 492 frauds (0.172%)
    Features: Time, V1-V28 (PCA), Amount, Class
    
    Args:
        url: URL to download from (if None, assumes local file)
        local_path: Path to local CSV file
    
    Returns:
        DataFrame with all features
    """
    if local_path:
        df = pd.read_csv(local_path)
    elif url:
        df = pd.read_csv(url)
    else:
        # Default: try to load from Kaggle API or local
        try:
            import kaggle
            kaggle.api.dataset_download_files(
                'mlg-ulb/creditcardfraud',
                path='./data',
                unzip=True
            )
            df = pd.read_csv('./data/creditcard.csv')
        except:
            raise FileNotFoundError(
                "Please provide local_path or url, or configure Kaggle API"
            )
    
    print(f"✅ Loaded {len(df):,} transactions")
    print(f"   Frauds: {df['Class'].sum():,} ({df['Class'].mean()*100:.3f}%)")
    
    return df


def profile_features(df: pd.DataFrame) -> Dict:
    """
    Generate comprehensive feature profile for fraud dataset.
    
    Analyzes:
    - PCA features (V1-V28): distribution, outliers
    - Time feature: temporal patterns
    - Amount feature: transaction values
    - Class distribution
    
    Args:
        df: Full dataset with Time, V1-V28, Amount, Class
    
    Returns:
        Dict with statistical profiles
    """
    profile = {
        'shape': df.shape,
        'missing': df.isnull().sum().to_dict(),
        'dtypes': df.dtypes.to_dict(),
        'class_distribution': df['Class'].value_counts().to_dict(),
        'fraud_rate': df['Class'].mean(),
    }
    
    # Time feature analysis
    profile['time_stats'] = {
        'min': df['Time'].min(),
        'max': df['Time'].max(),
        'duration_hours': (df['Time'].max() - df['Time'].min()) / 3600,
        'mean': df['Time'].mean(),
        'std': df['Time'].std(),
    }
    
    # Amount feature analysis
    profile['amount_stats'] = {
        'min': df['Amount'].min(),
        'max': df['Amount'].max(),
        'mean': df['Amount'].mean(),
        'median': df['Amount'].median(),
        'std': df['Amount'].std(),
        'fraud_mean': df[df['Class'] == 1]['Amount'].mean(),
        'legit_mean': df[df['Class'] == 0]['Amount'].mean(),
    }
    
    # PCA features analysis (V1-V28)
    pca_cols = [f'V{i}' for i in range(1, 29)]
    profile['pca_stats'] = {}
    for col in pca_cols:
        profile['pca_stats'][col] = {
            'mean': df[col].mean(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'fraud_mean': df[df['Class'] == 1][col].mean(),
            'legit_mean': df[df['Class'] == 0][col].mean(),
        }
    
    return profile


def temporal_split(df: pd.DataFrame, 
                   train_size: float = 0.6,
                   val_size: float = 0.2,
                   test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Temporal train/val/test split (NO shuffling for fraud detection).
    
    Critical: Temporal ordering prevents data leakage and simulates
    real-world deployment where we predict future frauds.
    
    Args:
        df: Dataset sorted by Time
        train_size: Proportion for training (default 0.6)
        val_size: Proportion for validation (default 0.2)
        test_size: Proportion for test (default 0.2)
    
    Returns:
        train_df, val_df, test_df (temporally ordered)
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, \
        "Split sizes must sum to 1.0"
    
    # Sort by time (should already be sorted, but ensure it)
    df = df.sort_values('Time').reset_index(drop=True)
    
    n = len(df)
    train_end = int(n * train_size)
    val_end = int(n * (train_size + val_size))
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    print("\n📊 Temporal Split:")
    print(f"  Train: {len(train_df):,} ({train_df['Class'].sum():,} frauds, "
          f"{train_df['Class'].mean()*100:.3f}%)")
    print(f"  Val:   {len(val_df):,} ({val_df['Class'].sum():,} frauds, "
          f"{val_df['Class'].mean()*100:.3f}%)")
    print(f"  Test:  {len(test_df):,} ({len(test_df) - test_df['Class'].sum():,} legit, "
          f"{test_df['Class'].sum():,} frauds, {test_df['Class'].mean()*100:.3f}%)")
    
    # Check temporal order
    assert train_df['Time'].max() <= val_df['Time'].min(), \
        "Train period overlaps with validation period!"
    assert val_df['Time'].max() <= test_df['Time'].min(), \
        "Validation period overlaps with test period!"
    
    print(f"  ✅ Temporal ordering verified (no data leakage)")
    
    return train_df, val_df, test_df


def plot_class_distribution(df: pd.DataFrame, 
                            splits: Dict[str, pd.DataFrame] = None,
                            figsize: Tuple[int, int] = (12, 4)) -> None:
    """
    Visualize extreme class imbalance.
    
    Args:
        df: Full dataset
        splits: Dict with 'train', 'val', 'test' DataFrames (optional)
        figsize: Figure size
    """
    if splits is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        
        counts = df['Class'].value_counts()
        ax.bar(['Legitimate', 'Fraud'], counts.values, color=['#2ecc71', '#e74c3c'])
        ax.set_ylabel('Count')
        ax.set_title(f'Class Distribution (Fraud Rate: {df["Class"].mean()*100:.3f}%)')
        
        for i, v in enumerate(counts.values):
            ax.text(i, v + 1000, f'{v:,}', ha='center', fontweight='bold')
    else:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        for ax, (name, data) in zip(axes, splits.items()):
            counts = data['Class'].value_counts()
            ax.bar(['Legitimate', 'Fraud'], counts.values, 
                   color=['#2ecc71', '#e74c3c'])
            ax.set_ylabel('Count')
            ax.set_title(f'{name.capitalize()} (Fraud: {data["Class"].mean()*100:.3f}%)')
            
            for i, v in enumerate(counts.values):
                ax.text(i, v + 500, f'{v:,}', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.show()


def plot_feature_comparison(df: pd.DataFrame, 
                            features: list,
                            figsize: Tuple[int, int] = (15, 8)) -> None:
    """
    Compare feature distributions between fraud and legitimate transactions.
    
    Args:
        df: Dataset with Class column
        features: List of feature names to compare (e.g., ['V1', 'V2', 'Amount'])
        figsize: Figure size
    """
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for idx, feature in enumerate(features):
        ax = axes[idx]
        
        # Box plot
        fraud = df[df['Class'] == 1][feature]
        legit = df[df['Class'] == 0][feature]
        
        ax.boxplot([legit, fraud], 
                   labels=['Legitimate', 'Fraud'],
                   patch_artist=True,
                   boxprops=dict(facecolor='#3498db', alpha=0.5))
        ax.set_title(f'{feature}')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
    
    # Hide extra subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()


def calculate_fraud_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate statistical differences between fraud and legitimate transactions.
    
    Uses Mann-Whitney U test (non-parametric) to test if distributions differ.
    
    Args:
        df: Dataset with Class column
    
    Returns:
        DataFrame with statistics for each feature
    """
    from scipy.stats import mannwhitneyu
    
    results = []
    
    # All features except Class
    features = [col for col in df.columns if col != 'Class']
    
    for feature in features:
        fraud = df[df['Class'] == 1][feature]
        legit = df[df['Class'] == 0][feature]
        
        # Mann-Whitney U test
        stat, p_value = mannwhitneyu(fraud, legit, alternative='two-sided')
        
        results.append({
            'Feature': feature,
            'Fraud_Mean': fraud.mean(),
            'Legit_Mean': legit.mean(),
            'Fraud_Std': fraud.std(),
            'Legit_Std': legit.std(),
            'Mean_Diff': fraud.mean() - legit.mean(),
            'Mann_Whitney_U': stat,
            'P_Value': p_value,
            'Significant': 'Yes' if p_value < 0.05 else 'No'
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('P_Value')
    
    return results_df
