"""
SEMMA Sampling Module
=====================
Functions for creating stratified samples and train/val/test splits.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple


def stratified_split(
    df: pd.DataFrame,
    target_col: str,
    train_size: float = 0.6,
    val_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create stratified train/val/test splits.
    
    Args:
        df: Input dataframe
        target_col: Name of target column
        train_size: Proportion for training (default 0.6)
        val_size: Proportion for validation (default 0.2)
        test_size: Proportion for test (default 0.2)
        random_state: Random seed for reproducibility
        
    Returns:
        train_df, val_df, test_df
    """
    assert train_size + val_size + test_size == 1.0, "Sizes must sum to 1.0"
    
    # First split: separate test set
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df[target_col],
        random_state=random_state
    )
    
    # Second split: separate train and validation
    val_proportion = val_size / (train_size + val_size)
    train, val = train_test_split(
        train_val,
        test_size=val_proportion,
        stratify=train_val[target_col],
        random_state=random_state
    )
    
    return train, val, test


def validate_stratification(
    original_df: pd.DataFrame,
    splits: Tuple[pd.DataFrame, ...],
    target_col: str
) -> dict:
    """
    Validate that stratification preserved class distribution.
    
    Args:
        original_df: Original dataframe
        splits: Tuple of (train, val, test) dataframes
        target_col: Target column name
        
    Returns:
        Dictionary with distribution statistics and chi-squared test
    """
    from scipy.stats import chisquare
    
    original_dist = original_df[target_col].value_counts(normalize=True).sort_index()
    
    results = {
        'original': original_dist.to_dict()
    }
    
    split_names = ['train', 'val', 'test']
    for name, split_df in zip(split_names, splits):
        split_dist = split_df[target_col].value_counts(normalize=True).sort_index()
        results[name] = split_dist.to_dict()
        
        # Chi-squared test
        observed = split_df[target_col].value_counts().sort_index().values
        expected_proportions = original_dist.values
        expected = expected_proportions * len(split_df)
        
        chi2, p_value = chisquare(observed, expected)
        results[f'{name}_chi2'] = chi2
        results[f'{name}_p_value'] = p_value
        
    return results


def compute_sample_size(
    population_size: int,
    confidence_level: float = 0.95,
    margin_of_error: float = 0.03,
    proportion: float = 0.5
) -> int:
    """
    Calculate required sample size for population proportion estimation.
    
    Args:
        population_size: Total population size
        confidence_level: Desired confidence (default 0.95)
        margin_of_error: Acceptable error (default 0.03 = 3%)
        proportion: Expected proportion (default 0.5 = most conservative)
        
    Returns:
        Required sample size
    """
    from scipy.stats import norm
    
    z_score = norm.ppf((1 + confidence_level) / 2)
    numerator = (z_score ** 2) * proportion * (1 - proportion)
    denominator = margin_of_error ** 2
    
    n0 = numerator / denominator
    
    # Finite population correction
    n = n0 / (1 + (n0 - 1) / population_size)
    
    return int(np.ceil(n))


def temporal_balance_check(
    df: pd.DataFrame,
    splits: Tuple[pd.DataFrame, ...],
    temporal_col: str
) -> dict:
    """
    Check if temporal features are balanced across splits.
    
    Args:
        df: Original dataframe
        splits: Tuple of splits
        temporal_col: Name of temporal column (e.g., 'month')
        
    Returns:
        Dictionary with temporal distributions
    """
    from scipy.stats import chi2_contingency
    
    original_dist = df[temporal_col].value_counts(normalize=True).sort_index()
    
    results = {'original': original_dist.to_dict()}
    
    split_names = ['train', 'val', 'test']
    for name, split_df in zip(split_names, splits):
        split_dist = split_df[temporal_col].value_counts(normalize=True).sort_index()
        results[name] = split_dist.to_dict()
    
    # Chi-squared contingency test (independence)
    contingency_table = pd.DataFrame({
        'train': splits[0][temporal_col].value_counts().sort_index(),
        'val': splits[1][temporal_col].value_counts().sort_index(),
        'test': splits[2][temporal_col].value_counts().sort_index()
    }).fillna(0)
    
    chi2, p_value, dof, expected = chi2_contingency(contingency_table.values)
    
    results['chi2_contingency'] = chi2
    results['p_value'] = p_value
    results['dof'] = dof
    
    return results
