"""
Tests for Imbalanced Learning Techniques
=========================================
Test SMOTE, ADASYN, and other imbalanced learning methods.
"""

import pytest
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler

import sys
sys.path.append('../src')
from transformation import (
    ImbalancedSampler, validate_synthetic_samples,
    check_test_contamination
)


@pytest.fixture
def imbalanced_data():
    """Create sample imbalanced dataset (10% minority class)."""
    np.random.seed(42)
    n_majority = 900
    n_minority = 100
    
    X_majority = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_majority),
        'feature2': np.random.normal(0, 1, n_majority),
    })
    y_majority = pd.Series([0] * n_majority)
    
    X_minority = pd.DataFrame({
        'feature1': np.random.normal(2, 1, n_minority),
        'feature2': np.random.normal(2, 1, n_minority),
    })
    y_minority = pd.Series([1] * n_minority)
    
    X = pd.concat([X_majority, X_minority], ignore_index=True)
    y = pd.concat([y_majority, y_minority], ignore_index=True)
    
    return X, y


class TestImbalancedSampler:
    """Test suite for ImbalancedSampler class."""
    
    def test_smote_increases_minority_class(self, imbalanced_data):
        """Test that SMOTE increases minority class samples."""
        X, y = imbalanced_data
        original_minority = y.sum()
        
        sampler = ImbalancedSampler(method='smote', sampling_strategy=0.5)
        X_res, y_res = sampler.fit_resample(X, y)
        
        new_minority = y_res.sum()
        assert new_minority > original_minority, \
            f"SMOTE should increase minority class: {original_minority} -> {new_minority}"
    
    def test_smote_sampling_strategy(self, imbalanced_data):
        """Test that SMOTE achieves target sampling strategy."""
        X, y = imbalanced_data
        target_strategy = 0.5  # 1 minority : 2 majority
        
        sampler = ImbalancedSampler(method='smote', sampling_strategy=target_strategy)
        X_res, y_res = sampler.fit_resample(X, y)
        
        minority_count = y_res.sum()
        majority_count = len(y_res) - minority_count
        actual_strategy = minority_count / majority_count
        
        assert abs(actual_strategy - target_strategy) < 0.01, \
            f"Expected {target_strategy}, got {actual_strategy}"
    
    def test_adasyn_increases_minority_class(self, imbalanced_data):
        """Test that ADASYN increases minority class samples."""
        X, y = imbalanced_data
        original_minority = y.sum()
        
        sampler = ImbalancedSampler(method='adasyn', sampling_strategy=0.5)
        X_res, y_res = sampler.fit_resample(X, y)
        
        new_minority = y_res.sum()
        assert new_minority > original_minority, \
            f"ADASYN should increase minority class: {original_minority} -> {new_minority}"
    
    def test_random_under_reduces_majority_class(self, imbalanced_data):
        """Test that RandomUnderSampler reduces majority class."""
        X, y = imbalanced_data
        original_size = len(X)
        
        sampler = ImbalancedSampler(method='random_under', sampling_strategy=0.5)
        X_res, y_res = sampler.fit_resample(X, y)
        
        new_size = len(X_res)
        assert new_size < original_size, \
            f"Under-sampling should reduce size: {original_size} -> {new_size}"
    
    def test_smote_preserves_feature_names(self, imbalanced_data):
        """Test that SMOTE preserves DataFrame column names."""
        X, y = imbalanced_data
        original_columns = X.columns.tolist()
        
        sampler = ImbalancedSampler(method='smote', sampling_strategy=0.5)
        X_res, y_res = sampler.fit_resample(X, y)
        
        assert X_res.columns.tolist() == original_columns, \
            "Column names should be preserved"
    
    def test_smote_reproducibility(self, imbalanced_data):
        """Test that SMOTE with same random_state produces same results."""
        X, y = imbalanced_data
        
        sampler1 = ImbalancedSampler(method='smote', sampling_strategy=0.5, random_state=42)
        X_res1, y_res1 = sampler1.fit_resample(X, y)
        
        sampler2 = ImbalancedSampler(method='smote', sampling_strategy=0.5, random_state=42)
        X_res2, y_res2 = sampler2.fit_resample(X, y)
        
        assert len(X_res1) == len(X_res2), "Same random_state should produce same size"
        assert y_res1.sum() == y_res2.sum(), "Same random_state should produce same minority count"


class TestSMOTEValidation:
    """Test suite for SMOTE validation functions."""
    
    def test_synthetic_samples_within_range(self, imbalanced_data):
        """Test that synthetic samples fall within original feature ranges."""
        X, y = imbalanced_data
        
        sampler = ImbalancedSampler(method='smote', sampling_strategy=0.5)
        X_res, y_res = sampler.fit_resample(X, y)
        
        # Identify synthetic samples
        n_original = len(X)
        X_synthetic = X_res.iloc[n_original:]
        
        # Check ranges
        for col in X.columns:
            original_min = X[col].min()
            original_max = X[col].max()
            synthetic_min = X_synthetic[col].min()
            synthetic_max = X_synthetic[col].max()
            
            # Allow small tolerance for floating point
            assert synthetic_min >= original_min - 0.1, \
                f"{col}: synthetic min {synthetic_min} < original min {original_min}"
            assert synthetic_max <= original_max + 0.1, \
                f"{col}: synthetic max {synthetic_max} > original max {original_max}"
    
    def test_no_duplicate_samples(self, imbalanced_data):
        """Test that SMOTE doesn't create exact duplicates."""
        X, y = imbalanced_data
        
        sampler = ImbalancedSampler(method='smote', sampling_strategy=0.5)
        X_res, y_res = sampler.fit_resample(X, y)
        
        # Check for duplicates
        n_duplicates = X_res.duplicated().sum()
        assert n_duplicates == 0, f"Found {n_duplicates} duplicate samples"
    
    def test_minority_class_preserved(self, imbalanced_data):
        """Test that original minority class samples are preserved."""
        X, y = imbalanced_data
        original_minority_count = y.sum()
        
        sampler = ImbalancedSampler(method='smote', sampling_strategy=0.5)
        X_res, y_res = sampler.fit_resample(X, y)
        
        # First n_original samples should include all original minority
        y_original = y_res.iloc[:len(y)]
        assert y_original.sum() == original_minority_count, \
            "Original minority samples should be preserved"


class TestClassDistribution:
    """Test suite for class distribution checks."""
    
    def test_class_distribution_calculation(self, imbalanced_data):
        """Test class distribution calculation."""
        X, y = imbalanced_data
        
        minority_count = y.sum()
        majority_count = len(y) - minority_count
        
        assert minority_count == 100, "Expected 100 minority samples"
        assert majority_count == 900, "Expected 900 majority samples"
    
    def test_class_balance_after_smote(self, imbalanced_data):
        """Test that class balance changes after SMOTE."""
        X, y = imbalanced_data
        original_ratio = y.sum() / (len(y) - y.sum())
        
        sampler = ImbalancedSampler(method='smote', sampling_strategy=0.5)
        X_res, y_res = sampler.fit_resample(X, y)
        
        new_ratio = y_res.sum() / (len(y_res) - y_res.sum())
        
        assert new_ratio > original_ratio, \
            f"Balance should improve: {original_ratio:.3f} -> {new_ratio:.3f}"
    
    def test_extreme_imbalance_handling(self):
        """Test SMOTE with extreme imbalance (0.1% minority)."""
        np.random.seed(42)
        n_majority = 9990
        n_minority = 10
        
        X = pd.DataFrame({
            'f1': np.random.normal(0, 1, n_majority + n_minority),
            'f2': np.random.normal(0, 1, n_majority + n_minority),
        })
        y = pd.Series([0] * n_majority + [1] * n_minority)
        
        sampler = ImbalancedSampler(method='smote', sampling_strategy=0.1)
        X_res, y_res = sampler.fit_resample(X, y)
        
        new_minority = y_res.sum()
        assert new_minority > n_minority, \
            f"Should increase minority from {n_minority} to {new_minority}"


class TestDataLeakage:
    """Test suite for data leakage prevention."""
    
    def test_test_set_not_contaminated(self, imbalanced_data):
        """Test that test set is not contaminated by SMOTE."""
        X, y = imbalanced_data
        
        # Split into train and test
        n_train = int(0.8 * len(X))
        X_train, X_test = X.iloc[:n_train], X.iloc[n_train:]
        y_train, y_test = y.iloc[:n_train], y.iloc[n_train:]
        
        # Apply SMOTE to train
        sampler = ImbalancedSampler(method='smote', sampling_strategy=0.5)
        X_train_smote, y_train_smote = sampler.fit_resample(X_train, y_train)
        
        # Check test set size unchanged
        assert len(X_test) == len(X) - n_train, \
            "Test set size should not change"
        assert len(y_test) == len(y) - n_train, \
            "Test target size should not change"
    
    def test_smote_only_on_train(self):
        """Test that SMOTE is only applied to training set."""
        np.random.seed(42)
        X = pd.DataFrame({
            'f1': np.random.normal(0, 1, 1000),
            'f2': np.random.normal(0, 1, 1000),
        })
        y = pd.Series([0] * 900 + [1] * 100)
        
        # Train/test split
        n_train = 800
        X_train, X_test = X.iloc[:n_train], X.iloc[n_train:]
        y_train, y_test = y.iloc[:n_train], y.iloc[n_train:]
        
        original_test_size = len(X_test)
        original_test_fraud_rate = y_test.mean()
        
        # Apply SMOTE to train only
        sampler = ImbalancedSampler(method='smote', sampling_strategy=0.5)
        X_train_smote, y_train_smote = sampler.fit_resample(X_train, y_train)
        
        # Verify test set unchanged
        assert len(X_test) == original_test_size, \
            "Test set size should remain unchanged"
        assert y_test.mean() == original_test_fraud_rate, \
            "Test set fraud rate should remain unchanged"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
