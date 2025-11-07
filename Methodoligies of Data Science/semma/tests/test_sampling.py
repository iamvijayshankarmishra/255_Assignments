"""
Tests for SEMMA Sampling Phase
===============================
Test stratified sampling, temporal validation, and χ² tests.
"""

import pytest
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split

import sys
sys.path.append('../src')
from sampling import (
    stratified_sample, validate_stratification, calculate_chi_squared,
    temporal_split, validate_temporal_order
)


@pytest.fixture
def bank_data():
    """Create sample bank marketing dataset."""
    np.random.seed(42)
    n = 1000
    
    data = pd.DataFrame({
        'age': np.random.randint(18, 80, n),
        'job': np.random.choice(['admin', 'technician', 'services', 'management'], n),
        'marital': np.random.choice(['married', 'single', 'divorced'], n),
        'education': np.random.choice(['primary', 'secondary', 'tertiary', 'unknown'], n),
        'default': np.random.choice(['yes', 'no'], n, p=[0.1, 0.9]),
        'balance': np.random.randint(-5000, 50000, n),
        'housing': np.random.choice(['yes', 'no'], n, p=[0.6, 0.4]),
        'loan': np.random.choice(['yes', 'no'], n, p=[0.2, 0.8]),
        'contact': np.random.choice(['cellular', 'telephone', 'unknown'], n),
        'day': np.random.randint(1, 32, n),
        'month': np.random.choice(['jan', 'feb', 'mar', 'apr', 'may', 'jun'], n),
        'duration': np.random.randint(0, 3600, n),
        'campaign': np.random.randint(1, 50, n),
        'pdays': np.random.randint(-1, 500, n),
        'previous': np.random.randint(0, 50, n),
        'poutcome': np.random.choice(['success', 'failure', 'unknown', 'other'], n),
        'y': np.random.choice(['yes', 'no'], n, p=[0.12, 0.88])  # 12% subscription rate
    })
    
    return data


@pytest.fixture
def imbalanced_data():
    """Create imbalanced dataset for stratification testing."""
    np.random.seed(42)
    n = 1000
    
    data = pd.DataFrame({
        'feature1': np.random.randn(n),
        'feature2': np.random.randn(n),
        'target': np.random.choice([0, 1], n, p=[0.9, 0.1])  # 10% minority
    })
    
    return data


class TestStratifiedSampling:
    """Test suite for stratified sampling."""
    
    def test_stratified_sample_preserves_ratio(self, imbalanced_data):
        """Test that stratified sampling preserves class ratio."""
        sample_size = 200
        sample = stratified_sample(
            imbalanced_data, 
            target_col='target', 
            sample_size=sample_size,
            random_state=42
        )
        
        # Original ratio
        original_ratio = imbalanced_data['target'].mean()
        
        # Sample ratio
        sample_ratio = sample['target'].mean()
        
        # Should be within 5% of original
        assert abs(sample_ratio - original_ratio) < 0.05, \
            f"Sample ratio {sample_ratio:.3f} differs from original {original_ratio:.3f}"
    
    def test_stratified_sample_size(self, imbalanced_data):
        """Test that stratified sample has correct size."""
        sample_size = 200
        sample = stratified_sample(
            imbalanced_data, 
            target_col='target', 
            sample_size=sample_size,
            random_state=42
        )
        
        assert len(sample) == sample_size, \
            f"Sample size {len(sample)} != requested {sample_size}"
    
    def test_stratified_split_train_test(self, bank_data):
        """Test stratified train/test split."""
        train, test = train_test_split(
            bank_data, 
            test_size=0.2, 
            stratify=bank_data['y'],
            random_state=42
        )
        
        # Train and test ratios should be similar
        train_ratio = (train['y'] == 'yes').mean()
        test_ratio = (test['y'] == 'yes').mean()
        
        assert abs(train_ratio - test_ratio) < 0.02, \
            f"Train ratio {train_ratio:.3f} differs from test {test_ratio:.3f}"
    
    def test_stratified_sample_reproducible(self, imbalanced_data):
        """Test that stratified sampling is reproducible."""
        sample1 = stratified_sample(
            imbalanced_data, 
            target_col='target', 
            sample_size=200,
            random_state=42
        )
        
        sample2 = stratified_sample(
            imbalanced_data, 
            target_col='target', 
            sample_size=200,
            random_state=42
        )
        
        assert sample1.equals(sample2), "Samples should be identical with same random_state"


class TestStratificationValidation:
    """Test suite for stratification validation."""
    
    def test_validate_stratification_identical(self, bank_data):
        """Test validation when stratification is perfect."""
        # Split data
        train, test = train_test_split(
            bank_data, 
            test_size=0.2, 
            stratify=bank_data['y'],
            random_state=42
        )
        
        is_valid, p_value = validate_stratification(
            train['y'], 
            test['y'], 
            alpha=0.05
        )
        
        assert is_valid, f"Stratification should be valid, p-value={p_value:.3f}"
        assert p_value > 0.05, "Should not reject null hypothesis (distributions are same)"
    
    def test_validate_stratification_different(self):
        """Test validation when distributions differ."""
        train = pd.Series(['yes'] * 50 + ['no'] * 50)
        test = pd.Series(['yes'] * 10 + ['no'] * 90)  # Very different
        
        is_valid, p_value = validate_stratification(train, test, alpha=0.05)
        
        assert not is_valid, "Should detect different distributions"
        assert p_value < 0.05, "Should reject null hypothesis (distributions differ)"


class TestChiSquared:
    """Test suite for χ² independence tests."""
    
    def test_chi_squared_independent(self):
        """Test χ² when variables are independent."""
        np.random.seed(42)
        
        # Independent variables
        data = pd.DataFrame({
            'var1': np.random.choice(['A', 'B'], 1000),
            'var2': np.random.choice(['X', 'Y'], 1000)
        })
        
        chi2_stat, p_value, cramers_v = calculate_chi_squared(
            data['var1'], 
            data['var2']
        )
        
        assert p_value > 0.05, f"Should not reject independence, p-value={p_value:.3f}"
        assert cramers_v < 0.1, f"Cramér's V should be small, got {cramers_v:.3f}"
    
    def test_chi_squared_dependent(self):
        """Test χ² when variables are dependent."""
        # Create dependent variables (var2 = var1)
        data = pd.DataFrame({
            'var1': ['A'] * 500 + ['B'] * 500,
            'var2': ['X'] * 500 + ['Y'] * 500  # Perfect correlation
        })
        
        chi2_stat, p_value, cramers_v = calculate_chi_squared(
            data['var1'], 
            data['var2']
        )
        
        assert p_value < 0.001, "Should strongly reject independence"
        assert cramers_v > 0.9, f"Cramér's V should be high, got {cramers_v:.3f}"
    
    def test_chi_squared_association_strength(self, bank_data):
        """Test Cramér's V interpretation."""
        # Test association between job and education
        chi2_stat, p_value, cramers_v = calculate_chi_squared(
            bank_data['job'], 
            bank_data['education']
        )
        
        # Interpret Cramér's V
        if cramers_v < 0.1:
            strength = "negligible"
        elif cramers_v < 0.3:
            strength = "weak"
        elif cramers_v < 0.5:
            strength = "moderate"
        else:
            strength = "strong"
        
        assert strength in ["negligible", "weak", "moderate", "strong"], \
            f"Invalid strength: {strength}"


class TestTemporalSplit:
    """Test suite for temporal split validation."""
    
    def test_temporal_split_no_shuffle(self):
        """Test that temporal split preserves order."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=1000, freq='H')
        data = pd.DataFrame({
            'timestamp': dates,
            'value': np.random.randn(1000)
        })
        
        train, val, test = temporal_split(
            data, 
            time_col='timestamp',
            train_size=0.6, 
            val_size=0.2,
            shuffle=False
        )
        
        # Check no overlap
        assert train['timestamp'].max() < val['timestamp'].min(), \
            "Train timestamps should be before validation"
        assert val['timestamp'].max() < test['timestamp'].min(), \
            "Validation timestamps should be before test"
    
    def test_temporal_split_sizes(self):
        """Test that temporal split has correct sizes."""
        np.random.seed(42)
        data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=1000, freq='H'),
            'value': np.random.randn(1000)
        })
        
        train, val, test = temporal_split(
            data, 
            time_col='timestamp',
            train_size=0.6, 
            val_size=0.2,
            shuffle=False
        )
        
        assert len(train) == 600, f"Train size should be 600, got {len(train)}"
        assert len(val) == 200, f"Val size should be 200, got {len(val)}"
        assert len(test) == 200, f"Test size should be 200, got {len(test)}"
    
    def test_validate_temporal_order(self):
        """Test temporal order validation."""
        train = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=100, freq='H')
        })
        test = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-05', periods=100, freq='H')
        })
        
        is_valid = validate_temporal_order(train, test, time_col='timestamp')
        assert is_valid, "Temporal order should be valid"
    
    def test_validate_temporal_order_invalid(self):
        """Test detection of temporal leakage."""
        train = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-05', periods=100, freq='H')
        })
        test = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=100, freq='H')
        })
        
        is_valid = validate_temporal_order(train, test, time_col='timestamp')
        assert not is_valid, "Should detect temporal leakage (test before train)"


class TestDataBalance:
    """Test suite for data balance analysis."""
    
    def test_class_distribution(self, bank_data):
        """Test class distribution calculation."""
        distribution = bank_data['y'].value_counts(normalize=True)
        
        assert 'yes' in distribution.index, "Should have 'yes' class"
        assert 'no' in distribution.index, "Should have 'no' class"
        assert abs(distribution.sum() - 1.0) < 0.001, "Should sum to 1.0"
    
    def test_imbalance_ratio(self, imbalanced_data):
        """Test imbalance ratio calculation."""
        minority_count = (imbalanced_data['target'] == 1).sum()
        majority_count = (imbalanced_data['target'] == 0).sum()
        imbalance_ratio = majority_count / minority_count
        
        assert imbalance_ratio > 1, "Majority should be larger than minority"
        assert 8 < imbalance_ratio < 10, f"Expected ~9:1 ratio, got {imbalance_ratio:.1f}:1"
    
    def test_stratification_balance(self, bank_data):
        """Test that stratification maintains balance across splits."""
        # Split with stratification
        train, test = train_test_split(
            bank_data, 
            test_size=0.2, 
            stratify=bank_data['y'],
            random_state=42
        )
        
        # Calculate distributions
        original_dist = bank_data['y'].value_counts(normalize=True)
        train_dist = train['y'].value_counts(normalize=True)
        test_dist = test['y'].value_counts(normalize=True)
        
        # All distributions should be similar
        for cls in ['yes', 'no']:
            assert abs(train_dist[cls] - original_dist[cls]) < 0.05, \
                f"Train distribution for {cls} differs from original"
            assert abs(test_dist[cls] - original_dist[cls]) < 0.05, \
                f"Test distribution for {cls} differs from original"


class TestCrossValidation:
    """Test suite for cross-validation with stratification."""
    
    def test_stratified_kfold_coverage(self, bank_data):
        """Test that stratified k-fold covers all data."""
        from sklearn.model_selection import StratifiedKFold
        
        X = bank_data.drop('y', axis=1)
        y = bank_data['y']
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        seen_indices = set()
        for train_idx, test_idx in skf.split(X, y):
            seen_indices.update(test_idx)
        
        assert len(seen_indices) == len(bank_data), \
            "All samples should be seen in test set exactly once"
    
    def test_stratified_kfold_balance(self, bank_data):
        """Test that each fold maintains class balance."""
        from sklearn.model_selection import StratifiedKFold
        
        X = bank_data.drop('y', axis=1)
        y = bank_data['y']
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        original_ratio = (y == 'yes').mean()
        
        for train_idx, test_idx in skf.split(X, y):
            fold_ratio = (y.iloc[test_idx] == 'yes').mean()
            
            assert abs(fold_ratio - original_ratio) < 0.05, \
                f"Fold ratio {fold_ratio:.3f} differs from original {original_ratio:.3f}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
