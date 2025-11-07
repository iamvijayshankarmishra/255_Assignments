"""
Tests for SEMMA Modification Phase
===================================
Test feature engineering, VIF calculation, and encoding validation.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

import sys
sys.path.append('../src')
from modification import (
    BankFeatureEngineer, calculate_vif, detect_multicollinearity,
    one_hot_encode, label_encode, validate_encoding
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
        'y': np.random.choice(['yes', 'no'], n, p=[0.12, 0.88])
    })
    
    return data


@pytest.fixture
def multicollinear_data():
    """Create dataset with multicollinearity."""
    np.random.seed(42)
    n = 500
    
    x1 = np.random.randn(n)
    x2 = x1 + np.random.randn(n) * 0.1  # Highly correlated with x1
    x3 = np.random.randn(n)  # Independent
    
    data = pd.DataFrame({
        'x1': x1,
        'x2': x2,  # Multicollinear
        'x3': x3
    })
    
    return data


class TestBankFeatureEngineer:
    """Test suite for BankFeatureEngineer."""
    
    def test_creates_age_group(self, bank_data):
        """Test that age_group feature is created."""
        engineer = BankFeatureEngineer()
        df = engineer.fit_transform(bank_data.copy())
        
        assert 'age_group' in df.columns, "Should create age_group feature"
        assert df['age_group'].nunique() > 1, "Should have multiple age groups"
    
    def test_creates_balance_category(self, bank_data):
        """Test that balance_category feature is created."""
        engineer = BankFeatureEngineer()
        df = engineer.fit_transform(bank_data.copy())
        
        assert 'balance_category' in df.columns, "Should create balance_category"
        categories = ['negative', 'low', 'medium', 'high']
        assert all(cat in df['balance_category'].values for cat in categories), \
            f"Should have categories: {categories}"
    
    def test_creates_contact_success_rate(self, bank_data):
        """Test that contact_success_rate feature is created."""
        engineer = BankFeatureEngineer()
        df = engineer.fit_transform(bank_data.copy())
        
        assert 'contact_success_rate' in df.columns, \
            "Should create contact_success_rate"
        assert df['contact_success_rate'].between(0, 1).all(), \
            "Success rate should be in [0, 1]"
    
    def test_creates_campaign_intensity(self, bank_data):
        """Test that campaign_intensity feature is created."""
        engineer = BankFeatureEngineer()
        df = engineer.fit_transform(bank_data.copy())
        
        assert 'campaign_intensity' in df.columns, \
            "Should create campaign_intensity"
        intensities = ['low', 'medium', 'high']
        assert all(df['campaign_intensity'].isin(intensities)), \
            f"Should have intensities: {intensities}"
    
    def test_creates_previous_contact_success(self, bank_data):
        """Test that previous_contact_success feature is created."""
        engineer = BankFeatureEngineer()
        df = engineer.fit_transform(bank_data.copy())
        
        assert 'previous_contact_success' in df.columns, \
            "Should create previous_contact_success"
        assert df['previous_contact_success'].isin([0, 1]).all(), \
            "Should be binary (0 or 1)"
    
    def test_preserves_original_columns(self, bank_data):
        """Test that original columns are preserved."""
        engineer = BankFeatureEngineer()
        df = engineer.fit_transform(bank_data.copy())
        
        for col in bank_data.columns:
            assert col in df.columns, f"Should preserve {col}"
    
    def test_reproducible(self, bank_data):
        """Test that feature engineering is reproducible."""
        engineer = BankFeatureEngineer()
        
        df1 = engineer.fit_transform(bank_data.copy())
        df2 = engineer.fit_transform(bank_data.copy())
        
        assert df1.equals(df2), "Should produce identical results"


class TestVIFCalculation:
    """Test suite for VIF (Variance Inflation Factor) calculation."""
    
    def test_vif_independent_variables(self):
        """Test VIF for independent variables (should be low)."""
        np.random.seed(42)
        data = pd.DataFrame({
            'x1': np.random.randn(500),
            'x2': np.random.randn(500),
            'x3': np.random.randn(500)
        })
        
        vif_df = calculate_vif(data)
        
        assert 'feature' in vif_df.columns, "Should have feature column"
        assert 'VIF' in vif_df.columns, "Should have VIF column"
        
        # All VIF should be low (< 5)
        assert (vif_df['VIF'] < 5).all(), "VIF should be < 5 for independent variables"
    
    def test_vif_multicollinear_variables(self, multicollinear_data):
        """Test VIF for multicollinear variables (should be high)."""
        vif_df = calculate_vif(multicollinear_data)
        
        # x1 and x2 are highly correlated, should have high VIF
        x1_vif = vif_df[vif_df['feature'] == 'x1']['VIF'].values[0]
        x2_vif = vif_df[vif_df['feature'] == 'x2']['VIF'].values[0]
        
        assert x1_vif > 5, f"x1 VIF should be > 5, got {x1_vif:.2f}"
        assert x2_vif > 5, f"x2 VIF should be > 5, got {x2_vif:.2f}"
    
    def test_detect_multicollinearity(self, multicollinear_data):
        """Test multicollinearity detection."""
        has_multicollinearity, high_vif_features = detect_multicollinearity(
            multicollinear_data, threshold=5
        )
        
        assert has_multicollinearity, "Should detect multicollinearity"
        assert 'x1' in high_vif_features or 'x2' in high_vif_features, \
            "Should identify x1 or x2 as multicollinear"


class TestOneHotEncoding:
    """Test suite for one-hot encoding."""
    
    def test_one_hot_encode_creates_columns(self, bank_data):
        """Test that one-hot encoding creates correct columns."""
        df = one_hot_encode(bank_data.copy(), columns=['job', 'marital'])
        
        # Original columns should be dropped
        assert 'job' not in df.columns, "Original job column should be dropped"
        assert 'marital' not in df.columns, "Original marital column should be dropped"
        
        # Encoded columns should exist
        assert 'job_admin' in df.columns or 'job_technician' in df.columns, \
            "Should create job_* columns"
        assert 'marital_married' in df.columns or 'marital_single' in df.columns, \
            "Should create marital_* columns"
    
    def test_one_hot_encode_binary_values(self, bank_data):
        """Test that one-hot encoded values are binary."""
        df = one_hot_encode(bank_data.copy(), columns=['job'])
        
        job_cols = [col for col in df.columns if col.startswith('job_')]
        
        for col in job_cols:
            assert df[col].isin([0, 1]).all(), f"{col} should be binary (0 or 1)"
    
    def test_one_hot_encode_exactly_one_active(self, bank_data):
        """Test that exactly one one-hot encoded column is active per row."""
        df = one_hot_encode(bank_data.copy(), columns=['job'])
        
        job_cols = [col for col in df.columns if col.startswith('job_')]
        row_sums = df[job_cols].sum(axis=1)
        
        assert (row_sums == 1).all(), "Exactly one job column should be 1 per row"
    
    def test_one_hot_encode_drop_first(self, bank_data):
        """Test one-hot encoding with drop_first=True (avoid dummy trap)."""
        df = one_hot_encode(bank_data.copy(), columns=['marital'], drop_first=True)
        
        marital_cols = [col for col in df.columns if col.startswith('marital_')]
        
        # Should have n-1 columns (3 categories -> 2 columns)
        assert len(marital_cols) == 2, \
            f"Should have 2 marital columns (drop_first=True), got {len(marital_cols)}"


class TestLabelEncoding:
    """Test suite for label encoding."""
    
    def test_label_encode_creates_numeric(self, bank_data):
        """Test that label encoding creates numeric values."""
        df = label_encode(bank_data.copy(), columns=['education'])
        
        assert pd.api.types.is_numeric_dtype(df['education']), \
            "Education should be numeric after label encoding"
    
    def test_label_encode_consistent_mapping(self, bank_data):
        """Test that label encoding uses consistent mapping."""
        df1 = label_encode(bank_data.copy(), columns=['education'])
        df2 = label_encode(bank_data.copy(), columns=['education'])
        
        assert (df1['education'] == df2['education']).all(), \
            "Label encoding should be consistent"
    
    def test_label_encode_preserves_categories(self, bank_data):
        """Test that label encoding preserves number of categories."""
        original_categories = bank_data['education'].nunique()
        
        df = label_encode(bank_data.copy(), columns=['education'])
        encoded_categories = df['education'].nunique()
        
        assert encoded_categories == original_categories, \
            f"Should preserve {original_categories} categories, got {encoded_categories}"
    
    def test_label_encode_starts_from_zero(self, bank_data):
        """Test that label encoding starts from 0."""
        df = label_encode(bank_data.copy(), columns=['education'])
        
        assert df['education'].min() == 0, "Label encoding should start from 0"
        assert df['education'].max() == bank_data['education'].nunique() - 1, \
            "Label encoding should end at n_categories - 1"


class TestEncodingValidation:
    """Test suite for encoding validation."""
    
    def test_validate_encoding_detects_categorical(self, bank_data):
        """Test that validation detects remaining categorical columns."""
        needs_encoding = validate_encoding(bank_data)
        
        assert len(needs_encoding) > 0, "Should detect categorical columns"
        assert 'job' in needs_encoding, "Should detect job as categorical"
        assert 'marital' in needs_encoding, "Should detect marital as categorical"
    
    def test_validate_encoding_after_encoding(self, bank_data):
        """Test that validation passes after encoding."""
        df = one_hot_encode(bank_data.copy(), columns=['job', 'marital', 'education'])
        df = label_encode(df, columns=['default', 'housing', 'loan', 'contact', 
                                       'poutcome', 'y', 'month'])
        
        needs_encoding = validate_encoding(df)
        
        assert len(needs_encoding) == 0, \
            f"Should have no categorical columns, found: {needs_encoding}"


class TestScaling:
    """Test suite for feature scaling."""
    
    def test_standard_scaler_mean_zero(self, bank_data):
        """Test that StandardScaler results in mean ~0."""
        scaler = StandardScaler()
        scaled = scaler.fit_transform(bank_data[['age', 'balance', 'duration']])
        
        means = scaled.mean(axis=0)
        assert np.allclose(means, 0, atol=1e-10), \
            f"Scaled features should have mean ~0, got {means}"
    
    def test_standard_scaler_std_one(self, bank_data):
        """Test that StandardScaler results in std ~1."""
        scaler = StandardScaler()
        scaled = scaler.fit_transform(bank_data[['age', 'balance', 'duration']])
        
        stds = scaled.std(axis=0)
        assert np.allclose(stds, 1, atol=1e-10), \
            f"Scaled features should have std ~1, got {stds}"
    
    def test_scaler_preserves_shape(self, bank_data):
        """Test that scaling preserves data shape."""
        scaler = StandardScaler()
        original_shape = bank_data[['age', 'balance', 'duration']].shape
        scaled = scaler.fit_transform(bank_data[['age', 'balance', 'duration']])
        
        assert scaled.shape == original_shape, \
            f"Shape should be preserved, got {scaled.shape} != {original_shape}"
    
    def test_scaler_inverse_transform(self, bank_data):
        """Test that inverse transform recovers original data."""
        scaler = StandardScaler()
        original = bank_data[['age', 'balance', 'duration']].values
        scaled = scaler.fit_transform(original)
        recovered = scaler.inverse_transform(scaled)
        
        assert np.allclose(original, recovered), \
            "Inverse transform should recover original data"


class TestFeatureInteractions:
    """Test suite for feature interactions."""
    
    def test_polynomial_features(self, bank_data):
        """Test polynomial feature generation."""
        from sklearn.preprocessing import PolynomialFeatures
        
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X = bank_data[['age', 'balance']].values
        X_poly = poly.fit_transform(X)
        
        # Should have original + interaction + squares
        # (age, balance) -> (age, balance, age^2, age*balance, balance^2)
        assert X_poly.shape[1] == 5, \
            f"Should have 5 features with degree=2, got {X_poly.shape[1]}"
    
    def test_interaction_terms(self, bank_data):
        """Test creation of specific interaction terms."""
        df = bank_data.copy()
        df['age_balance_interaction'] = df['age'] * df['balance']
        
        assert 'age_balance_interaction' in df.columns, \
            "Should create interaction term"
        assert (df['age_balance_interaction'] == df['age'] * df['balance']).all(), \
            "Interaction should be product of features"


class TestBinning:
    """Test suite for feature binning."""
    
    def test_equal_width_binning(self, bank_data):
        """Test equal-width binning."""
        df = bank_data.copy()
        df['age_binned'] = pd.cut(df['age'], bins=5, labels=['very_young', 'young', 
                                                              'middle', 'senior', 'elderly'])
        
        assert df['age_binned'].nunique() == 5, "Should have 5 bins"
    
    def test_equal_frequency_binning(self, bank_data):
        """Test equal-frequency binning (quantiles)."""
        df = bank_data.copy()
        df['balance_binned'] = pd.qcut(df['balance'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], 
                                       duplicates='drop')
        
        # Each bin should have ~25% of data (may vary due to duplicates)
        bin_counts = df['balance_binned'].value_counts(normalize=True)
        assert all(0.15 < freq < 0.35 for freq in bin_counts), \
            "Each quantile should have ~25% of data"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
