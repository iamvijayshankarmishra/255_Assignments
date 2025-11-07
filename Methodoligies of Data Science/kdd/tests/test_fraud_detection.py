"""
Tests for Fraud Detection Models
==================================
Test PR-AUC calculation, threshold tuning, and cost-sensitive metrics.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

import sys
sys.path.append('../src')
from mining import (
    calculate_pr_auc, find_optimal_threshold, compare_models
)
from evaluation import (
    calculate_cost_sensitive_profit, calculate_business_roi
)


@pytest.fixture
def binary_predictions():
    """Create sample binary classification predictions."""
    np.random.seed(42)
    n_samples = 1000
    n_frauds = 10  # 1% fraud rate
    
    # Generate true labels
    y_true = np.array([0] * (n_samples - n_frauds) + [1] * n_frauds)
    
    # Generate probabilistic predictions (frauds have higher prob)
    y_proba = np.random.beta(2, 10, n_samples)  # Most values near 0
    y_proba[y_true == 1] = np.random.beta(8, 2, n_frauds)  # Frauds near 1
    
    return y_true, y_proba


class TestPRAUC:
    """Test suite for PR-AUC calculation."""
    
    def test_pr_auc_range(self, binary_predictions):
        """Test that PR-AUC is between 0 and 1."""
        y_true, y_proba = binary_predictions
        pr_auc = calculate_pr_auc(y_true, y_proba)
        
        assert 0 <= pr_auc <= 1, f"PR-AUC should be in [0, 1], got {pr_auc}"
    
    def test_pr_auc_perfect_classifier(self):
        """Test PR-AUC for perfect classifier."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_proba = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        
        pr_auc = calculate_pr_auc(y_true, y_proba)
        assert pr_auc == 1.0, f"Perfect classifier should have PR-AUC=1.0, got {pr_auc}"
    
    def test_pr_auc_random_classifier(self):
        """Test PR-AUC for random classifier (should be close to baseline)."""
        np.random.seed(42)
        y_true = np.array([0] * 990 + [1] * 10)
        y_proba = np.random.uniform(0, 1, 1000)
        
        pr_auc = calculate_pr_auc(y_true, y_proba)
        baseline = y_true.mean()  # 0.01
        
        # Random classifier should be close to baseline (±50%)
        assert 0.5 * baseline < pr_auc < 2 * baseline, \
            f"Random classifier PR-AUC {pr_auc} far from baseline {baseline}"
    
    def test_pr_auc_vs_roc_auc_imbalanced(self, binary_predictions):
        """Test that PR-AUC is more informative than ROC-AUC for imbalanced data."""
        y_true, y_proba = binary_predictions
        
        pr_auc = calculate_pr_auc(y_true, y_proba)
        roc_auc = roc_auc_score(y_true, y_proba)
        
        # For imbalanced data, ROC-AUC is typically higher
        assert roc_auc > pr_auc, \
            f"For imbalanced data, expect ROC-AUC ({roc_auc:.3f}) > PR-AUC ({pr_auc:.3f})"


class TestThresholdTuning:
    """Test suite for threshold optimization."""
    
    def test_optimal_threshold_found(self, binary_predictions):
        """Test that optimal threshold is found."""
        y_true, y_proba = binary_predictions
        
        threshold, metrics = find_optimal_threshold(y_true, y_proba, metric='f1')
        
        assert 0 <= threshold <= 1, f"Threshold should be in [0, 1], got {threshold}"
        assert 'precision' in metrics, "Metrics should include precision"
        assert 'recall' in metrics, "Metrics should include recall"
        assert 'f1' in metrics, "Metrics should include F1"
    
    def test_threshold_not_default(self, binary_predictions):
        """Test that optimal threshold is not default 0.5."""
        y_true, y_proba = binary_predictions
        
        threshold, _ = find_optimal_threshold(y_true, y_proba, metric='f1', min_recall=0.8)
        
        # For imbalanced data, optimal threshold should be lower than 0.5
        assert threshold < 0.5, \
            f"For imbalanced data, expect threshold < 0.5, got {threshold}"
    
    def test_min_recall_constraint(self, binary_predictions):
        """Test that min_recall constraint is respected."""
        y_true, y_proba = binary_predictions
        min_recall = 0.9
        
        threshold, metrics = find_optimal_threshold(
            y_true, y_proba, metric='f1', min_recall=min_recall
        )
        
        assert metrics['recall'] >= min_recall - 0.01, \
            f"Recall {metrics['recall']} should be >= {min_recall}"
    
    def test_f2_favors_recall(self, binary_predictions):
        """Test that F2 metric favors recall over precision."""
        y_true, y_proba = binary_predictions
        
        f1_threshold, f1_metrics = find_optimal_threshold(y_true, y_proba, metric='f1')
        f2_threshold, f2_metrics = find_optimal_threshold(y_true, y_proba, metric='f2')
        
        # F2 should favor higher recall
        assert f2_metrics['recall'] >= f1_metrics['recall'], \
            f"F2 recall ({f2_metrics['recall']}) should be >= F1 recall ({f1_metrics['recall']})"


class TestModelComparison:
    """Test suite for model comparison."""
    
    def test_compare_models_output_format(self, binary_predictions):
        """Test that compare_models returns DataFrame with correct columns."""
        y_true, y_proba = binary_predictions
        
        models = {
            'Model1': y_proba,
            'Model2': y_proba * 0.9,  # Slightly worse
        }
        
        comparison = compare_models(y_true, models, threshold=0.5)
        
        assert isinstance(comparison, pd.DataFrame), "Should return DataFrame"
        assert 'Model' in comparison.columns, "Should have Model column"
        assert 'PR-AUC' in comparison.columns, "Should have PR-AUC column"
        assert 'Precision' in comparison.columns, "Should have Precision column"
        assert 'Recall' in comparison.columns, "Should have Recall column"
    
    def test_compare_models_sorted_by_pr_auc(self, binary_predictions):
        """Test that models are sorted by PR-AUC."""
        y_true, y_proba = binary_predictions
        
        models = {
            'Worse': y_proba * 0.5,
            'Better': y_proba,
            'Best': y_proba * 1.5,
        }
        # Clip to [0, 1]
        models = {k: np.clip(v, 0, 1) for k, v in models.items()}
        
        comparison = compare_models(y_true, models, threshold=0.5)
        
        # First row should have highest PR-AUC
        pr_aucs = comparison['PR-AUC'].values
        assert all(pr_aucs[i] >= pr_aucs[i+1] for i in range(len(pr_aucs)-1)), \
            "Models should be sorted by PR-AUC descending"
    
    def test_confusion_matrix_components(self, binary_predictions):
        """Test that confusion matrix components are included."""
        y_true, y_proba = binary_predictions
        
        models = {'Model': y_proba}
        comparison = compare_models(y_true, models, threshold=0.5)
        
        assert 'TP' in comparison.columns, "Should include True Positives"
        assert 'FP' in comparison.columns, "Should include False Positives"
        assert 'TN' in comparison.columns, "Should include True Negatives"
        assert 'FN' in comparison.columns, "Should include False Negatives"


class TestCostSensitiveEvaluation:
    """Test suite for cost-sensitive evaluation."""
    
    def test_cost_sensitive_profit_calculation(self):
        """Test cost-sensitive profit calculation."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 1, 0])  # 1 FP, 1 FN
        
        results = calculate_cost_sensitive_profit(
            y_true, y_pred,
            fn_cost=1000.0,  # Missing fraud
            fp_cost=100.0    # False alarm
        )
        
        assert results['fn'] == 1, "Should have 1 false negative"
        assert results['fp'] == 1, "Should have 1 false positive"
        assert results['fn_cost'] == 1000.0, "FN cost should be 1000"
        assert results['fp_cost'] == 100.0, "FP cost should be 100"
        assert results['total_cost'] == 1100.0, "Total cost should be 1100"
    
    def test_cost_sensitive_perfect_classifier(self):
        """Test cost for perfect classifier (no FN, no FP)."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 1, 1])
        
        results = calculate_cost_sensitive_profit(
            y_true, y_pred,
            fn_cost=1000.0,
            fp_cost=100.0
        )
        
        assert results['total_cost'] == 0.0, "Perfect classifier should have zero cost"
    
    def test_cost_sensitive_all_fraud_flagged(self):
        """Test cost when all transactions are flagged as fraud."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 1, 1, 1])  # Flag everything
        
        results = calculate_cost_sensitive_profit(
            y_true, y_pred,
            fn_cost=1000.0,
            fp_cost=100.0
        )
        
        assert results['fn'] == 0, "Should have no false negatives"
        assert results['fp'] == 3, "Should have 3 false positives"
        assert results['fn_cost'] == 0.0, "No FN cost"
        assert results['fp_cost'] == 300.0, "FP cost = 3 * 100"
    
    def test_fn_cost_greater_than_fp_cost(self):
        """Test that missing fraud costs more than false alarm."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred_miss_fraud = np.array([0, 0, 0, 0, 1, 1])  # Miss 1 fraud
        y_pred_false_alarm = np.array([1, 0, 0, 1, 1, 1])  # 1 false alarm
        
        cost_miss = calculate_cost_sensitive_profit(
            y_true, y_pred_miss_fraud, fn_cost=1000.0, fp_cost=100.0
        )
        cost_alarm = calculate_cost_sensitive_profit(
            y_true, y_pred_false_alarm, fn_cost=1000.0, fp_cost=100.0
        )
        
        assert cost_miss['total_cost'] > cost_alarm['total_cost'], \
            f"Missing fraud (€{cost_miss['total_cost']}) should cost more than false alarm (€{cost_alarm['total_cost']})"


class TestBusinessROI:
    """Test suite for business ROI calculation."""
    
    def test_roi_vs_no_detection(self):
        """Test ROI compared to no fraud detection."""
        y_true = np.array([0] * 990 + [1] * 10)
        y_pred = np.array([0] * 990 + [1] * 9 + [0] * 1)  # Catch 9/10 frauds
        
        roi = calculate_business_roi(y_true, y_pred, fn_cost=1000.0, fp_cost=100.0)
        
        # No detection = all frauds succeed = 10 * 1000 = 10,000
        assert roi['baseline_no_detection']['cost'] == 10000.0, \
            "No detection baseline should cost 10,000"
        
        # Model catches 9/10, misses 1 = 1 * 1000 = 1,000
        assert roi['model']['net_profit'] < 0, "Model should have negative profit (cost)"
        assert abs(roi['model']['net_profit']) < 10000.0, \
            "Model cost should be less than no detection"
    
    def test_roi_vs_flag_all(self):
        """Test ROI compared to flagging all transactions."""
        y_true = np.array([0] * 990 + [1] * 10)
        y_pred = np.array([0] * 980 + [1] * 20)  # Flag 20 (10 frauds + 10 legit)
        
        roi = calculate_business_roi(y_true, y_pred, fn_cost=1000.0, fp_cost=100.0)
        
        # Flag all = 1000 * 100 = 100,000 investigation cost
        assert roi['baseline_flag_all']['cost'] == 100000.0, \
            "Flag all baseline should cost 100,000"
        
        # Model flags 20, has 10 FP = 10 * 100 = 1,000 cost
        assert roi['model']['total_cost'] < 100000.0, \
            "Model should be cheaper than flag all"
    
    def test_model_positive_savings(self):
        """Test that model provides positive savings vs baselines."""
        y_true = np.array([0] * 990 + [1] * 10)
        y_pred = np.array([0] * 985 + [1] * 15)  # Flag 15, catch all 10 frauds
        
        roi = calculate_business_roi(y_true, y_pred, fn_cost=1000.0, fp_cost=100.0)
        
        # Model should save money vs no detection
        assert roi['baseline_no_detection']['savings'] > 0, \
            "Model should save money vs no detection"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
