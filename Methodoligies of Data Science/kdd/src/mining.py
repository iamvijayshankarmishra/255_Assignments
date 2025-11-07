"""
KDD Phase 4: Data Mining
=========================
Model training with focus on imbalanced learning and fraud detection.

Models:
- Isolation Forest (unsupervised anomaly detection)
- Random Forest (with class_weight)
- XGBoost (with scale_pos_weight)
- LightGBM (with class_weight)

Evaluation:
- PR-AUC (primary metric for imbalanced data)
- ROC-AUC (secondary metric)
- Precision-Recall curves
- Threshold tuning for business constraints
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    roc_auc_score, roc_curve, confusion_matrix,
    classification_report, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns


def train_isolation_forest(X_train: pd.DataFrame,
                           contamination: float = 0.002,
                           random_state: int = 42) -> IsolationForest:
    """
    Train Isolation Forest for anomaly detection.
    
    Unsupervised method that doesn't use labels during training.
    Good baseline for fraud detection.
    
    Args:
        X_train: Training features (ignores Class labels)
        contamination: Expected fraud rate (default 0.002 = 0.2%)
        random_state: Random seed
    
    Returns:
        Fitted Isolation Forest model
    """
    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100,
        n_jobs=-1
    )
    
    model.fit(X_train)
    
    print(f"✅ Isolation Forest trained (contamination={contamination})")
    
    return model


def train_random_forest(X_train: pd.DataFrame,
                       y_train: pd.Series,
                       class_weight: str = 'balanced',
                       random_state: int = 42) -> RandomForestClassifier:
    """
    Train Random Forest with class weighting.
    
    Args:
        X_train: Training features
        y_train: Training labels
        class_weight: 'balanced' or dict of class weights
        random_state: Random seed
    
    Returns:
        Fitted Random Forest model
    """
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    print(f"✅ Random Forest trained (class_weight={class_weight})")
    
    return model


def train_xgboost(X_train: pd.DataFrame,
                 y_train: pd.Series,
                 scale_pos_weight: float = None,
                 random_state: int = 42):
    """
    Train XGBoost with scale_pos_weight for imbalance.
    
    Args:
        X_train: Training features
        y_train: Training labels
        scale_pos_weight: Weight for positive class (if None, auto-compute)
        random_state: Random seed
    
    Returns:
        Fitted XGBoost model
    """
    import xgboost as xgb
    
    # Auto-compute scale_pos_weight if not provided
    if scale_pos_weight is None:
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        scale_pos_weight = n_neg / n_pos
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=-1,
        eval_metric='aucpr'
    )
    
    model.fit(X_train, y_train)
    
    print(f"✅ XGBoost trained (scale_pos_weight={scale_pos_weight:.1f})")
    
    return model


def train_lightgbm(X_train: pd.DataFrame,
                  y_train: pd.Series,
                  class_weight: str = 'balanced',
                  random_state: int = 42):
    """
    Train LightGBM with class weighting.
    
    Args:
        X_train: Training features
        y_train: Training labels
        class_weight: 'balanced' or None
        random_state: Random seed
    
    Returns:
        Fitted LightGBM model
    """
    import lightgbm as lgb
    
    model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1
    )
    
    model.fit(X_train, y_train)
    
    print(f"✅ LightGBM trained (class_weight={class_weight})")
    
    return model


def calculate_pr_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Calculate Precision-Recall AUC (primary metric for imbalanced data).
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities for positive class
    
    Returns:
        PR-AUC score
    """
    return average_precision_score(y_true, y_proba)


def plot_pr_curve(y_true: np.ndarray,
                 y_proba_dict: Dict[str, np.ndarray],
                 figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Plot Precision-Recall curves for multiple models.
    
    Args:
        y_true: True labels
        y_proba_dict: Dict of {model_name: predicted_probabilities}
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    # Baseline (random classifier)
    baseline = y_true.mean()
    plt.axhline(baseline, color='gray', linestyle='--', 
                label=f'Baseline (Random): {baseline:.3f}')
    
    # Plot each model
    for model_name, y_proba in y_proba_dict.items():
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = average_precision_score(y_true, y_proba)
        
        plt.plot(recall, precision, label=f'{model_name} (PR-AUC: {pr_auc:.3f})')
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves (PRIMARY METRIC)', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_roc_curve(y_true: np.ndarray,
                  y_proba_dict: Dict[str, np.ndarray],
                  figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Plot ROC curves for multiple models (secondary metric).
    
    Args:
        y_true: True labels
        y_proba_dict: Dict of {model_name: predicted_probabilities}
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    # Diagonal (random classifier)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random')
    
    # Plot each model
    for model_name, y_proba in y_proba_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = roc_auc_score(y_true, y_proba)
        
        plt.plot(fpr, tpr, label=f'{model_name} (ROC-AUC: {roc_auc:.3f})')
    
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves (Secondary Metric)', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def find_optimal_threshold(y_true: np.ndarray,
                          y_proba: np.ndarray,
                          metric: str = 'f1',
                          min_recall: float = 0.9) -> Tuple[float, Dict]:
    """
    Find optimal classification threshold.
    
    For fraud detection, we often want:
    - High recall (catch most frauds)
    - Acceptable precision (minimize false alarms)
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        metric: 'f1', 'precision', 'recall', 'f2' (favors recall)
        min_recall: Minimum acceptable recall (e.g., 0.9 = catch 90% of frauds)
    
    Returns:
        optimal_threshold, metrics_at_threshold
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    
    # Remove last element (precision and recall have one extra element)
    precision = precision[:-1]
    recall = recall[:-1]
    
    if metric == 'f1':
        # F1 score
        with np.errstate(divide='ignore', invalid='ignore'):
            f1_scores = 2 * (precision * recall) / (precision + recall)
            f1_scores = np.nan_to_num(f1_scores)
        
        # Filter by minimum recall
        valid_idx = recall >= min_recall
        if valid_idx.sum() == 0:
            print(f"⚠️ No threshold achieves recall>={min_recall}")
            valid_idx = np.ones(len(recall), dtype=bool)
        
        best_idx = np.argmax(f1_scores[valid_idx])
        optimal_threshold = thresholds[valid_idx][best_idx]
        
        metrics = {
            'threshold': optimal_threshold,
            'precision': precision[valid_idx][best_idx],
            'recall': recall[valid_idx][best_idx],
            'f1': f1_scores[valid_idx][best_idx]
        }
    
    elif metric == 'f2':
        # F2 score (favors recall over precision)
        beta = 2
        with np.errstate(divide='ignore', invalid='ignore'):
            f2_scores = (1 + beta**2) * (precision * recall) / \
                       (beta**2 * precision + recall)
            f2_scores = np.nan_to_num(f2_scores)
        
        valid_idx = recall >= min_recall
        if valid_idx.sum() == 0:
            valid_idx = np.ones(len(recall), dtype=bool)
        
        best_idx = np.argmax(f2_scores[valid_idx])
        optimal_threshold = thresholds[valid_idx][best_idx]
        
        metrics = {
            'threshold': optimal_threshold,
            'precision': precision[valid_idx][best_idx],
            'recall': recall[valid_idx][best_idx],
            'f2': f2_scores[valid_idx][best_idx]
        }
    
    return optimal_threshold, metrics


def compare_models(y_true: np.ndarray,
                  y_proba_dict: Dict[str, np.ndarray],
                  threshold: float = 0.5) -> pd.DataFrame:
    """
    Compare model performance with multiple metrics.
    
    Args:
        y_true: True labels
        y_proba_dict: Dict of {model_name: predicted_probabilities}
        threshold: Classification threshold (default 0.5)
    
    Returns:
        DataFrame with comparison results
    """
    results = []
    
    for model_name, y_proba in y_proba_dict.items():
        y_pred = (y_proba >= threshold).astype(int)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Metrics
        pr_auc = average_precision_score(y_true, y_proba)
        roc_auc = roc_auc_score(y_true, y_proba)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results.append({
            'Model': model_name,
            'PR-AUC': pr_auc,
            'ROC-AUC': roc_auc,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'TP': tp,
            'FP': fp,
            'TN': tn,
            'FN': fn,
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('PR-AUC', ascending=False)
    
    return results_df


def plot_feature_importance(model,
                           feature_names: List[str],
                           top_n: int = 20,
                           figsize: Tuple[int, int] = (10, 8)) -> None:
    """
    Plot feature importance for tree-based models.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        top_n: Number of top features to show
        figsize: Figure size
    """
    if not hasattr(model, 'feature_importances_'):
        print("⚠️ Model doesn't have feature_importances_ attribute")
        return
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    })
    importance_df = importance_df.sort_values('Importance', ascending=False).head(top_n)
    
    plt.figure(figsize=figsize)
    plt.barh(range(len(importance_df)), importance_df['Importance'], color='#3498db')
    plt.yticks(range(len(importance_df)), importance_df['Feature'])
    plt.xlabel('Importance', fontsize=12)
    plt.title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()
