"""
KDD Phase 5: Interpretation & Evaluation
=========================================
Cost-sensitive evaluation and business impact analysis for fraud detection.

Key Metrics:
- Cost-Sensitive Profit: Consider FN cost >> FP cost
- Business ROI vs baseline
- Confusion matrix at optimal threshold
- Pattern discovery in frauds
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_cost_sensitive_profit(y_true: np.ndarray,
                                    y_pred: np.ndarray,
                                    fn_cost: float = 1000.0,
                                    fp_cost: float = 100.0,
                                    tp_gain: float = 0.0,
                                    tn_gain: float = 0.0) -> Dict:
    """
    Calculate profit/loss with cost-sensitive analysis.
    
    Fraud Detection Costs:
    - False Negative (missed fraud): ~€1000 (fraud loss)
    - False Positive (false alarm): ~€100 (investigation cost)
    - True Positive (caught fraud): €0 (prevented loss, no gain)
    - True Negative (correctly identified legit): €0
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        fn_cost: Cost of missing a fraud
        fp_cost: Cost of false alarm
        tp_gain: Gain from catching fraud (usually 0)
        tn_gain: Gain from correct rejection (usually 0)
    
    Returns:
        Dict with cost breakdown
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate costs
    fn_total_cost = fn * fn_cost
    fp_total_cost = fp * fp_cost
    tp_total_gain = tp * tp_gain
    tn_total_gain = tn * tn_gain
    
    total_cost = fn_total_cost + fp_total_cost
    total_gain = tp_total_gain + tn_total_gain
    net_profit = total_gain - total_cost
    
    return {
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp,
        'fn_cost': fn_total_cost,
        'fp_cost': fp_total_cost,
        'tp_gain': tp_total_gain,
        'tn_gain': tn_total_gain,
        'total_cost': total_cost,
        'total_gain': total_gain,
        'net_profit': net_profit,
    }


def compare_cost_sensitive_models(y_true: np.ndarray,
                                 y_pred_dict: Dict[str, np.ndarray],
                                 fn_cost: float = 1000.0,
                                 fp_cost: float = 100.0) -> pd.DataFrame:
    """
    Compare models using cost-sensitive profit.
    
    Args:
        y_true: True labels
        y_pred_dict: Dict of {model_name: predictions}
        fn_cost: Cost of missing fraud
        fp_cost: Cost of false alarm
    
    Returns:
        DataFrame with cost comparison
    """
    results = []
    
    for model_name, y_pred in y_pred_dict.items():
        metrics = calculate_cost_sensitive_profit(
            y_true, y_pred, fn_cost, fp_cost
        )
        
        results.append({
            'Model': model_name,
            'Net_Profit': metrics['net_profit'],
            'Total_Cost': metrics['total_cost'],
            'FN_Cost': metrics['fn_cost'],
            'FP_Cost': metrics['fp_cost'],
            'TP': metrics['tp'],
            'FP': metrics['fp'],
            'TN': metrics['tn'],
            'FN': metrics['fn'],
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Net_Profit', ascending=False)
    
    return results_df


def calculate_business_roi(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          fn_cost: float = 1000.0,
                          fp_cost: float = 100.0) -> Dict:
    """
    Calculate ROI compared to baselines.
    
    Baselines:
    - No fraud detection: All frauds succeed (max loss)
    - Flag all: All transactions flagged (max investigation cost)
    
    Args:
        y_true: True labels
        y_pred: Model predictions
        fn_cost: Cost per missed fraud
        fp_cost: Cost per false alarm
    
    Returns:
        Dict with ROI comparison
    """
    n_frauds = y_true.sum()
    n_legit = len(y_true) - n_frauds
    
    # Model performance
    model_metrics = calculate_cost_sensitive_profit(y_true, y_pred, fn_cost, fp_cost)
    
    # Baseline 1: No fraud detection (all frauds succeed)
    no_detection_cost = n_frauds * fn_cost
    
    # Baseline 2: Flag all transactions (investigate everything)
    flag_all_cost = len(y_true) * fp_cost
    
    # Savings vs baselines
    savings_vs_no_detection = no_detection_cost + model_metrics['net_profit']
    savings_vs_flag_all = flag_all_cost + model_metrics['net_profit']
    
    return {
        'model': {
            'net_profit': model_metrics['net_profit'],
            'total_cost': model_metrics['total_cost'],
            'tp': model_metrics['tp'],
            'fp': model_metrics['fp'],
            'fn': model_metrics['fn'],
        },
        'baseline_no_detection': {
            'cost': no_detection_cost,
            'savings': savings_vs_no_detection,
        },
        'baseline_flag_all': {
            'cost': flag_all_cost,
            'savings': savings_vs_flag_all,
        }
    }


def plot_cost_sensitivity_analysis(y_true: np.ndarray,
                                   y_proba: np.ndarray,
                                   fn_cost_range: Tuple[float, float] = (500, 2000),
                                   fp_cost: float = 100.0,
                                   figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Analyze how optimal threshold changes with FN cost.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        fn_cost_range: Range of FN costs to test (min, max)
        fp_cost: Fixed FP cost
        figsize: Figure size
    """
    from sklearn.metrics import precision_recall_curve
    
    fn_costs = np.linspace(fn_cost_range[0], fn_cost_range[1], 20)
    optimal_thresholds = []
    max_profits = []
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    precision = precision[:-1]
    recall = recall[:-1]
    
    for fn_cost in fn_costs:
        best_profit = -np.inf
        best_threshold = 0.5
        
        for i, thresh in enumerate(thresholds):
            y_pred = (y_proba >= thresh).astype(int)
            metrics = calculate_cost_sensitive_profit(y_true, y_pred, fn_cost, fp_cost)
            
            if metrics['net_profit'] > best_profit:
                best_profit = metrics['net_profit']
                best_threshold = thresh
        
        optimal_thresholds.append(best_threshold)
        max_profits.append(best_profit)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Optimal threshold vs FN cost
    axes[0].plot(fn_costs, optimal_thresholds, marker='o', color='#3498db')
    axes[0].set_xlabel('False Negative Cost (€)', fontsize=12)
    axes[0].set_ylabel('Optimal Threshold', fontsize=12)
    axes[0].set_title('Optimal Threshold vs FN Cost')
    axes[0].grid(True, alpha=0.3)
    
    # Max profit vs FN cost
    axes[1].plot(fn_costs, max_profits, marker='o', color='#2ecc71')
    axes[1].set_xlabel('False Negative Cost (€)', fontsize=12)
    axes[1].set_ylabel('Maximum Profit (€)', fontsize=12)
    axes[1].set_title('Achievable Profit vs FN Cost')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true: np.ndarray,
                         y_pred: np.ndarray,
                         labels: list = ['Legitimate', 'Fraud'],
                         figsize: Tuple[int, int] = (8, 6)) -> None:
    """
    Plot confusion matrix with business interpretation.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels
        figsize: Figure size
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'})
    
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    # Add business interpretation
    tn, fp, fn, tp = cm.ravel()
    
    plt.text(0.5, -0.15, 
             f'TN={tn:,} (✅ Correct), FP={fp:,} (💰 Investigation cost)\n'
             f'FN={fn:,} (💸 Fraud loss), TP={tp:,} (✅ Fraud caught)',
             ha='center', va='top', transform=plt.gca().transAxes,
             fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()


def discover_fraud_patterns(df: pd.DataFrame,
                           fraud_col: str = 'Class',
                           features: list = None,
                           top_n: int = 10) -> pd.DataFrame:
    """
    Discover characteristic patterns in fraud transactions.
    
    Args:
        df: Dataset with fraud_col
        fraud_col: Column name for fraud indicator
        features: List of features to analyze (if None, use all)
        top_n: Number of top distinguishing features
    
    Returns:
        DataFrame with fraud vs legitimate statistics
    """
    from scipy.stats import mannwhitneyu
    
    if features is None:
        features = [col for col in df.columns if col != fraud_col]
    
    results = []
    
    for feature in features:
        fraud = df[df[fraud_col] == 1][feature]
        legit = df[df[fraud_col] == 0][feature]
        
        # Statistical test
        stat, p_value = mannwhitneyu(fraud, legit, alternative='two-sided')
        
        results.append({
            'Feature': feature,
            'Fraud_Mean': fraud.mean(),
            'Legit_Mean': legit.mean(),
            'Fraud_Median': fraud.median(),
            'Legit_Median': legit.median(),
            'Mean_Ratio': fraud.mean() / legit.mean() if legit.mean() != 0 else np.inf,
            'P_Value': p_value,
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('P_Value').head(top_n)
    
    print(f"\n🔍 Top {top_n} Features Distinguishing Fraud:")
    print(results_df.to_string(index=False))
    
    return results_df


def generate_model_card(model_name: str,
                       metrics: Dict,
                       dataset_info: Dict,
                       limitations: list,
                       use_cases: list) -> str:
    """
    Generate model card documentation.
    
    Args:
        model_name: Name of the model
        metrics: Dict with performance metrics
        dataset_info: Dict with dataset information
        limitations: List of known limitations
        use_cases: List of intended use cases
    
    Returns:
        Formatted model card string
    """
    card = f"""
# Model Card: {model_name}

## Model Details
- **Model Type**: {model_name}
- **Task**: Credit Card Fraud Detection
- **Date**: {pd.Timestamp.now().strftime('%Y-%m-%d')}

## Intended Use
### Primary Use Cases
{chr(10).join(f'- {use_case}' for use_case in use_cases)}

### Out-of-Scope Uses
- High-stakes decisions without human review
- Deployment without monitoring
- Use on non-credit-card transactions

## Training Data
- **Dataset**: {dataset_info.get('name', 'Credit Card Fraud Detection')}
- **Size**: {dataset_info.get('size', 'N/A')} transactions
- **Fraud Rate**: {dataset_info.get('fraud_rate', 'N/A')}%
- **Time Period**: {dataset_info.get('time_period', '48 hours')}
- **Features**: {dataset_info.get('n_features', 'N/A')} (V1-V28 PCA, Time, Amount)

## Performance Metrics
- **PR-AUC**: {metrics.get('pr_auc', 'N/A'):.3f}
- **ROC-AUC**: {metrics.get('roc_auc', 'N/A'):.3f}
- **Precision**: {metrics.get('precision', 'N/A'):.3f}
- **Recall**: {metrics.get('recall', 'N/A'):.3f}
- **F1-Score**: {metrics.get('f1', 'N/A'):.3f}

## Limitations
{chr(10).join(f'- {limitation}' for limitation in limitations)}

## Ethical Considerations
- **Fairness**: PCA features prevent demographic fairness analysis
- **Privacy**: Model uses anonymized features only
- **Bias**: Temporal bias possible (trained on 48-hour window)

## Monitoring Plan
- Track PR-AUC weekly (alert if drops below {metrics.get('pr_auc', 0) * 0.9:.3f})
- Monitor false positive rate (investigation cost)
- Monitor false negative rate (fraud loss)
- Retrain if concept drift detected

## Contact
- Maintainer: Data Science Team
- Last Updated: {pd.Timestamp.now().strftime('%Y-%m-%d')}
"""
    
    return card
