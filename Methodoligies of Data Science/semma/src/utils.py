"""
SEMMA Utilities Module
======================
Helper functions for data loading, statistical tests, visualization, and logging.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime


def download_bank_marketing_data(data_dir: str = "../data") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download Bank Marketing dataset from UCI ML Repository.
    
    Args:
        data_dir: Directory to save data
        
    Returns:
        Tuple of (data_df, metadata_dict)
    """
    import urllib.request
    import zipfile
    
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
    zip_path = data_path / "bank-additional.zip"
    
    print(f"Downloading Bank Marketing dataset from {url}...")
    urllib.request.urlretrieve(url, zip_path)
    
    print(f"Extracting files...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_path)
    
    # Load the full dataset (bank-additional-full.csv)
    df = pd.read_csv(data_path / "bank-additional" / "bank-additional-full.csv", sep=';')
    
    print(f"✓ Loaded {len(df)} records, {df.shape[1]} features")
    print(f"  Target distribution: {df['y'].value_counts(normalize=True).to_dict()}")
    
    metadata = {
        'source': 'UCI ML Repository',
        'url': url,
        'records': len(df),
        'features': df.shape[1],
        'target': 'y',
        'positive_class': (df['y'] == 'yes').sum(),
        'negative_class': (df['y'] == 'no').sum(),
        'download_date': datetime.now().isoformat()
    }
    
    return df, metadata


def statistical_profile(df: pd.DataFrame, target_col: str = 'y') -> Dict:
    """
    Generate comprehensive statistical profile of dataset.
    
    Args:
        df: Input dataframe
        target_col: Name of target column
        
    Returns:
        Dictionary with statistical tests and summaries
    """
    from scipy import stats
    
    profile = {}
    
    # Continuous features
    continuous_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in continuous_cols:
        continuous_cols.remove(target_col)
    
    profile['continuous'] = {}
    for col in continuous_cols:
        # Normality test (Shapiro-Wilk)
        stat, p_value = stats.shapiro(df[col].dropna().sample(min(5000, len(df[col].dropna()))))
        
        # Descriptive stats
        profile['continuous'][col] = {
            'mean': df[col].mean(),
            'std': df[col].std(),
            'median': df[col].median(),
            'min': df[col].min(),
            'max': df[col].max(),
            'skewness': df[col].skew(),
            'kurtosis': df[col].kurt(),
            'shapiro_p_value': p_value,
            'is_normal': p_value > 0.05
        }
        
        # Test association with target (t-test or Mann-Whitney U)
        if target_col in df.columns:
            group1 = df[df[target_col] == df[target_col].unique()[0]][col].dropna()
            group2 = df[df[target_col] == df[target_col].unique()[1]][col].dropna()
            
            # T-test
            t_stat, t_p = stats.ttest_ind(group1, group2)
            profile['continuous'][col]['ttest_p_value'] = t_p
            profile['continuous'][col]['is_significant'] = t_p < 0.05
            
            # Mann-Whitney U (non-parametric)
            u_stat, u_p = stats.mannwhitneyu(group1, group2)
            profile['continuous'][col]['mannwhitney_p_value'] = u_p
    
    # Categorical features
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if target_col in categorical_cols and target_col != target_col:
        categorical_cols.remove(target_col)
    
    profile['categorical'] = {}
    for col in categorical_cols:
        if col == target_col:
            continue
            
        # Frequency table
        freq = df[col].value_counts()
        profile['categorical'][col] = {
            'unique_values': df[col].nunique(),
            'mode': df[col].mode()[0],
            'mode_frequency': freq.iloc[0] / len(df),
            'value_counts': freq.to_dict()
        }
        
        # Chi-squared test with target
        if target_col in df.columns:
            contingency = pd.crosstab(df[col], df[target_col])
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
            
            # Cramér's V (effect size)
            n = contingency.sum().sum()
            cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))
            
            profile['categorical'][col]['chi2_p_value'] = p_value
            profile['categorical'][col]['is_significant'] = p_value < 0.05
            profile['categorical'][col]['cramers_v'] = cramers_v
            profile['categorical'][col]['effect_size'] = 'small' if cramers_v < 0.1 else ('medium' if cramers_v < 0.3 else 'large')
    
    return profile


def plot_lift_chart(y_true, y_proba, n_bins=10):
    """
    Plot lift chart for classification model.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        n_bins: Number of bins (default 10)
    """
    # Create dataframe
    df = pd.DataFrame({'y_true': y_true, 'y_proba': y_proba})
    df = df.sort_values('y_proba', ascending=False)
    
    # Create bins
    df['decile'] = pd.qcut(df['y_proba'], n_bins, labels=False, duplicates='drop')
    
    # Calculate lift
    lift_data = df.groupby('decile').agg({
        'y_true': ['sum', 'count']
    })
    lift_data.columns = ['positives', 'total']
    lift_data['response_rate'] = lift_data['positives'] / lift_data['total']
    lift_data['cumulative_positives'] = lift_data['positives'].cumsum()
    lift_data['cumulative_total'] = lift_data['total'].cumsum()
    lift_data['cumulative_response_rate'] = lift_data['cumulative_positives'] / lift_data['cumulative_total']
    
    baseline_rate = y_true.sum() / len(y_true)
    lift_data['lift'] = lift_data['response_rate'] / baseline_rate
    lift_data['cumulative_lift'] = lift_data['cumulative_response_rate'] / baseline_rate
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Decile-wise lift
    axes[0].bar(lift_data.index, lift_data['lift'], color='steelblue', alpha=0.7)
    axes[0].axhline(1.0, color='red', linestyle='--', label='Baseline')
    axes[0].set_xlabel('Decile (0=highest probability)')
    axes[0].set_ylabel('Lift')
    axes[0].set_title('Lift Chart by Decile')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Cumulative lift
    axes[1].plot(range(1, len(lift_data) + 1), lift_data['cumulative_lift'], marker='o', color='darkgreen')
    axes[1].axhline(1.0, color='red', linestyle='--', label='Baseline')
    axes[1].set_xlabel('Top N Deciles')
    axes[1].set_ylabel('Cumulative Lift')
    axes[1].set_title('Cumulative Lift Chart')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return lift_data


def plot_calibration_curve(y_true, y_proba, n_bins=10):
    """
    Plot calibration (reliability) diagram.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        n_bins: Number of bins (default 10)
    """
    from sklearn.calibration import calibration_curve
    
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy='uniform')
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.plot(mean_predicted_value, fraction_of_positives, marker='s', label='Model', color='steelblue', linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated', color='red', linewidth=1.5)
    
    ax.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax.set_ylabel('Fraction of Positives', fontsize=12)
    ax.set_title('Calibration (Reliability) Diagram', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Calculate Brier score
    from sklearn.metrics import brier_score_loss
    brier = brier_score_loss(y_true, y_proba)
    print(f"Brier Score: {brier:.4f} (lower is better, <0.10 is good)")
    
    return brier


def compute_business_roi(y_true, y_pred, cost_per_call=5, revenue_per_sub=200):
    """
    Calculate business ROI for campaign.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        cost_per_call: Cost per call (€)
        revenue_per_sub: Revenue per subscription (€)
        
    Returns:
        Dictionary with ROI metrics
    """
    from sklearn.metrics import confusion_matrix
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Costs
    total_calls = tp + fp
    total_cost = total_calls * cost_per_call
    
    # Revenue
    total_revenue = tp * revenue_per_sub
    
    # Profit
    profit = total_revenue - total_cost
    roi = (profit / total_cost) * 100 if total_cost > 0 else 0
    
    # Baseline (call everyone)
    baseline_calls = len(y_true)
    baseline_cost = baseline_calls * cost_per_call
    baseline_revenue = y_true.sum() * revenue_per_sub
    baseline_profit = baseline_revenue - baseline_cost
    baseline_roi = (baseline_profit / baseline_cost) * 100 if baseline_cost > 0 else 0
    
    # Random (call proportionally)
    random_calls = baseline_calls * 0.2  # Assume 20% calling capacity
    random_cost = random_calls * cost_per_call
    random_tp = y_true.sum() * 0.2
    random_revenue = random_tp * revenue_per_sub
    random_profit = random_revenue - random_cost
    random_roi = (random_profit / random_cost) * 100 if random_cost > 0 else 0
    
    return {
        'model': {
            'calls': total_calls,
            'true_positives': tp,
            'false_positives': fp,
            'precision': tp / total_calls if total_calls > 0 else 0,
            'cost': total_cost,
            'revenue': total_revenue,
            'profit': profit,
            'roi': roi
        },
        'baseline': {
            'calls': baseline_calls,
            'cost': baseline_cost,
            'revenue': baseline_revenue,
            'profit': baseline_profit,
            'roi': baseline_roi
        },
        'random': {
            'calls': random_calls,
            'cost': random_cost,
            'revenue': random_revenue,
            'profit': random_profit,
            'roi': random_roi
        },
        'improvement_vs_random': roi - random_roi
    }


def log_critique_to_file(phase: str, critique: str, response: str, output_dir: str):
    """
    Log critic checkpoint to file.
    
    Args:
        phase: Phase name (e.g., "Sample")
        critique: Critique text
        response: Response text
        output_dir: Directory to save log
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_path / f"{timestamp}_{phase.lower().replace(' ', '_')}_critique.md"
    
    content = f"""# {phase} Phase Critique
    
**Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Dr. Raymond Hettinger's Critique

{critique}

## Our Response

{response}

---
*Logged automatically from SEMMA.ipynb*
"""
    
    with open(filename, 'w') as f:
        f.write(content)
    
    print(f"✓ Critique saved to {filename}")
