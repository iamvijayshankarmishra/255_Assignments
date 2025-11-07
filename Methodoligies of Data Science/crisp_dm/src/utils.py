"""
Utility functions for CRISP-DM Rossmann Sales project.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any
from sklearn.metrics import mean_absolute_error, mean_squared_error


def download_rossmann_data(data_dir: str = 'data/raw') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Download Rossmann Store Sales data from Kaggle.
    
    Args:
        data_dir: Directory to save data
    
    Returns:
        Tuple of (train_df, test_df, store_df)
    """
    import os
    from pathlib import Path
    
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    
    # Check if data already exists
    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join(data_dir, 'test.csv')
    store_path = os.path.join(data_dir, 'store.csv')
    
    if not all([os.path.exists(p) for p in [train_path, test_path, store_path]]):
        print("Downloading Rossmann data from Kaggle...")
        try:
            import kaggle
            kaggle.api.competition_download_files('rossmann-store-sales', path=data_dir)
            
            # Unzip
            import zipfile
            zip_path = os.path.join(data_dir, 'rossmann-store-sales.zip')
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            os.remove(zip_path)
            print("✓ Data downloaded successfully!")
        except Exception as e:
            print(f"❌ Kaggle download failed: {e}")
            print("Please download manually from: https://www.kaggle.com/c/rossmann-store-sales/data")
            raise
    else:
        print("✓ Data already exists locally")
    
    # Load data
    train_df = pd.read_csv(train_path, parse_dates=['Date'])
    test_df = pd.read_csv(test_path, parse_dates=['Date'])
    store_df = pd.read_csv(store_path)
    
    return train_df, test_df, store_df


def rmspe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Square Percentage Error (Kaggle competition metric).
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
    
    Returns:
        RMSPE value
    """
    # Exclude zero sales (store closed)
    mask = y_true != 0
    return np.sqrt(np.mean(((y_true[mask] - y_pred[mask]) / y_true[mask]) ** 2))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Symmetric Mean Absolute Percentage Error.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
    
    Returns:
        sMAPE value (percentage)
    """
    mask = (y_true != 0) | (y_pred != 0)
    return 100 * np.mean(2 * np.abs(y_pred[mask] - y_true[mask]) / 
                         (np.abs(y_true[mask]) + np.abs(y_pred[mask])))


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Weighted Absolute Percentage Error.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
    
    Returns:
        WAPE value (percentage)
    """
    return 100 * np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, 
                   model_name: str = "Model") -> Dict[str, float]:
    """
    Comprehensive evaluation metrics for forecasting.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        model_name: Name of the model
    
    Returns:
        Dictionary of metrics
    """
    results = {
        'Model': model_name,
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'RMSPE': rmspe(y_true, y_pred),
        'sMAPE': smape(y_true, y_pred),
        'WAPE': wape(y_true, y_pred)
    }
    
    return results


def plot_predictions_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, 
                                 dates: pd.Series = None,
                                 title: str = "Predictions vs Actual") -> None:
    """
    Plot predicted vs actual values over time.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        dates: Date index (optional)
        title: Plot title
    """
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    
    x = dates if dates is not None else range(len(y_true))
    
    # Time series plot
    axes[0].plot(x, y_true, label='Actual', alpha=0.7, linewidth=1)
    axes[0].plot(x, y_pred, label='Predicted', alpha=0.7, linewidth=1)
    axes[0].set_title(title)
    axes[0].set_xlabel('Date' if dates is not None else 'Index')
    axes[0].set_ylabel('Sales')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Scatter plot
    axes[1].scatter(y_true, y_pred, alpha=0.3, s=10)
    axes[1].plot([y_true.min(), y_true.max()], 
                 [y_true.min(), y_true.max()], 
                 'r--', lw=2, label='Perfect Prediction')
    axes[1].set_xlabel('Actual Sales')
    axes[1].set_ylabel('Predicted Sales')
    axes[1].set_title('Predicted vs Actual (Scatter)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Plot residual diagnostics.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Residuals over time
    axes[0, 0].plot(residuals, alpha=0.5)
    axes[0, 0].axhline(0, color='r', linestyle='--')
    axes[0, 0].set_title('Residuals Over Time')
    axes[0, 0].set_ylabel('Residual')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Residuals vs predictions
    axes[0, 1].scatter(y_pred, residuals, alpha=0.3, s=10)
    axes[0, 1].axhline(0, color='r', linestyle='--')
    axes[0, 1].set_title('Residuals vs Predictions')
    axes[0, 1].set_xlabel('Predicted Sales')
    axes[0, 1].set_ylabel('Residual')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Residual histogram
    axes[1, 0].hist(residuals, bins=50, edgecolor='black')
    axes[1, 0].set_title('Residual Distribution')
    axes[1, 0].set_xlabel('Residual')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def time_series_train_test_split(df: pd.DataFrame, 
                                  test_size: int = 48,
                                  date_col: str = 'Date') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split time series data maintaining temporal order.
    
    Args:
        df: Dataframe with time series data
        test_size: Number of days in test set
        date_col: Name of date column
    
    Returns:
        Tuple of (train_df, test_df)
    """
    df = df.sort_values(date_col)
    split_date = df[date_col].max() - pd.Timedelta(days=test_size)
    
    train = df[df[date_col] <= split_date].copy()
    test = df[df[date_col] > split_date].copy()
    
    print(f"Train: {train[date_col].min()} to {train[date_col].max()} ({len(train)} rows)")
    print(f"Test:  {test[date_col].min()} to {test[date_col].max()} ({len(test)} rows)")
    
    return train, test


def check_data_leakage(train: pd.DataFrame, test: pd.DataFrame, 
                        date_col: str = 'Date') -> None:
    """
    Verify no temporal overlap between train and test.
    
    Args:
        train: Training dataframe
        test: Test dataframe
        date_col: Name of date column
    """
    train_max = train[date_col].max()
    test_min = test[date_col].min()
    
    if train_max >= test_min:
        raise ValueError(f"❌ DATA LEAKAGE DETECTED: Train max date ({train_max}) >= Test min date ({test_min})")
    else:
        gap_days = (test_min - train_max).days
        print(f"✓ No temporal leakage. Gap between train and test: {gap_days} days")


def log_critique_to_file(phase: str, critique: str, response: str, 
                         output_dir: str = 'prompts/executed') -> None:
    """
    Save critic feedback and response to timestamped file.
    
    Args:
        phase: CRISP-DM phase name
        critique: Critic's questions/concerns
        response: Your response and actions taken
        output_dir: Directory to save logs
    """
    from datetime import datetime
    from pathlib import Path
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/{timestamp}_{phase.lower().replace(' ', '_')}_critique.md"
    
    content = f"""# Critique Log: {phase}

**Timestamp**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## Dr. Foster Provost's Critique

{critique}

---

## My Response & Actions Taken

{response}

---

## Status
- [ ] Addressed all concerns
- [ ] Updated code/documentation
- [ ] Ready to proceed to next phase
"""
    
    with open(filename, 'w') as f:
        f.write(content)
    
    print(f"✓ Critique logged to: {filename}")
