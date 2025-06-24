# src/utils.py

import joblib
import yaml
import pandas as pd
import numpy as np
from typing import Dict, Any
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str = 'configs/main_config.yaml') -> Dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    logger.info(f"Configuration loaded from {config_path}")
    return config


def save_model(model: Any, path: str) -> None:
    """
    Save model to disk using joblib.

    Args:
        model: Model object to save
        path: Path to save the model
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"Model saved to {path}")


def load_model(path: str) -> Any:
    """
    Load model from disk using joblib.

    Args:
        path: Path to the saved model

    Returns:
        Loaded model object
    """
    model = joblib.load(path)
    logger.info(f"Model loaded from {path}")
    return model


def ensure_directory_exists(path: str) -> None:
    """
    Ensure that a directory exists, creating it if necessary.

    Args:
        path: Directory path
    """
    os.makedirs(path, exist_ok=True)


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Get list of feature columns from DataFrame.

    Args:
        df: DataFrame containing features and targets

    Returns:
        List of feature column names
    """
    exclude_cols = [
        'account_id', 'trade_date', 'target_pnl', 'target_large_loss',
        'daily_pnl', 'large_loss_threshold', 'date_idx'
    ]

    feature_cols = [col for col in df.columns if col not in exclude_cols]
    return feature_cols


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, metric_type: str = 'regression') -> Dict[str, float]:
    """
    Calculate performance metrics.

    Args:
        y_true: True values
        y_pred: Predicted values
        metric_type: 'regression' or 'classification'

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    if metric_type == 'regression':
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['r2'] = r2_score(y_true, y_pred)

    elif metric_type == 'classification':
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

        # For binary predictions
        y_pred_binary = (y_pred > 0.5).astype(int)

        metrics['accuracy'] = accuracy_score(y_true, y_pred_binary)
        metrics['precision'] = precision_score(y_true, y_pred_binary, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred_binary, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred_binary, zero_division=0)

        # AUC requires probabilities
        if y_pred.min() >= 0 and y_pred.max() <= 1:
            metrics['auc'] = roc_auc_score(y_true, y_pred)

    return metrics


def create_performance_summary(backtest_results: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Create a summary of model performance from backtest results.

    Args:
        backtest_results: DataFrame with backtest predictions
        config: Configuration dictionary

    Returns:
        DataFrame with performance summary
    """
    summary_data = []

    # Overall metrics
    overall_metrics = {
        'level': 'Overall',
        'n_samples': len(backtest_results),
        'var_violation_rate': (backtest_results['true_pnl'] < backtest_results['pred_var']).mean(),
        'expected_violation_rate': config['production_model']['var_model']['alpha']
    }

    # Add loss model AUC if available
    if 'pred_loss_proba' in backtest_results.columns:
        from sklearn.metrics import roc_auc_score
        overall_metrics['loss_auc'] = roc_auc_score(
            backtest_results['true_large_loss'],
            backtest_results['pred_loss_proba']
        )

    summary_data.append(overall_metrics)

    # Per-trader metrics
    for account_id in backtest_results['account_id'].unique():
        trader_data = backtest_results[backtest_results['account_id'] == account_id]

        trader_metrics = {
            'level': f'Trader_{account_id}',
            'n_samples': len(trader_data),
            'var_violation_rate': (trader_data['true_pnl'] < trader_data['pred_var']).mean(),
            'expected_violation_rate': config['production_model']['var_model']['alpha']
        }

        if 'pred_loss_proba' in trader_data.columns and len(trader_data['true_large_loss'].unique()) > 1:
            trader_metrics['loss_auc'] = roc_auc_score(
                trader_data['true_large_loss'],
                trader_data['pred_loss_proba']
            )

        summary_data.append(trader_metrics)

    summary_df = pd.DataFrame(summary_data)
    return summary_df


def validate_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate data quality and return diagnostics.

    Args:
        df: DataFrame to validate

    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'n_rows': len(df),
        'n_columns': len(df.columns),
        'missing_values': {},
        'infinite_values': {},
        'data_types': {},
        'warnings': []
    }

    # Check for missing values
    missing_counts = df.isnull().sum()
    validation_results['missing_values'] = missing_counts[missing_counts > 0].to_dict()

    # Check for infinite values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            validation_results['infinite_values'][col] = inf_count

    # Data types
    validation_results['data_types'] = df.dtypes.astype(str).to_dict()

    # Warnings
    if len(validation_results['missing_values']) > 0:
        validation_results['warnings'].append(
            f"Found missing values in {len(validation_results['missing_values'])} columns"
        )

    if len(validation_results['infinite_values']) > 0:
        validation_results['warnings'].append(
            f"Found infinite values in {len(validation_results['infinite_values'])} columns"
        )

    # Check for duplicate indices
    if df.index.duplicated().any():
        validation_results['warnings'].append("Duplicate indices detected")

    return validation_results
