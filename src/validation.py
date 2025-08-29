"""
Validation Strategy (CLAUDE.md implementation)
Proper time series cross-validation with purging
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from scipy import stats
from typing import Tuple, Dict, Generator
from config import config


def time_series_cv_with_purging(X: pd.DataFrame, y: np.ndarray,
                               n_splits: int = 5, purge_days: int = 5) -> Generator:
    """
    Proper time series cross-validation with purging
    Prevents look-ahead bias
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    for train_idx, test_idx in tscv.split(X):
        # Purge boundary points to prevent leakage
        train_idx = train_idx[:-purge_days]
        test_idx = test_idx[purge_days:]

        yield train_idx, test_idx


def evaluate_model_statistical_significance(model, X: pd.DataFrame, y: np.ndarray,
                                           rules_predictions: Dict[str, float]) -> Dict:
    """
    Test if model beats baseline with statistical significance
    """
    # Simple baseline: restrict after N consecutive losses
    baseline_preds = []
    model_preds = model.predict(X)

    # Convert rules predictions to array format matching X
    trader_ids = X['trader_id'] if 'trader_id' in X.columns else range(len(X))

    for trader_id in trader_ids:
        trader_str = str(trader_id)
        if trader_str in rules_predictions:
            baseline_preds.append(rules_predictions[trader_str]['reduction_pct'])
        else:
            baseline_preds.append(0)

    baseline_preds = np.array(baseline_preds)

    # Handle model predictions (could be dict or array)
    if isinstance(model_preds, dict):
        model_pred_array = []
        for trader_id in trader_ids:
            trader_str = str(trader_id)
            if trader_str in model_preds:
                model_pred_array.append(model_preds[trader_str])
            else:
                model_pred_array.append(0)
        model_preds = np.array(model_pred_array)

    # Calculate improvements
    baseline_errors = np.abs(y - baseline_preds)
    model_errors = np.abs(y - model_preds)

    # Paired t-test (same data, different methods)
    t_stat, p_value = stats.ttest_rel(baseline_errors, model_errors)

    improvement = (baseline_errors.mean() - model_errors.mean()) / (baseline_errors.mean() + 1e-8)

    return {
        'improvement_pct': improvement * 100,
        'p_value': p_value,
        'is_significant': p_value < 0.05,
        'baseline_mae': baseline_errors.mean(),
        'model_mae': model_errors.mean(),
        'meets_threshold': improvement > config.MIN_IMPROVEMENT_FOR_ML
    }


def calculate_risk_metrics(predictions: Dict[str, float], actuals: Dict[str, float],
                          restrictions_applied: Dict[str, bool]) -> Dict:
    """
    Professional risk metrics for model evaluation
    """
    metrics = {}

    # Convert to aligned arrays
    pred_values = []
    actual_values = []
    applied_values = []

    for trader_id in predictions.keys():
        if trader_id in actuals:
            pred_values.append(predictions[trader_id])
            actual_values.append(actuals[trader_id])
            applied_values.append(restrictions_applied.get(trader_id, False))

    pred_array = np.array(pred_values)
    actual_array = np.array(actual_values)
    applied_array = np.array(applied_values)

    # Confusion matrix for restriction decisions
    restricted = pred_array > 20  # 20% reduction threshold
    had_large_loss = actual_array < -2000  # Define "large loss"

    # Key metrics
    if np.any(had_large_loss):
        metrics['hit_rate'] = np.mean(restricted[had_large_loss])  # Caught actual risks
    else:
        metrics['hit_rate'] = 0

    if np.any(~had_large_loss):
        metrics['false_positive_rate'] = np.mean(restricted[~had_large_loss])
    else:
        metrics['false_positive_rate'] = 0

    # Economic metrics (what matters)
    prevented_losses = []
    opportunity_costs = []

    for pred, actual, applied in zip(pred_array, actual_array, applied_array):
        if applied and actual < 0:
            # Prevented part of the loss
            prevented = max(0, abs(actual) - (config.DEFAULT_LIMIT * (1 - pred/100)))
            prevented_losses.append(prevented)
        elif applied and actual > 0:
            # Restricted a winning day
            opportunity_costs.append(actual * (pred/100))

    metrics['total_prevented_losses'] = sum(prevented_losses) if prevented_losses else 0
    metrics['total_opportunity_cost'] = sum(opportunity_costs) if opportunity_costs else 0
    metrics['net_benefit'] = metrics['total_prevented_losses'] - metrics['total_opportunity_cost']

    # Sharpe-like ratio for restrictions
    if len(prevented_losses) > 0:
        benefit_per_restriction = np.array(prevented_losses) - np.mean(opportunity_costs or [0])
        if benefit_per_restriction.std() > 0:
            metrics['restriction_sharpe'] = benefit_per_restriction.mean() / benefit_per_restriction.std()
        else:
            metrics['restriction_sharpe'] = 0
    else:
        metrics['restriction_sharpe'] = 0

    return metrics


def backtest_simple(predictions: Dict[str, Dict], actual_data: pd.DataFrame) -> Dict:
    """
    Did we prevent losses without over-restricting?
    """
    results = {
        'restrictions_applied': 0,
        'losses_prevented': 0,
        'false_restrictions': 0,
        'missed_blowups': 0
    }

    for date in actual_data['date'].unique():
        day_data = actual_data[actual_data['date'] == date]

        for trader_id in day_data['trader_id'].unique():
            trader_str = str(trader_id)
            pred_reduction = predictions.get(trader_str, {}).get('reduction_pct', 0)
            trader_day_data = day_data[day_data['trader_id'] == trader_id]

            if len(trader_day_data) == 0:
                continue

            actual_pnl = trader_day_data['pnl'].sum()

            if pred_reduction > 20:
                results['restrictions_applied'] += 1

                if actual_pnl < -2000:
                    # Good call
                    results['losses_prevented'] += abs(actual_pnl) - 2000
                elif actual_pnl > 0:
                    # Over-restricted
                    results['false_restrictions'] += 1

            elif actual_pnl < -5000:
                # Missed a big loss
                results['missed_blowups'] += 1

    # Simple metrics
    if results['restrictions_applied'] > 0:
        results['precision'] = (
            results['losses_prevented'] /
            (results['losses_prevented'] + results['false_restrictions'] * 1000)
        )
    else:
        results['precision'] = 0

    return results
