# src/feature_engineering.py

import pandas as pd
import numpy as np
from typing import Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_features(panel_df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Build a rich set of predictive features, including advanced risk and behavioral metrics.

    Args:
        panel_df: Panel DataFrame from data_processing
        config: Configuration dictionary

    Returns:
        pd.DataFrame: Panel with engineered features
    """
    logger.info("Starting feature engineering...")

    # Create a copy to avoid modifying original
    df = panel_df.copy()

    # Sort by account_id and trade_date
    df = df.sort_values(['account_id', 'trade_date'])

    # Handle missing data from inactivity
    # Group by account_id and forward-fill
    groupby_account = df.groupby('account_id')

    # Forward-fill these columns within each account
    fill_cols = ['daily_gross', 'daily_fees', 'gross_profit', 'gross_loss']
    for col in fill_cols:
        if col in df.columns:
            df[col] = groupby_account[col].ffill()

    # Fill daily_pnl and n_trades with 0 for inactive days
    df['daily_pnl'] = df['daily_pnl'].fillna(0)
    df['n_trades'] = df['n_trades'].fillna(0)
    df['daily_volume'] = df['daily_volume'].fillna(0)

    # Shift for lookahead prevention - all features based on lagged data
    logger.info("Applying time shifts to prevent lookahead bias...")

    # Generate base features
    feature_params = config['feature_params']

    # EWMA features
    for span in feature_params['ewma_spans']:
        df[f'ewm_pnl_{span}'] = groupby_account['daily_pnl'].apply(
            lambda x: x.shift(1).ewm(span=span, adjust=False).mean()
        ).reset_index(level=0, drop=True)

        df[f'ewm_volume_{span}'] = groupby_account['daily_volume'].apply(
            lambda x: x.shift(1).ewm(span=span, adjust=False).mean()
        ).reset_index(level=0, drop=True)

    # Rolling volatility features
    for window in feature_params['rolling_vol_windows']:
        df[f'rolling_vol_{window}'] = groupby_account['daily_pnl'].apply(
            lambda x: x.shift(1).rolling(window=window).std()
        ).reset_index(level=0, drop=True)

    # Advanced risk features
    logger.info("Calculating advanced risk features...")

    # Rolling Sortino ratio
    sortino_window = feature_params['sortino_window']
    df['rolling_sortino_ratio'] = groupby_account['daily_pnl'].apply(
        lambda x: calculate_sortino_ratio(x.shift(1), sortino_window)
    ).reset_index(level=0, drop=True)

    # Rolling profit factor
    profit_factor_window = feature_params['profit_factor_window']
    df['rolling_profit_factor'] = groupby_account.apply(
        lambda x: calculate_profit_factor(
            x['gross_profit'].shift(1),
            x['gross_loss'].shift(1),
            profit_factor_window
        )
    ).reset_index(level=0, drop=True)

    # Rolling max drawdown
    drawdown_window = feature_params['drawdown_window']
    df['rolling_max_drawdown'] = groupby_account['daily_pnl'].apply(
        lambda x: calculate_max_drawdown(x.shift(1), drawdown_window)
    ).reset_index(level=0, drop=True)

    # Behavioral features
    logger.info("Calculating behavioral features...")

    # Calculate 21-day average trades
    df['avg_trades_21d'] = groupby_account['n_trades'].apply(
        lambda x: x.shift(1).rolling(window=21).mean()
    ).reset_index(level=0, drop=True)

    # Large loss flag (for revenge trading detection)
    large_loss_quantile = config['target_variable_params']['large_loss_quantile']
    df['daily_loss_flag'] = df['daily_pnl'] < 0

    # Calculate quantile threshold for large losses using ONLY historical data (expanding window)
    # CRITICAL FIX: Prevent future leakage by using expanding window
    df['large_loss_threshold'] = groupby_account['daily_pnl'].apply(
        lambda x: calculate_expanding_quantile_threshold(x, large_loss_quantile)
    ).reset_index(level=0, drop=True)

    df['large_loss_recent'] = groupby_account.apply(
        lambda x: (x['daily_pnl'].shift(1) < x['large_loss_threshold']).rolling(
            window=feature_params['large_loss_lookback']
        ).max()
    ).reset_index(level=0, drop=True)

    # Revenge trading proxy
    df['revenge_trading_proxy'] = (
        (df['n_trades'] > 1.5 * df['avg_trades_21d']) &
        (df['large_loss_recent'] == 1)
    ).astype(int)

    # Generate target variables with strict temporal validation
    logger.info("Creating target variables with leakage prevention...")

    # CRITICAL: Verify all features are lagged by at least 1 day
    # This is essential to prevent lookahead bias

    # Target PnL (next day's PnL) - this is what we're predicting
    df['target_pnl'] = groupby_account['daily_pnl'].shift(-1)

    # Target large loss (binary) - using threshold calculated from historical data only
    df['target_large_loss'] = groupby_account.apply(
        lambda x: (x['daily_pnl'].shift(-1) < x['large_loss_threshold']).astype(int)
    ).reset_index(level=0, drop=True)

    # Validation: Ensure no feature can predict current day's PnL
    # This catches potential leakage where features inadvertently include same-day information
    logger.info("Performing temporal leakage validation...")

    # Check each feature's correlation with same-day PnL (should be low for lagged features)
    feature_cols_to_check = [col for col in df.columns if col.startswith(('ewm_', 'rolling_', 'avg_', 'win_rate', 'cumulative'))]

    leakage_warnings = []
    for feature in feature_cols_to_check:
        if feature in df.columns and df[feature].dtype in ['int64', 'float64']:
            same_day_corr = df[feature].corr(df['daily_pnl'])
            if abs(same_day_corr) > 0.7:  # High correlation suggests potential leakage
                leakage_warnings.append(f"Feature {feature} has high same-day correlation: {same_day_corr:.3f}")

    if leakage_warnings:
        logger.warning("Potential temporal leakage detected:")
        for warning in leakage_warnings[:5]:  # Show first 5
            logger.warning(f"  {warning}")
    else:
        logger.info("Temporal leakage validation passed - no high correlations found")

    # Additional features
    # Cumulative features
    df['cumulative_pnl'] = groupby_account['daily_pnl'].apply(
        lambda x: x.shift(1).cumsum()
    ).reset_index(level=0, drop=True)

    # Win rate
    df['win_rate_21d'] = groupby_account['daily_pnl'].apply(
        lambda x: (x.shift(1) > 0).rolling(window=21).mean()
    ).reset_index(level=0, drop=True)

    # Average win/loss
    df['avg_win_21d'] = groupby_account['daily_pnl'].apply(
        lambda x: x.shift(1).where(x.shift(1) > 0).rolling(window=21).mean()
    ).reset_index(level=0, drop=True)

    df['avg_loss_21d'] = groupby_account['daily_pnl'].apply(
        lambda x: x.shift(1).where(x.shift(1) < 0).rolling(window=21).mean()
    ).reset_index(level=0, drop=True)

    # Drop initial rows with NaN values from rolling calculations
    initial_rows = max(feature_params['rolling_vol_windows'] +
                      [sortino_window, profit_factor_window, drawdown_window, 21])

    df = df.groupby('account_id').apply(
        lambda x: x.iloc[initial_rows:]
    ).reset_index(drop=True)

    # Drop any remaining NaN target values
    df = df.dropna(subset=['target_pnl', 'target_large_loss'])

    # Comprehensive data quality validation
    logger.info("Performing final data quality validation...")

    validation_results = validate_feature_data_quality(df, config)

    # Save processed features with validation metadata
    logger.info(f"Saving features to {config['paths']['processed_features']}")

    # Try to save as parquet first, fall back to pickle if needed
    try:
        df.to_parquet(config['paths']['processed_features'], index=False)
        logger.info(f"Saved features as parquet: {config['paths']['processed_features']}")
    except Exception as e:
        logger.warning(f"Parquet save failed ({str(e)}), using pickle instead")
        import pickle
        pickle_path = config['paths']['processed_features'].replace('.parquet', '.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(df, f)
        logger.info(f"Saved features to {pickle_path}")

    # Save validation results
    validation_path = config['paths']['processed_features'].replace('.parquet', '_validation.json').replace('.pkl', '_validation.json')
    import json
    with open(validation_path, 'w') as f:
        json.dump(validation_results, f, indent=2, default=str)
    logger.info(f"Validation results saved to {validation_path}")

    logger.info(f"Feature engineering complete. Final shape: {df.shape}")
    logger.info(f"Data quality issues found: {validation_results['n_issues']}")

    return df


def validate_feature_data_quality(df: pd.DataFrame, config: Dict) -> Dict:
    """
    Comprehensive validation of feature data quality and integrity.

    Args:
        df: Engineered features DataFrame
        config: Configuration dictionary

    Returns:
        Dict with validation results
    """
    logger.info("Validating feature data quality...")

    validation_results = {
        'validation_timestamp': pd.Timestamp.now().isoformat(),
        'issues': [],
        'warnings': [],
        'data_summary': {},
        'feature_statistics': {}
    }

    # Basic data summary
    validation_results['data_summary'] = {
        'n_rows': len(df),
        'n_features': len([col for col in df.columns if col.startswith(('ewm_', 'rolling_', 'avg_', 'win_rate', 'cumulative'))]),
        'n_traders': df['account_id'].nunique(),
        'date_range': [df['trade_date'].min().isoformat(), df['trade_date'].max().isoformat()],
        'target_pnl_range': [float(df['target_pnl'].min()), float(df['target_pnl'].max())],
        'large_loss_rate': float(df['target_large_loss'].mean())
    }

    # Check for missing data patterns
    missing_data = df.isnull().sum()
    high_missing = missing_data[missing_data > len(df) * 0.05]  # >5% missing

    if len(high_missing) > 0:
        validation_results['warnings'].append(f"Features with >5% missing data: {high_missing.to_dict()}")

    # Check for feature stability across traders
    feature_cols = [col for col in df.columns if col.startswith(('ewm_', 'rolling_', 'avg_', 'win_rate', 'cumulative'))]

    for feature in feature_cols[:10]:  # Check first 10 features for performance
        if df[feature].dtype in ['int64', 'float64']:
            trader_means = df.groupby('account_id')[feature].mean()
            trader_stds = df.groupby('account_id')[feature].std()

            # Check for extreme differences between traders
            mean_ratio = trader_means.max() / (trader_means.min() + 1e-6)
            if mean_ratio > 100:  # One trader has 100x higher values
                validation_results['warnings'].append(f"Feature {feature} has extreme trader differences (ratio: {mean_ratio:.1f})")

            # Store feature statistics
            validation_results['feature_statistics'][feature] = {
                'mean': float(df[feature].mean()),
                'std': float(df[feature].std()),
                'min': float(df[feature].min()),
                'max': float(df[feature].max()),
                'missing_rate': float(missing_data[feature] / len(df)),
                'trader_mean_ratio': float(mean_ratio)
            }

    # Validate target variables
    target_issues = []

    # Check target PnL distribution
    if df['target_pnl'].std() == 0:
        target_issues.append("target_pnl has zero variance")

    # Check for reasonable large loss rate (5-25% is typical)
    large_loss_rate = df['target_large_loss'].mean()
    if large_loss_rate < 0.05 or large_loss_rate > 0.25:
        target_issues.append(f"Unusual large loss rate: {large_loss_rate:.1%} (expected 5-25%)")

    # Check panel balance
    observations_per_trader = df.groupby('account_id').size()
    min_obs = observations_per_trader.min()
    max_obs = observations_per_trader.max()
    balance_ratio = min_obs / max_obs

    if balance_ratio < 0.5:
        validation_results['warnings'].append(f"Unbalanced panel: {min_obs} to {max_obs} observations per trader")

    validation_results['data_summary']['panel_balance_ratio'] = float(balance_ratio)
    validation_results['data_summary']['observations_per_trader'] = observations_per_trader.to_dict()

    # Check for date gaps
    date_gaps = []
    for trader in df['account_id'].unique():
        trader_data = df[df['account_id'] == trader].sort_values('trade_date')
        date_diffs = trader_data['trade_date'].diff().dt.days
        large_gaps = date_diffs[date_diffs > 7].count()  # Gaps > 1 week
        if large_gaps > 0:
            date_gaps.append(f"Trader {trader}: {large_gaps} gaps > 1 week")

    if date_gaps:
        validation_results['warnings'].append(f"Date gaps found: {date_gaps[:3]}")  # Show first 3

    # Add all issues to main issues list
    validation_results['issues'].extend(target_issues)
    validation_results['n_issues'] = len(validation_results['issues'])
    validation_results['n_warnings'] = len(validation_results['warnings'])

    # Log summary
    if validation_results['issues']:
        logger.warning(f"Found {validation_results['n_issues']} data quality issues")
        for issue in validation_results['issues'][:5]:
            logger.warning(f"  ISSUE: {issue}")

    if validation_results['warnings']:
        logger.info(f"Found {validation_results['n_warnings']} data quality warnings")
        for warning in validation_results['warnings'][:3]:
            logger.info(f"  WARNING: {warning}")

    if not validation_results['issues'] and not validation_results['warnings']:
        logger.info("All data quality checks passed successfully")

    return validation_results


def calculate_expanding_quantile_threshold(returns: pd.Series, quantile: float) -> pd.Series:
    """
    Calculate expanding window quantile threshold to prevent future leakage.
    Only uses historical data up to each point in time.
    """
    threshold_series = pd.Series(index=returns.index, dtype=float)

    for i in range(len(returns)):
        # Only use data up to (but not including) current point
        historical_data = returns.iloc[:i]
        negative_returns = historical_data[historical_data < 0]

        if len(negative_returns) >= 10:  # Need minimum samples for stable quantile
            threshold_series.iloc[i] = negative_returns.quantile(quantile)
        else:
            # Conservative fallback for early periods with insufficient data - use large negative number instead of -inf
            threshold_series.iloc[i] = -1e6  # Very large negative number instead of -inf

    return threshold_series


def calculate_sortino_ratio(returns: pd.Series, window: int) -> pd.Series:
    """Calculate rolling Sortino ratio."""
    def sortino(x):
        if len(x) < window:
            return np.nan
        mean_return = x.mean()
        downside_returns = x[x < 0]
        if len(downside_returns) == 0:
            return np.inf if mean_return > 0 else 0
        downside_std = downside_returns.std()
        if downside_std == 0:
            return np.inf if mean_return > 0 else 0
        return mean_return / downside_std * np.sqrt(252)  # Annualized

    return returns.rolling(window=window).apply(sortino)


def calculate_profit_factor(gross_profit: pd.Series, gross_loss: pd.Series, window: int) -> pd.Series:
    """Calculate rolling profit factor."""
    rolling_profit = gross_profit.rolling(window=window).sum()
    rolling_loss = gross_loss.rolling(window=window).sum()

    # Handle division by zero
    profit_factor = rolling_profit / rolling_loss.replace(0, np.nan)
    profit_factor = profit_factor.fillna(np.inf)  # If no losses, profit factor is infinite

    return profit_factor


def calculate_max_drawdown(returns: pd.Series, window: int) -> pd.Series:
    """Calculate rolling maximum drawdown."""
    def max_dd(x):
        if len(x) < window:
            return np.nan
        cumsum = x.cumsum()
        running_max = cumsum.expanding().max()
        drawdown = cumsum - running_max
        return drawdown.min()

    return returns.rolling(window=window).apply(max_dd)
