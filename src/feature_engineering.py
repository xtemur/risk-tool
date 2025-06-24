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

    # Calculate quantile threshold for large losses within each trader
    df['large_loss_threshold'] = groupby_account['daily_pnl'].transform(
        lambda x: x[x < 0].quantile(large_loss_quantile) if (x < 0).any() else -np.inf
    )

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

    # Generate target variables
    logger.info("Creating target variables...")

    # Target PnL (next day's PnL)
    df['target_pnl'] = groupby_account['daily_pnl'].shift(-1)

    # Target large loss (binary)
    df['target_large_loss'] = groupby_account.apply(
        lambda x: (x['daily_pnl'].shift(-1) < x['large_loss_threshold']).astype(int)
    ).reset_index(level=0, drop=True)

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

    # Save processed features
    logger.info(f"Saving features to {config['paths']['processed_features']}")
    # Use pickle temporarily due to PyArrow issues
    import pickle
    pickle_path = config['paths']['processed_features'].replace('.parquet', '.pkl')
    with open(pickle_path, 'wb') as f:
        pickle.dump(df, f)
    logger.info(f"Saved features to {pickle_path} (using pickle due to PyArrow issues)")

    logger.info(f"Feature engineering complete. Final shape: {df.shape}")

    return df


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
