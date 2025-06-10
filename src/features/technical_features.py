"""
Technical Features for Trading
Price-based indicators with proper handling of financial time series
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
import logging
from scipy import stats

from src.features.base_features import BaseFeatures
from src.core.constants import TradingConstants as TC, DataQualityLimits as DQL

logger = logging.getLogger(__name__)


class TechnicalFeatures(BaseFeatures):
    """
    Technical indicators for trading strategy
    Implements momentum, volatility, and microstructure features
    """

    def __init__(self):
        super().__init__(feature_prefix='tech', lookback_days=60)

    def create_features(self,
                       totals_df: pd.DataFrame,
                       fills_df: Optional[pd.DataFrame] = None,
                       as_of_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """
        Create technical features from daily totals and fills

        Features include:
        - Returns at multiple horizons
        - Volatility measures (realized, EWMA, GARCH-like)
        - Risk metrics (Sharpe, Sortino, Calmar)
        - Microstructure (if fills available)
        """
        # Prepare data
        df = self._ensure_datetime_index(totals_df.copy())
        df = self._apply_point_in_time(df, as_of_date)

        if df.empty:
            return pd.DataFrame()

        # Initialize feature container
        all_features = []

        # Process each account separately to maintain independence
        for account_id in df['account_id'].unique():
            acc_df = df[df['account_id'] == account_id].sort_index()

            if len(acc_df) < 5:  # Need minimum history
                continue

            # Create features for this account
            features = pd.DataFrame(index=acc_df.index)
            features['account_id'] = account_id

            # 1. Returns and momentum features
            features = self._add_return_features(features, acc_df)

            # 2. Volatility features
            features = self._add_volatility_features(features, acc_df)

            # 3. Risk-adjusted performance metrics
            features = self._add_risk_metrics(features, acc_df)

            # 4. Market microstructure features (if fills available)
            if fills_df is not None and not fills_df.empty:
                features = self._add_microstructure_features(
                    features, acc_df, fills_df[fills_df['account_id'] == account_id]
                )

            # 5. Technical patterns
            features = self._add_pattern_features(features, acc_df)

            all_features.append(features)

        # Combine all accounts
        if not all_features:
            return pd.DataFrame()

        result = pd.concat(all_features, axis=0).sort_index()

        # Handle missing values
        result = self._handle_missing_data(result, method='forward_fill')

        # Add feature prefix
        result = self._add_feature_prefix(result)

        # Validate
        result = self._validate_features(result)

        return result

    def _add_return_features(self, features: pd.DataFrame, acc_df: pd.DataFrame) -> pd.DataFrame:
        """Add return-based features"""

        # Simple returns at different horizons
        for window in TC.FEATURE_WINDOWS:
            if window <= len(acc_df):
                # Cumulative return
                features[f'return_{window}d'] = (
                    acc_df['net_pnl'].rolling(window).sum()
                )

                # Average daily return
                features[f'avg_return_{window}d'] = (
                    acc_df['net_pnl'].rolling(window).mean()
                )

                # Return momentum (current vs previous period)
                if window * 2 <= len(acc_df):
                    current = acc_df['net_pnl'].rolling(window).sum()
                    previous = acc_df['net_pnl'].shift(window).rolling(window).sum()
                    features[f'momentum_{window}d'] = current - previous

        # Log returns for better statistical properties
        # Add small epsilon to avoid log(0)
        cumulative_pnl = acc_df['net_pnl'].cumsum() + acc_df['net_pnl'].abs().sum() + 1000
        log_returns = np.log(cumulative_pnl / cumulative_pnl.shift(1))

        features['log_return_1d'] = log_returns
        features['log_return_5d'] = log_returns.rolling(5).sum()

        # Relative Strength Index (RSI) - momentum oscillator
        features['rsi_14d'] = self._calculate_rsi(acc_df['net_pnl'], 14)

        # Moving Average Convergence Divergence (MACD)
        ema_12 = acc_df['net_pnl'].ewm(span=12, adjust=False).mean()
        ema_26 = acc_df['net_pnl'].ewm(span=26, adjust=False).mean()
        features['macd'] = ema_12 - ema_26
        features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
        features['macd_diff'] = features['macd'] - features['macd_signal']

        return features

    def _add_volatility_features(self, features: pd.DataFrame, acc_df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based features"""

        # Realized volatility at different horizons
        for window in TC.FEATURE_WINDOWS:
            if window <= len(acc_df):
                # Standard deviation
                features[f'volatility_{window}d'] = (
                    acc_df['net_pnl'].rolling(window).std()
                )

                # Downside volatility (Sortino denominator)
                downside_returns = acc_df['net_pnl'].copy()
                downside_returns[downside_returns > 0] = 0
                features[f'downside_vol_{window}d'] = (
                    downside_returns.rolling(window).std()
                )

                # Upside volatility
                upside_returns = acc_df['net_pnl'].copy()
                upside_returns[upside_returns < 0] = 0
                features[f'upside_vol_{window}d'] = (
                    upside_returns.rolling(window).std()
                )

        # EWMA volatility (reacts faster to changes)
        features['ewma_vol_10d'] = (
            acc_df['net_pnl'].ewm(span=10, adjust=False).std()
        )

        # Volatility of volatility (vol clustering)
        if 'volatility_20d' in features.columns:
            features['vol_of_vol_20d'] = (
                features['volatility_20d'].rolling(20).std()
            )

        # Parkinson volatility (if we have high/low from fills)
        # This is a placeholder - would need intraday data

        # GARCH-like volatility persistence
        if len(acc_df) >= 60:
            vol_20 = acc_df['net_pnl'].rolling(20).std()
            features['vol_persistence'] = vol_20.autocorr(lag=1)

        return features

    def _add_risk_metrics(self, features: pd.DataFrame, acc_df: pd.DataFrame) -> pd.DataFrame:
        """Add risk-adjusted performance metrics"""

        # Sharpe ratio at different horizons
        for window in TC.FEATURE_WINDOWS:
            if window <= len(acc_df):
                returns = acc_df['net_pnl'].rolling(window).mean()
                vol = acc_df['net_pnl'].rolling(window).std()

                # Annualized Sharpe
                features[f'sharpe_{window}d'] = (
                    returns / (vol + TC.MIN_VARIANCE) * np.sqrt(TC.TRADING_DAYS_PER_YEAR)
                )

                # Sortino ratio (downside deviation)
                downside = acc_df['net_pnl'].copy()
                downside[downside > 0] = 0
                downside_std = downside.rolling(window).std()

                features[f'sortino_{window}d'] = (
                    returns / (downside_std + TC.MIN_VARIANCE) * np.sqrt(TC.TRADING_DAYS_PER_YEAR)
                )

        # Maximum drawdown
        cumsum = acc_df['net_pnl'].cumsum()
        running_max = cumsum.expanding().max()
        drawdown = cumsum - running_max

        for window in [20, 60]:
            if window <= len(acc_df):
                features[f'max_drawdown_{window}d'] = drawdown.rolling(window).min()
                features[f'max_drawdown_pct_{window}d'] = (
                    drawdown / (running_max + TC.MIN_VARIANCE)
                ).rolling(window).min()

        # Calmar ratio (return / max drawdown)
        if 'return_20d' in features.columns and 'max_drawdown_20d' in features.columns:
            annual_return = features['return_20d'] * (TC.TRADING_DAYS_PER_YEAR / 20)
            features['calmar_20d'] = (
                annual_return / (features['max_drawdown_20d'].abs() + TC.MIN_VARIANCE)
            )

        # Win rate and profit factor
        for window in [10, 20]:
            if window <= len(acc_df):
                wins = (acc_df['net_pnl'] > 0).rolling(window).sum()
                features[f'win_rate_{window}d'] = wins / window

                # Profit factor
                profits = acc_df['net_pnl'].copy()
                profits[profits < 0] = 0
                losses = acc_df['net_pnl'].copy()
                losses[losses > 0] = 0

                total_profits = profits.rolling(window).sum()
                total_losses = losses.abs().rolling(window).sum()

                features[f'profit_factor_{window}d'] = (
                    total_profits / (total_losses + TC.MIN_VARIANCE)
                )

        # Risk of ruin approximation
        if 'win_rate_20d' in features.columns and 'avg_return_20d' in features.columns:
            win_rate = features['win_rate_20d']
            avg_win = features['avg_return_20d'].where(features['avg_return_20d'] > 0).fillna(0)
            avg_loss = features['avg_return_20d'].where(features['avg_return_20d'] < 0).abs().fillna(0)

            # Kelly criterion
            features['kelly_fraction'] = (
                (win_rate * avg_win - (1 - win_rate) * avg_loss) /
                (avg_win + TC.MIN_VARIANCE)
            ).clip(-1, 1)

        return features

    def _add_microstructure_features(self,
                                    features: pd.DataFrame,
                                    acc_df: pd.DataFrame,
                                    fills_df: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features from fills data"""

        if fills_df.empty:
            return features

        # Ensure fills have datetime
        fills_df = fills_df.copy()
        if 'datetime' in fills_df.columns:
            fills_df['date'] = pd.to_datetime(fills_df['datetime']).dt.date
            fills_df['hour'] = pd.to_datetime(fills_df['datetime']).dt.hour
            fills_df['minute'] = pd.to_datetime(fills_df['datetime']).dt.minute

        # Daily aggregations
        for date in features.index:
            date_fills = fills_df[fills_df['date'] == date.date()]

            if date_fills.empty:
                continue

            # Trading intensity
            features.loc[date, 'trade_count'] = len(date_fills)
            features.loc[date, 'unique_symbols'] = date_fills['symbol'].nunique()

            # Average trade size
            features.loc[date, 'avg_trade_size'] = date_fills['quantity'].abs().mean()
            features.loc[date, 'trade_size_std'] = date_fills['quantity'].abs().std()

            # Price dispersion
            features.loc[date, 'avg_price'] = date_fills['price'].mean()
            features.loc[date, 'price_std'] = date_fills['price'].std()

            # Intraday patterns
            if 'hour' in date_fills.columns:
                # Morning vs afternoon activity
                morning_trades = date_fills[date_fills['hour'] < 12]
                afternoon_trades = date_fills[date_fills['hour'] >= 12]

                features.loc[date, 'morning_activity_pct'] = (
                    len(morning_trades) / (len(date_fills) + TC.MIN_VARIANCE)
                )

                # First and last hour activity
                first_hour = date_fills[date_fills['hour'] == 9]
                last_hour = date_fills[date_fills['hour'] == 15]

                features.loc[date, 'first_hour_pct'] = (
                    len(first_hour) / (len(date_fills) + TC.MIN_VARIANCE)
                )
                features.loc[date, 'last_hour_pct'] = (
                    len(last_hour) / (len(date_fills) + TC.MIN_VARIANCE)
                )

            # Effective spread proxy (using price changes)
            if len(date_fills) > 1:
                price_changes = date_fills['price'].diff().abs()
                features.loc[date, 'effective_spread'] = price_changes.mean()

        # Rolling microstructure metrics
        for col in ['trade_count', 'unique_symbols', 'avg_trade_size']:
            if col in features.columns:
                features[f'{col}_ma5'] = features[col].rolling(5).mean()
                features[f'{col}_std5'] = features[col].rolling(5).std()

        return features

    def _add_pattern_features(self, features: pd.DataFrame, acc_df: pd.DataFrame) -> pd.DataFrame:
        """Add technical pattern features"""

        # Streak features
        features['win_streak'] = self._calculate_streak(acc_df['net_pnl'] > 0)
        features['loss_streak'] = self._calculate_streak(acc_df['net_pnl'] < 0)

        # Autocorrelation features (momentum vs mean reversion)
        for lag in [1, 5, 10]:
            if len(acc_df) > lag + 10:
                features[f'autocorr_lag{lag}'] = (
                    acc_df['net_pnl'].rolling(20).apply(
                        lambda x: x.autocorr(lag=lag) if len(x.dropna()) > lag else np.nan
                    )
                )

        # Higher moments
        for window in [20, 60]:
            if window <= len(acc_df):
                # Skewness (asymmetry)
                features[f'skewness_{window}d'] = (
                    acc_df['net_pnl'].rolling(window).skew()
                )

                # Kurtosis (tail risk)
                features[f'kurtosis_{window}d'] = (
                    acc_df['net_pnl'].rolling(window).kurt()
                )

        # Hurst exponent (trending vs mean reverting)
        if len(acc_df) >= 100:
            features['hurst_100d'] = acc_df['net_pnl'].rolling(100).apply(
                self._calculate_hurst_exponent
            )

        return features

    def _calculate_rsi(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / (loss + TC.MIN_VARIANCE)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_streak(self, condition: pd.Series) -> pd.Series:
        """Calculate consecutive true values in boolean series"""
        # Create groups where condition changes
        groups = (condition != condition.shift()).cumsum()

        # Count consecutive values within each group
        streak = condition.groupby(groups).cumsum()

        # Set to 0 where condition is false
        return streak.where(condition, 0)

    def _calculate_hurst_exponent(self, series: pd.Series) -> float:
        """
        Calculate Hurst exponent using R/S analysis
        H < 0.5: Mean reverting
        H = 0.5: Random walk
        H > 0.5: Trending
        """
        if len(series.dropna()) < 20:
            return np.nan

        try:
            series = series.dropna().values

            # Calculate R/S for different lags
            lags = range(2, min(len(series) // 2, 20))
            rs_values = []

            for lag in lags:
                # Divide series into chunks
                chunks = [series[i:i+lag] for i in range(0, len(series), lag)]
                chunks = [c for c in chunks if len(c) == lag]

                if not chunks:
                    continue

                rs_list = []
                for chunk in chunks:
                    # Calculate mean-adjusted series
                    mean_adj = chunk - np.mean(chunk)

                    # Calculate cumulative sum
                    cum_sum = np.cumsum(mean_adj)

                    # Calculate range
                    R = np.max(cum_sum) - np.min(cum_sum)

                    # Calculate standard deviation
                    S = np.std(chunk, ddof=1)

                    if S > 0:
                        rs_list.append(R / S)

                if rs_list:
                    rs_values.append(np.mean(rs_list))

            if len(rs_values) > 3:
                # Fit log(R/S) = log(c) + H*log(n)
                log_lags = np.log(list(lags)[:len(rs_values)])
                log_rs = np.log(rs_values)

                # Linear regression
                H, _ = np.polyfit(log_lags, log_rs, 1)
                return H

        except Exception as e:
            logger.debug(f"Hurst calculation failed: {e}")

        return np.nan
