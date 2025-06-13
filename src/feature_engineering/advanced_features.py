"""
Advanced Feature Engineering for Trading Risk Models

Implements Phase 1 improvements:
- Alternative target variables (multi-day, direction, risk-adjusted)
- Outlier treatment and normalization
- Cross-sectional and behavioral features
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class AdvancedFeatureEngineer:
    """
    Advanced feature engineering focusing on target transformation
    and behavioral/cross-sectional features
    """

    def __init__(self,
                 target_smoothing_windows: List[int] = [3, 5, 7],
                 outlier_percentile: float = 0.05,
                 volatility_window: int = 20):
        """
        Initialize advanced feature engineer

        Args:
            target_smoothing_windows: Days to smooth target PnL
            outlier_percentile: Percentile for outlier treatment (0.05 = 5% each tail)
            volatility_window: Window for volatility calculations
        """
        self.target_smoothing_windows = target_smoothing_windows
        self.outlier_percentile = outlier_percentile
        self.volatility_window = volatility_window

    def create_alternative_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create alternative target variables to improve predictability

        Args:
            df: DataFrame with net PnL column

        Returns:
            DataFrame with additional target columns
        """
        logger.info("Creating alternative target variables...")
        df = df.copy()

        # Ensure proper sorting
        df = df.sort_values(['account_id', 'date']).reset_index(drop=True)

        # 1. Multi-day smoothed targets (reduce noise)
        for window in self.target_smoothing_windows:
            # Forward-looking smoothed PnL
            df[f'target_pnl_{window}d'] = (
                df.groupby('account_id')['net']
                .rolling(window, min_periods=1)
                .mean()
                .shift(-window + 1)
                .reset_index(level=0, drop=True)
            )

        # 2. Direction/Classification targets
        # Next day direction (up/down)
        df['target_direction_1d'] = (
            df.groupby('account_id')['net'].shift(-1) > 0
        ).astype(int)

        # Multi-day direction (majority positive days)
        for window in [3, 5]:
            future_returns = (
                df.groupby('account_id')['net']
                .rolling(window, min_periods=1)
                .apply(lambda x: (x > 0).sum())
                .shift(-window + 1)
                .reset_index(level=0, drop=True)
            )
            df[f'target_direction_{window}d'] = (
                future_returns > window / 2
            ).astype(int)

        # 3. Risk-adjusted targets
        # Rolling volatility for normalization
        df['rolling_volatility'] = (
            df.groupby('account_id')['net']
            .rolling(self.volatility_window, min_periods=5)
            .std()
            .reset_index(level=0, drop=True)
        )

        # Volatility-adjusted target
        df['target_vol_adjusted_1d'] = (
            df.groupby('account_id')['net'].shift(-1) /
            (df['rolling_volatility'] + 1e-6)
        )

        # 4. Relative performance targets
        # Target relative to trader's historical mean
        trader_mean = (
            df.groupby('account_id')['net']
            .expanding(min_periods=10)
            .mean()
            .reset_index(level=0, drop=True)
        )

        df['target_relative_1d'] = (
            df.groupby('account_id')['net'].shift(-1) - trader_mean
        )

        # 5. Quantile-based targets (robust to outliers)
        # Is next day performance above median?
        trader_median = (
            df.groupby('account_id')['net']
            .expanding(min_periods=10)
            .median()
            .reset_index(level=0, drop=True)
        )

        df['target_above_median_1d'] = (
            df.groupby('account_id')['net'].shift(-1) > trader_median
        ).astype(int)

        # Is next day in top quartile?
        trader_75th = (
            df.groupby('account_id')['net']
            .expanding(min_periods=20)
            .quantile(0.75)
            .reset_index(level=0, drop=True)
        )

        df['target_top_quartile_1d'] = (
            df.groupby('account_id')['net'].shift(-1) > trader_75th
        ).astype(int)

        logger.info(f"Created {len([c for c in df.columns if c.startswith('target_')])} target variables")

        return df

    def apply_outlier_treatment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply outlier treatment to target variables

        Args:
            df: DataFrame with target variables

        Returns:
            DataFrame with outlier-treated targets
        """
        logger.info("Applying outlier treatment to targets...")
        df = df.copy()

        target_cols = [col for col in df.columns if col.startswith('target_pnl')]

        for col in target_cols:
            if col in df.columns:
                # Calculate percentile thresholds
                lower_bound = df[col].quantile(self.outlier_percentile)
                upper_bound = df[col].quantile(1 - self.outlier_percentile)

                # Create winsorized version
                df[f'{col}_winsorized'] = df[col].clip(lower_bound, upper_bound)

                # Create outlier indicators
                df[f'{col}_is_outlier'] = (
                    (df[col] < lower_bound) | (df[col] > upper_bound)
                ).astype(int)

                outlier_pct = df[f'{col}_is_outlier'].mean()
                logger.info(f"{col}: {outlier_pct:.1%} outliers treated")

        return df

    def create_cross_sectional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create cross-sectional features comparing traders

        Args:
            df: DataFrame with multiple traders

        Returns:
            DataFrame with cross-sectional features
        """
        logger.info("Creating cross-sectional features...")
        df = df.copy()

        # Ensure unique account-date combinations and reset index
        df = df.drop_duplicates(subset=['account_id', 'date'], keep='first').reset_index(drop=True)

        # Daily cross-sectional features
        daily_stats = df.groupby('date').agg({
            'net': ['mean', 'median', 'std', 'count']
        }).round(2)
        daily_stats.columns = ['market_pnl_mean', 'market_pnl_median',
                              'market_pnl_std', 'active_traders']

        # Reset index to make date a column for merging
        daily_stats = daily_stats.reset_index()

        # Merge back to main dataframe
        df = df.merge(daily_stats, on='date', how='left')

        # Relative performance features
        df['pnl_vs_market_mean'] = df['net'] - df['market_pnl_mean']
        df['pnl_vs_market_median'] = df['net'] - df['market_pnl_median']

        # Percentile rank among active traders
        df['daily_pnl_rank'] = (
            df.groupby('date')['net']
            .rank(pct=True)
        )

        # Rolling cross-sectional features (5-day)
        for window in [5, 10]:
            # Rolling relative performance
            df[f'avg_relative_performance_{window}d'] = (
                df.groupby('account_id')['pnl_vs_market_mean']
                .rolling(window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )

            # Rolling rank consistency
            df[f'avg_rank_{window}d'] = (
                df.groupby('account_id')['daily_pnl_rank']
                .rolling(window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )

        # Market correlation (using simpler approach)
        for window in [10, 20]:
            correlation_values = []

            for account_id in df['account_id'].unique():
                account_data = df[df['account_id'] == account_id].copy()
                account_corr = account_data[['net', 'market_pnl_mean']].rolling(
                    window, min_periods=5
                ).corr().iloc[0::2, -1].values  # Get correlation values
                correlation_values.extend(account_corr)

            df[f'market_correlation_{window}d'] = correlation_values

        return df

    def create_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create behavioral finance features

        Args:
            df: DataFrame with trading data

        Returns:
            DataFrame with behavioral features
        """
        logger.info("Creating behavioral features...")
        df = df.copy()

        # Ensure proper sorting
        df = df.sort_values(['account_id', 'date']).reset_index(drop=True)

        # 1. Overconfidence indicators
        # Recent wins leading to increased position sizes
        df['is_positive'] = (df['net'] > 0).astype(int)
        df['recent_win_streak'] = (
            df.groupby('account_id')['is_positive']
            .rolling(5, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )

        # Position size after wins vs losses
        df['is_win'] = (df['net'] > 0).astype(int)
        df['prev_day_win'] = (
            df.groupby('account_id')['is_win'].shift(1)
        )

        # 2. Loss aversion patterns
        # Behavior change after large losses (simplified)
        df['large_loss_indicator'] = 0
        for account_id in df['account_id'].unique():
            mask = df['account_id'] == account_id
            account_data = df.loc[mask, 'net']
            tenth_percentile = account_data.expanding().quantile(0.1)
            df.loc[mask, 'large_loss_indicator'] = (account_data < tenth_percentile).astype(int)

        # Days since last large loss (simplified)
        df['days_since_large_loss'] = 0
        for account_id in df['account_id'].unique():
            mask = df['account_id'] == account_id
            large_loss_series = df.loc[mask, 'large_loss_indicator']
            days_since = (large_loss_series == 0).cumsum()
            df.loc[mask, 'days_since_large_loss'] = days_since

        # 3. Momentum vs contrarian behavior
        # Trading in same direction as recent performance (simplified)
        df['recent_pnl_trend'] = (
            df.groupby('account_id')['net']
            .rolling(3, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        df['recent_pnl_trend'] = np.where(df['recent_pnl_trend'] > 0, 1, -1)

        # Position sizing trend (create simple proxy if not available)
        if 'trade_size_proxy' not in df.columns:
            df['trade_size_proxy'] = abs(df['net'])

        df['position_size_trend'] = (
            df.groupby('account_id')['trade_size_proxy']
            .rolling(3, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        df['position_size_trend'] = np.where(
            df['position_size_trend'] > df.groupby('account_id')['trade_size_proxy'].shift(3), 1, -1
        )

        # Momentum indicator (same direction)
        df['momentum_behavior'] = (
            df['recent_pnl_trend'] == df['position_size_trend']
        ).astype(int)

        # 4. Risk-taking patterns
        # Volatility of recent performance
        df['recent_pnl_volatility'] = (
            df.groupby('account_id')['net']
            .rolling(10, min_periods=3)
            .std()
            .reset_index(level=0, drop=True)
        )

        # Risk-taking after wins vs losses
        df['risk_taking_after_win'] = np.where(
            df['prev_day_win'] == 1,
            df['recent_pnl_volatility'],
            np.nan
        )

        df['risk_taking_after_loss'] = np.where(
            df['prev_day_win'] == 0,
            df['recent_pnl_volatility'],
            np.nan
        )

        # 5. Consistency indicators
        # Coefficient of variation (risk-adjusted consistency)
        df['performance_consistency'] = (
            df.groupby('account_id')['net']
            .rolling(20, min_periods=5)
            .apply(lambda x: abs(x.mean()) / (x.std() + 1e-6))
            .reset_index(level=0, drop=True)
        )

        return df

    def create_market_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create market regime and microstructure features

        Args:
            df: DataFrame with trading data

        Returns:
            DataFrame with market features
        """
        logger.info("Creating market regime features...")
        df = df.copy()

        # 1. Volatility regime indicators
        market_vol = (
            df.groupby('date')['net'].std().reset_index()
            .rename(columns={'net': 'market_daily_vol'})
        )
        df = df.merge(market_vol, on='date', how='left')

        # Volatility regime (high/low based on rolling percentile)
        df['market_vol_percentile'] = (
            df['market_daily_vol']
            .rolling(60, min_periods=20)
            .rank(pct=True)
        )

        df['high_vol_regime'] = (df['market_vol_percentile'] > 0.7).astype(int)
        df['low_vol_regime'] = (df['market_vol_percentile'] < 0.3).astype(int)

        # 2. Calendar effects
        df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
        df['is_monday'] = (df['day_of_week'] == 0).astype(int)
        df['is_friday'] = (df['day_of_week'] == 4).astype(int)

        df['month'] = pd.to_datetime(df['date']).dt.month
        df['is_month_end'] = (
            pd.to_datetime(df['date']).dt.is_month_end
        ).astype(int)

        # 3. Market stress indicators
        # Dispersion of trader performance (market stress proxy)
        daily_dispersion = (
            df.groupby('date')['net'].std().reset_index()
            .rename(columns={'net': 'trader_dispersion'})
        )
        df = df.merge(daily_dispersion, on='date', how='left')

        # High dispersion = market stress
        df['market_stress'] = (
            df['trader_dispersion'] >
            df['trader_dispersion'].rolling(30, min_periods=10).quantile(0.8)
        ).astype(int)

        return df

    def process_all_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process all advanced features in correct order

        Args:
            df: Input DataFrame with basic features

        Returns:
            DataFrame with all advanced features
        """
        logger.info("Processing all advanced features...")

        # Phase 1: Target transformations
        df = self.create_alternative_targets(df)
        df = self.apply_outlier_treatment(df)

        # Phase 2: Advanced feature engineering
        df = self.create_cross_sectional_features(df)
        df = self.create_behavioral_features(df)
        df = self.create_market_regime_features(df)

        # Fill NaN values with appropriate methods
        df = self._safe_fillna_advanced(df)

        logger.info(f"Final dataset shape: {df.shape}")
        logger.info(f"Total features: {len([c for c in df.columns if c not in ['account_id', 'date']])}")

        return df

    def _safe_fillna_advanced(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Safe NaN filling for advanced features
        """
        # Numeric columns: use forward fill then median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['account_id', 'date']:
                df[col] = df.groupby('account_id')[col].transform(
                    lambda x: x.fillna(method='ffill').fillna(x.median())
                )

        return df

    def get_target_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary of target variable improvements
        """
        target_cols = [col for col in df.columns if col.startswith('target_')]

        summary = {}
        for col in target_cols:
            if col in df.columns and df[col].notna().sum() > 100:
                # Signal-to-noise ratio
                signal_to_noise = abs(df[col].mean()) / (df[col].std() + 1e-6)

                summary[col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'signal_to_noise': signal_to_noise,
                    'non_null_count': df[col].notna().sum()
                }

                # For classification targets, add accuracy metrics
                if 'direction' in col or 'above_median' in col or 'quartile' in col:
                    summary[col]['class_balance'] = df[col].mean()

        return summary
