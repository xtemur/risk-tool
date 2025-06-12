# src/features/market_regime_features.py
"""
Market Regime Features - Simplified for single trader analysis
Focus on personal trading regimes rather than market-wide analysis
"""

import pandas as pd
import numpy as np
from typing import Optional
import logging

from src.features.base_features import BaseFeatures

logger = logging.getLogger(__name__)


class MarketRegimeFeatures(BaseFeatures):
    """
    Regime features based on individual trader's patterns
    """

    def __init__(self):
        super().__init__(feature_prefix='regime', min_periods=20)

    def create_features(self,
                       daily_summary: pd.DataFrame,
                       fills: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Create regime features from trader's own history"""

        # Validate required columns
        required_cols = ['date', 'net', 'gross', 'orders', 'qty', 'end_balance']
        if not self._validate_data(daily_summary, required_cols):
            return pd.DataFrame()

        # Sort by date and set as index
        df = daily_summary.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').set_index('date')

        features = pd.DataFrame(index=df.index)

        # 1. Personal volatility regimes
        features = self._add_volatility_regimes(features, df)

        # 2. Performance regimes
        features = self._add_performance_regimes(features, df)

        # 3. Activity regimes
        features = self._add_activity_regimes(features, df)

        # 4. Capital regimes
        features = self._add_capital_regimes(features, df)

        # Reset index to have date as column
        features = features.reset_index()

        # Add prefix
        features = self._add_feature_prefix(features)

        return features

    def _add_volatility_regimes(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Personal volatility regime features"""

        # Calculate rolling volatility
        vol_5d = df['net'].rolling(5).std()
        vol_20d = df['net'].rolling(20).std()
        vol_60d = df['net'].rolling(60).std()

        # Volatility percentiles (personal history)
        features['vol_percentile_20d'] = vol_20d.rolling(60).rank(pct=True)
        features['vol_percentile_60d'] = vol_20d.rolling(252).rank(pct=True)

        # Volatility regime (low/medium/high based on personal history)
        features['vol_regime_low'] = (features['vol_percentile_60d'] < 0.33).astype(int)
        features['vol_regime_medium'] = (
            (features['vol_percentile_60d'] >= 0.33) &
            (features['vol_percentile_60d'] < 0.67)
        ).astype(int)
        features['vol_regime_high'] = (features['vol_percentile_60d'] >= 0.67).astype(int)

        # Volatility trend
        features['vol_expanding'] = (vol_5d > vol_20d).astype(int)
        features['vol_contracting'] = (vol_5d < vol_20d * 0.8).astype(int)

        # Volatility regime persistence
        vol_high = features['vol_regime_high'] == 1
        features['vol_regime_days'] = self._calculate_regime_duration(vol_high)

        return features

    def _add_performance_regimes(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Performance-based regime features"""

        # Rolling performance metrics
        ret_5d = df['net'].rolling(5).sum()
        ret_20d = df['net'].rolling(20).sum()
        ret_60d = df['net'].rolling(60).sum()

        # Performance percentiles
        features['perf_percentile_20d'] = ret_20d.rolling(60).rank(pct=True)
        features['perf_percentile_60d'] = ret_20d.rolling(252).rank(pct=True)

        # Performance regimes
        features['perf_regime_strong'] = (features['perf_percentile_60d'] > 0.7).astype(int)
        features['perf_regime_weak'] = (features['perf_percentile_60d'] < 0.3).astype(int)

        # Momentum regime
        features['momentum_positive'] = (ret_5d > 0) & (ret_20d > 0)
        features['momentum_negative'] = (ret_5d < 0) & (ret_20d < 0)
        features['momentum_days'] = self._calculate_regime_duration(features['momentum_positive'])

        # Recovery/drawdown regime
        cumsum = df['net'].cumsum()
        running_max = cumsum.expanding().max()
        drawdown = cumsum - running_max

        features['in_drawdown'] = (drawdown < 0).astype(int)
        features['drawdown_days'] = self._calculate_regime_duration(drawdown < 0)
        features['recovery_distance'] = -drawdown / (running_max + 1e-8)

        return features

    def _add_activity_regimes(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Trading activity regime features"""

        # Activity levels
        orders_ma5 = df['orders'].rolling(5).mean()
        orders_ma20 = df['orders'].rolling(20).mean()

        # Activity percentiles
        features['activity_percentile'] = df['orders'].rolling(60).rank(pct=True)

        # Activity regimes
        features['activity_high'] = (df['orders'] > orders_ma20 * 1.5).astype(int)
        features['activity_low'] = (df['orders'] < orders_ma20 * 0.5).astype(int)
        features['activity_normal'] = (
            ~features['activity_high'] & ~features['activity_low']
        ).astype(int)

        # Size regime
        size_ma20 = df['qty'].rolling(20).mean()
        features['size_regime_large'] = (df['qty'] > size_ma20 * 1.5).astype(int)
        features['size_regime_small'] = (df['qty'] < size_ma20 * 0.5).astype(int)

        # Efficiency regime (fill rate)
        fill_rate = df['fills'] / (df['orders'] + 1e-8)
        fill_rate_ma = fill_rate.rolling(20).mean()
        features['efficiency_high'] = (fill_rate > fill_rate_ma * 1.1).astype(int)
        features['efficiency_low'] = (fill_rate < fill_rate_ma * 0.9).astype(int)

        return features

    def _add_capital_regimes(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Capital utilization regime features"""

        # Capital metrics
        if 'end_balance' in df.columns and 'cash' in df.columns:
            # Capital utilization
            capital_used = (df['end_balance'] - df['cash']) / (df['end_balance'] + 1e-8)
            features['capital_utilization'] = capital_used

            # Leverage regime
            features['leverage_high'] = (capital_used > 0.7).astype(int)
            features['leverage_low'] = (capital_used < 0.3).astype(int)

            # Capital growth
            balance_pct_change = df['end_balance'].pct_change(20)
            features['capital_growing'] = (balance_pct_change > 0.05).astype(int)
            features['capital_declining'] = (balance_pct_change < -0.05).astype(int)

        # Risk capacity (based on recent performance)
        recent_vol = df['net'].rolling(20).std()
        recent_balance = df['end_balance'].rolling(20).mean()
        features['risk_capacity'] = recent_balance / (recent_vol * 20 + 1e-8)  # Can survive 20 std moves

        features['risk_capacity_high'] = (features['risk_capacity'] > 10).astype(int)
        features['risk_capacity_low'] = (features['risk_capacity'] < 5).astype(int)

        return features

    def _calculate_regime_duration(self, condition: pd.Series) -> pd.Series:
        """Calculate how long current regime has persisted"""
        # Create regime change indicator
        regime_change = condition != condition.shift(1)
        # Create regime groups
        regime_groups = regime_change.cumsum()
        # Count duration within each regime
        duration = condition.groupby(regime_groups).cumcount() + 1
        # Only count when condition is True
        return duration.where(condition, 0)
