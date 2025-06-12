# src/features/technical_features.py
"""
Technical Features - Simplified and fixed for actual data
Focus on proven indicators that work with limited data
"""

import pandas as pd
import numpy as np
from typing import Optional
import logging

from src.features.base_features import BaseFeatures

logger = logging.getLogger(__name__)


class TechnicalFeatures(BaseFeatures):
    """
    Technical indicators focused on what works with day trading data
    """

    def __init__(self):
        super().__init__(feature_prefix='tech', min_periods=20)

    def create_features(self,
                       daily_summary: pd.DataFrame,
                       fills: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Create technical features from daily summary data"""

        # Validate required columns
        required_cols = ['date', 'net', 'gross', 'orders', 'fills', 'qty']
        if not self._validate_data(daily_summary, required_cols):
            return pd.DataFrame()

        # Sort by date and set as index
        df = daily_summary.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').set_index('date')

        features = pd.DataFrame(index=df.index)

        # 1. Returns and momentum
        features = self._add_return_features(features, df)

        # 2. Volatility
        features = self._add_volatility_features(features, df)

        # 3. Volume and activity
        features = self._add_volume_features(features, df)

        # 4. Efficiency metrics
        features = self._add_efficiency_features(features, df)

        # 5. Simple risk metrics
        features = self._add_risk_features(features, df)

        # Reset index to have date as column
        features = features.reset_index()

        # Add prefix
        features = self._add_feature_prefix(features)

        return features

    def _add_return_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add return-based features"""

        # Simple returns
        for window in [1, 3, 5, 10, 20]:
            features[f'return_{window}d'] = df['net'].rolling(window).sum()
            features[f'return_mean_{window}d'] = df['net'].rolling(window).mean()

        # Momentum
        features['momentum_5d'] = features['return_5d'] - features['return_5d'].shift(5)
        features['momentum_20d'] = features['return_20d'] - features['return_20d'].shift(20)

        # Win rate
        features['win_rate_20d'] = (df['net'] > 0).rolling(20).mean()
        features['win_rate_60d'] = (df['net'] > 0).rolling(60).mean()

        # Streaks
        is_win = df['net'] > 0
        features['current_streak'] = self._calculate_streak(is_win)

        return features

    def _add_volatility_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility features"""

        # Standard deviation
        for window in [5, 10, 20]:
            features[f'volatility_{window}d'] = self._safe_rolling(df['net'], window, 'std')

        # Normalized volatility (coefficient of variation)
        features['volatility_normalized_20d'] = (
            features['volatility_20d'] /
            (df['net'].rolling(20).mean().abs() + 1e-8)
        )

        # EWMA volatility (reacts faster)
        features['volatility_ewm_10d'] = df['net'].ewm(span=10, adjust=False).std()

        # Volatility of volatility
        features['vol_of_vol_20d'] = features['volatility_10d'].rolling(20).std()

        return features

    def _add_volume_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume and activity features"""

        # Order activity
        features['orders_mean_20d'] = self._safe_rolling(df['orders'], 20, 'mean')
        features['orders_vs_avg'] = df['orders'] / (features['orders_mean_20d'] + 1e-8)

        # Fill rate
        features['fill_rate'] = df['fills'] / (df['orders'] + 1e-8)
        features['fill_rate_20d'] = self._safe_rolling(features['fill_rate'], 20, 'mean')

        # Quantity patterns
        features['qty_mean_20d'] = self._safe_rolling(df['qty'], 20, 'mean')
        features['qty_std_20d'] = self._safe_rolling(df['qty'], 20, 'std')
        features['qty_vs_avg'] = df['qty'] / (features['qty_mean_20d'] + 1e-8)

        return features

    def _add_efficiency_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add trading efficiency features"""

        # Cost efficiency
        total_fees = df.get('trade_fees', df.get('comm', 0))
        features['fee_rate'] = total_fees / (df['gross'].abs() + 1e-8)
        features['fee_rate_20d'] = self._safe_rolling(features['fee_rate'], 20, 'mean')

        # P&L per trade
        features['pnl_per_fill'] = df['net'] / (df['fills'] + 1e-8)
        features['pnl_per_fill_20d'] = self._safe_rolling(features['pnl_per_fill'], 20, 'mean')

        # Gross to net ratio (impact of fees)
        features['gross_net_ratio'] = df['net'] / (df['gross'] + 1e-8)

        return features

    def _add_risk_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add simple risk metrics"""

        # Drawdown
        cumsum = df['net'].cumsum()
        running_max = cumsum.expanding().max()
        drawdown = cumsum - running_max

        features['drawdown'] = drawdown
        features['drawdown_pct'] = drawdown / (running_max + 1e-8)
        features['days_since_high'] = (drawdown < 0).groupby((drawdown == 0).cumsum()).cumsum()

        # Simple Sharpe (annualized)
        features['sharpe_20d'] = (
            features['return_mean_20d'] / (features['volatility_20d'] + 1e-8) * np.sqrt(252)
        )

        # Downside deviation
        downside_returns = df['net'].copy()
        downside_returns[downside_returns > 0] = 0
        features['downside_vol_20d'] = self._safe_rolling(downside_returns, 20, 'std')

        # Simple Sortino
        features['sortino_20d'] = (
            features['return_mean_20d'] / (features['downside_vol_20d'] + 1e-8) * np.sqrt(252)
        )

        return features

    def _calculate_streak(self, condition: pd.Series) -> pd.Series:
        """Calculate consecutive True values"""
        # Create groups where condition changes
        groups = (condition != condition.shift()).cumsum()
        # Count within groups, reset when condition is False
        streak = condition.groupby(groups).cumsum()
        return streak.where(condition, 0)
