# src/features/behavioral_features.py
"""
Behavioral Features - Simplified to focus on key psychological patterns
"""

import pandas as pd
import numpy as np
from typing import Optional
import logging

from src.features.base_features import BaseFeatures

logger = logging.getLogger(__name__)


class BehavioralFeatures(BaseFeatures):
    """
    Behavioral features capturing trader psychology
    Simplified to work with limited data
    """

    def __init__(self):
        super().__init__(feature_prefix='behav', min_periods=20)

    def create_features(self,
                       daily_summary: pd.DataFrame,
                       fills: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Create behavioral features"""

        # Validate required columns
        required_cols = ['date', 'net', 'orders', 'fills', 'qty', 'gross']
        if not self._validate_data(daily_summary, required_cols):
            return pd.DataFrame()

        # Sort by date and set as index
        df = daily_summary.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').set_index('date')

        features = pd.DataFrame(index=df.index)

        # 1. Loss aversion
        features = self._add_loss_aversion_features(features, df)

        # 2. Overconfidence
        features = self._add_overconfidence_features(features, df)

        # 3. Consistency
        features = self._add_consistency_features(features, df)

        # 4. Stress indicators
        features = self._add_stress_features(features, df)

        # 5. Time patterns (if fills available)
        if fills is not None and not fills.empty:
            features = self._add_time_features(features, df, fills)

        # Reset index to have date as column
        features = features.reset_index()

        # Add prefix
        features = self._add_feature_prefix(features)

        return features

    def _add_loss_aversion_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Loss aversion indicators"""

        # Post-loss behavior
        is_loss = df['net'] < 0
        features['post_loss_activity'] = (
            df['orders'].where(is_loss.shift(1)).rolling(5).mean() /
            (df['orders'].rolling(20).mean() + 1e-8)
        )

        # Consecutive losses
        features['loss_streak'] = self._calculate_streak(is_loss)
        features['max_loss_streak_20d'] = features['loss_streak'].rolling(20).max()

        # Recovery pressure (cumulative recent losses)
        features['cum_loss_5d'] = df['net'].where(is_loss, 0).rolling(5).sum()
        features['cum_loss_20d'] = df['net'].where(is_loss, 0).rolling(20).sum()

        # Break-even distance
        cumsum = df['net'].cumsum()
        running_max = cumsum.expanding().max()
        features['distance_from_peak'] = cumsum - running_max
        features['days_below_peak'] = (features['distance_from_peak'] < 0).groupby(
            (features['distance_from_peak'] >= 0).cumsum()
        ).cumsum()

        return features

    def _add_overconfidence_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Overconfidence indicators"""

        # Win streaks
        is_win = df['net'] > 0
        features['win_streak'] = self._calculate_streak(is_win)

        # Post-win behavior
        features['post_win_activity'] = (
            df['orders'].where(is_win.shift(1)).rolling(5).mean() /
            (df['orders'].rolling(20).mean() + 1e-8)
        )

        # Position sizing after wins
        features['post_win_size'] = (
            df['qty'].where(is_win.shift(1)).rolling(5).mean() /
            (df['qty'].rolling(20).mean() + 1e-8)
        )

        # Trading frequency changes
        features['order_frequency_change'] = (
            df['orders'].rolling(5).mean() -
            df['orders'].rolling(20).mean()
        ) / (df['orders'].rolling(20).std() + 1e-8)

        return features

    def _add_consistency_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Trading consistency metrics"""

        # P&L consistency
        features['pnl_volatility'] = df['net'].rolling(20).std()
        features['pnl_consistency'] = 1 / (
            features['pnl_volatility'] / (df['net'].rolling(20).mean().abs() + 1e-8) + 1
        )

        # Activity consistency
        features['orders_volatility'] = df['orders'].rolling(20).std()
        features['activity_consistency'] = 1 / (
            features['orders_volatility'] / (df['orders'].rolling(20).mean() + 1e-8) + 1
        )

        # Size consistency
        features['size_volatility'] = df['qty'].rolling(20).std()
        features['size_consistency'] = 1 / (
            features['size_volatility'] / (df['qty'].rolling(20).mean() + 1e-8) + 1
        )

        return features

    def _add_stress_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Stress and tilt indicators"""

        # Drawdown stress
        cumsum = df['net'].cumsum()
        running_max = cumsum.expanding().max()
        drawdown_pct = (cumsum - running_max) / (running_max + 1e-8)

        features['drawdown_stress'] = drawdown_pct.rolling(10).mean().abs()
        features['max_stress_20d'] = drawdown_pct.rolling(20).min().abs()

        # Revenge trading (increased activity after losses)
        is_loss = df['net'] < 0
        features['revenge_score'] = (
            df['orders'].where(is_loss.shift(1)).rolling(3).mean() /
            (df['orders'].rolling(10).mean() + 1e-8)
        ).fillna(1)

        # Tilt indicator (erratic behavior)
        features['order_variance'] = df['orders'].rolling(10).std()
        features['tilt_score'] = (
            features['order_variance'] /
            (df['orders'].rolling(30).std() + 1e-8)
        ).clip(0, 3)

        return features

    def _add_time_features(self, features: pd.DataFrame, df: pd.DataFrame,
                          fills: pd.DataFrame) -> pd.DataFrame:
        """Time-based behavioral patterns"""

        # Ensure fills has datetime
        fills = fills.copy()
        fills['datetime'] = pd.to_datetime(fills['datetime'])
        fills['date'] = fills['datetime'].dt.date
        fills['hour'] = fills['datetime'].dt.hour

        # Daily time metrics
        daily_stats = fills.groupby('date').agg({
            'hour': ['mean', 'std', 'min', 'max'],
            'fill_id': 'count'
        })

        daily_stats.columns = ['avg_trade_hour', 'hour_std', 'first_hour', 'last_hour', 'n_fills']
        daily_stats.index = pd.to_datetime(daily_stats.index)

        # Merge with features
        for col in daily_stats.columns:
            if col in features.columns:
                features[col] = daily_stats[col]
            else:
                features = features.join(daily_stats[[col]], how='left')

        # Forward fill time features
        time_cols = ['avg_trade_hour', 'hour_std', 'first_hour', 'last_hour']
        features[time_cols] = features[time_cols].fillna(method='ffill', limit=5)

        # Session concentration
        features['morning_bias'] = (features['avg_trade_hour'] < 12).astype(int)
        features['session_spread'] = features['last_hour'] - features['first_hour']

        return features

    def _calculate_streak(self, condition: pd.Series) -> pd.Series:
        """Calculate consecutive True values"""
        groups = (condition != condition.shift()).cumsum()
        streak = condition.groupby(groups).cumsum()
        return streak.where(condition, 0)
