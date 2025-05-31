"""
Simplified Feature Engineering for Risk Management MVP
Focus on essential features for personal models only
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Simplified feature engineering for personal trader models"""

    def __init__(self):
        self.feature_windows = [3, 5, 10, 20]  # Rolling windows for features

    def create_features(self, totals_df: pd.DataFrame, fills_df: pd.DataFrame) -> pd.DataFrame:
        """Create features from totals and fills data"""
        if totals_df.empty:
            return pd.DataFrame()

        # Start with totals data
        features = totals_df.copy()
        features = features.sort_values('date')

        # 1. Basic trading metrics
        features['win_rate'] = (features['net_pnl'] > 0).astype(int)
        features['loss_rate'] = (features['net_pnl'] < 0).astype(int)
        features['fee_ratio'] = features['total_fees'] / (features['gross_pnl'].abs() + 1e-8)

        # 2. Rolling statistics
        for window in self.feature_windows:
            # P&L statistics
            features[f'pnl_mean_{window}d'] = features['net_pnl'].rolling(window, min_periods=1).mean()
            features[f'pnl_std_{window}d'] = features['net_pnl'].rolling(window, min_periods=1).std()
            features[f'pnl_sum_{window}d'] = features['net_pnl'].rolling(window, min_periods=1).sum()

            # Win rate
            features[f'win_rate_{window}d'] = features['win_rate'].rolling(window, min_periods=1).mean()

            # Trading activity
            features[f'orders_mean_{window}d'] = features['orders_count'].rolling(window, min_periods=1).mean()
            features[f'volume_mean_{window}d'] = features['quantity'].rolling(window, min_periods=1).mean()

        # 3. Momentum indicators
        features['pnl_momentum_3d'] = features['net_pnl'].rolling(3, min_periods=1).sum()
        features['pnl_momentum_5d'] = features['net_pnl'].rolling(5, min_periods=1).sum()

        # 4. Streak features
        features['win_streak'] = self._calculate_streak(features['win_rate'])
        features['loss_streak'] = self._calculate_streak(features['loss_rate'])

        # 5. Behavioral features from fills if available
        if not fills_df.empty:
            behavioral_features = self._create_behavioral_features(fills_df, features['date'].unique())
            features = features.merge(behavioral_features, on='date', how='left')

        # 6. Time-based features
        features['day_of_week'] = pd.to_datetime(features['date']).dt.dayofweek
        features['is_monday'] = (features['day_of_week'] == 0).astype(int)
        features['is_friday'] = (features['day_of_week'] == 4).astype(int)
        features['day_of_month'] = pd.to_datetime(features['date']).dt.day

        # 7. Create target variable (next day's P&L)
        features['target'] = features['net_pnl'].shift(-1)
        features['target_binary'] = (features['target'] > 0).astype(int)

        # Clean up
        features = features.replace([np.inf, -np.inf], 0)
        features = features.fillna(0)

        # Remove last row (no target)
        features = features[:-1]

        logger.info(f"Created {len(features.columns)} features for {len(features)} trading days")

        return features

    def _calculate_streak(self, series: pd.Series) -> pd.Series:
        """Calculate consecutive streaks"""
        streak = series.groupby((series != series.shift()).cumsum()).cumcount() + 1
        return streak * series

    def _create_behavioral_features(self, fills_df: pd.DataFrame, dates: np.ndarray) -> pd.DataFrame:
        """Create behavioral features from fills data"""
        fills_df = fills_df.copy()
        fills_df['date'] = pd.to_datetime(fills_df['datetime']).dt.date

        behavioral = []

        for date in dates:
            date_fills = fills_df[fills_df['date'] == date]

            if date_fills.empty:
                behavioral.append({
                    'date': date,
                    'trading_hours': 0,
                    'symbols_traded': 0,
                    'avg_trade_size': 0,
                    'trade_frequency': 0,
                    'morning_activity': 0,
                    'afternoon_activity': 0
                })
            else:
                # Extract hour from datetime
                date_fills['hour'] = pd.to_datetime(date_fills['datetime']).dt.hour

                behavioral.append({
                    'date': date,
                    'trading_hours': date_fills['hour'].nunique(),
                    'symbols_traded': date_fills['symbol'].nunique(),
                    'avg_trade_size': date_fills['quantity'].mean(),
                    'trade_frequency': len(date_fills) / date_fills['hour'].nunique() if date_fills['hour'].nunique() > 0 else 0,
                    'morning_activity': (date_fills['hour'] < 12).sum() / len(date_fills),
                    'afternoon_activity': (date_fills['hour'] >= 12).sum() / len(date_fills)
                })

        return pd.DataFrame(behavioral)

    def get_feature_columns(self) -> List[str]:
        """Get list of feature columns for modeling"""
        # Base features
        features = [
            'orders_count', 'fills_count', 'quantity', 'gross_pnl', 'net_pnl',
            'total_fees', 'fee_ratio', 'win_rate', 'loss_rate',
            'pnl_momentum_3d', 'pnl_momentum_5d', 'win_streak', 'loss_streak',
            'day_of_week', 'is_monday', 'is_friday', 'day_of_month'
        ]

        # Add rolling features
        for window in [3, 5, 10, 20]:
            features.extend([
                f'pnl_mean_{window}d',
                f'pnl_std_{window}d',
                f'pnl_sum_{window}d',
                f'win_rate_{window}d',
                f'orders_mean_{window}d',
                f'volume_mean_{window}d'
            ])

        # Add behavioral features if available
        behavioral = [
            'trading_hours', 'symbols_traded', 'avg_trade_size',
            'trade_frequency', 'morning_activity', 'afternoon_activity'
        ]
        features.extend(behavioral)

        return features
