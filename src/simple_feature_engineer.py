import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


class SimpleFeatureEngineer:
    """Simplified feature engineering focused on essential behavioral signals"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def aggregate_fills_behavioral_signals(self, fills: pd.DataFrame) -> pd.DataFrame:
        """Extract 5 essential behavioral/emotional signals from fills data"""

        fills = fills.copy()
        fills['datetime'] = pd.to_datetime(fills['datetime'])
        fills['date'] = fills['datetime'].dt.date
        fills['hour'] = fills['datetime'].dt.hour
        fills['time_of_day'] = fills['hour'].apply(lambda x: 'morning' if x < 12 else 'afternoon')

        # Calculate trade-level metrics first
        fills['trade_size'] = fills['qty'] * fills['price']

        daily_behavioral = fills.groupby(['date', 'account_id']).agg({
            # 1. Trading Frequency - High frequency often indicates emotional trading
            'order_id': 'nunique',

            # 2. Trading Concentration - How spread out trading is during the day
            'hour': lambda x: x.nunique() / 24,  # Trading hour spread

            # 3. Symbol Diversification - Trading many symbols may indicate chasing
            'symbol': 'nunique',

            # 4. Trade Size Variability - Large variance indicates inconsistent sizing
            'trade_size': lambda x: x.std() / (x.mean() + 1e-8),

            # 5. Morning vs Afternoon behavior - Different patterns may indicate fatigue
            'time_of_day': lambda x: (x == 'morning').sum() / len(x) if len(x) > 0 else 0.5
        }).reset_index()

        daily_behavioral.columns = [
            'date', 'account_id',
            'trading_frequency',      # Signal 1: Number of orders
            'hour_concentration',     # Signal 2: How concentrated trading is
            'symbol_diversity',       # Signal 3: Number of different symbols
            'size_inconsistency',     # Signal 4: Trade size coefficient of variation
            'morning_bias'           # Signal 5: Proportion of morning trades
        ]

        return daily_behavioral

    def extract_essential_totals_metrics(self, totals: pd.DataFrame) -> pd.DataFrame:
        """Extract 10 essential metrics from totals data"""

        totals = totals.copy()
        totals = totals.sort_values(['account_id', 'date'])

        # Core performance metrics
        metrics = totals[['date', 'account_id', 'net_pnl', 'gross_pnl',
                         'total_fees', 'qty', 'orders_count', 'fills_count']].copy()

        # Additional calculated metrics
        metrics['fee_ratio'] = metrics['total_fees'] / (np.abs(metrics['gross_pnl']) + 1e-8)
        metrics['avg_fill_size'] = metrics['qty'] / (metrics['fills_count'] + 1e-8)
        metrics['fills_per_order'] = metrics['fills_count'] / (metrics['orders_count'] + 1e-8)

        # Rolling metrics (5-day window for responsiveness)
        for col in ['net_pnl', 'qty']:
            metrics[f'{col}_5d_avg'] = metrics.groupby('account_id')[col].transform(
                lambda x: x.rolling(5, min_periods=1).mean()
            )

        # Cumulative win rate
        metrics['is_win'] = (metrics['net_pnl'] > 0).astype(int)
        metrics['cum_win_rate'] = metrics.groupby('account_id')['is_win'].transform(
            lambda x: x.expanding().mean()
        )

        # Recent momentum (3-day)
        metrics['momentum_3d'] = metrics.groupby('account_id')['net_pnl'].transform(
            lambda x: x.rolling(3, min_periods=1).sum()
        )

        # Select final 10 metrics
        final_metrics = metrics[[
            'date', 'account_id',
            'net_pnl',           # 1. Daily net P&L
            'gross_pnl',         # 2. Daily gross P&L
            'total_fees',        # 3. Total fees paid
            'fee_ratio',         # 4. Fees as ratio of gross
            'qty',               # 5. Total quantity traded
            'orders_count',      # 6. Number of orders
            'avg_fill_size',     # 7. Average fill size
            'net_pnl_5d_avg',    # 8. 5-day average P&L
            'cum_win_rate',      # 9. Cumulative win rate
            'momentum_3d'        # 10. 3-day momentum
        ]]

        return final_metrics

    def create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create binary target: will next day be profitable?"""

        df = df.copy()
        df = df.sort_values(['account_id', 'date'])

        # Next day's P&L
        df['next_day_pnl'] = df.groupby('account_id')['net_pnl'].shift(-1)

        # Binary target: 1 if profitable, 0 if loss
        df['target'] = (df['next_day_pnl'] > 0).astype(int)

        # Also keep the actual value for evaluation
        df['target_value'] = df['next_day_pnl']

        # Remove last day per trader (no target)
        df = df.dropna(subset=['target'])

        return df

    def engineer_features(self, totals: pd.DataFrame, fills: pd.DataFrame) -> pd.DataFrame:
        """Main pipeline combining behavioral signals and essential metrics"""

        self.logger.info("Engineering simplified features...")

        # Get behavioral signals from fills
        behavioral_signals = self.aggregate_fills_behavioral_signals(fills)

        # Get essential metrics from totals
        essential_metrics = self.extract_essential_totals_metrics(totals)

        # Merge behavioral and totals data
        essential_metrics['date'] = pd.to_datetime(essential_metrics['date'])
        behavioral_signals['date'] = pd.to_datetime(behavioral_signals['date'])

        features = essential_metrics.merge(
            behavioral_signals,
            on=['date', 'account_id'],
            how='left'
        )

        # Fill missing behavioral signals with defaults
        behavioral_cols = ['trading_frequency', 'hour_concentration',
                          'symbol_diversity', 'size_inconsistency', 'morning_bias']
        for col in behavioral_cols:
            features[col] = features[col].fillna(features[col].median())

        # Add temporal features
        features['date'] = pd.to_datetime(features['date'])
        features['day_of_week'] = features['date'].dt.dayofweek
        features['is_monday'] = (features['day_of_week'] == 0).astype(int)
        features['is_friday'] = (features['day_of_week'] == 4).astype(int)

        # Create target
        features = self.create_target(features)

        # Final cleanup
        features = features.replace([np.inf, -np.inf], 0)
        features = features.fillna(0)

        self.logger.info(f"Feature engineering complete. Shape: {features.shape}")

        return features

    def get_feature_columns(self) -> List[str]:
        """Return list of feature columns for modeling"""

        return [
            # Behavioral signals (5)
            'trading_frequency', 'hour_concentration', 'symbol_diversity',
            'size_inconsistency', 'morning_bias',

            # Essential metrics (10)
            'net_pnl', 'gross_pnl', 'total_fees', 'fee_ratio', 'qty',
            'orders_count', 'avg_fill_size', 'net_pnl_5d_avg',
            'cum_win_rate', 'momentum_3d',

            # Temporal features (3)
            'day_of_week', 'is_monday', 'is_friday'
        ]
