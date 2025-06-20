import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime, timedelta

class FeatureEngineer:
    def __init__(self, ewma_spans: List[int] = [5, 20]):
        """
        Initialize feature engineer with EWMA spans for rolling features.

        Args:
            ewma_spans: List of spans for exponentially weighted moving averages
        """
        self.ewma_spans = ewma_spans

    def create_daily_features(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive daily feature set for each trader.
        Focus on realized PNL prediction only.
        """
        # Aggregate trades to daily level first
        daily_agg = self._aggregate_daily_trades(trades_df)

        # Create complete daily timeline for each trader
        complete_timeline = self._create_complete_timeline(daily_agg)

        # Generate core features
        features_df = self._generate_core_features(complete_timeline)

        # Add EWMA rolling features
        features_df = self._add_ewma_features(features_df)

        # Add lagged features
        features_df = self._add_lagged_features(features_df)

        # Add volatility features
        features_df = self._add_volatility_features(features_df)

        return features_df

    def _aggregate_daily_trades(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate individual trades to daily level per trader."""
        daily_agg = trades_df.groupby(['account_id', 'trade_date']).agg({
            'realized_pnl': ['sum', 'count', 'mean', 'std'],
            'is_winner': ['sum', 'mean']
        }).reset_index()

        # Flatten column names
        daily_agg.columns = [
            'account_id', 'trade_date', 'realized_pnl', 'num_trades',
            'avg_trade_pnl', 'pnl_std', 'num_winners', 'win_rate'
        ]

        # Handle NaN values
        daily_agg['pnl_std'] = daily_agg['pnl_std'].fillna(0)

        return daily_agg

    def _create_complete_timeline(self, daily_agg: pd.DataFrame) -> pd.DataFrame:
        """
        Create complete daily timeline for each trader, filling gaps.
        Activity-based metrics filled with 0, state-based with forward-fill.
        """
        # Get date range
        min_date = daily_agg['trade_date'].min()
        max_date = daily_agg['trade_date'].max()

        # Create complete date range
        date_range = pd.date_range(start=min_date, end=max_date, freq='D')

        # Get all unique traders
        traders = daily_agg['account_id'].unique()

        # Create complete timeline
        complete_timeline = []
        for trader in traders:
            trader_dates = pd.DataFrame({
                'account_id': trader,
                'trade_date': date_range
            })
            complete_timeline.append(trader_dates)

        complete_df = pd.concat(complete_timeline, ignore_index=True)

        # Merge with actual data
        merged_df = complete_df.merge(daily_agg, on=['account_id', 'trade_date'], how='left')

        # Fill activity-based metrics with 0
        activity_cols = ['realized_pnl', 'num_trades', 'num_winners']
        merged_df[activity_cols] = merged_df[activity_cols].fillna(0)

        # Fill rate-based metrics properly
        merged_df['win_rate'] = merged_df['win_rate'].fillna(0)
        merged_df['avg_trade_pnl'] = merged_df['avg_trade_pnl'].fillna(0)
        merged_df['pnl_std'] = merged_df['pnl_std'].fillna(0)

        return merged_df.sort_values(['account_id', 'trade_date']).reset_index(drop=True)

    def _generate_core_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate core daily features."""
        features_df = df.copy()

        # Days since last trade feature
        features_df['days_since_last_trade'] = 0

        for account_id in features_df['account_id'].unique():
            mask = features_df['account_id'] == account_id
            account_data = features_df[mask].copy()

            days_since = []
            last_trade_day = 0

            for idx, row in account_data.iterrows():
                if row['num_trades'] > 0:
                    last_trade_day = 0
                else:
                    last_trade_day += 1
                days_since.append(last_trade_day)

            features_df.loc[mask, 'days_since_last_trade'] = days_since

        # Profit factor (avoid division by zero)
        total_wins = features_df['num_winners'] * features_df['avg_trade_pnl']
        total_losses = (features_df['num_trades'] - features_df['num_winners']) * features_df['avg_trade_pnl']

        features_df['profit_factor'] = np.where(
            total_losses < 0,
            total_wins / abs(total_losses),
            0
        )

        return features_df

    def _add_ewma_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add exponentially weighted moving average features."""
        features_df = df.copy()

        for span in self.ewma_spans:
            for account_id in features_df['account_id'].unique():
                mask = features_df['account_id'] == account_id
                account_data = features_df[mask]

                # EWMA features
                features_df.loc[mask, f'realized_pnl_ewma_{span}'] = (
                    account_data['realized_pnl'].ewm(span=span).mean()
                )
                features_df.loc[mask, f'win_rate_ewma_{span}'] = (
                    account_data['win_rate'].ewm(span=span).mean()
                )
                features_df.loc[mask, f'profit_factor_ewma_{span}'] = (
                    account_data['profit_factor'].ewm(span=span).mean()
                )
                features_df.loc[mask, f'num_trades_ewma_{span}'] = (
                    account_data['num_trades'].ewm(span=span).mean()
                )

        return features_df

    def _add_lagged_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged features."""
        features_df = df.copy()

        lag_features = ['realized_pnl', 'win_rate', 'profit_factor', 'num_trades']
        lags = [1, 2, 3]

        for feature in lag_features:
            for lag in lags:
                features_df[f'{feature}_lag_{lag}'] = (
                    features_df.groupby('account_id')[feature].shift(lag)
                )

        return features_df

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling volatility features."""
        features_df = df.copy()

        for span in [10, 20]:
            for account_id in features_df['account_id'].unique():
                mask = features_df['account_id'] == account_id
                account_data = features_df[mask]

                # Rolling volatility using EWMA
                features_df.loc[mask, f'pnl_volatility_{span}'] = (
                    account_data['realized_pnl'].ewm(span=span).std()
                )

        return features_df

    def create_target_variable(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create target variable: next day's realized PNL normalized by recent volatility.
        """
        target_df = features_df.copy()

        # Create next day's realized PNL
        target_df['next_day_realized_pnl'] = (
            target_df.groupby('account_id')['realized_pnl'].shift(-1)
        )

        # Normalize by 20-day volatility (use 10-day if 20-day not available)
        vol_20 = target_df['pnl_volatility_20'].fillna(target_df['pnl_volatility_10'])
        vol_20 = vol_20.fillna(1)  # Fallback to 1 if no volatility available

        # Avoid division by very small numbers
        vol_20 = np.where(vol_20 < 0.01, 0.01, vol_20)

        target_df['target'] = target_df['next_day_realized_pnl'] / vol_20

        return target_df
