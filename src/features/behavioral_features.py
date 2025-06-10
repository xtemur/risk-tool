"""
Behavioral Features for Trading
Captures trader psychology, habits, and behavioral patterns
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List
import logging
from datetime import time

from src.features.base_features import BaseFeatures
from src.core.constants import TradingConstants as TC

logger = logging.getLogger(__name__)


class BehavioralFeatures(BaseFeatures):
    """
    Behavioral features capturing trader psychology and habits
    Based on behavioral finance research
    """

    def __init__(self):
        super().__init__(feature_prefix='behav', lookback_days=60)

    def create_features(self,
                       totals_df: pd.DataFrame,
                       fills_df: Optional[pd.DataFrame] = None,
                       as_of_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """
        Create behavioral features from trading data

        Features include:
        - Loss aversion indicators
        - Overconfidence measures
        - Disposition effect
        - Trading time preferences
        - Emotional state proxies
        """
        # Prepare data
        df = self._ensure_datetime_index(totals_df.copy())
        df = self._apply_point_in_time(df, as_of_date)

        if df.empty:
            return pd.DataFrame()

        # Initialize feature container
        all_features = []

        # Process each account
        for account_id in df['account_id'].unique():
            acc_df = df[df['account_id'] == account_id].sort_index()

            if len(acc_df) < 5:
                continue

            features = pd.DataFrame(index=acc_df.index)
            features['account_id'] = account_id

            # 1. Loss aversion features
            features = self._add_loss_aversion_features(features, acc_df)

            # 2. Overconfidence indicators
            features = self._add_overconfidence_features(features, acc_df)

            # 3. Disposition effect
            features = self._add_disposition_features(features, acc_df)

            # 4. Trading time patterns (if fills available)
            if fills_df is not None and not fills_df.empty:
                acc_fills = fills_df[fills_df['account_id'] == account_id]
                features = self._add_time_pattern_features(features, acc_df, acc_fills)

            # 5. Emotional state proxies
            features = self._add_emotional_features(features, acc_df)

            # 6. Consistency metrics
            features = self._add_consistency_features(features, acc_df)

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

    def _add_loss_aversion_features(self, features: pd.DataFrame, acc_df: pd.DataFrame) -> pd.DataFrame:
        """
        Features related to loss aversion behavior
        Research shows traders feel losses ~2x more than equivalent gains
        """

        # Reaction to losses
        is_loss = acc_df['net_pnl'] < 0

        # Post-loss behavior
        features['post_loss_activity'] = is_loss.shift(1).rolling(5).mean()

        # Loss recovery pressure (cumulative losses)
        cumulative_loss = acc_df['net_pnl'].copy()
        cumulative_loss[cumulative_loss > 0] = 0
        features['cumulative_loss_5d'] = cumulative_loss.rolling(5).sum()
        features['cumulative_loss_20d'] = cumulative_loss.rolling(20).sum()

        # Break-even effect (distance from high water mark)
        cumsum = acc_df['net_pnl'].cumsum()
        running_max = cumsum.expanding().max()
        features['distance_from_hwm'] = cumsum - running_max
        features['days_below_hwm'] = (features['distance_from_hwm'] < 0).rolling(20).sum()

        # Loss magnitude sensitivity
        avg_loss = acc_df['net_pnl'][is_loss].rolling(20).mean()
        avg_gain = acc_df['net_pnl'][~is_loss].rolling(20).mean()
        features['loss_gain_ratio'] = avg_loss.abs() / (avg_gain + TC.MIN_VARIANCE)

        # Consecutive loss impact
        loss_streak = self._calculate_streak(is_loss)
        features['current_loss_streak'] = loss_streak
        features['max_loss_streak_20d'] = loss_streak.rolling(20).max()

        # Risk taking after losses (measured by position size changes)
        features['post_loss_volume_change'] = (
            acc_df['quantity'].pct_change().where(is_loss.shift(1), np.nan)
        ).rolling(10).mean()

        return features

    def _add_overconfidence_features(self, features: pd.DataFrame, acc_df: pd.DataFrame) -> pd.DataFrame:
        """
        Features indicating overconfidence
        Overconfident traders trade more and take larger positions after wins
        """

        is_win = acc_df['net_pnl'] > 0

        # Hot hand fallacy (winning streak effects)
        win_streak = self._calculate_streak(is_win)
        features['current_win_streak'] = win_streak
        features['max_win_streak_20d'] = win_streak.rolling(20).max()

        # Post-win behavior
        features['post_win_activity'] = is_win.shift(1).rolling(5).mean()

        # Overtrading indicators
        features['trade_frequency'] = acc_df['orders_count'].rolling(20).mean()
        features['trade_frequency_change'] = (
            acc_df['orders_count'].rolling(5).mean() /
            (acc_df['orders_count'].rolling(20).mean() + TC.MIN_VARIANCE)
        )

        # Position sizing after wins
        features['post_win_size_change'] = (
            acc_df['quantity'].pct_change().where(is_win.shift(1), np.nan)
        ).rolling(10).mean()

        # Volatility of returns (overconfident traders have more volatile results)
        features['return_volatility_ratio'] = (
            acc_df['net_pnl'].rolling(10).std() /
            (acc_df['net_pnl'].rolling(30).std() + TC.MIN_VARIANCE)
        )

        # Attribution bias (wins vs losses frequency)
        features['win_rate_20d'] = is_win.rolling(20).mean()
        features['win_rate_change'] = (
            is_win.rolling(5).mean() - is_win.rolling(20).mean()
        )

        # Illusion of control (trading on specific days/times)
        features['dow_concentration'] = self._calculate_dow_concentration(acc_df)

        return features

    def _add_disposition_features(self, features: pd.DataFrame, acc_df: pd.DataFrame) -> pd.DataFrame:
        """
        Disposition effect: tendency to sell winners too early and hold losers too long
        """

        # Profit taking behavior
        is_profitable = acc_df['net_pnl'] > 0

        # Quick profit taking (small wins)
        small_win_threshold = acc_df['net_pnl'].quantile(0.6)  # 60th percentile
        small_wins = (acc_df['net_pnl'] > 0) & (acc_df['net_pnl'] < small_win_threshold)
        features['small_win_frequency'] = small_wins.rolling(20).mean()

        # Large loss frequency (holding losers)
        large_loss_threshold = acc_df['net_pnl'].quantile(0.1)  # 10th percentile
        large_losses = acc_df['net_pnl'] < large_loss_threshold
        features['large_loss_frequency'] = large_losses.rolling(20).mean()

        # Win/loss size ratio
        avg_win = acc_df['net_pnl'][is_profitable].rolling(20).mean()
        avg_loss = acc_df['net_pnl'][~is_profitable].rolling(20).mean()
        features['avg_win_loss_ratio'] = avg_win / (avg_loss.abs() + TC.MIN_VARIANCE)

        # Realized vs unrealized P&L patterns
        if 'unrealized_delta' in acc_df.columns:
            features['unrealized_ratio'] = (
                acc_df['unrealized_delta'] /
                (acc_df['net_pnl'].abs() + TC.MIN_VARIANCE)
            )

            # Tendency to realize gains vs losses
            realize_gains = (acc_df['net_pnl'] > 0) & (acc_df['unrealized_delta'] < 0)
            realize_losses = (acc_df['net_pnl'] < 0) & (acc_df['unrealized_delta'] > 0)

            features['gain_realization_rate'] = realize_gains.rolling(20).mean()
            features['loss_realization_rate'] = realize_losses.rolling(20).mean()

        return features

    def _add_time_pattern_features(self,
                                  features: pd.DataFrame,
                                  acc_df: pd.DataFrame,
                                  fills_df: pd.DataFrame) -> pd.DataFrame:
        """
        Trading time patterns and preferences
        """

        if fills_df.empty:
            return features

        fills_df = fills_df.copy()
        if 'datetime' in fills_df.columns:
            fills_df['date'] = pd.to_datetime(fills_df['datetime']).dt.date
            fills_df['hour'] = pd.to_datetime(fills_df['datetime']).dt.hour
            fills_df['minute'] = pd.to_datetime(fills_df['datetime']).dt.minute
            fills_df['time_of_day'] = (
                fills_df['hour'] + fills_df['minute'] / 60
            )

        # Daily patterns
        for date in features.index:
            date_fills = fills_df[fills_df['date'] == date.date()]

            if date_fills.empty or 'hour' not in date_fills.columns:
                continue

            # Time of day preferences
            features.loc[date, 'avg_trade_time'] = date_fills['time_of_day'].mean()
            features.loc[date, 'trade_time_std'] = date_fills['time_of_day'].std()

            # Opening bell activity (first 30 minutes)
            opening_trades = date_fills[
                (date_fills['hour'] == 9) & (date_fills['minute'] <= 30)
            ]
            features.loc[date, 'opening_activity'] = (
                len(opening_trades) / (len(date_fills) + TC.MIN_VARIANCE)
            )

            # Closing bell activity (last 30 minutes)
            closing_trades = date_fills[
                (date_fills['hour'] == 15) & (date_fills['minute'] >= 30)
            ]
            features.loc[date, 'closing_activity'] = (
                len(closing_trades) / (len(date_fills) + TC.MIN_VARIANCE)
            )

            # Lunch hour avoidance
            lunch_trades = date_fills[
                (date_fills['hour'] >= 12) & (date_fills['hour'] < 13)
            ]
            features.loc[date, 'lunch_avoidance'] = 1 - (
                len(lunch_trades) / (len(date_fills) + TC.MIN_VARIANCE)
            )

            # Trading session distribution entropy
            session_dist = date_fills['hour'].value_counts(normalize=True)
            features.loc[date, 'time_entropy'] = -(session_dist * np.log(session_dist + 1e-8)).sum()

        # Fill forward time-based features
        time_cols = ['avg_trade_time', 'trade_time_std', 'opening_activity',
                    'closing_activity', 'lunch_avoidance', 'time_entropy']
        for col in time_cols:
            if col in features.columns:
                features[col] = features[col].fillna(method='ffill', limit=5)

        # Day of week patterns
        features['day_of_week'] = features.index.dayofweek
        features['is_monday'] = (features['day_of_week'] == 0).astype(int)
        features['is_friday'] = (features['day_of_week'] == 4).astype(int)

        # Weekend effect (Monday vs other days)
        monday_pnl = acc_df['net_pnl'][features['is_monday'] == 1].mean()
        other_pnl = acc_df['net_pnl'][features['is_monday'] == 0].mean()
        features['monday_effect'] = monday_pnl - other_pnl

        return features

    def _add_emotional_features(self, features: pd.DataFrame, acc_df: pd.DataFrame) -> pd.DataFrame:
        """
        Proxies for emotional state and its impact on trading
        """

        # Revenge trading (increased activity after losses)
        is_loss = acc_df['net_pnl'] < 0

        features['post_loss_orders'] = (
            acc_df['orders_count'].where(is_loss.shift(1), np.nan)
        ).rolling(5).mean()

        features['revenge_trading_score'] = (
            features['post_loss_orders'] /
            (acc_df['orders_count'].rolling(20).mean() + TC.MIN_VARIANCE)
        )

        # Tilt indicator (emotional instability)
        # High variance in order frequency suggests emotional trading
        features['order_variance'] = acc_df['orders_count'].rolling(10).std()
        features['tilt_score'] = (
            features['order_variance'] /
            (acc_df['orders_count'].rolling(30).std() + TC.MIN_VARIANCE)
        )

        # Fear indicator (reduced activity after large losses)
        large_loss = acc_df['net_pnl'] < acc_df['net_pnl'].rolling(60).quantile(0.1)
        features['post_large_loss_activity'] = (
            acc_df['orders_count'].where(large_loss.shift(1), np.nan)
        ).rolling(5).mean()

        # Greed indicator (increased size after wins)
        large_win = acc_df['net_pnl'] > acc_df['net_pnl'].rolling(60).quantile(0.9)
        features['post_large_win_size'] = (
            acc_df['quantity'].where(large_win.shift(1), np.nan)
        ).rolling(5).mean()

        # Stress indicator (based on drawdown)
        cumsum = acc_df['net_pnl'].cumsum()
        running_max = cumsum.expanding().max()
        drawdown_pct = (cumsum - running_max) / (running_max + TC.MIN_VARIANCE)

        features['stress_level'] = drawdown_pct.abs().rolling(10).mean()
        features['max_stress_20d'] = drawdown_pct.abs().rolling(20).max()

        # Recovery time from losses
        features['recovery_speed'] = self._calculate_recovery_speed(acc_df['net_pnl'])

        return features

    def _add_consistency_features(self, features: pd.DataFrame, acc_df: pd.DataFrame) -> pd.DataFrame:
        """
        Trading consistency and discipline metrics
        """

        # P&L consistency
        features['pnl_consistency'] = (
            acc_df['net_pnl'].rolling(20).std() /
            (acc_df['net_pnl'].abs().rolling(20).mean() + TC.MIN_VARIANCE)
        )

        # Trading frequency consistency
        features['frequency_consistency'] = (
            acc_df['orders_count'].rolling(20).std() /
            (acc_df['orders_count'].rolling(20).mean() + TC.MIN_VARIANCE)
        )

        # Size consistency
        features['size_consistency'] = (
            acc_df['quantity'].rolling(20).std() /
            (acc_df['quantity'].rolling(20).mean() + TC.MIN_VARIANCE)
        )

        # Time consistency (if we have fill times)
        if 'avg_trade_time' in features.columns:
            features['time_consistency'] = features['avg_trade_time'].rolling(10).std()

        # Strategy drift (changes in trading patterns)
        # Compare recent to historical behavior
        for metric in ['orders_count', 'quantity', 'net_pnl']:
            recent = acc_df[metric].rolling(10).mean()
            historical = acc_df[metric].rolling(60).mean()
            features[f'{metric}_drift'] = (
                (recent - historical) / (historical.abs() + TC.MIN_VARIANCE)
            )

        # Discipline score (inverse of emotional indicators)
        if all(col in features.columns for col in ['tilt_score', 'revenge_trading_score']):
            features['discipline_score'] = 1 - (
                features['tilt_score'] + features['revenge_trading_score']
            ) / 2

        return features

    def _calculate_dow_concentration(self, acc_df: pd.DataFrame) -> pd.Series:
        """Calculate concentration of trading on specific days of week"""
        dow_counts = pd.Series(index=acc_df.index, dtype=float)

        # Rolling window calculation
        for i in range(20, len(acc_df)):
            window = acc_df.iloc[i-20:i]
            window_dow = pd.to_datetime(window.index).dayofweek

            # Calculate entropy of day-of-week distribution
            dow_dist = pd.Series(window_dow).value_counts(normalize=True)
            entropy = -(dow_dist * np.log(dow_dist + 1e-8)).sum()

            # Normalize to 0-1 (lower entropy = higher concentration)
            max_entropy = -np.log(1/5)  # 5 trading days
            concentration = 1 - (entropy / max_entropy)

            dow_counts.iloc[i] = concentration

        return dow_counts

    def _calculate_recovery_speed(self, pnl_series: pd.Series) -> pd.Series:
        """Calculate how quickly trader recovers from losses"""
        recovery_speed = pd.Series(index=pnl_series.index, dtype=float)

        cumsum = pnl_series.cumsum()

        for i in range(1, len(pnl_series)):
            if pnl_series.iloc[i-1] < 0:  # Previous day was a loss
                # Find how many days to recover
                loss_amount = abs(pnl_series.iloc[i-1])
                recovery_sum = 0
                days_to_recover = 0

                for j in range(i, min(i+20, len(pnl_series))):  # Look ahead max 20 days
                    recovery_sum += pnl_series.iloc[j]
                    days_to_recover += 1

                    if recovery_sum >= loss_amount:
                        break

                # Inverse of days (faster recovery = higher score)
                recovery_speed.iloc[i] = 1 / (days_to_recover + 1)

        return recovery_speed.rolling(10).mean()
