#!/usr/bin/env python3
"""
Feature Engineering
Migrated from step2_feature_engineering.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    def __init__(self, daily_df_path='data/daily_aggregated.pkl'):
        self.daily_df = pd.read_pickle(daily_df_path)
        self.feature_df = None

    def create_basic_features(self):
        """Create basic features as specified in CLAUDE.md"""
        print("=== FEATURE ENGINEERING ===")
        print("Creating basic features...")

        # Sort by trader and date
        self.daily_df = self.daily_df.sort_values(['account_id', 'trade_date'])

        feature_dfs = []

        for trader_id in self.daily_df['account_id'].unique():
            trader_df = self.daily_df[self.daily_df['account_id'] == trader_id].copy()

            # Basic features (already have some)
            trader_df['realized_pnl'] = trader_df['realized_pnl']  # Already exists
            trader_df['num_trades'] = trader_df['num_trades']      # Already exists
            trader_df['win_rate'] = trader_df['win_rate']          # Already exists

            # Calculate additional basic features
            trader_df['avg_trade_size'] = trader_df['realized_pnl'] / np.maximum(trader_df['num_trades'], 1)

            # Profit factor (handle division by zero)
            wins = trader_df['realized_pnl'] * (trader_df['realized_pnl'] > 0)
            losses = abs(trader_df['realized_pnl'] * (trader_df['realized_pnl'] < 0))

            trader_df['daily_wins'] = wins
            trader_df['daily_losses'] = losses

            # Calculate cumulative wins/losses for profit factor
            trader_df['cum_wins'] = trader_df['daily_wins'].cumsum()
            trader_df['cum_losses'] = trader_df['daily_losses'].cumsum()
            trader_df['profit_factor'] = trader_df['cum_wins'] / np.maximum(trader_df['cum_losses'], 1)

            # Consecutive wins/losses
            trader_df['is_win_day'] = (trader_df['realized_pnl'] > 0).astype(int)
            trader_df['is_loss_day'] = (trader_df['realized_pnl'] < 0).astype(int)

            # Calculate consecutive streaks
            trader_df['consecutive_wins'] = self._calculate_consecutive(trader_df['is_win_day'])
            trader_df['consecutive_losses'] = self._calculate_consecutive(trader_df['is_loss_day'])

            # Max single day gains/losses (rolling)
            trader_df['max_single_day_gain'] = trader_df['realized_pnl'].expanding().max()
            trader_df['max_single_day_loss'] = trader_df['realized_pnl'].expanding().min()

            # Days since last trade
            active_days = trader_df[trader_df['num_trades'] > 0]['trade_date']
            trader_df['days_since_last_trade'] = 0

            for idx, row in trader_df.iterrows():
                if row['num_trades'] == 0:
                    last_active = active_days[active_days < row['trade_date']]
                    if len(last_active) > 0:
                        trader_df.loc[idx, 'days_since_last_trade'] = (
                            row['trade_date'] - last_active.max()
                        ).days
                    else:
                        trader_df.loc[idx, 'days_since_last_trade'] = 999  # No previous trading

            # Trading frequency (trades per week over rolling window)
            trader_df['trading_frequency'] = trader_df['num_trades'].rolling(window=7, min_periods=1).sum()

            feature_dfs.append(trader_df)

        self.feature_df = pd.concat(feature_dfs, ignore_index=True)
        print(f"✓ Created basic features for {self.feature_df['account_id'].nunique()} traders")

    def create_rolling_features_ewma(self):
        """Create rolling features using EWMA as specified"""
        print("Creating EWMA rolling features...")

        feature_dfs = []

        for trader_id in self.feature_df['account_id'].unique():
            trader_df = self.feature_df[self.feature_df['account_id'] == trader_id].copy()
            trader_df = trader_df.sort_values('trade_date')

            # EWMA parameters
            alpha_5 = 2 / (5 + 1)   # 5-day EWMA
            alpha_20 = 2 / (20 + 1)  # 20-day EWMA

            # 5-day EWMA features
            trader_df['realized_pnl_ewma5'] = trader_df['realized_pnl'].ewm(alpha=alpha_5).mean()
            trader_df['win_rate_ewma5'] = trader_df['win_rate'].ewm(alpha=alpha_5).mean()
            trader_df['profit_factor_ewma5'] = trader_df['profit_factor'].ewm(alpha=alpha_5).mean()
            trader_df['volatility_ewma5'] = trader_df['realized_pnl'].ewm(alpha=alpha_5).std()
            trader_df['num_trades_ewma5'] = trader_df['num_trades'].ewm(alpha=alpha_5).mean()

            # 20-day EWMA features
            trader_df['realized_pnl_ewma20'] = trader_df['realized_pnl'].ewm(alpha=alpha_20).mean()
            trader_df['win_rate_ewma20'] = trader_df['win_rate'].ewm(alpha=alpha_20).mean()
            trader_df['profit_factor_ewma20'] = trader_df['profit_factor'].ewm(alpha=alpha_20).mean()
            trader_df['volatility_ewma20'] = trader_df['realized_pnl'].ewm(alpha=alpha_20).std()
            trader_df['num_trades_ewma20'] = trader_df['num_trades'].ewm(alpha=alpha_20).mean()

            # Fill NaN values from initial periods
            ewma_cols = [col for col in trader_df.columns if 'ewma' in col]
            for col in ewma_cols:
                trader_df[col] = trader_df[col].fillna(method='bfill').fillna(0)

            feature_dfs.append(trader_df)

        self.feature_df = pd.concat(feature_dfs, ignore_index=True)
        print(f"✓ Created EWMA features")

    def create_lagged_features(self):
        """Create lagged features for important variables"""
        print("Creating lagged features...")

        feature_dfs = []

        for trader_id in self.feature_df['account_id'].unique():
            trader_df = self.feature_df[self.feature_df['account_id'] == trader_id].copy()
            trader_df = trader_df.sort_values('trade_date')

            # Key features to lag
            lag_features = [
                'realized_pnl', 'win_rate', 'profit_factor', 'num_trades',
                'realized_pnl_ewma5', 'win_rate_ewma5', 'volatility_ewma5'
            ]

            # Create 1-3 day lags
            for feature in lag_features:
                if feature in trader_df.columns:
                    trader_df[f'{feature}_lag1'] = trader_df[feature].shift(1)
                    trader_df[f'{feature}_lag2'] = trader_df[feature].shift(2)
                    trader_df[f'{feature}_lag3'] = trader_df[feature].shift(3)

            # Fill NaN values for initial periods
            lag_cols = [col for col in trader_df.columns if '_lag' in col]
            for col in lag_cols:
                trader_df[col] = trader_df[col].fillna(0)

            feature_dfs.append(trader_df)

        self.feature_df = pd.concat(feature_dfs, ignore_index=True)
        print(f"✓ Created lagged features")

    def create_advanced_features(self):
        """Create advanced features only if basic features work"""
        print("Creating advanced features...")

        feature_dfs = []

        for trader_id in self.feature_df['account_id'].unique():
            trader_df = self.feature_df[self.feature_df['account_id'] == trader_id].copy()
            trader_df = trader_df.sort_values('trade_date')

            # Volatility regime indicators
            vol_rolling = trader_df['realized_pnl'].rolling(window=20, min_periods=5).std()
            vol_mean = vol_rolling.mean()
            trader_df['high_vol_regime'] = (vol_rolling > vol_mean * 1.5).astype(int)
            trader_df['low_vol_regime'] = (vol_rolling < vol_mean * 0.5).astype(int)

            # Drawdown features
            cumulative_pnl = trader_df['realized_pnl'].cumsum()
            running_max = cumulative_pnl.expanding().max()
            drawdown = cumulative_pnl - running_max
            trader_df['current_drawdown'] = drawdown
            trader_df['max_drawdown'] = drawdown.expanding().min()

            # Drawdown recovery
            trader_df['drawdown_recovery'] = (drawdown > drawdown.shift(1)).astype(int)

            # Risk-adjusted performance (Sharpe-like)
            rolling_mean = trader_df['realized_pnl'].rolling(window=20, min_periods=5).mean()
            rolling_std = trader_df['realized_pnl'].rolling(window=20, min_periods=5).std()
            trader_df['sharpe_ratio'] = rolling_mean / np.maximum(rolling_std, 1)

            # Performance relative to recent history
            trader_df['pnl_vs_recent'] = (
                trader_df['realized_pnl'] - trader_df['realized_pnl_ewma20']
            ) / np.maximum(trader_df['volatility_ewma20'], 1)

            # Trading intensity features
            trader_df['high_activity'] = (trader_df['num_trades'] > trader_df['num_trades_ewma20'] * 1.5).astype(int)
            trader_df['low_activity'] = (trader_df['num_trades'] < trader_df['num_trades_ewma20'] * 0.5).astype(int)

            # Fill NaN values
            advanced_cols = [
                'high_vol_regime', 'low_vol_regime', 'current_drawdown', 'max_drawdown',
                'drawdown_recovery', 'sharpe_ratio', 'pnl_vs_recent', 'high_activity', 'low_activity'
            ]
            for col in advanced_cols:
                if col in trader_df.columns:
                    trader_df[col] = trader_df[col].fillna(0)

            feature_dfs.append(trader_df)

        self.feature_df = pd.concat(feature_dfs, ignore_index=True)
        print(f"✓ Created advanced features")

    def _calculate_consecutive(self, series):
        """Helper function to calculate consecutive streaks"""
        consecutive = []
        current_streak = 0

        for value in series:
            if value == 1:
                current_streak += 1
            else:
                current_streak = 0
            consecutive.append(current_streak)

        return consecutive

    def validate_features(self):
        """Validate feature correlation with targets"""
        print("\n=== FEATURE VALIDATION ===")

        # Create a simple next-day target for validation
        validation_results = {}

        for trader_id in self.feature_df['account_id'].unique()[:10]:  # Sample validation
            trader_df = self.feature_df[self.feature_df['account_id'] == trader_id].copy()
            trader_df = trader_df.sort_values('trade_date')

            # Create next-day target
            trader_df['next_day_pnl'] = trader_df['realized_pnl'].shift(-1)

            # Remove rows with missing target
            valid_data = trader_df.dropna(subset=['next_day_pnl'])

            if len(valid_data) < 20:
                continue

            # Test key feature correlations
            feature_correlations = {}
            key_features = [
                'realized_pnl_lag1', 'win_rate_lag1', 'profit_factor_lag1',
                'volatility_ewma5', 'consecutive_losses', 'current_drawdown'
            ]

            for feature in key_features:
                if feature in valid_data.columns:
                    corr = valid_data[feature].corr(valid_data['next_day_pnl'])
                    feature_correlations[feature] = corr

            validation_results[trader_id] = feature_correlations

        # Summarize validation results
        if validation_results:
            print(f"✓ Validated features for {len(validation_results)} traders")

            # Average correlations across traders
            all_features = set()
            for trader_corrs in validation_results.values():
                all_features.update(trader_corrs.keys())

            avg_correlations = {}
            for feature in all_features:
                corrs = [validation_results[trader].get(feature, 0)
                        for trader in validation_results.keys()]
                corrs = [c for c in corrs if not pd.isna(c)]
                if corrs:
                    avg_correlations[feature] = np.mean(corrs)

            print("Average feature correlations with next-day PnL:")
            for feature, corr in sorted(avg_correlations.items(), key=lambda x: abs(x[1]), reverse=True):
                print(f"  {feature}: {corr:.3f}")

        return validation_results

    def finalize_features(self):
        """Finalize feature set and prepare for modeling"""
        print("\n=== FINALIZING FEATURES ===")

        # Select final feature set
        basic_features = [
            'account_id', 'trade_date', 'realized_pnl', 'num_trades', 'win_rate',
            'avg_trade_size', 'profit_factor', 'consecutive_wins', 'consecutive_losses',
            'max_single_day_gain', 'max_single_day_loss', 'days_since_last_trade',
            'trading_frequency'
        ]

        ewma_features = [col for col in self.feature_df.columns if 'ewma' in col]
        lag_features = [col for col in self.feature_df.columns if '_lag' in col]
        advanced_features = [
            'high_vol_regime', 'low_vol_regime', 'current_drawdown', 'max_drawdown',
            'drawdown_recovery', 'sharpe_ratio', 'pnl_vs_recent', 'high_activity', 'low_activity'
        ]

        final_features = basic_features + ewma_features + lag_features + advanced_features

        # Ensure all features exist
        available_features = [f for f in final_features if f in self.feature_df.columns]

        self.feature_df = self.feature_df[available_features].copy()

        # Remove infinite values
        self.feature_df = self.feature_df.replace([np.inf, -np.inf], 0)

        # Final quality check
        print(f"✓ Final feature set: {len(available_features)} features")
        print(f"✓ Total observations: {len(self.feature_df)}")
        print(f"✓ Traders: {self.feature_df['account_id'].nunique()}")

        # Check for traders with sufficient data
        trader_counts = self.feature_df.groupby('account_id').size()
        viable_traders = trader_counts[trader_counts >= 60].index

        print(f"✓ Traders with >60 observations: {len(viable_traders)}")

        return available_features
