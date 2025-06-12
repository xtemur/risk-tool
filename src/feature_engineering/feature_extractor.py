"""
Advanced Feature Extractor for Trading Behavior Analysis

Extracts psychological and behavioral features from trading data:
- Revenge trading patterns
- Loss aversion behaviors
- Rolling performance metrics
- Aggressive trading patterns
- Risk-taking behaviors
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Advanced feature extractor for behavioral trading patterns
    """

    def __init__(self, lookback_window: int = 10):
        """
        Initialize feature extractor

        Args:
            lookback_window: Number of days to look back for rolling features
        """
        self.lookback_window = lookback_window

    def extract_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all advanced features

        Args:
            df: Daily trading dataframe (sorted by account_id, date)

        Returns:
            Dataframe with advanced features added
        """
        if df.empty:
            return df

        df = df.copy()

        # Ensure proper sorting
        df = df.sort_values(['account_id', 'date']).reset_index(drop=True)

        # Extract features by category
        df = self._extract_revenge_trading_features(df)
        df = self._extract_loss_aversion_features(df)
        df = self._extract_rolling_performance_features(df)
        df = self._extract_aggressive_trading_features(df)
        df = self._extract_risk_behavior_features(df)
        df = self._extract_consistency_features(df)
        df = self._extract_market_timing_features(df)

        logger.info(f"Extracted advanced features, final shape: {df.shape}")

        return df

    def _extract_revenge_trading_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract revenge trading features (trading more after losses)
        """
        if 'net' not in df.columns:
            return df

        # Create loss indicator
        df['is_loss_day'] = (df['net'] < 0).astype(int)

        # Revenge trading: increased volume/trades after loss
        for window in [1, 2, 3]:
            df[f'loss_streak_{window}d'] = (
                df.groupby('account_id')['is_loss_day']
                .rolling(window=window, min_periods=1)
                .sum()
                .reset_index(0, drop=True)
            )

            # Volume after loss
            if 'qty' in df.columns:
                df[f'volume_after_loss_{window}d'] = np.where(
                    df[f'loss_streak_{window}d'] > 0,
                    df['qty'],
                    0
                )

        # Trading intensity after losses
        if 'fills' in df.columns:
            df['prev_day_loss'] = df.groupby('account_id')['is_loss_day'].shift(1).fillna(0)
            df['trades_after_loss'] = df['fills'] * df['prev_day_loss']

            # Revenge ratio: trades today / avg trades when prev day was loss vs profit
            avg_trades_after_loss = df.groupby('account_id')['trades_after_loss'].transform('mean')
            avg_trades_after_profit = df.groupby('account_id').apply(
                lambda x: x[x['prev_day_loss'] == 0]['fills'].mean()
            ).reindex(df['account_id']).values

            df['revenge_ratio'] = avg_trades_after_loss / (avg_trades_after_profit + 1e-6)

        return df

    def _extract_loss_aversion_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract loss aversion features (holding losers, selling winners)
        """
        if 'net' not in df.columns:
            return df

        # Win/loss patterns
        df['is_win_day'] = (df['net'] > 0).astype(int)

        # Loss aversion: tendency to hold unrealized losses
        if 'unrealized' in df.columns:
            df['unrealized_loss'] = np.where(df['unrealized'] < 0, abs(df['unrealized']), 0)
            df['unrealized_gain'] = np.where(df['unrealized'] > 0, df['unrealized'], 0)

            # Rolling average of unrealized losses vs gains
            for window in [3, 7, 10]:
                df[f'avg_unrealized_loss_{window}d'] = (
                    df.groupby('account_id')['unrealized_loss']
                    .rolling(window=window, min_periods=1)
                    .mean()
                    .reset_index(0, drop=True)
                )
                df[f'avg_unrealized_gain_{window}d'] = (
                    df.groupby('account_id')['unrealized_gain']
                    .rolling(window=window, min_periods=1)
                    .mean()
                    .reset_index(0, drop=True)
                )

                # Loss aversion ratio
                df[f'loss_aversion_ratio_{window}d'] = (
                    df[f'avg_unrealized_loss_{window}d'] /
                    (df[f'avg_unrealized_gain_{window}d'] + 1e-6)
                )

        # Win/loss streak behavior
        df['win_streak'] = df.groupby('account_id')['is_win_day'].apply(
            lambda x: x * (x.groupby((x != x.shift()).cumsum()).cumcount() + 1)
        ).reset_index(0, drop=True)

        df['loss_streak'] = df.groupby('account_id')['is_loss_day'].apply(
            lambda x: x * (x.groupby((x != x.shift()).cumsum()).cumcount() + 1)
        ).reset_index(0, drop=True)

        return df

    def _extract_rolling_performance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract rolling performance features
        """
        if 'net' not in df.columns:
            return df

        # Rolling PnL statistics
        for window in [3, 7, 10, 20]:
            # Rolling sum
            df[f'rolling_pnl_{window}d'] = (
                df.groupby('account_id')['net']
                .rolling(window=window, min_periods=1)
                .sum()
                .reset_index(0, drop=True)
            )

            # Rolling mean
            df[f'rolling_avg_pnl_{window}d'] = (
                df.groupby('account_id')['net']
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(0, drop=True)
            )

            # Rolling std (volatility)
            df[f'rolling_vol_{window}d'] = (
                df.groupby('account_id')['net']
                .rolling(window=window, min_periods=2)
                .std()
                .reset_index(0, drop=True)
                .fillna(0)
            )

            # Sharpe-like ratio
            df[f'rolling_sharpe_{window}d'] = (
                df[f'rolling_avg_pnl_{window}d'] / (df[f'rolling_vol_{window}d'] + 1e-6)
            )

            # Win rate
            df[f'win_rate_{window}d'] = (
                df.groupby('account_id')['is_win_day']
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(0, drop=True)
            )

        # Performance momentum
        df['pnl_momentum_3d'] = df['rolling_pnl_3d'].diff()
        df['pnl_momentum_7d'] = df['rolling_pnl_7d'].diff()

        return df

    def _extract_aggressive_trading_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract aggressive trading behavior features
        """
        # Volume-based aggression
        if 'qty' in df.columns:
            # Volume spikes
            for window in [5, 10]:
                df[f'avg_volume_{window}d'] = (
                    df.groupby('account_id')['qty']
                    .rolling(window=window, min_periods=1)
                    .mean()
                    .reset_index(0, drop=True)
                )

                df[f'volume_spike_{window}d'] = (
                    df['qty'] / (df[f'avg_volume_{window}d'] + 1e-6)
                )

        # Trading frequency aggression
        if 'fills' in df.columns:
            # Trade frequency spikes
            for window in [5, 10]:
                df[f'avg_trades_{window}d'] = (
                    df.groupby('account_id')['fills']
                    .rolling(window=window, min_periods=1)
                    .mean()
                    .reset_index(0, drop=True)
                )

                df[f'trade_spike_{window}d'] = (
                    df['fills'] / (df[f'avg_trades_{window}d'] + 1e-6)
                )

        # Market exposure aggression
        if 'total_trade_value' in df.columns:
            for window in [5, 10]:
                df[f'avg_exposure_{window}d'] = (
                    df.groupby('account_id')['total_trade_value']
                    .rolling(window=window, min_periods=1)
                    .mean()
                    .reset_index(0, drop=True)
                )

                df[f'exposure_spike_{window}d'] = (
                    df['total_trade_value'] / (df[f'avg_exposure_{window}d'] + 1e-6)
                )

        return df

    def _extract_risk_behavior_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract risk-taking behavior features
        """
        # Diversification risk
        if 'unique_symbols' in df.columns:
            df['concentration_risk'] = 1 / (df['unique_symbols'] + 1)

            # Rolling average symbols
            df['avg_symbols_10d'] = (
                df.groupby('account_id')['unique_symbols']
                .rolling(window=10, min_periods=1)
                .mean()
                .reset_index(0, drop=True)
            )

            df['symbol_diversification_change'] = (
                df['unique_symbols'] - df['avg_symbols_10d']
            )

        # Position sizing risk
        if 'avg_trade_value' in df.columns:
            # Position size consistency
            df['avg_trade_size_10d'] = (
                df.groupby('account_id')['avg_trade_value']
                .rolling(window=10, min_periods=1)
                .mean()
                .reset_index(0, drop=True)
            )

            df['position_size_deviation'] = (
                abs(df['avg_trade_value'] - df['avg_trade_size_10d']) /
                (df['avg_trade_size_10d'] + 1e-6)
            )

        # Risk-adjusted returns
        if 'net' in df.columns and 'std_trade_value' in df.columns:
            df['risk_adjusted_return'] = (
                df['net'] / (df['std_trade_value'] + 1e-6)
            )

        return df

    def _extract_consistency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract trading consistency features
        """
        if 'net' not in df.columns:
            return df

        # PnL consistency - only use windows that exist from rolling performance
        for window in [10]:  # Only use 10d as it's created in rolling performance
            if f'rolling_vol_{window}d' in df.columns:
                df[f'pnl_consistency_{window}d'] = (
                    1 / (df[f'rolling_vol_{window}d'] + 1e-6)
                )

        # Trading pattern consistency
        if 'fills' in df.columns:
            df['trade_consistency_10d'] = (
                1 / (df.groupby('account_id')['fills']
                     .rolling(window=10, min_periods=2)
                     .std()
                     .reset_index(0, drop=True)
                     .fillna(1) + 1e-6)
            )

        # Drawdown features
        df['cumulative_pnl'] = df.groupby('account_id')['net'].cumsum()
        df['rolling_max_pnl'] = (
            df.groupby('account_id')['cumulative_pnl']
            .expanding()
            .max()
            .reset_index(0, drop=True)
        )
        df['drawdown'] = df['cumulative_pnl'] - df['rolling_max_pnl']
        df['drawdown_pct'] = df['drawdown'] / (abs(df['rolling_max_pnl']) + 1e-6)

        return df

    def _extract_market_timing_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract market timing and calendar features
        """
        if 'date' not in df.columns:
            return df

        # Calendar features
        df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
        df['day_of_month'] = pd.to_datetime(df['date']).dt.day
        df['is_month_end'] = (pd.to_datetime(df['date']).dt.day >= 28).astype(int)
        df['is_monday'] = (df['day_of_week'] == 0).astype(int)
        df['is_friday'] = (df['day_of_week'] == 4).astype(int)

        # Performance by day patterns
        if 'net' in df.columns:
            df['monday_performance'] = df['net'] * df['is_monday']
            df['friday_performance'] = df['net'] * df['is_friday']

            # Month-end behavior
            df['month_end_performance'] = df['net'] * df['is_month_end']

        return df

    def select_top_features(self, df: pd.DataFrame,
                           target_col: str = 'target_next_pnl',
                           n_features: int = 10,
                           method: str = 'f_regression') -> Tuple[pd.DataFrame, List[str]]:
        """
        Select top N features for modeling

        Args:
            df: Feature dataframe
            target_col: Target column name
            n_features: Number of features to select
            method: Selection method ('f_regression' or 'mutual_info')

        Returns:
            Tuple of (dataframe with selected features, list of selected feature names)
        """
        if df.empty or target_col not in df.columns:
            return df, []

        # Get feature columns (numeric only, excluding target and identifiers)
        exclude_cols = ['account_id', 'date', target_col]
        feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns
                       if col not in exclude_cols]

        if len(feature_cols) <= n_features:
            logger.info(f"Only {len(feature_cols)} features available, returning all")
            return df, feature_cols

        # Prepare data
        X = df[feature_cols].fillna(0)
        y = df[target_col].fillna(0)

        # Remove any infinite values
        X = X.replace([np.inf, -np.inf], 0)

        # Feature selection
        if method == 'f_regression':
            selector = SelectKBest(score_func=f_regression, k=n_features)
        elif method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_regression, k=n_features)
        else:
            raise ValueError(f"Unknown method: {method}")

        try:
            X_selected = selector.fit_transform(X, y)
            selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]

            # Create result dataframe
            result_df = df[exclude_cols + selected_features].copy()

            logger.info(f"Selected {len(selected_features)} features using {method}")
            logger.info(f"Selected features: {selected_features}")

            return result_df, selected_features

        except Exception as e:
            logger.error(f"Feature selection failed: {e}")
            # Return top N features by variance as fallback
            feature_vars = X.var().sort_values(ascending=False)
            top_features = feature_vars.head(n_features).index.tolist()
            result_df = df[exclude_cols + top_features].copy()

            logger.info(f"Fallback: selected top {len(top_features)} features by variance")
            return result_df, top_features

    def get_feature_importance_summary(self, df: pd.DataFrame,
                                     target_col: str = 'target_next_pnl') -> Dict[str, Any]:
        """
        Get feature importance summary

        Args:
            df: Feature dataframe
            target_col: Target column name

        Returns:
            Feature importance summary
        """
        if df.empty or target_col not in df.columns:
            return {}

        # Get feature columns
        exclude_cols = ['account_id', 'date', target_col]
        feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns
                       if col not in exclude_cols]

        X = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        y = df[target_col].fillna(0)

        try:
            # Calculate F-scores
            f_scores = f_regression(X, y)[0]

            # Create importance ranking
            importance_df = pd.DataFrame({
                'feature': feature_cols,
                'f_score': f_scores
            }).sort_values('f_score', ascending=False)

            return {
                'total_features': len(feature_cols),
                'top_10_features': importance_df.head(10).to_dict('records'),
                'feature_categories': self._categorize_features(feature_cols)
            }

        except Exception as e:
            logger.error(f"Feature importance calculation failed: {e}")
            return {'error': str(e)}

    def _categorize_features(self, feature_names: List[str]) -> Dict[str, List[str]]:
        """
        Categorize features by type
        """
        categories = {
            'revenge_trading': [],
            'loss_aversion': [],
            'rolling_performance': [],
            'aggressive_trading': [],
            'risk_behavior': [],
            'consistency': [],
            'market_timing': [],
            'basic': []
        }

        for feature in feature_names:
            if any(keyword in feature.lower() for keyword in ['revenge', 'loss_streak', 'after_loss']):
                categories['revenge_trading'].append(feature)
            elif any(keyword in feature.lower() for keyword in ['loss_aversion', 'unrealized', 'win_streak']):
                categories['loss_aversion'].append(feature)
            elif any(keyword in feature.lower() for keyword in ['rolling', 'momentum', 'sharpe', 'win_rate']):
                categories['rolling_performance'].append(feature)
            elif any(keyword in feature.lower() for keyword in ['spike', 'aggressive', 'volume_after']):
                categories['aggressive_trading'].append(feature)
            elif any(keyword in feature.lower() for keyword in ['risk', 'concentration', 'diversification', 'deviation']):
                categories['risk_behavior'].append(feature)
            elif any(keyword in feature.lower() for keyword in ['consistency', 'drawdown']):
                categories['consistency'].append(feature)
            elif any(keyword in feature.lower() for keyword in ['day_of', 'month', 'monday', 'friday']):
                categories['market_timing'].append(feature)
            else:
                categories['basic'].append(feature)

        return {k: v for k, v in categories.items() if v}
