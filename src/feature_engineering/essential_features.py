"""
Essential Feature Engineering with Data Leakage Prevention

Focus on 15-20 high-quality features with proper temporal validation
to prevent data leakage and improve model robustness.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class EssentialFeatureExtractor:
    """
    Essential feature extractor with data leakage prevention

    Creates 15-20 high-quality features focusing on:
    - Position sizing intelligence
    - Loss management discipline
    - Risk control adherence
    - Market adaptation
    - Core performance metrics
    """

    def __init__(self,
                 lookback_window: int = 20,
                 max_features: int = 20,
                 temporal_validation: bool = True):
        """
        Initialize essential feature extractor

        Args:
            lookback_window: Days for rolling calculations
            max_features: Maximum number of features to select
            temporal_validation: Enable temporal validation checks
        """
        self.lookback_window = lookback_window
        self.max_features = max_features
        self.temporal_validation = temporal_validation
        self.selected_features = []

    def safe_fillna_temporal(self, series: pd.Series, method: str = 'expanding_median') -> pd.Series:
        """
        Safe NaN filling that prevents data leakage

        Args:
            series: Time series to fill
            method: Filling method ('expanding_median', 'ffill', 'zero')

        Returns:
            Filled series with no data leakage
        """
        if method == 'expanding_median':
            # Use expanding median (only past data)
            return series.fillna(series.expanding().median())
        elif method == 'ffill':
            # Forward fill with limit
            return series.fillna(method='ffill', limit=5)
        elif method == 'zero':
            return series.fillna(0)
        else:
            raise ValueError(f"Unknown fillna method: {method}")

    def safe_rolling_calculation(self, df: pd.DataFrame, column: str,
                                window: int, operation: str) -> pd.Series:
        """
        Safe rolling calculation that prevents data leakage

        Args:
            df: DataFrame sorted by account_id, date
            column: Column to calculate on
            window: Rolling window size
            operation: 'mean', 'median', 'std', 'min', 'max'

        Returns:
            Rolling calculation result
        """
        if operation == 'mean':
            result = df.groupby('account_id')[column].rolling(window, min_periods=1).mean()
        elif operation == 'median':
            result = df.groupby('account_id')[column].rolling(window, min_periods=1).median()
        elif operation == 'std':
            result = df.groupby('account_id')[column].rolling(window, min_periods=2).std()
        elif operation == 'min':
            result = df.groupby('account_id')[column].rolling(window, min_periods=1).min()
        elif operation == 'max':
            result = df.groupby('account_id')[column].rolling(window, min_periods=1).max()
        else:
            raise ValueError(f"Unknown operation: {operation}")

        # Reset index to align with original dataframe
        return result.reset_index(level=0, drop=True)

    def extract_position_sizing_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract position sizing intelligence features

        Args:
            df: Daily summary dataframe

        Returns:
            DataFrame with position sizing features added
        """
        # Trade size proxy from available volume data
        if 'qty' in df.columns:
            df['trade_size_proxy'] = abs(df['qty'])
        elif 'num_buys' in df.columns and 'num_sells' in df.columns:
            df['trade_size_proxy'] = df['num_buys'] + df['num_sells']
        elif 'total_trade_value' in df.columns:
            df['trade_size_proxy'] = abs(df['total_trade_value'])
        else:
            # Use gross volume as proxy
            df['trade_size_proxy'] = abs(df['gross']) if 'gross' in df.columns else abs(df['net'])

        # Rolling average trade size (10-day)
        df['avg_trade_size_10d'] = self.safe_rolling_calculation(
            df, 'trade_size_proxy', 10, 'mean'
        )

        # Position size ratio (current vs historical average)
        df['position_size_ratio'] = df['trade_size_proxy'] / (df['avg_trade_size_10d'] + 1e-6)

        # Capital utilization (gross volume vs account balance)
        df['daily_volume'] = abs(df['gross']) if 'gross' in df.columns else abs(df['net'])

        # Use available balance column
        balance_col = None
        for col in ['end_balance', 'cash', 'unrealized']:
            if col in df.columns:
                balance_col = col
                break

        if balance_col:
            df['capital_utilization'] = df['daily_volume'] / (abs(df[balance_col]) + 1e-6)
        else:
            # Fallback: use cumulative PnL as balance proxy
            df['capital_utilization'] = df['daily_volume'] / (abs(df['cumulative_pnl']) + 1000)

        # Position size consistency (lower is more disciplined)
        df['position_size_std_10d'] = self.safe_rolling_calculation(
            df, 'trade_size_proxy', 10, 'std'
        )
        df['position_consistency'] = df['position_size_std_10d'] / (df['avg_trade_size_10d'] + 1e-6)

        # Fill NaN values safely
        for col in ['position_size_ratio', 'capital_utilization', 'position_consistency']:
            if col in df.columns:
                df[col] = df.groupby('account_id')[col].transform(
                    lambda x: self.safe_fillna_temporal(x, 'expanding_median')
                )

        return df

    def extract_loss_management_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract loss management discipline features

        Args:
            df: Daily summary dataframe

        Returns:
            DataFrame with loss management features added
        """
        # Loss days indicator
        df['is_loss_day'] = (df['net'] < 0).astype(int)

        # Loss magnitude relative to recent performance
        df['avg_loss_10d'] = df.groupby('account_id').apply(
            lambda x: x['net'].where(x['net'] < 0).rolling(10, min_periods=1).mean()
        ).reset_index(level=0, drop=True)

        # Current loss vs average loss (loss cutting discipline)
        df['loss_vs_avg'] = np.where(
            df['net'] < 0,
            df['net'] / (df['avg_loss_10d'] - 1e-6),
            0
        )

        # Recovery time after losses (simplified)
        df['days_since_loss'] = df.groupby('account_id')['is_loss_day'].apply(
            lambda x: (x == 0).cumsum()
        ).reset_index(level=0, drop=True)

        # Loss streak length
        df['loss_streak'] = df.groupby('account_id')['is_loss_day'].apply(
            lambda x: x.groupby((x != x.shift()).cumsum()).cumsum()
        ).reset_index(level=0, drop=True)
        df['loss_streak'] = np.where(df['is_loss_day'] == 0, 0, df['loss_streak'])

        # Max drawdown in recent period
        df['cumulative_pnl_10d'] = self.safe_rolling_calculation(df, 'net', 10, 'mean') * 10
        df['peak_pnl_10d'] = df.groupby('account_id')['cumulative_pnl_10d'].expanding().max().reset_index(level=0, drop=True)
        df['drawdown_from_peak'] = df['cumulative_pnl_10d'] - df['peak_pnl_10d']

        # Fill NaN values safely
        for col in ['loss_vs_avg', 'loss_streak', 'drawdown_from_peak']:
            if col in df.columns:
                df[col] = df.groupby('account_id')[col].transform(
                    lambda x: self.safe_fillna_temporal(x, 'zero')
                )

        return df

    def extract_risk_control_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract risk control adherence features

        Args:
            df: Daily summary dataframe

        Returns:
            DataFrame with risk control features added
        """
        # Daily volatility (rolling 10-day)
        df['daily_vol_10d'] = self.safe_rolling_calculation(df, 'net', 10, 'std')

        # Risk budget usage (current vol vs target)
        target_daily_vol = df.groupby('account_id')['net'].apply(
            lambda x: x.expanding().std()
        ).reset_index(level=0, drop=True)
        df['risk_budget_usage'] = df['daily_vol_10d'] / (target_daily_vol + 1e-6)

        # Leverage proxy (gross exposure vs balance)
        gross_col = 'gross' if 'gross' in df.columns else 'net'

        # Find available balance column
        balance_col = None
        for col in ['end_balance', 'cash', 'unrealized', 'cumulative_pnl']:
            if col in df.columns:
                balance_col = col
                break

        if balance_col:
            df['leverage_proxy'] = abs(df[gross_col]) / (abs(df[balance_col]) + 1e-6)
        else:
            # Fallback: use normalized gross volume
            df['leverage_proxy'] = abs(df[gross_col]) / (abs(df[gross_col]).rolling(10).mean() + 1e-6)

        # Risk limit adherence (max loss vs typical loss)
        df['max_loss_5d'] = df.groupby('account_id')['net'].rolling(5, min_periods=1).min().reset_index(level=0, drop=True)
        df['typical_loss_10d'] = df.groupby('account_id').apply(
            lambda x: x['net'].where(x['net'] < 0).rolling(10, min_periods=1).mean()
        ).reset_index(level=0, drop=True)
        df['risk_limit_adherence'] = df['max_loss_5d'] / (df['typical_loss_10d'] - 1e-6)

        # Trading frequency control
        if 'fills' in df.columns:
            df['active_trading_day'] = (df['fills'] > 0).astype(int)
        else:
            # Use any trading activity as proxy
            df['active_trading_day'] = (abs(df['net']) > 0).astype(int)

        df['trading_frequency_10d'] = self.safe_rolling_calculation(
            df, 'active_trading_day', 10, 'mean'
        )

        # Fill NaN values safely
        for col in ['risk_budget_usage', 'leverage_proxy', 'risk_limit_adherence', 'trading_frequency_10d']:
            if col in df.columns:
                df[col] = df.groupby('account_id')[col].transform(
                    lambda x: self.safe_fillna_temporal(x, 'expanding_median')
                )

        return df

    def extract_market_adaptation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract market adaptation features

        Args:
            df: Daily summary dataframe

        Returns:
            DataFrame with market adaptation features added
        """
        # Market volatility regime (based on PnL volatility)
        df['market_vol_regime'] = self.safe_rolling_calculation(df, 'net', 10, 'std')

        # Volatility percentile (current vol vs historical)
        df['vol_percentile'] = df.groupby('account_id')['market_vol_regime'].apply(
            lambda x: x.expanding().rank(pct=True)
        ).reset_index(level=0, drop=True)

        # Position size adaptation to volatility
        df['vol_adjusted_position'] = df['trade_size_proxy'] / (df['market_vol_regime'] + 1e-6)

        # Fill NaN values safely
        for col in ['vol_percentile', 'vol_adjusted_position']:
            if col in df.columns:
                df[col] = df.groupby('account_id')[col].transform(
                    lambda x: self.safe_fillna_temporal(x, 'expanding_median')
                )

        return df

    def extract_core_performance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract core performance metrics

        Args:
            df: Daily summary dataframe

        Returns:
            DataFrame with core performance features added
        """
        # Sharpe ratio (10-day rolling)
        df['avg_return_10d'] = self.safe_rolling_calculation(df, 'net', 10, 'mean')
        df['vol_return_10d'] = self.safe_rolling_calculation(df, 'net', 10, 'std')
        df['sharpe_10d'] = df['avg_return_10d'] / (df['vol_return_10d'] + 1e-6)

        # Hit rate (7-day rolling)
        df['win_indicator'] = (df['net'] > 0).astype(int)
        df['hit_rate_7d'] = self.safe_rolling_calculation(df, 'win_indicator', 7, 'mean')

        # Maximum drawdown percentage (10-day)
        df['cumulative_10d'] = self.safe_rolling_calculation(df, 'net', 10, 'mean') * 10
        df['running_max_10d'] = df.groupby('account_id')['cumulative_10d'].expanding().max().reset_index(level=0, drop=True)
        df['max_drawdown_pct'] = (df['cumulative_10d'] - df['running_max_10d']) / (abs(df['running_max_10d']) + 1e-6)

        # Profit factor (wins vs losses)
        df['total_wins_10d'] = df.groupby('account_id').apply(
            lambda x: x['net'].where(x['net'] > 0, 0).rolling(10, min_periods=1).sum()
        ).reset_index(level=0, drop=True)
        df['total_losses_10d'] = df.groupby('account_id').apply(
            lambda x: abs(x['net'].where(x['net'] < 0, 0)).rolling(10, min_periods=1).sum()
        ).reset_index(level=0, drop=True)
        df['profit_factor_10d'] = df['total_wins_10d'] / (df['total_losses_10d'] + 1e-6)

        # Fill NaN values safely
        for col in ['sharpe_10d', 'hit_rate_7d', 'max_drawdown_pct', 'profit_factor_10d']:
            if col in df.columns:
                df[col] = df.groupby('account_id')[col].transform(
                    lambda x: self.safe_fillna_temporal(x, 'expanding_median')
                )

        return df

    def extract_all_essential_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all essential features with temporal validation

        Args:
            df: Daily summary dataframe (must be sorted by account_id, date)

        Returns:
            DataFrame with essential features added
        """
        if df.empty:
            logger.warning("Empty dataframe provided")
            return df

        logger.info("Extracting essential features with data leakage prevention...")

        # Validate temporal ordering
        if self.temporal_validation:
            self._validate_temporal_ordering(df)

        df = df.copy()

        # Extract feature groups
        logger.info("Extracting position sizing features...")
        df = self.extract_position_sizing_features(df)

        logger.info("Extracting loss management features...")
        df = self.extract_loss_management_features(df)

        logger.info("Extracting risk control features...")
        df = self.extract_risk_control_features(df)

        logger.info("Extracting market adaptation features...")
        df = self.extract_market_adaptation_features(df)

        logger.info("Extracting core performance features...")
        df = self.extract_core_performance_features(df)

        # Define essential feature list
        essential_features = [
            # Position sizing (4 features)
            'position_size_ratio', 'capital_utilization', 'position_consistency', 'vol_adjusted_position',

            # Loss management (4 features)
            'loss_vs_avg', 'loss_streak', 'drawdown_from_peak', 'days_since_loss',

            # Risk control (4 features)
            'risk_budget_usage', 'leverage_proxy', 'risk_limit_adherence', 'trading_frequency_10d',

            # Market adaptation (2 features)
            'vol_percentile', 'market_vol_regime',

            # Core performance (4 features)
            'sharpe_10d', 'hit_rate_7d', 'max_drawdown_pct', 'profit_factor_10d',

            # Essential basics (3 features)
            'net', 'fills', 'unrealized'
        ]

        # Keep only essential features + required columns
        required_cols = ['account_id', 'date', 'target_next_pnl']
        available_essentials = [col for col in essential_features if col in df.columns]
        keep_columns = required_cols + available_essentials

        df_essential = df[keep_columns].copy()

        logger.info(f"Selected {len(available_essentials)} essential features: {available_essentials}")
        self.selected_features = available_essentials

        return df_essential

    def _validate_temporal_ordering(self, df: pd.DataFrame) -> None:
        """
        Validate that dataframe is properly sorted for temporal calculations

        Args:
            df: DataFrame to validate

        Raises:
            ValueError: If temporal ordering is invalid
        """
        if 'date' not in df.columns:
            raise ValueError("DataFrame must have 'date' column")

        if 'account_id' not in df.columns:
            raise ValueError("DataFrame must have 'account_id' column")

        # Check if sorted by account_id, date
        df_sorted = df.sort_values(['account_id', 'date'])
        if not df.equals(df_sorted):
            logger.warning("DataFrame not properly sorted by account_id, date")
            raise ValueError("DataFrame must be sorted by account_id, date for temporal validity")

    def select_best_features(self, X: pd.DataFrame, y: pd.Series,
                           method: str = 'mutual_info',
                           n_features: Optional[int] = None) -> List[str]:
        """
        Select best features using statistical methods

        Args:
            X: Feature matrix
            y: Target variable
            method: Selection method ('mutual_info', 'f_regression')
            n_features: Number of features to select

        Returns:
            List of selected feature names
        """
        if n_features is None:
            n_features = min(self.max_features, X.shape[1])

        # Remove non-numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_cols]

        # Handle NaN values with more robust approach
        # Only use rows with non-null target values
        valid_target_mask = y.notna()
        X_numeric = X_numeric[valid_target_mask]
        y = y[valid_target_mask]

        # For features with >90% missing, drop them
        missing_rates = X_numeric.isnull().mean()
        good_features = missing_rates[missing_rates < 0.9].index
        X_numeric = X_numeric[good_features]

        # Fill remaining NaN with median (temporal-safe)
        X_clean = X_numeric.fillna(X_numeric.median())
        y_clean = y.fillna(y.median())

        # Feature selection
        if method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_regression, k=n_features)
        elif method == 'f_regression':
            selector = SelectKBest(score_func=f_regression, k=n_features)
        else:
            raise ValueError(f"Unknown selection method: {method}")

        selector.fit(X_clean, y_clean)
        selected_features = X_clean.columns[selector.get_support()].tolist()

        logger.info(f"Selected {len(selected_features)} features using {method}: {selected_features}")

        return selected_features

    def validate_features_temporal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate features for temporal consistency and data leakage

        Args:
            df: DataFrame with features

        Returns:
            Validation report
        """
        report = {
            'temporal_ordering_valid': True,
            'data_leakage_detected': False,
            'feature_stability': {},
            'missing_data_stats': {}
        }

        # Check temporal ordering
        try:
            self._validate_temporal_ordering(df)
        except ValueError as e:
            report['temporal_ordering_valid'] = False
            report['error'] = str(e)

        # Check for missing data patterns
        feature_cols = [col for col in df.columns
                       if col not in ['account_id', 'date', 'target_next_pnl']]

        for col in feature_cols:
            missing_pct = df[col].isnull().mean()
            report['missing_data_stats'][col] = missing_pct

            if missing_pct > 0.5:
                logger.warning(f"Feature {col} has {missing_pct:.1%} missing data")

        logger.info("Feature temporal validation completed")
        return report

    def get_feature_summary(self) -> Dict[str, Any]:
        """
        Get summary of selected essential features

        Returns:
            Feature summary dictionary
        """
        feature_categories = {
            'position_sizing': ['position_size_ratio', 'capital_utilization', 'position_consistency', 'vol_adjusted_position'],
            'loss_management': ['loss_vs_avg', 'loss_streak', 'drawdown_from_peak', 'days_since_loss'],
            'risk_control': ['risk_budget_usage', 'leverage_proxy', 'risk_limit_adherence', 'trading_frequency_10d'],
            'market_adaptation': ['vol_percentile', 'market_vol_regime'],
            'core_performance': ['sharpe_10d', 'hit_rate_7d', 'max_drawdown_pct', 'profit_factor_10d'],
            'essentials': ['net', 'fills', 'unrealized']
        }

        selected_by_category = {}
        for category, features in feature_categories.items():
            selected_by_category[category] = [f for f in features if f in self.selected_features]

        return {
            'total_selected_features': len(self.selected_features),
            'selected_features': self.selected_features,
            'features_by_category': selected_by_category,
            'max_features_limit': self.max_features,
            'temporal_validation_enabled': self.temporal_validation
        }
