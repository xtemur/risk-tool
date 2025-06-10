"""
Base Feature Engineering Class
Foundation for all feature generators with time series safety
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta

from src.core.constants import TradingConstants as TC, DataQualityLimits as DQL

logger = logging.getLogger(__name__)


class BaseFeatures(ABC):
    """
    Abstract base class for feature engineering
    Ensures temporal integrity and provides common utilities
    """

    def __init__(self,
                 feature_prefix: str,
                 lookback_days: Optional[int] = None,
                 min_periods: Optional[int] = None):
        """
        Initialize base feature generator

        Args:
            feature_prefix: Prefix for all features from this generator
            lookback_days: Maximum lookback period for features
            min_periods: Minimum data points required for calculation
        """
        self.feature_prefix = feature_prefix
        self.lookback_days = lookback_days or max(TC.FEATURE_WINDOWS)
        self.min_periods = min_periods or TC.MIN_SAMPLES_PER_TRADER

        # Track feature metadata
        self.feature_names_: List[str] = []
        self.feature_importance_: Dict[str, float] = {}

    @abstractmethod
    def create_features(self,
                       totals_df: pd.DataFrame,
                       fills_df: Optional[pd.DataFrame] = None,
                       as_of_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """
        Create features from input data

        Args:
            totals_df: Daily totals data
            fills_df: Optional fills data for intraday features
            as_of_date: Create features as of this date (for point-in-time)

        Returns:
            DataFrame with features indexed by date and account_id
        """
        pass

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit any parameters needed for feature generation
        Default implementation does nothing (stateless features)
        """
        return self

    def transform(self,
                  totals_df: pd.DataFrame,
                  fills_df: Optional[pd.DataFrame] = None,
                  as_of_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """
        Transform data to features (scikit-learn compatible)
        """
        return self.create_features(totals_df, fills_df, as_of_date)

    def fit_transform(self,
                      totals_df: pd.DataFrame,
                      fills_df: Optional[pd.DataFrame] = None,
                      y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fit and transform in one step
        """
        self.fit(totals_df, y)
        return self.transform(totals_df, fills_df)

    # ==================== Utility Methods ====================

    def _ensure_datetime_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure DataFrame has datetime index"""
        if 'date' in df.columns and df.index.name != 'date':
            df = df.set_index('date')

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        return df.sort_index()

    def _apply_point_in_time(self,
                            df: pd.DataFrame,
                            as_of_date: pd.Timestamp) -> pd.DataFrame:
        """
        Filter data to only include information available as of given date
        Critical for preventing lookahead bias
        """
        if as_of_date is None:
            return df

        # Only use data up to (but not including) as_of_date
        return df[df.index < as_of_date]

    def _calculate_rolling_window(self,
                                 series: pd.Series,
                                 window: int,
                                 func: str = 'mean',
                                 min_periods: Optional[int] = None) -> pd.Series:
        """
        Calculate rolling window statistics with proper handling

        Args:
            series: Input series
            window: Window size in periods
            func: Function to apply ('mean', 'std', 'sum', etc.)
            min_periods: Minimum periods for calculation

        Returns:
            Series with rolling calculation
        """
        min_periods = min_periods or max(1, window // 2)

        if func == 'mean':
            return series.rolling(window=window, min_periods=min_periods).mean()
        elif func == 'std':
            return series.rolling(window=window, min_periods=min_periods).std()
        elif func == 'sum':
            return series.rolling(window=window, min_periods=min_periods).sum()
        elif func == 'max':
            return series.rolling(window=window, min_periods=min_periods).max()
        elif func == 'min':
            return series.rolling(window=window, min_periods=min_periods).min()
        elif func == 'skew':
            return series.rolling(window=window, min_periods=min_periods).skew()
        elif func == 'kurt':
            return series.rolling(window=window, min_periods=min_periods).kurt()
        else:
            raise ValueError(f"Unknown function: {func}")

    def _calculate_ewm(self,
                      series: pd.Series,
                      span: int,
                      min_periods: Optional[int] = None) -> pd.Series:
        """
        Calculate exponentially weighted statistics
        """
        min_periods = min_periods or max(1, span // 2)
        return series.ewm(span=span, min_periods=min_periods, adjust=False).mean()

    def _handle_missing_data(self,
                           df: pd.DataFrame,
                           method: str = 'forward_fill',
                           limit: int = TC.MAX_CONSECUTIVE_MISSING) -> pd.DataFrame:
        """
        Handle missing data in features

        Args:
            df: DataFrame with potential missing values
            method: Method to handle missing data
            limit: Maximum consecutive values to fill

        Returns:
            DataFrame with missing values handled
        """
        if method == 'forward_fill':
            # Forward fill with limit (no future information)
            return df.fillna(method='ffill', limit=limit)
        elif method == 'zero':
            return df.fillna(0)
        elif method == 'mean':
            # Use expanding mean (no future information)
            return df.fillna(df.expanding(min_periods=1).mean())
        else:
            raise ValueError(f"Unknown method: {method}")

    def _clip_outliers(self,
                      series: pd.Series,
                      method: str = 'zscore',
                      threshold: float = TC.Z_SCORE_THRESHOLD) -> pd.Series:
        """
        Clip outliers in series

        Args:
            series: Input series
            method: Method to detect outliers ('zscore' or 'iqr')
            threshold: Threshold for outlier detection

        Returns:
            Series with outliers clipped
        """
        if method == 'zscore':
            mean = series.mean()
            std = series.std()
            lower = mean - threshold * std
            upper = mean + threshold * std
        elif method == 'iqr':
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr
        else:
            raise ValueError(f"Unknown method: {method}")

        return series.clip(lower=lower, upper=upper)

    def _normalize_features(self,
                          df: pd.DataFrame,
                          method: str = 'zscore') -> pd.DataFrame:
        """
        Normalize features

        Args:
            df: DataFrame with features
            method: Normalization method ('zscore', 'minmax', 'robust')

        Returns:
            Normalized DataFrame
        """
        if method == 'zscore':
            return (df - df.mean()) / (df.std() + TC.MIN_VARIANCE)
        elif method == 'minmax':
            return (df - df.min()) / (df.max() - df.min() + TC.MIN_VARIANCE)
        elif method == 'robust':
            median = df.median()
            mad = (df - median).abs().median()
            return (df - median) / (mad + TC.MIN_VARIANCE)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _create_target(self,
                      df: pd.DataFrame,
                      target_col: str = 'net_pnl',
                      horizon: int = 1,
                      target_type: str = 'regression') -> pd.Series:
        """
        Create target variable with proper shift

        Args:
            df: DataFrame with data
            target_col: Column to use as target
            horizon: Prediction horizon (days ahead)
            target_type: 'regression' or 'classification'

        Returns:
            Series with target values
        """
        # Shift by negative horizon (future values)
        target = df[target_col].shift(-horizon)

        if target_type == 'classification':
            # Binary classification: profit or loss
            target = (target > 0).astype(int)
        elif target_type == 'multiclass':
            # Three classes: large loss, small change, large profit
            conditions = [
                target < -df[target_col].std(),
                target > df[target_col].std()
            ]
            choices = [0, 2]  # 1 is default (small change)
            target = np.select(conditions, choices, default=1)

        return target

    def _add_feature_prefix(self,
                           df: pd.DataFrame,
                           exclude_cols: List[str] = None) -> pd.DataFrame:
        """Add prefix to all feature columns"""
        exclude_cols = exclude_cols or ['date', 'account_id']

        rename_dict = {
            col: f"{self.feature_prefix}_{col}"
            for col in df.columns
            if col not in exclude_cols
        }

        df = df.rename(columns=rename_dict)
        self.feature_names_ = list(rename_dict.values())

        return df

    def _validate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate features for common issues

        Checks:
        - No infinite values
        - No extremely large values
        - Reasonable number of missing values
        """
        # Check for infinite values
        inf_cols = df.columns[np.isinf(df).any()].tolist()
        if inf_cols:
            logger.warning(f"Infinite values found in columns: {inf_cols}")
            df[inf_cols] = df[inf_cols].replace([np.inf, -np.inf], np.nan)

        # Check for extremely large values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].abs().max() > 1e6:
                logger.warning(f"Large values found in {col}, clipping...")
                df[col] = self._clip_outliers(df[col])

        # Check missing data percentage
        missing_pct = df.isnull().sum() / len(df)
        high_missing = missing_pct[missing_pct > TC.MAX_MISSING_PCT]
        if not high_missing.empty:
            logger.warning(f"High missing data in columns: {high_missing.to_dict()}")

        return df

    def get_feature_names(self) -> List[str]:
        """Get list of feature names created by this generator"""
        return self.feature_names_

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance if available"""
        return self.feature_importance_
