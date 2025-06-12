# src/features/base_features.py
"""
Base Feature Engineering Class - Fixed for actual database schema
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class BaseFeatures(ABC):
    """
    Abstract base class for feature engineering
    Fixed to work with actual database schema
    """

    def __init__(self,
                 feature_prefix: str,
                 min_periods: int = 20):
        """
        Args:
            feature_prefix: Prefix for all features from this generator
            min_periods: Minimum data points required
        """
        self.feature_prefix = feature_prefix
        self.min_periods = min_periods
        self.feature_names_ = []

    @abstractmethod
    def create_features(self,
                       daily_summary: pd.DataFrame,
                       fills: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Create features from data

        Args:
            daily_summary: From account_daily_summary table
            fills: From fills table (optional)

        Returns:
            DataFrame with features, indexed by date
        """
        pass

    def fit(self, X: pd.DataFrame, y=None):
        """Compatibility with sklearn interface"""
        return self

    def transform(self,
                  daily_summary: pd.DataFrame,
                  fills: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Transform method for sklearn compatibility"""
        return self.create_features(daily_summary, fills)

    def _validate_data(self, df: pd.DataFrame, required_cols: List[str]) -> bool:
        """Validate that dataframe has required columns"""
        missing = set(required_cols) - set(df.columns)
        if missing:
            logger.warning(f"Missing columns: {missing}")
            return False
        return True

    def _safe_rolling(self, series: pd.Series, window: int,
                     func: str = 'mean', min_periods: Optional[int] = None) -> pd.Series:
        """
        Safe rolling window calculation

        Args:
            series: Input series
            window: Window size
            func: Function to apply
            min_periods: Minimum periods (default window//2)
        """
        if min_periods is None:
            min_periods = max(1, window // 2)

        try:
            if func == 'mean':
                return series.rolling(window, min_periods=min_periods).mean()
            elif func == 'std':
                return series.rolling(window, min_periods=min_periods).std()
            elif func == 'sum':
                return series.rolling(window, min_periods=min_periods).sum()
            elif func == 'max':
                return series.rolling(window, min_periods=min_periods).max()
            elif func == 'min':
                return series.rolling(window, min_periods=min_periods).min()
            else:
                raise ValueError(f"Unknown function: {func}")
        except Exception as e:
            logger.error(f"Rolling calculation failed: {e}")
            return pd.Series(index=series.index, dtype=float)

    def _add_feature_prefix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add prefix to all columns except date"""
        rename_dict = {
            col: f"{self.feature_prefix}_{col}"
            for col in df.columns if col != 'date'
        }
        df = df.rename(columns=rename_dict)
        self.feature_names_ = list(rename_dict.values())
        return df
