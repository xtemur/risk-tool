"""
Time Series Cross-Validation
Proper cross-validation for financial time series with purging and embargo
"""

import pandas as pd
import numpy as np
from typing import Iterator, Tuple, Optional, List, Union, Dict
from dataclasses import dataclass
import logging
from datetime import timedelta

from src.core.constants import TradingConstants as TC

logger = logging.getLogger(__name__)


@dataclass
class CVFold:
    """Container for a single CV fold"""
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: Optional[np.ndarray] = None
    fold_id: int = 0
    train_start: Optional[pd.Timestamp] = None
    train_end: Optional[pd.Timestamp] = None
    val_start: Optional[pd.Timestamp] = None
    val_end: Optional[pd.Timestamp] = None
    test_start: Optional[pd.Timestamp] = None
    test_end: Optional[pd.Timestamp] = None


class TimeSeriesSplit:
    """
    Time series cross-validation with financial data considerations

    Features:
    - Purging: Gap between train and validation to prevent leakage
    - Embargo: Prevent using same time period in multiple folds
    - Walk-forward: Expanding or rolling window
    - Group awareness: Handle multiple assets/traders
    """

    def __init__(self,
                 n_splits: int = 5,
                 train_size: Optional[int] = None,
                 val_size: Optional[int] = None,
                 test_size: Optional[int] = None,
                 purge_days: int = TC.PURGE_DAYS,
                 embargo_days: int = 0,
                 mode: str = 'expanding'):
        """
        Initialize time series splitter

        Args:
            n_splits: Number of CV folds
            train_size: Training set size (days). If None, expanding window
            val_size: Validation set size (days). If None, computed from data
            test_size: Test set size (days). If provided, creates train/val/test split
            purge_days: Gap between train and val to prevent leakage
            embargo_days: Days to embargo after each fold
            mode: 'expanding' or 'rolling' window
        """
        self.n_splits = n_splits
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.purge_days = purge_days
        self.embargo_days = embargo_days
        self.mode = mode

        if mode not in ['expanding', 'rolling']:
            raise ValueError(f"Mode must be 'expanding' or 'rolling', got {mode}")

    def split(self,
              X: pd.DataFrame,
              y: Optional[pd.Series] = None,
              groups: Optional[pd.Series] = None) -> Iterator[CVFold]:
        """
        Generate train/validation indices for time series CV

        Args:
            X: Features DataFrame with datetime index
            y: Optional target series
            groups: Optional group labels (e.g., trader IDs)

        Yields:
            CVFold objects with train/val indices
        """
        # Ensure datetime index
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("X must have DatetimeIndex")

        # Get unique dates
        dates = X.index.unique().sort_values()
        n_dates = len(dates)

        # Calculate sizes if not provided
        if self.val_size is None:
            # Use remaining data divided by splits
            remaining = n_dates - (self.train_size or n_dates // 2)
            self.val_size = max(20, remaining // (self.n_splits + 1))

        if self.test_size and self.test_size > 0:
            # Reserve test data at the end
            n_dates = n_dates - self.test_size
            dates = dates[:-self.test_size]

        # Minimum training size
        min_train = self.train_size or TC.MIN_TRAINING_DAYS

        # Generate folds
        fold_id = 0
        current_pos = min_train

        while fold_id < self.n_splits and current_pos + self.val_size <= n_dates:
            # Training dates
            if self.mode == 'expanding':
                train_start_idx = 0
            else:  # rolling
                train_start_idx = max(0, current_pos - (self.train_size or min_train))

            train_end_idx = current_pos

            # Apply purge
            val_start_idx = train_end_idx + self.purge_days
            val_end_idx = min(val_start_idx + self.val_size, n_dates)

            # Skip if not enough validation data
            if val_end_idx - val_start_idx < 10:
                break

            # Get actual dates
            train_dates = dates[train_start_idx:train_end_idx]
            val_dates = dates[val_start_idx:val_end_idx]

            # Get indices
            train_idx = X.index.isin(train_dates)
            val_idx = X.index.isin(val_dates)

            # Handle groups if provided
            if groups is not None:
                train_idx, val_idx = self._apply_group_constraints(
                    X, train_idx, val_idx, groups
                )

            # Create fold
            fold = CVFold(
                train_idx=np.where(train_idx)[0],
                val_idx=np.where(val_idx)[0],
                fold_id=fold_id,
                train_start=train_dates[0],
                train_end=train_dates[-1],
                val_start=val_dates[0],
                val_end=val_dates[-1]
            )

            # Add test indices if test_size specified
            if self.test_size and self.test_size > 0:
                test_dates = dates[-self.test_size:]
                test_idx = X.index.isin(test_dates)
                fold.test_idx = np.where(test_idx)[0]
                fold.test_start = test_dates[0]
                fold.test_end = test_dates[-1]

            yield fold

            # Move to next fold
            fold_id += 1
            current_pos = val_end_idx + self.embargo_days

    def _apply_group_constraints(self,
                                X: pd.DataFrame,
                                train_idx: pd.Series,
                                val_idx: pd.Series,
                                groups: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Apply group constraints to prevent leakage
        E.g., same trader shouldn't appear in train and val on same date
        """
        # This is a simplified version
        # In practice, might need more sophisticated group handling
        return train_idx, val_idx

    def get_n_splits(self, X: pd.DataFrame) -> int:
        """Get actual number of splits for given data"""
        splits = list(self.split(X))
        return len(splits)


class WalkForwardAnalysis:
    """
    Walk-forward analysis for trading strategies
    Combines training, validation, and out-of-sample testing
    """

    def __init__(self,
                 train_days: int = TC.WALK_FORWARD_TRAIN_DAYS,
                 val_days: int = TC.WALK_FORWARD_VAL_DAYS,
                 test_days: int = TC.WALK_FORWARD_TEST_DAYS,
                 retrain_frequency: int = 21,  # Retrain every 21 days (monthly)
                 min_train_days: int = TC.MIN_TRAINING_DAYS):
        """
        Initialize walk-forward analysis

        Args:
            train_days: Days for training
            val_days: Days for validation (hyperparameter tuning)
            test_days: Days for out-of-sample testing
            retrain_frequency: How often to retrain (days)
            min_train_days: Minimum training days required
        """
        self.train_days = train_days
        self.val_days = val_days
        self.test_days = test_days
        self.retrain_frequency = retrain_frequency
        self.min_train_days = min_train_days

    def generate_splits(self,
                       data: pd.DataFrame,
                       start_date: Optional[pd.Timestamp] = None,
                       end_date: Optional[pd.Timestamp] = None) -> List[CVFold]:
        """
        Generate walk-forward splits

        Args:
            data: DataFrame with datetime index
            start_date: Start date for analysis
            end_date: End date for analysis

        Returns:
            List of CVFold objects
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have DatetimeIndex")

        # Get date range
        dates = data.index.unique().sort_values()

        if start_date:
            dates = dates[dates >= start_date]
        if end_date:
            dates = dates[dates <= end_date]

        splits = []

        # Initial position after minimum training
        current_date_idx = self.min_train_days
        fold_id = 0

        while current_date_idx + self.val_days + self.test_days <= len(dates):
            # Only retrain at specified frequency
            if fold_id > 0 and fold_id % (self.retrain_frequency // self.test_days) != 0:
                # Shift test window without retraining
                test_start_idx = current_date_idx
                test_end_idx = test_start_idx + self.test_days

                # Use previous fold's model
                if splits:
                    prev_fold = splits[-1]
                    fold = CVFold(
                        train_idx=prev_fold.train_idx,
                        val_idx=prev_fold.val_idx,
                        test_idx=np.where(data.index.isin(dates[test_start_idx:test_end_idx]))[0],
                        fold_id=fold_id,
                        train_start=prev_fold.train_start,
                        train_end=prev_fold.train_end,
                        val_start=prev_fold.val_start,
                        val_end=prev_fold.val_end,
                        test_start=dates[test_start_idx],
                        test_end=dates[test_end_idx - 1]
                    )
                    splits.append(fold)
            else:
                # Full retrain
                # Training period
                train_end_idx = current_date_idx
                train_start_idx = max(0, train_end_idx - self.train_days)

                # Validation period (with purge)
                val_start_idx = train_end_idx + TC.PURGE_DAYS
                val_end_idx = val_start_idx + self.val_days

                # Test period (with purge)
                test_start_idx = val_end_idx + TC.PURGE_DAYS
                test_end_idx = test_start_idx + self.test_days

                if test_end_idx > len(dates):
                    break

                # Get indices
                train_dates = dates[train_start_idx:train_end_idx]
                val_dates = dates[val_start_idx:val_end_idx]
                test_dates = dates[test_start_idx:test_end_idx]

                fold = CVFold(
                    train_idx=np.where(data.index.isin(train_dates))[0],
                    val_idx=np.where(data.index.isin(val_dates))[0],
                    test_idx=np.where(data.index.isin(test_dates))[0],
                    fold_id=fold_id,
                    train_start=train_dates[0],
                    train_end=train_dates[-1],
                    val_start=val_dates[0],
                    val_end=val_dates[-1],
                    test_start=test_dates[0],
                    test_end=test_dates[-1]
                )

                splits.append(fold)

            # Move forward
            current_date_idx += self.test_days
            fold_id += 1

        logger.info(f"Generated {len(splits)} walk-forward splits")
        return splits

    def analyze_stability(self,
                         results: Dict[int, Dict[str, float]]) -> pd.DataFrame:
        """
        Analyze stability of results across walk-forward periods

        Args:
            results: Dictionary mapping fold_id to performance metrics

        Returns:
            DataFrame with stability analysis
        """
        if not results:
            return pd.DataFrame()

        # Convert to DataFrame
        results_df = pd.DataFrame.from_dict(results, orient='index')

        # Calculate stability metrics
        stability_metrics = {
            'mean': results_df.mean(),
            'std': results_df.std(),
            'sharpe': results_df.mean() / (results_df.std() + TC.MIN_VARIANCE),
            'min': results_df.min(),
            'max': results_df.max(),
            'cv': results_df.std() / (results_df.mean().abs() + TC.MIN_VARIANCE),
            'trend': self._calculate_trend(results_df)
        }

        return pd.DataFrame(stability_metrics)

    def _calculate_trend(self, results_df: pd.DataFrame) -> pd.Series:
        """Calculate trend in metrics over time"""
        trends = {}

        for col in results_df.columns:
            # Simple linear regression
            x = np.arange(len(results_df))
            y = results_df[col].values

            # Remove NaN values
            mask = ~np.isnan(y)
            if mask.sum() < 2:
                trends[col] = 0
                continue

            x = x[mask]
            y = y[mask]

            # Calculate slope
            slope = np.polyfit(x, y, 1)[0]

            # Normalize by mean
            trends[col] = slope / (np.abs(y.mean()) + TC.MIN_VARIANCE)

        return pd.Series(trends)


class PurgedKFold:
    """
    K-Fold CV with purging for time series
    Ensures no temporal leakage between folds
    """

    def __init__(self,
                 n_splits: int = 5,
                 purge_days: int = TC.PURGE_DAYS):
        """
        Initialize purged K-fold

        Args:
            n_splits: Number of folds
            purge_days: Days to purge around test fold
        """
        self.n_splits = n_splits
        self.purge_days = purge_days

    def split(self,
              X: pd.DataFrame,
              y: Optional[pd.Series] = None,
              groups: Optional[pd.Series] = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate purged train/test splits

        Args:
            X: Features with datetime index
            y: Optional target
            groups: Optional groups

        Yields:
            Tuples of (train_idx, test_idx)
        """
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("X must have DatetimeIndex")

        # Get unique dates
        dates = X.index.unique().sort_values()
        n_dates = len(dates)

        # Calculate fold size
        fold_size = n_dates // self.n_splits

        for fold in range(self.n_splits):
            # Test fold dates
            test_start_idx = fold * fold_size
            test_end_idx = (fold + 1) * fold_size if fold < self.n_splits - 1 else n_dates

            test_dates = dates[test_start_idx:test_end_idx]

            # Purge around test dates
            purge_start = test_dates[0] - timedelta(days=self.purge_days)
            purge_end = test_dates[-1] + timedelta(days=self.purge_days)

            # Training dates (everything except test and purged)
            train_mask = ~((dates >= purge_start) & (dates <= purge_end))
            train_dates = dates[train_mask]

            # Get indices
            train_idx = np.where(X.index.isin(train_dates))[0]
            test_idx = np.where(X.index.isin(test_dates))[0]

            yield train_idx, test_idx
