"""
Time Series Validator

Handles temporal data splitting, walk-forward validation, and holdout testing
for trading data to prevent future data leakage.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Iterator, Optional, Any
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import logging

from .config import ModelConfig

logger = logging.getLogger(__name__)


class TimeSeriesValidator:
    """
    Time series validation with proper temporal splitting
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize the validator

        Args:
            config: Model configuration object
        """
        self.config = config or ModelConfig()

    def create_holdout_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and holdout sets based on time

        Args:
            df: Complete dataset with 'date' column

        Returns:
            Tuple of (training_data, holdout_data)
        """
        if df.empty:
            return pd.DataFrame(), pd.DataFrame()

        # Ensure date column is datetime
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])

        # Get holdout dates
        holdout_start, holdout_end = self.config.get_holdout_dates()

        # Split data
        training_data = df[df['date'] < holdout_start].copy()
        holdout_data = df[df['date'] >= holdout_start].copy()

        logger.info(f"Holdout split created:")
        logger.info(f"  Training: {len(training_data)} samples ({training_data['date'].min()} to {training_data['date'].max()})")
        logger.info(f"  Holdout: {len(holdout_data)} samples ({holdout_data['date'].min()} to {holdout_data['date'].max()})")

        return training_data, holdout_data

    def create_trader_splits(self, df: pd.DataFrame) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Create holdout splits for each trader individually

        Args:
            df: Complete dataset with 'account_id' and 'date' columns

        Returns:
            Dictionary mapping account_id to (training_data, holdout_data) tuples
        """
        if df.empty or 'account_id' not in df.columns:
            return {}

        trader_splits = {}

        for account_id in df['account_id'].unique():
            trader_data = df[df['account_id'] == account_id].copy()

            # Check if trader has enough data
            if len(trader_data) < self.config.MIN_SAMPLES_PER_TRADER:
                logger.warning(f"Trader {account_id} has only {len(trader_data)} samples, skipping")
                continue

            # Create split for this trader
            train_data, holdout_data = self.create_holdout_split(trader_data)

            if len(train_data) > 0 and len(holdout_data) > 0:
                trader_splits[account_id] = (train_data, holdout_data)
                logger.info(f"Trader {account_id}: {len(train_data)} train, {len(holdout_data)} holdout")

        return trader_splits

    def walk_forward_split(self, training_data: pd.DataFrame) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Generate walk-forward validation splits

        Args:
            training_data: Training dataset (before holdout)

        Yields:
            Tuples of (train_fold, validation_fold)
        """
        if training_data.empty:
            return

        # Ensure data is sorted by date
        training_data = training_data.sort_values('date').copy()
        training_data['date'] = pd.to_datetime(training_data['date'])

        # Get validation period
        val_start, val_end = self.config.get_validation_dates()

        # Start walk-forward from validation start date
        current_date = val_start

        while current_date < val_end:
            # Define validation period (1 month)
            val_period_start = current_date
            val_period_end = current_date + relativedelta(months=self.config.VALIDATION_STEP_MONTHS)

            # Define training period (preceding months)
            train_period_end = val_period_start
            train_period_start = train_period_end - relativedelta(months=self.config.TRAINING_WINDOW_MONTHS)

            # Create splits
            train_fold = training_data[
                (training_data['date'] >= train_period_start) &
                (training_data['date'] < train_period_end)
            ].copy()

            val_fold = training_data[
                (training_data['date'] >= val_period_start) &
                (training_data['date'] < val_period_end)
            ].copy()

            # Check if we have enough training data
            if len(train_fold) < self.config.MIN_SAMPLES_PER_TRADER:
                logger.warning(f"Not enough training data for period {train_period_start} to {train_period_end}")
                current_date = val_period_end
                continue

            # Check if we have validation data
            if len(val_fold) == 0:
                logger.warning(f"No validation data for period {val_period_start} to {val_period_end}")
                current_date = val_period_end
                continue

            logger.info(f"Walk-forward fold: Train {len(train_fold)} samples ({train_period_start.date()} to {train_period_end.date()}), Val {len(val_fold)} samples ({val_period_start.date()} to {val_period_end.date()})")

            yield train_fold, val_fold

            # Move to next period
            current_date = val_period_end

    def trader_walk_forward_split(self, training_data: pd.DataFrame, account_id: str) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Generate walk-forward validation splits for a specific trader

        Args:
            training_data: Training dataset for the trader
            account_id: Trader account ID

        Yields:
            Tuples of (train_fold, validation_fold) for the trader
        """
        trader_data = training_data[training_data['account_id'] == account_id].copy()

        if trader_data.empty:
            logger.warning(f"No data found for trader {account_id}")
            return

        # Generate walk-forward splits for this trader
        for train_fold, val_fold in self.walk_forward_split(trader_data):
            if len(train_fold) >= self.config.MIN_SAMPLES_PER_TRADER and len(val_fold) > 0:
                yield train_fold, val_fold

    def get_temporal_cv_splits(self, df: pd.DataFrame, n_splits: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create temporal cross-validation splits (for hyperparameter tuning)

        Args:
            df: Dataset to split
            n_splits: Number of splits to create

        Returns:
            List of (train_indices, test_indices) tuples
        """
        if df.empty:
            return []

        df = df.sort_values('date').reset_index(drop=True)
        df['date'] = pd.to_datetime(df['date'])

        splits = []
        total_samples = len(df)

        # Calculate split points
        for i in range(n_splits):
            # Each split uses progressively more data for training
            train_size = int(total_samples * (i + 1) / (n_splits + 1))
            test_start = train_size
            test_size = int(total_samples / (n_splits + 1))
            test_end = min(test_start + test_size, total_samples)

            if test_start >= total_samples:
                break

            train_indices = np.arange(train_size)
            test_indices = np.arange(test_start, test_end)

            if len(train_indices) > 0 and len(test_indices) > 0:
                splits.append((train_indices, test_indices))

        logger.info(f"Created {len(splits)} temporal CV splits")
        return splits

    def validate_temporal_integrity(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> bool:
        """
        Validate that there's no temporal leakage between train and test sets

        Args:
            train_data: Training dataset
            test_data: Test dataset

        Returns:
            True if temporal integrity is maintained
        """
        if train_data.empty or test_data.empty:
            return True

        train_max_date = pd.to_datetime(train_data['date']).max()
        test_min_date = pd.to_datetime(test_data['date']).min()

        is_valid = train_max_date < test_min_date

        if not is_valid:
            logger.error(f"Temporal leakage detected! Train max date: {train_max_date}, Test min date: {test_min_date}")
        else:
            logger.debug(f"Temporal integrity validated. Train max: {train_max_date}, Test min: {test_min_date}")

        return is_valid

    def get_validation_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary of validation setup

        Args:
            df: Complete dataset

        Returns:
            Summary statistics
        """
        if df.empty:
            return {}

        df['date'] = pd.to_datetime(df['date'])

        # Overall data summary
        summary = {
            'total_samples': len(df),
            'total_traders': df['account_id'].nunique() if 'account_id' in df.columns else 1,
            'date_range': {
                'start': df['date'].min(),
                'end': df['date'].max(),
                'days': (df['date'].max() - df['date'].min()).days
            }
        }

        # Holdout split summary
        train_data, holdout_data = self.create_holdout_split(df)
        summary['holdout_split'] = {
            'training_samples': len(train_data),
            'holdout_samples': len(holdout_data),
            'holdout_ratio': len(holdout_data) / len(df) if len(df) > 0 else 0
        }

        # Walk-forward summary
        if not train_data.empty:
            wf_splits = list(self.walk_forward_split(train_data))
            summary['walk_forward'] = {
                'num_splits': len(wf_splits),
                'avg_train_samples': np.mean([len(train) for train, _ in wf_splits]) if wf_splits else 0,
                'avg_val_samples': np.mean([len(val) for _, val in wf_splits]) if wf_splits else 0
            }

        # Per-trader summary
        if 'account_id' in df.columns:
            trader_splits = self.create_trader_splits(df)
            summary['per_trader'] = {
                'valid_traders': len(trader_splits),
                'avg_train_samples': np.mean([len(train) for train, _ in trader_splits.values()]) if trader_splits else 0,
                'avg_holdout_samples': np.mean([len(holdout) for _, holdout in trader_splits.values()]) if trader_splits else 0
            }

        return summary
