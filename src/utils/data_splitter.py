# src/utils/data_splitter.py
"""
Data Splitter
Manages train/validation/test splits with proper temporal ordering
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path

from src.core.constants import TradingConstants as TC

logger = logging.getLogger(__name__)


class DataSplitter:
    """
    Manages data splits ensuring temporal integrity
    Once test set is created, it's locked and never used in training
    """

    def __init__(self,
                 test_size: float = 0.15,
                 val_size: float = 0.15,
                 min_test_days: int = 60,
                 split_config_path: str = "data/split_config.json"):
        """
        Args:
            test_size: Proportion of data for final testing
            val_size: Proportion of data for validation
            min_test_days: Minimum days required in test set
            split_config_path: Path to save split configuration
        """
        self.test_size = test_size
        self.val_size = val_size
        self.min_test_days = min_test_days
        self.split_config_path = Path(split_config_path)

        # Load existing split if available
        self.split_dates = self._load_split_config()

    def create_splits(self,
                     data: pd.DataFrame,
                     date_column: str = 'date',
                     force_new: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Create train/val/test splits

        Args:
            data: Full dataset
            date_column: Name of date column
            force_new: Force creation of new splits

        Returns:
            Dictionary with 'train', 'val', 'test' DataFrames
        """
        if self.split_dates and not force_new:
            logger.info("Using existing split configuration")
            return self._apply_splits(data, date_column)

        # Create new splits
        logger.info("Creating new data splits")

        # Ensure data is sorted by date
        data = data.sort_values(date_column)

        # Get unique dates
        unique_dates = sorted(data[date_column].unique())
        n_dates = len(unique_dates)

        # Calculate split indices
        test_days = max(int(n_dates * self.test_size), self.min_test_days)
        val_days = max(int(n_dates * self.val_size), 30)

        # Split dates
        train_end_idx = n_dates - test_days - val_days
        val_end_idx = n_dates - test_days

        self.split_dates = {
            'train_start': unique_dates[0],
            'train_end': unique_dates[train_end_idx - 1],
            'val_start': unique_dates[train_end_idx],
            'val_end': unique_dates[val_end_idx - 1],
            'test_start': unique_dates[val_end_idx],
            'test_end': unique_dates[-1],
            'created_at': datetime.now().isoformat(),
            'n_train_days': train_end_idx,
            'n_val_days': val_days,
            'n_test_days': test_days
        }

        # Save configuration
        self._save_split_config()

        return self._apply_splits(data, date_column)

    def _apply_splits(self, data: pd.DataFrame, date_column: str) -> Dict[str, pd.DataFrame]:
        """Apply saved splits to data"""

        # Ensure date column is datetime
        data[date_column] = pd.to_datetime(data[date_column])

        train_mask = (data[date_column] >= self.split_dates['train_start']) & \
                    (data[date_column] <= self.split_dates['train_end'])
        val_mask = (data[date_column] >= self.split_dates['val_start']) & \
                   (data[date_column] <= self.split_dates['val_end'])
        test_mask = (data[date_column] >= self.split_dates['test_start']) & \
                    (data[date_column] <= self.split_dates['test_end'])

        splits = {
            'train': data[train_mask].copy(),
            'val': data[val_mask].copy(),
            'test': data[test_mask].copy()
        }

        # Log split sizes
        for split_name, split_data in splits.items():
            logger.info(f"{split_name}: {len(split_data)} samples, "
                       f"{split_data[date_column].min()} to {split_data[date_column].max()}")

        return splits

    def _save_split_config(self):
        """Save split configuration"""
        self.split_config_path.parent.mkdir(exist_ok=True, parents=True)

        with open(self.split_config_path, 'w') as f:
            json.dump(self.split_dates, f, indent=2, default=str)

        logger.info(f"Split configuration saved to {self.split_config_path}")

    def _load_split_config(self) -> Optional[Dict]:
        """Load existing split configuration"""
        if self.split_config_path.exists():
            with open(self.split_config_path, 'r') as f:
                config = json.load(f)

            # Convert date strings back to datetime
            for key in ['train_start', 'train_end', 'val_start', 'val_end', 'test_start', 'test_end']:
                config[key] = pd.to_datetime(config[key])

            logger.info(f"Loaded existing split configuration from {self.split_config_path}")
            return config

        return None

    def get_test_dates(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """Get test set date range"""
        if not self.split_dates:
            raise ValueError("No splits created yet")

        return self.split_dates['test_start'], self.split_dates['test_end']
