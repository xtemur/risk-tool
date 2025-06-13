"""
Configuration for Modeling Pipeline

Centralized configuration for model training, validation, and evaluation.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import numpy as np


class ModelConfig:
    """
    Configuration class for modeling pipeline
    """

    # Time series validation settings
    HOLDOUT_MONTHS: int = 2  # Last 2 months for holdout testing
    TRAINING_WINDOW_MONTHS: int = 12  # Rolling training window
    VALIDATION_STEP_MONTHS: int = 1  # Step size for walk-forward
    MIN_TRAINING_MONTHS: int = 6  # Minimum training data required
    MIN_SAMPLES_PER_TRADER: int = 30  # Minimum samples per trader

    # Data splitting
    HOLDOUT_START_DATE: str = "2025-04-01"  # Start of holdout period
    VALIDATION_START_DATE: str = "2024-04-01"  # Start of walk-forward validation

    # Model types and configurations
    MODEL_TYPES: Dict[str, Dict[str, Any]] = {
        'linear': {
            'class': 'LinearRegression',
            'params': {}
        },
        'ridge': {
            'class': 'Ridge',
            'params': {'alpha': 1.0}
        },
        'lasso': {
            'class': 'Lasso',
            'params': {'alpha': 0.1}
        },
        'elastic_net': {
            'class': 'ElasticNet',
            'params': {'alpha': 0.1, 'l1_ratio': 0.5}
        },
        'xgboost': {
            'class': 'XGBRegressor',
            'params': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42
            }
        }
    }

    # Hyperparameter grids for optimization
    HYPERPARAMETER_GRIDS: Dict[str, Dict[str, List]] = {
        'ridge': {
            'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
        },
        'lasso': {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
        },
        'elastic_net': {
            'alpha': [0.01, 0.1, 1.0],
            'l1_ratio': [0.1, 0.5, 0.7, 0.9]
        },
        'xgboost': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.05, 0.1, 0.2]
        }
    }

    # Feature preprocessing
    FEATURE_SCALING: str = 'standard'  # 'standard', 'minmax', 'robust', 'none'
    HANDLE_MISSING: str = 'median'  # 'median', 'mean', 'zero', 'drop'
    HANDLE_OUTLIERS: bool = True  # Whether to cap outliers
    OUTLIER_THRESHOLD: float = 3.0  # Z-score threshold for outliers

    # Feature selection
    FEATURE_SELECTION: bool = True
    MAX_FEATURES: int = 20  # Maximum features to use
    FEATURE_SELECTION_METHOD: str = 'f_regression'  # 'f_regression', 'mutual_info'

    # Cross-validation
    CV_FOLDS: int = 3  # Number of CV folds for hyperparameter tuning
    CV_SCORING: str = 'neg_mean_absolute_error'  # Scoring metric for CV

    # Performance evaluation
    METRICS: List[str] = [
        'mae', 'rmse', 'r2', 'mape',  # Statistical metrics
        'hit_rate', 'sharpe_ratio', 'max_drawdown', 'profit_factor'  # Financial metrics
    ]

    # Model persistence
    SAVE_MODELS: bool = True
    MODEL_SAVE_PATH: str = "models/saved"
    SAVE_PREDICTIONS: bool = True
    PREDICTIONS_SAVE_PATH: str = "results/predictions"

    # Logging and monitoring
    LOG_LEVEL: str = "INFO"
    SAVE_TRAINING_HISTORY: bool = True
    PLOT_RESULTS: bool = True
    RESULTS_PATH: str = "results"

    @classmethod
    def get_custom_config(cls, **kwargs) -> 'ModelConfig':
        """
        Create a custom configuration

        Args:
            **kwargs: Configuration overrides

        Returns:
            Custom ModelConfig instance
        """
        config = cls()
        for key, value in kwargs.items():
            if hasattr(config, key.upper()):
                setattr(config, key.upper(), value)
        return config

    @classmethod
    def for_quick_testing(cls) -> 'ModelConfig':
        """
        Configuration for quick testing/development
        """
        config = cls()
        config.TRAINING_WINDOW_MONTHS = 6  # Shorter training windows
        config.MIN_TRAINING_MONTHS = 3
        config.MIN_SAMPLES_PER_TRADER = 10
        config.CV_FOLDS = 2  # Faster CV
        config.SAVE_MODELS = False  # Don't save during testing
        config.PLOT_RESULTS = False
        return config

    @classmethod
    def for_production(cls) -> 'ModelConfig':
        """
        Configuration for production training
        """
        config = cls()
        config.TRAINING_WINDOW_MONTHS = 18  # Longer training windows
        config.MIN_TRAINING_MONTHS = 12
        config.CV_FOLDS = 5  # More thorough CV
        config.FEATURE_SELECTION = True
        config.HANDLE_OUTLIERS = True
        config.SAVE_MODELS = True
        config.SAVE_TRAINING_HISTORY = True
        return config

    @classmethod
    def for_baseline_only(cls) -> 'ModelConfig':
        """
        Configuration for baseline linear models only
        """
        config = cls()
        # Remove XGBoost from available models
        config.MODEL_TYPES = {k: v for k, v in config.MODEL_TYPES.items()
                             if k != 'xgboost'}
        config.HYPERPARAMETER_GRIDS = {k: v for k, v in config.HYPERPARAMETER_GRIDS.items()
                                      if k != 'xgboost'}
        return config

    def get_holdout_dates(self) -> tuple:
        """
        Get holdout period start and end dates

        Returns:
            Tuple of (start_date, end_date) as datetime objects
        """
        holdout_start = datetime.strptime(self.HOLDOUT_START_DATE, "%Y-%m-%d")
        # Assume holdout goes to end of available data
        holdout_end = datetime(2025, 6, 30)  # Based on the data we saw
        return holdout_start, holdout_end

    def get_validation_dates(self) -> tuple:
        """
        Get validation period start and end dates

        Returns:
            Tuple of (start_date, end_date) as datetime objects
        """
        val_start = datetime.strptime(self.VALIDATION_START_DATE, "%Y-%m-%d")
        val_end = datetime.strptime(self.HOLDOUT_START_DATE, "%Y-%m-%d")
        return val_start, val_end
