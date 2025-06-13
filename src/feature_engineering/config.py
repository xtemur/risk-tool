"""
Configuration for Feature Engineering Pipeline

Centralized configuration for customizing feature engineering behavior.
"""

from typing import List, Dict, Any


class FeatureConfig:
    """
    Configuration class for feature engineering
    """

    # Columns to drop from daily summary data
    COLUMNS_TO_DROP: List[str] = [
        'cash',
        'transfers',
        'transfer_deposit',
        'end_balance',
        'fee_software_md',
        'fee_vat',
        'fee_daily_interest',
        'id',
        'created_at',
        'unrealized_delta',
        'total_delta',
        'adj_fees',
        'adj_net'
    ]

    # Columns to sum into total_sum
    SUM_COLUMNS: List[str] = [
        'comm',
        'ecn_fee',
        'sec',
        'orf',
        'cat',
        'taf',
        'ftt',
        'nscc',
        'acc',
        'clr',
        'misc',
        'trade_fees'
    ]

    # Rolling window sizes for features
    ROLLING_WINDOWS: List[int] = [3, 7, 10, 20]

    # Short-term windows for behavioral features
    SHORT_WINDOWS: List[int] = [1, 2, 3, 5]

    # Default number of features to select
    DEFAULT_N_FEATURES: int = 10

    # Minimum samples required per trader
    MIN_SAMPLES_PER_TRADER: int = 30

    # Feature selection method
    FEATURE_SELECTION_METHOD: str = 'f_regression'  # or 'mutual_info'

    # Target column name
    TARGET_COLUMN: str = 'net'

    # Feature categories and their importance weights
    FEATURE_CATEGORIES: Dict[str, float] = {
        'revenge_trading': 1.0,
        'loss_aversion': 1.0,
        'rolling_performance': 0.8,
        'aggressive_trading': 0.7,
        'risk_behavior': 0.6,
        'consistency': 0.5,
        'market_timing': 0.4,
        'basic': 0.3
    }

    @classmethod
    def get_custom_config(cls, **kwargs) -> 'FeatureConfig':
        """
        Create a custom configuration

        Args:
            **kwargs: Configuration overrides

        Returns:
            Custom FeatureConfig instance
        """
        config = cls()
        for key, value in kwargs.items():
            if hasattr(config, key.upper()):
                setattr(config, key.upper(), value)
        return config

    @classmethod
    def for_options_trading(cls) -> 'FeatureConfig':
        """
        Configuration optimized for options trading
        """
        config = cls()

        # Options traders typically have different risk patterns
        config.ROLLING_WINDOWS = [3, 7, 10]  # Shorter windows
        config.DEFAULT_N_FEATURES = 12  # More features for complex options strategies

        # Different feature importance for options
        config.FEATURE_CATEGORIES = {
            'risk_behavior': 1.0,  # Higher importance for options
            'revenge_trading': 0.9,
            'loss_aversion': 0.9,
            'aggressive_trading': 0.8,
            'rolling_performance': 0.7,
            'consistency': 0.6,
            'market_timing': 0.5,
            'basic': 0.3
        }

        return config

    @classmethod
    def for_equity_trading(cls) -> 'FeatureConfig':
        """
        Configuration optimized for equity trading
        """
        config = cls()

        # Equity traders benefit from longer-term patterns
        config.ROLLING_WINDOWS = [5, 10, 20, 30]  # Longer windows
        config.DEFAULT_N_FEATURES = 8  # Fewer features for cleaner signals

        # Different feature importance for equities
        config.FEATURE_CATEGORIES = {
            'rolling_performance': 1.0,  # Higher importance for equities
            'consistency': 0.9,
            'revenge_trading': 0.8,
            'loss_aversion': 0.8,
            'market_timing': 0.7,
            'aggressive_trading': 0.6,
            'risk_behavior': 0.5,
            'basic': 0.3
        }

        return config
