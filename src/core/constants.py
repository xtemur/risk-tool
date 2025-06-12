"""
Trading System Constants
All magic numbers and configuration values in one place
"""

from datetime import time
from typing import Dict, List


class TradingConstants:
    """Core trading system constants"""

    # ==================== Time Constants ====================
    # Market hours (EST)
    MARKET_OPEN = time(9, 30)
    MARKET_CLOSE = time(16, 0)
    PRE_MARKET_OPEN = time(4, 0)
    AFTER_MARKET_CLOSE = time(20, 0)

    # Trading days
    TRADING_DAYS_PER_YEAR = 252
    TRADING_DAYS_PER_MONTH = 21
    TRADING_DAYS_PER_WEEK = 5

    # ==================== Risk Parameters ====================
    # Position limits
    MAX_POSITION_SIZE_PCT = 0.1  # Max 10% of capital per position
    MAX_DAILY_LOSS_PCT = 0.02    # Max 2% daily loss
    MAX_DRAWDOWN_PCT = 0.1       # Max 10% drawdown

    # Risk scoring thresholds
    HIGH_RISK_SCORE = 0.8
    MEDIUM_RISK_SCORE = 0.5
    LOW_RISK_SCORE = 0.2

    # ==================== Transaction Costs ====================
    # Brokerage fees (per share)
    COMMISSION_PER_SHARE = 0.005
    MIN_COMMISSION = 1.0

    # Market impact and slippage
    SLIPPAGE_BPS = 2  # 2 basis points
    MARKET_IMPACT_BPS = 1  # 1 basis point for small orders

    # SEC and other regulatory fees
    SEC_FEE_RATE = 0.0000278  # Current SEC fee rate
    TAF_FEE_RATE = 0.000145   # FINRA TAF

    # ==================== Model Parameters ====================
    # Data requirements
    MIN_TRAINING_DAYS = 252      # 1 year minimum
    MIN_SAMPLES_PER_TRADER = 100 # Minimum samples to include trader

    # Feature windows
    FEATURE_WINDOWS = [3, 5, 10, 20, 60]  # Days
    INTRADAY_WINDOWS = [5, 15, 30, 60]   # Minutes

    # Model validation
    WALK_FORWARD_TRAIN_DAYS = 252
    WALK_FORWARD_VAL_DAYS = 63    # 3 months
    WALK_FORWARD_TEST_DAYS = 21   # 1 month
    PURGE_DAYS = 2                # Gap between train/test

    # ==================== Statistical Parameters ====================
    # Outlier detection
    Z_SCORE_THRESHOLD = 4.0       # Extreme outliers
    IQR_MULTIPLIER = 3.0          # For IQR-based outlier detection

    # Minimum variance (to avoid division by zero)
    MIN_VARIANCE = 1e-8

    # Correlation thresholds
    HIGH_CORRELATION = 0.8
    FEATURE_IMPORTANCE_THRESHOLD = 0.01

    # ==================== Data Quality ====================
    # Missing data thresholds
    MAX_MISSING_PCT = 0.2         # Max 20% missing data
    MAX_CONSECUTIVE_MISSING = 5   # Max 5 consecutive missing days

    # Data staleness
    MAX_DATA_AGE_DAYS = 3        # Data older than 3 days is stale

    # ==================== Performance Metrics ====================
    # Risk-free rate (annual)
    RISK_FREE_RATE = 0.05        # 5% annual

    # Target metrics
    TARGET_SHARPE = 1.5
    TARGET_SORTINO = 2.0
    TARGET_CALMAR = 1.0

    # ==================== System Parameters ====================
    # Retry logic
    MAX_API_RETRIES = 3
    API_RETRY_DELAY = 1.0  # seconds

    # Logging
    LOG_LEVEL = "INFO"
    MAX_LOG_SIZE_MB = 100
    LOG_RETENTION_DAYS = 30

    # ==================== Feature Categories ====================
    # Used for feature organization and selection
    FEATURE_CATEGORIES = {
        'returns': ['return_1d', 'return_3d', 'return_5d', 'return_20d'],
        'volatility': ['volatility_5d', 'volatility_20d', 'volatility_60d'],
        'volume': ['volume_ratio_5d', 'volume_ratio_20d'],
        'momentum': ['rsi_14', 'macd_signal', 'momentum_20d'],
        'behavioral': ['win_rate', 'loss_streak', 'time_of_day_bias'],
        'risk': ['max_drawdown_20d', 'value_at_risk', 'expected_shortfall']
    }

    # ==================== Trader Categories ====================
    # For stratified analysis
    TRADER_CATEGORIES = {
        'high_frequency': {'min_daily_trades': 100},
        'day_trader': {'min_daily_trades': 10, 'max_overnight_position': 0},
        'swing_trader': {'min_holding_period_days': 2, 'max_holding_period_days': 10},
        'position_trader': {'min_holding_period_days': 10}
    }


class DataQualityLimits:
    """Data quality check limits"""

    # Price limits (detect bad data)
    MAX_DAILY_RETURN = 0.5       # 50% daily move is suspicious
    MIN_PRICE = 0.01             # Penny stock threshold
    MAX_PRICE = 10000            # Sanity check

    # Volume limits
    MIN_DAILY_VOLUME = 100       # Minimum liquidity
    MAX_VOLUME_SPIKE = 10        # 10x average volume is suspicious

    # P&L limits (per day)
    MAX_DAILY_PNL = 100000       # $100k daily P&L limit
    MIN_DAILY_PNL = -100000      # -$100k daily loss limit

    # Fee sanity checks
    MAX_FEE_PCT = 0.02           # Fees > 2% of gross P&L is suspicious

    # Time gaps
    MAX_TIME_GAP_MINUTES = 390   # Full trading day
    MAX_WEEKEND_GAP_HOURS = 72   # Friday close to Monday open


class ModelConfig:
    """Model configuration parameters"""

    # LightGBM default parameters (conservative for finance)
    LIGHTGBM_PARAMS = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'min_child_samples': 20,
        'min_split_gain': 0.001,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'max_depth': 6,
        'n_estimators': 100,
        'early_stopping_rounds': None,
        'random_state': 42
    }

    # Optuna hyperparameter search space
    OPTUNA_SEARCH_SPACE = {
        'num_leaves': (20, 300),
        'learning_rate': (0.01, 0.3),
        'feature_fraction': (0.5, 1.0),
        'bagging_fraction': (0.5, 1.0),
        'lambda_l1': (0, 1.0),
        'lambda_l2': (0, 1.0),
        'min_child_samples': (10, 100),
        'max_depth': (3, 10)
    }

    # Ensemble weights
    ENSEMBLE_MIN_WEIGHT = 0.1    # Minimum 10% weight
    ENSEMBLE_MAX_WEIGHT = 0.5    # Maximum 50% weight


# Convenience aliases
TC = TradingConstants
DQL = DataQualityLimits
MC = ModelConfig
