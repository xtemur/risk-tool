# config.py - No JSON, no YAML, just Python (as per CLAUDE.md)
class Config:
    # Data
    DATA_PATH = 'data/risk_tool.db'

    # Model
    MODEL_TYPE = 'ridge'  # Start with simplest
    RETRAIN_DAYS = 30  # Monthly is enough

    # Risk parameters
    DEFAULT_LIMIT = 5000
    MAX_REDUCTION = 80  # Never reduce more than 80%
    MIN_SAMPLES_FOR_ML = 200  # REDUCED: 15 traders * ~15 days = ~225 samples is realistic minimum

    # Features
    FEATURE_WINDOWS = [3, 5, 10, 20]  # Don't need more

    # Validation
    CV_SPLITS = 5
    MIN_IMPROVEMENT_FOR_ML = 0.10  # ML must beat rules by 10%

    # Email
    RECIPIENTS = ['temurbekkhujaev@gmail.com', 'risk_manager@firm.com']
    SEND_TIME = '08:00'

    # Account filtering
    EXCLUDE_LEGACY_ACCOUNTS = True  # Exclude accounts with '_OLD' suffix

config = Config()
