"""Central constants for the risk tool."""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
CONFIGS_DIR = PROJECT_ROOT / "configs"
LOGS_DIR = PROJECT_ROOT / "logs"

# Database
DB_PATH = DATA_DIR / "risk_tool.db"
DB_BACKUP_PREFIX = "risk_tool.db.backup"

# Model paths
TRADER_MODELS_DIR = MODELS_DIR / "trader_specific"
MODEL_FILE_SUFFIX = "_tuned_validated.pkl"

# Data processing
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRADER_SPLITS_DIR = PROCESSED_DATA_DIR / "trader_splits"

# Risk thresholds
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_VAR_CONFIDENCE = 0.95
DEFAULT_LOOKBACK_DAYS = 30
MIN_TRADES_FOR_ANALYSIS = 10

# Model parameters
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
CV_FOLDS = 5
EARLY_STOPPING_ROUNDS = 50

# Trading metrics thresholds
HIGH_RISK_SHARPE_THRESHOLD = 0.5
HIGH_RISK_WIN_RATE_THRESHOLD = 40.0
HIGH_RISK_WL_RATIO_THRESHOLD = 1.0

# Risk levels
RISK_SCORE_LOW = 25
RISK_SCORE_MEDIUM = 50
RISK_SCORE_HIGH = 75
RISK_SCORE_CRITICAL = 100

# Email configuration
EMAIL_TEMPLATE_DIR = PROJECT_ROOT / "inference" / "templates"
EMAIL_SUBJECT_PREFIX = "[Risk Alert]"
MAX_EMAIL_RETRIES = 3

# Active traders (to be moved to config)
ACTIVE_TRADERS = [
    3942, 3943, 3946, 3950, 3951,
    3956, 4003, 4004, 4396, 5093, 5580
]

# Feature engineering
ROLLING_WINDOWS = [5, 10, 20, 30]
TECHNICAL_INDICATORS = ['rsi', 'macd', 'bollinger', 'ema']

# Logging
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
LOG_FILE_MAX_BYTES = 10_485_760  # 10MB
LOG_FILE_BACKUP_COUNT = 5

# Performance thresholds
MIN_SHARPE_FOR_TRADING = 0.0
MAX_DRAWDOWN_PERCENT = 20.0
MIN_DAILY_TRADES = 5

# Causal impact analysis
CAUSAL_PRE_PERIOD_DAYS = 60
CAUSAL_POST_PERIOD_DAYS = 30
CAUSAL_MODEL_SAMPLES = 1000

# API limits
MAX_BATCH_SIZE = 1000
MAX_CONCURRENT_REQUESTS = 10
REQUEST_TIMEOUT_SECONDS = 30

# Data validation
MAX_MISSING_DATA_PERCENT = 20.0
MIN_DATA_POINTS = 100
OUTLIER_STD_THRESHOLD = 4.0
