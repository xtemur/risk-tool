"""Configuration management for the risk tool."""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import logging

from ..constants import CONFIGS_DIR, DB_PATH, ACTIVE_TRADERS

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration."""
    path: str
    backup_enabled: bool = True
    backup_retention_days: int = 30


@dataclass
class ModelConfig:
    """Model configuration."""
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    early_stopping_rounds: int = 50
    n_estimators: int = 100
    max_depth: int = 5
    learning_rate: float = 0.1


@dataclass
class EmailConfig:
    """Email configuration."""
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    use_tls: bool = True
    sender: Optional[str] = None
    recipients: List[str] = None

    def __post_init__(self):
        if self.recipients is None:
            self.recipients = []


@dataclass
class RiskConfig:
    """Risk assessment configuration."""
    confidence_level: float = 0.95
    var_confidence: float = 0.95
    lookback_days: int = 30
    min_trades: int = 10
    high_risk_sharpe: float = 0.5
    high_risk_win_rate: float = 40.0
    high_risk_wl_ratio: float = 1.0


class ConfigManager:
    """Manages application configuration."""

    def __init__(self, env: str = 'production', config_file: Optional[str] = None):
        """Initialize configuration manager."""
        self.env = env
        self.config_file = config_file or CONFIGS_DIR / "main_config.yaml"
        self._config = self._load_config()
        self._override_from_env()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        if Path(self.config_file).exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = yaml.safe_load(f)
                logger.info(f"Loaded configuration from {self.config_file}")

                # Convert old format to new format if needed
                if 'paths' in config and 'database' not in config:
                    config['database'] = {
                        'path': config['paths'].get('db_path', str(DB_PATH)),
                        'backup_enabled': True,
                        'backup_retention_days': 30
                    }

                # Ensure required sections exist
                if 'database' not in config:
                    config['database'] = {}
                if 'email' not in config:
                    config['email'] = {}
                if 'model' not in config:
                    config['model'] = {}
                if 'risk' not in config:
                    config['risk'] = {}

                return config
            except Exception as e:
                logger.error(f"Error loading config: {e}")
                return self._get_default_config()
        else:
            logger.warning(f"Config file not found: {self.config_file}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'database': {
                'path': str(DB_PATH),
                'backup_enabled': True,
                'backup_retention_days': 30
            },
            'model': {
                'test_size': 0.2,
                'random_state': 42,
                'cv_folds': 5,
                'early_stopping_rounds': 50,
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.1
            },
            'email': {
                'smtp_host': 'smtp.gmail.com',
                'smtp_port': 587,
                'use_tls': True,
                'sender': None,
                'recipients': []
            },
            'risk': {
                'confidence_level': 0.95,
                'var_confidence': 0.95,
                'lookback_days': 30,
                'min_trades': 10,
                'high_risk_sharpe': 0.5,
                'high_risk_win_rate': 40.0,
                'high_risk_wl_ratio': 1.0
            },
            'active_traders': ACTIVE_TRADERS,
            'features': {
                'rolling_windows': [5, 10, 20, 30],
                'technical_indicators': ['rsi', 'macd', 'bollinger', 'ema']
            }
        }

    def _override_from_env(self):
        """Override configuration from environment variables."""
        # Ensure required config sections exist
        if 'database' not in self._config:
            self._config['database'] = {}
        if 'email' not in self._config:
            self._config['email'] = {}
        if 'model' not in self._config:
            self._config['model'] = {}

        # Database path
        if db_path := os.getenv('RISK_TOOL_DB_PATH'):
            self._config['database']['path'] = db_path

        # Email settings
        if smtp_host := os.getenv('RISK_TOOL_SMTP_HOST'):
            self._config['email']['smtp_host'] = smtp_host

        if smtp_port := os.getenv('RISK_TOOL_SMTP_PORT'):
            self._config['email']['smtp_port'] = int(smtp_port)

        if email_sender := os.getenv('RISK_TOOL_EMAIL_SENDER'):
            self._config['email']['sender'] = email_sender

        if email_recipients := os.getenv('RISK_TOOL_EMAIL_RECIPIENTS'):
            self._config['email']['recipients'] = email_recipients.split(',')

        # Model settings
        if test_size := os.getenv('RISK_TOOL_MODEL_TEST_SIZE'):
            self._config['model']['test_size'] = float(test_size)

    @property
    def database(self) -> DatabaseConfig:
        """Get database configuration."""
        db_config = self._config.get('database', {})
        # Ensure path is always provided
        if 'path' not in db_config:
            db_config['path'] = str(DB_PATH)
        return DatabaseConfig(**db_config)

    @property
    def model(self) -> ModelConfig:
        """Get model configuration."""
        return ModelConfig(**self._config.get('model', {}))

    @property
    def email(self) -> EmailConfig:
        """Get email configuration."""
        return EmailConfig(**self._config.get('email', {}))

    @property
    def risk(self) -> RiskConfig:
        """Get risk configuration."""
        return RiskConfig(**self._config.get('risk', {}))

    @property
    def active_traders(self) -> List[int]:
        """Get list of active traders."""
        return self._config.get('active_traders', ACTIVE_TRADERS)

    @property
    def features(self) -> Dict[str, Any]:
        """Get feature engineering configuration."""
        return self._config.get('features', {})

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    def reload(self):
        """Reload configuration from file."""
        self._config = self._load_config()
        self._override_from_env()
        logger.info("Configuration reloaded")

    def save(self, path: Optional[str] = None):
        """Save current configuration to file."""
        save_path = path or self.config_file

        try:
            with open(save_path, 'w') as f:
                yaml.dump(self._config, f, default_flow_style=False, indent=2)
            logger.info(f"Configuration saved to {save_path}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            raise

    def to_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary."""
        return self._config.copy()

    def validate(self) -> bool:
        """Validate configuration."""
        errors = []

        # Check database path
        if not Path(self.database.path).parent.exists():
            errors.append(f"Database directory does not exist: {Path(self.database.path).parent}")

        # Check model parameters
        if not 0 < self.model.test_size < 1:
            errors.append(f"Invalid test_size: {self.model.test_size}")

        if self.model.cv_folds < 2:
            errors.append(f"Invalid cv_folds: {self.model.cv_folds}")

        # Check risk thresholds
        if not 0 < self.risk.confidence_level <= 1:
            errors.append(f"Invalid confidence_level: {self.risk.confidence_level}")

        if errors:
            for error in errors:
                logger.error(f"Configuration validation error: {error}")
            return False

        logger.info("Configuration validation passed")
        return True
