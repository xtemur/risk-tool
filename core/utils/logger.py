"""Logging infrastructure for the Trader Risk Management System."""

import os
import sys
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import json
import structlog
from pythonjsonlogger import jsonlogger

try:
    from ..config import get_config
except ImportError:
    # Fallback for when running as script
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import get_config


class RiskManagementLogger:
    """Centralized logging configuration for the system."""

    _instance: Optional['RiskManagementLogger'] = None
    _initialized: bool = False

    def __new__(cls):
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize logger configuration."""
        if not self._initialized:
            self.config = get_config()
            self.setup_logging()
            self._initialized = True

    def setup_logging(self):
        """Configure logging based on configuration settings."""
        log_config = self.config['logging']

        # Create log directory if it doesn't exist
        if log_config['handlers']['file']['enabled']:
            log_path = Path(log_config['handlers']['file']['path'])
            log_path.parent.mkdir(parents=True, exist_ok=True)

        # Configure structlog for structured logging
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer() if log_config['format'] == "json" else structlog.dev.ConsoleRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

        # Configure standard logging
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_config['level']))

        # Remove existing handlers
        root_logger.handlers = []

        # Setup formatters
        if log_config['format'] == "json":
            formatter = jsonlogger.JsonFormatter(
                '%(timestamp)s %(level)s %(name)s %(message)s',
                timestamp=True
            )
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )

        # Console handler
        if log_config['handlers']['console']['enabled']:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            console_handler.setLevel(getattr(logging, log_config['level']))
            root_logger.addHandler(console_handler)

        # File handler with rotation
        if log_config['handlers']['file']['enabled']:
            file_handler = logging.handlers.RotatingFileHandler(
                filename=log_config['handlers']['file']['path'],
                maxBytes=log_config['handlers']['file']['max_bytes'],
                backupCount=log_config['handlers']['file']['backup_count']
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(getattr(logging, log_config['level']))
            root_logger.addHandler(file_handler)

    @staticmethod
    def get_logger(name: str) -> structlog.BoundLogger:
        """
        Get a logger instance for a specific module.

        Args:
            name: Module name (typically __name__)

        Returns:
            Configured logger instance
        """
        return structlog.get_logger(name)

    @staticmethod
    def log_model_performance(
        trader_id: str,
        accuracy: float,
        f1_score: float,
        confusion_matrix: Dict[str, Any],
        additional_metrics: Optional[Dict[str, Any]] = None
    ):
        """Log model performance metrics."""
        logger = structlog.get_logger("model_performance")

        log_data = {
            "trader_id": trader_id,
            "accuracy": accuracy,
            "f1_score": f1_score,
            "confusion_matrix": confusion_matrix,
            "timestamp": datetime.utcnow().isoformat()
        }

        if additional_metrics:
            log_data.update(additional_metrics)

        logger.info("Model performance recorded", **log_data)

    @staticmethod
    def log_signal_generation(
        trader_id: str,
        signal: int,
        confidence: float,
        features: Optional[Dict[str, Any]] = None
    ):
        """Log signal generation event."""
        logger = structlog.get_logger("signal_generation")

        log_data = {
            "trader_id": trader_id,
            "signal": signal,
            "confidence": confidence,
            "timestamp": datetime.utcnow().isoformat()
        }

        if features:
            log_data["features"] = features

        logger.info("Risk signal generated", **log_data)

    @staticmethod
    def log_trading_impact(
        trader_id: str,
        date: str,
        signal: int,
        actual_pnl: float,
        avoided_loss: Optional[float] = None,
        strategy: str = "trade_filtering"
    ):
        """Log trading impact of risk signals."""
        logger = structlog.get_logger("trading_impact")

        log_data = {
            "trader_id": trader_id,
            "date": date,
            "signal": signal,
            "actual_pnl": actual_pnl,
            "strategy": strategy,
            "timestamp": datetime.utcnow().isoformat()
        }

        if avoided_loss is not None:
            log_data["avoided_loss"] = avoided_loss

        logger.info("Trading impact recorded", **log_data)

    @staticmethod
    def log_error(
        error_type: str,
        error_message: str,
        context: Optional[Dict[str, Any]] = None,
        exc_info: bool = True
    ):
        """Log error with context."""
        logger = structlog.get_logger("error")

        log_data = {
            "error_type": error_type,
            "error_message": error_message,
            "timestamp": datetime.utcnow().isoformat()
        }

        if context:
            log_data["context"] = context

        logger.error("Error occurred", exc_info=exc_info, **log_data)

    @staticmethod
    def log_data_quality_issue(
        issue_type: str,
        trader_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log data quality issues."""
        logger = structlog.get_logger("data_quality")

        log_data = {
            "issue_type": issue_type,
            "timestamp": datetime.utcnow().isoformat()
        }

        if trader_id:
            log_data["trader_id"] = trader_id

        if details:
            log_data["details"] = details

        logger.warning("Data quality issue detected", **log_data)

    @staticmethod
    def log_system_event(
        event_type: str,
        description: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log system-level events."""
        logger = structlog.get_logger("system")

        log_data = {
            "event_type": event_type,
            "description": description,
            "timestamp": datetime.utcnow().isoformat()
        }

        if metadata:
            log_data["metadata"] = metadata

        logger.info("System event", **log_data)


# Initialize logger on module import
_logger_instance = RiskManagementLogger()


# Convenience functions
def get_logger(name: str) -> structlog.BoundLogger:
    """Get a logger instance."""
    return RiskManagementLogger.get_logger(name)


def log_model_performance(*args, **kwargs):
    """Log model performance metrics."""
    RiskManagementLogger.log_model_performance(*args, **kwargs)


def log_signal_generation(*args, **kwargs):
    """Log signal generation event."""
    RiskManagementLogger.log_signal_generation(*args, **kwargs)


def log_trading_impact(*args, **kwargs):
    """Log trading impact."""
    RiskManagementLogger.log_trading_impact(*args, **kwargs)


def log_error(*args, **kwargs):
    """Log error with context."""
    RiskManagementLogger.log_error(*args, **kwargs)


def log_data_quality_issue(*args, **kwargs):
    """Log data quality issues."""
    RiskManagementLogger.log_data_quality_issue(*args, **kwargs)


def log_system_event(*args, **kwargs):
    """Log system events."""
    RiskManagementLogger.log_system_event(*args, **kwargs)
