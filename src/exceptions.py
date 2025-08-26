"""Custom exceptions for the risk tool."""

from typing import Any, Optional


class RiskToolException(Exception):
    """Base exception for all risk tool errors."""

    def __init__(self, message: str, details: Optional[dict] = None):
        """Initialize exception with message and optional details."""
        super().__init__(message)
        self.message = message
        self.details = details or {}


class DataException(RiskToolException):
    """Base exception for data-related errors."""
    pass


class ModelException(RiskToolException):
    """Base exception for model-related errors."""
    pass


class ConfigurationException(RiskToolException):
    """Base exception for configuration errors."""
    pass


# Data Exceptions
class InsufficientDataError(DataException):
    """Raised when there's not enough data for analysis."""

    def __init__(self, trader_id: int, required: int, actual: int):
        """Initialize with trader info and data counts."""
        message = f"Insufficient data for trader {trader_id}: required {required}, got {actual}"
        super().__init__(message, {
            'trader_id': trader_id,
            'required_count': required,
            'actual_count': actual
        })


class DataValidationError(DataException):
    """Raised when data validation fails."""

    def __init__(self, field: str, reason: str, value: Any = None):
        """Initialize with validation details."""
        message = f"Data validation failed for {field}: {reason}"
        super().__init__(message, {
            'field': field,
            'reason': reason,
            'value': value
        })


class DatabaseConnectionError(DataException):
    """Raised when database connection fails."""

    def __init__(self, db_path: str, original_error: Exception):
        """Initialize with database path and original error."""
        message = f"Failed to connect to database at {db_path}: {str(original_error)}"
        super().__init__(message, {
            'db_path': db_path,
            'original_error': str(original_error)
        })


class TraderNotFoundError(DataException):
    """Raised when a trader is not found."""

    def __init__(self, trader_id: int):
        """Initialize with trader ID."""
        message = f"Trader {trader_id} not found"
        super().__init__(message, {'trader_id': trader_id})


# Model Exceptions
class ModelNotFoundError(ModelException):
    """Raised when a trader model is not found."""

    def __init__(self, trader_id: int, model_path: str = None):
        """Initialize with trader ID and optional model path."""
        message = f"Model for trader {trader_id} not found"
        if model_path:
            message += f" at {model_path}"
        super().__init__(message, {
            'trader_id': trader_id,
            'model_path': model_path
        })


class ModelLoadError(ModelException):
    """Raised when model loading fails."""

    def __init__(self, model_path: str, original_error: Exception):
        """Initialize with model path and original error."""
        message = f"Failed to load model from {model_path}: {str(original_error)}"
        super().__init__(message, {
            'model_path': model_path,
            'original_error': str(original_error)
        })


class PredictionError(ModelException):
    """Raised when prediction fails."""

    def __init__(self, trader_id: int, reason: str):
        """Initialize with trader ID and failure reason."""
        message = f"Prediction failed for trader {trader_id}: {reason}"
        super().__init__(message, {
            'trader_id': trader_id,
            'reason': reason
        })


class ModelTrainingError(ModelException):
    """Raised when model training fails."""

    def __init__(self, trader_id: int, reason: str, details: dict = None):
        """Initialize with training failure details."""
        message = f"Model training failed for trader {trader_id}: {reason}"
        super().__init__(message, {
            'trader_id': trader_id,
            'reason': reason,
            **(details or {})
        })


# Configuration Exceptions
class ConfigurationNotFoundError(ConfigurationException):
    """Raised when configuration file is not found."""

    def __init__(self, config_path: str):
        """Initialize with configuration path."""
        message = f"Configuration file not found: {config_path}"
        super().__init__(message, {'config_path': config_path})


class ConfigurationValidationError(ConfigurationException):
    """Raised when configuration validation fails."""

    def __init__(self, errors: list):
        """Initialize with validation errors."""
        message = f"Configuration validation failed with {len(errors)} errors"
        super().__init__(message, {'errors': errors})


class InvalidConfigurationError(ConfigurationException):
    """Raised when configuration contains invalid values."""

    def __init__(self, key: str, value: Any, reason: str):
        """Initialize with invalid configuration details."""
        message = f"Invalid configuration for {key}: {reason}"
        super().__init__(message, {
            'key': key,
            'value': value,
            'reason': reason
        })


# Risk Assessment Exceptions
class RiskAssessmentError(RiskToolException):
    """Raised when risk assessment fails."""

    def __init__(self, trader_id: int, reason: str):
        """Initialize with assessment failure details."""
        message = f"Risk assessment failed for trader {trader_id}: {reason}"
        super().__init__(message, {
            'trader_id': trader_id,
            'reason': reason
        })


class ThresholdOptimizationError(RiskToolException):
    """Raised when threshold optimization fails."""

    def __init__(self, reason: str, details: dict = None):
        """Initialize with optimization failure details."""
        message = f"Threshold optimization failed: {reason}"
        super().__init__(message, details)


# Email/Notification Exceptions
class NotificationError(RiskToolException):
    """Raised when notification sending fails."""

    def __init__(self, recipient: str, reason: str):
        """Initialize with notification failure details."""
        message = f"Failed to send notification to {recipient}: {reason}"
        super().__init__(message, {
            'recipient': recipient,
            'reason': reason
        })


# Feature Engineering Exceptions
class FeatureEngineeringError(RiskToolException):
    """Raised when feature engineering fails."""

    def __init__(self, feature_name: str, reason: str):
        """Initialize with feature engineering failure details."""
        message = f"Feature engineering failed for {feature_name}: {reason}"
        super().__init__(message, {
            'feature_name': feature_name,
            'reason': reason
        })


# Utility function for exception handling
def handle_exception(exception: Exception, logger=None, reraise: bool = True):
    """
    Centralized exception handler.

    Args:
        exception: The exception to handle
        logger: Optional logger instance
        reraise: Whether to re-raise the exception
    """
    if logger:
        if isinstance(exception, RiskToolException):
            logger.error(f"{exception.message} - Details: {exception.details}")
        else:
            logger.error(f"Unexpected error: {str(exception)}", exc_info=True)

    if reraise:
        raise exception
