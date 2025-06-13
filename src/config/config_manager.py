#!/usr/bin/env python3
"""
Configuration Manager
Handles loading and validation of system configuration
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import os

logger = logging.getLogger(__name__)

class ConfigManager:
    """
    Centralized configuration management for the trading signal system
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager

        Args:
            config_path: Path to config file (defaults to config.yaml in project root)
        """
        if config_path is None:
            # Default to config.yaml in project root
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config.yaml"

        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()

        logger.info(f"✓ Configuration loaded from {self.config_path}")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)

            if not config:
                raise ValueError("Configuration file is empty")

            return config

        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration file: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration: {e}")

    def _validate_config(self) -> None:
        """Validate configuration structure and values"""

        required_sections = ['database', 'models', 'signals', 'email', 'monitoring']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")

        # Validate database config
        db_config = self.config['database']
        if 'path' not in db_config:
            raise ValueError("Database path not specified in configuration")

        # Validate model config
        model_config = self.config['models']
        if 'directory' not in model_config:
            raise ValueError("Model directory not specified in configuration")

        # Validate signal thresholds
        signal_config = self.config['signals']
        if 'thresholds' not in signal_config:
            raise ValueError("Signal thresholds not specified in configuration")

        logger.debug("Configuration validation passed")

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation

        Args:
            key_path: Dot-separated path to config value (e.g., 'database.path')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key_path.split('.')
        value = self.config

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            if default is not None:
                return default
            raise KeyError(f"Configuration key not found: {key_path}")

    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration"""
        return self.config['database']

    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return self.config['models']

    def get_signal_config(self) -> Dict[str, Any]:
        """Get signal configuration"""
        return self.config['signals']

    def get_email_config(self) -> Dict[str, Any]:
        """Get email configuration"""
        return self.config['email']

    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring configuration"""
        return self.config['monitoring']

    def get_risk_config(self) -> Dict[str, Any]:
        """Get risk management configuration"""
        return self.config.get('risk', {})

    def is_debug_enabled(self) -> bool:
        """Check if debug mode is enabled"""
        return self.config.get('debug', {}).get('enabled', False)

    def get_log_level(self) -> str:
        """Get configured log level"""
        return self.config.get('monitoring', {}).get('log_level', 'INFO')

    def get_signal_thresholds(self) -> Dict[str, float]:
        """Get signal classification thresholds"""
        return self.config['signals']['thresholds']

    def get_validation_settings(self) -> Dict[str, Any]:
        """Get data validation settings"""
        return self.config['database'].get('validation', {})

    def update_config(self, key_path: str, value: Any) -> None:
        """
        Update configuration value (in memory only)

        Args:
            key_path: Dot-separated path to config value
            value: New value to set
        """
        keys = key_path.split('.')
        config_ref = self.config

        # Navigate to parent of target key
        for key in keys[:-1]:
            if key not in config_ref:
                config_ref[key] = {}
            config_ref = config_ref[key]

        # Set the value
        config_ref[keys[-1]] = value
        logger.debug(f"Updated config: {key_path} = {value}")

    def save_config(self, path: Optional[str] = None) -> None:
        """
        Save current configuration to file

        Args:
            path: Path to save to (defaults to original config file)
        """
        save_path = Path(path) if path else self.config_path

        try:
            with open(save_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)

            logger.info(f"Configuration saved to {save_path}")

        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise

    def reload_config(self) -> None:
        """Reload configuration from file"""
        logger.info("Reloading configuration...")
        self.config = self._load_config()
        self._validate_config()
        logger.info("✓ Configuration reloaded")

    def get_env_override(self, key_path: str, env_var: str) -> Any:
        """
        Get config value with environment variable override

        Args:
            key_path: Configuration key path
            env_var: Environment variable name

        Returns:
            Environment variable value if set, otherwise config value
        """
        env_value = os.getenv(env_var)
        if env_value is not None:
            logger.debug(f"Using environment override: {env_var} = {env_value}")
            return env_value

        return self.get(key_path)

    def __str__(self) -> str:
        """String representation of configuration"""
        return f"ConfigManager(path={self.config_path}, sections={list(self.config.keys())})"


# Global configuration instance
_config_instance: Optional[ConfigManager] = None

def get_config(config_path: Optional[str] = None) -> ConfigManager:
    """
    Get global configuration instance (singleton pattern)

    Args:
        config_path: Path to config file (only used on first call)

    Returns:
        ConfigManager instance
    """
    global _config_instance

    if _config_instance is None:
        _config_instance = ConfigManager(config_path)

    return _config_instance

def reload_config() -> ConfigManager:
    """
    Force reload of global configuration

    Returns:
        Reloaded ConfigManager instance
    """
    global _config_instance
    _config_instance = None
    return get_config()
