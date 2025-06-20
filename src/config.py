"""Configuration management for the Trader Risk Management System."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class Config:
    """Configuration container with attribute access."""

    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize configuration from dictionary."""
        for key, value in config_dict.items():
            if isinstance(value, dict):
                # Recursively convert nested dictionaries
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with dot notation support."""
        keys = key.split('.')
        value = self

        for k in keys:
            if hasattr(value, k):
                value = getattr(value, k)
            else:
                return default

        return value

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration back to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result


class ConfigurationManager:
    """Manages application configuration with environment override support."""

    def __init__(self, config_path: Optional[str] = None, env: Optional[str] = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to configuration file (defaults to config/config.yaml)
            env: Environment name (development, staging, production)
        """
        self.env = env or os.getenv('RISK_ENV', 'development')
        self.config_path = config_path or os.path.join(PROJECT_ROOT, 'config', 'config.yaml')
        self._config_dict: Dict[str, Any] = {}
        self._config: Optional[Config] = None

    def load(self) -> Config:
        """Load configuration from file and environment."""
        # Load base configuration
        self._load_yaml_config()

        # Load environment-specific overrides
        env_config_path = self.config_path.replace('.yaml', f'.{self.env}.yaml')
        if os.path.exists(env_config_path):
            self._load_yaml_config(env_config_path, override=True)

        # Load local overrides (not tracked in git)
        local_config_path = self.config_path.replace('.yaml', '.local.yaml')
        if os.path.exists(local_config_path):
            self._load_yaml_config(local_config_path, override=True)

        # Apply environment variable overrides
        self._apply_env_overrides()

        # Resolve paths relative to project root
        self._resolve_paths()

        # Create Config object
        self._config = Config(self._config_dict)

        return self._config

    def _load_yaml_config(self, path: Optional[str] = None, override: bool = False):
        """Load YAML configuration file."""
        config_path = path or self.config_path

        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)

            if override:
                self._merge_configs(self._config_dict, config_data)
            else:
                self._config_dict = config_data

        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration file: {e}")

    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]):
        """Recursively merge configuration dictionaries."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_configs(base[key], value)
            else:
                base[key] = value

    def _apply_env_overrides(self):
        """Apply environment variable overrides."""
        # Map of environment variables to config paths
        env_mappings = {
            'RISK_DB_PATH': 'data.database_path',
            'RISK_LOG_LEVEL': 'logging.level',
            'RISK_API_PORT': 'api.port',
            'RISK_API_KEY': 'api.api_key',
            'RISK_MODEL_PATH': 'deployment.model_storage.path',
        }

        for env_var, config_path in env_mappings.items():
            if env_var in os.environ:
                self._set_nested_value(config_path, os.environ[env_var])

    def _set_nested_value(self, path: str, value: Any):
        """Set a nested configuration value using dot notation."""
        keys = path.split('.')
        current = self._config_dict

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        # Convert string values to appropriate types
        if isinstance(value, str):
            if value.lower() in ('true', 'false'):
                value = value.lower() == 'true'
            elif value.isdigit():
                value = int(value)
            elif '.' in value and all(part.isdigit() for part in value.split('.', 1)):
                value = float(value)

        current[keys[-1]] = value

    def _resolve_paths(self):
        """Resolve relative paths to absolute paths."""
        path_keys = [
            'data.database_path',
            'paths.data_dir',
            'paths.models_dir',
            'paths.reports_dir',
            'paths.logs_dir',
            'paths.signals_dir',
            'paths.config_dir',
            'logging.handlers.file.path',
            'deployment.model_storage.path',
            'database.backup.path',
        ]

        for path_key in path_keys:
            value = self._get_nested_value(path_key)
            if value and not os.path.isabs(value):
                absolute_path = os.path.join(PROJECT_ROOT, value)
                self._set_nested_value(path_key, absolute_path)

    def _get_nested_value(self, path: str) -> Any:
        """Get a nested configuration value using dot notation."""
        keys = path.split('.')
        current = self._config_dict

        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None

        return current

    @property
    def config(self) -> Config:
        """Get the loaded configuration."""
        if self._config is None:
            self.load()
        return self._config

    def reload(self):
        """Reload configuration from files."""
        self._config_dict = {}
        self._config = None
        self.load()

    def save_local_override(self, overrides: Dict[str, Any]):
        """Save local configuration overrides."""
        local_config_path = self.config_path.replace('.yaml', '.local.yaml')

        # Load existing local config if it exists
        existing = {}
        if os.path.exists(local_config_path):
            with open(local_config_path, 'r') as f:
                existing = yaml.safe_load(f) or {}

        # Merge with new overrides
        self._merge_configs(existing, overrides)

        # Save to file
        with open(local_config_path, 'w') as f:
            yaml.dump(existing, f, default_flow_style=False, sort_keys=False)


# Global configuration instance
_config_manager = ConfigurationManager()


def get_config() -> Config:
    """Get the global configuration instance."""
    return _config_manager.config


def reload_config():
    """Reload the global configuration."""
    _config_manager.reload()


def set_config_override(overrides: Dict[str, Any], save_local: bool = False):
    """
    Set configuration overrides.

    Args:
        overrides: Dictionary of configuration overrides
        save_local: Whether to save overrides to local config file
    """
    _config_manager._merge_configs(_config_manager._config_dict, overrides)
    _config_manager._config = Config(_config_manager._config_dict)

    if save_local:
        _config_manager.save_local_override(overrides)


# Convenience function for testing
def load_test_config() -> Config:
    """Load test configuration."""
    test_config = ConfigurationManager(env='test')
    return test_config.load()
