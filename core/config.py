"""
Production Configuration Management
Unified configuration system loading from YAML with environment overrides
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any

def get_config() -> Dict[str, Any]:
    """
    Load configuration from YAML file with environment variable overrides

    Returns:
        dict: Complete configuration dictionary
    """

    # Get paths
    project_root = Path(__file__).parent.parent
    config_file = project_root / 'config' / 'config.yaml'

    # Load base configuration from YAML
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Fallback configuration if YAML file doesn't exist
        config = _get_fallback_config(project_root)

    # Apply environment-specific overrides
    _apply_environment_overrides(config)

    # Convert to flat structure for backward compatibility
    flat_config = _flatten_config(config, project_root)

    return flat_config

def _get_fallback_config(project_root: Path) -> Dict[str, Any]:
    """Fallback configuration if YAML file is missing"""
    return {
        'app': {
            'name': 'Trader Risk Management System',
            'version': '1.0.0',
            'environment': 'production'
        },
        'data': {
            'database_path': 'data/risk_tool.db',
            'min_trading_days': 60,
            'test_cutoff_date': '2025-04-01'
        },
        'models': {
            'xgboost': {
                'n_estimators': 200,
                'max_depth': 4,
                'learning_rate': 0.1,
                'objective': 'multi:softprob',
                'random_state': 42,
                'n_jobs': -1
            }
        },
        'paths': {
            'data_dir': 'data',
            'models_dir': 'outputs/models',
            'reports_dir': 'outputs/reports',
            'logs_dir': 'outputs/logs',
            'signals_dir': 'outputs/signals'
        },
        'logging': {
            'level': 'INFO',
            'format': 'text',
            'handlers': {
                'console': {'enabled': True},
                'file': {
                    'enabled': True,
                    'path': 'outputs/pipeline.log',
                    'max_bytes': 10485760,
                    'backup_count': 5
                }
            }
        }
    }

def _apply_environment_overrides(config: Dict[str, Any]) -> None:
    """Apply environment variable overrides"""

    # Environment-based overrides
    env = os.getenv('ENVIRONMENT', 'development')
    if env == 'production':
        config['app']['environment'] = 'production'
        config['logging']['level'] = 'WARNING'
    elif env == 'development':
        config['app']['environment'] = 'development'
        config['logging']['level'] = 'DEBUG'

    # Database path override
    if os.getenv('DATABASE_PATH'):
        config['data']['database_path'] = os.getenv('DATABASE_PATH')

    # API key override
    if os.getenv('LOGGING_API_KEY'):
        config['logging']['handlers']['external']['api_key'] = os.getenv('LOGGING_API_KEY')

def _flatten_config(config: Dict[str, Any], project_root: Path) -> Dict[str, str]:
    """
    Flatten nested config to flat dictionary for backward compatibility

    Args:
        config: Nested configuration dictionary
        project_root: Project root path for resolving relative paths

    Returns:
        dict: Flattened configuration
    """

    data_config = config.get('data', {})
    model_config = config.get('models', {}).get('xgboost', {})
    paths_config = config.get('paths', {})
    logging_config = config.get('logging', {})
    backtesting_config = config.get('backtesting', {})
    signals_config = config.get('signals', {})

    # Convert relative paths to absolute
    def resolve_path(path_str: str) -> str:
        if path_str.startswith('/'):
            return path_str
        return str(project_root / path_str)

    return {
        # Data paths
        'db_path': resolve_path(data_config.get('database_path', 'data/risk_tool.db')),
        'output_dir': resolve_path(paths_config.get('signals_dir', 'outputs')),
        'checkpoint_dir': resolve_path('outputs/checkpoints'),

        # Model settings
        'min_trading_days': data_config.get('min_trading_days', 60),
        'test_start_date': data_config.get('test_cutoff_date', '2025-04-01'),

        # Feature engineering
        'ewma_spans': config.get('features', {}).get('ewma_spans', [5, 20]),
        'lag_days': config.get('features', {}).get('lag_days', [1, 2, 3]),

        # Model parameters
        'xgboost_params': {
            'objective': model_config.get('objective', 'multi:softprob'),
            'num_class': 3,
            'n_estimators': model_config.get('n_estimators', 200),
            'max_depth': model_config.get('max_depth', 4),
            'learning_rate': model_config.get('learning_rate', 0.1),
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': model_config.get('random_state', 42),
            'n_jobs': model_config.get('n_jobs', -1)
        },

        # Validation settings
        'walk_forward_test_size': backtesting_config.get('test_size', 30) / 100.0,  # Convert days to percentage
        'min_train_size': backtesting_config.get('min_train_size', 50),

        # Risk thresholds
        'high_risk_threshold': signals_config.get('confidence_threshold', 0.6),
        'low_risk_threshold': 0.3,

        # Strategy settings
        'position_reduction_factor': config.get('strategies', {}).get('position_sizing', {}).get('high_risk_reduction', 0.5),
        'max_position_increase': config.get('strategies', {}).get('position_sizing', {}).get('low_risk_increase', 1.2),

        # Logging
        'logging': logging_config,

        # Runtime
        'timestamp': config.get('app', {}).get('version', '1.0.0'),

        # Additional paths for full pipeline
        'models_dir': resolve_path(paths_config.get('models_dir', 'outputs/models')),
        'reports_dir': resolve_path(paths_config.get('reports_dir', 'outputs/reports')),
        'logs_dir': resolve_path(paths_config.get('logs_dir', 'outputs/logs'))
    }

def get_yaml_config() -> Dict[str, Any]:
    """
    Get the full YAML configuration for advanced use cases

    Returns:
        dict: Complete nested configuration
    """
    project_root = Path(__file__).parent.parent
    config_file = project_root / 'config' / 'config.yaml'

    if config_file.exists():
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        _apply_environment_overrides(config)
        return config
    else:
        return _get_fallback_config(project_root)

def save_config_updates(updates: Dict[str, Any]) -> None:
    """
    Save configuration updates back to YAML file

    Args:
        updates: Dictionary of configuration updates to apply
    """
    project_root = Path(__file__).parent.parent
    config_file = project_root / 'config' / 'config.yaml'

    # Load current config
    current_config = get_yaml_config()

    # Apply updates
    def deep_update(d: Dict, u: Dict) -> Dict:
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = deep_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    updated_config = deep_update(current_config, updates)

    # Save back to file
    with open(config_file, 'w') as f:
        yaml.dump(updated_config, f, default_flow_style=False, indent=2)

# Maintain backward compatibility
def get_config_legacy():
    """Legacy function name for backward compatibility"""
    return get_config()
