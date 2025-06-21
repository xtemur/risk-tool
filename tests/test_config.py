"""Tests for configuration management."""

import os
import tempfile
import unittest
from pathlib import Path
import yaml

from core.config import ConfigurationManager, Config, get_config, set_config_override


class TestConfigurationManager(unittest.TestCase):
    """Test configuration loading and management."""

    def setUp(self):
        """Create temporary config file for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, 'test_config.yaml')

        # Create test configuration
        self.test_config = {
            'app': {
                'name': 'Test App',
                'version': '1.0.0',
                'environment': 'test'
            },
            'data': {
                'database_path': 'test/data.db',
                'min_trading_days': 30
            },
            'models': {
                'xgboost': {
                    'n_estimators': 100,
                    'max_depth': 3
                }
            },
            'paths': {
                'data_dir': 'test/data',
                'models_dir': 'test/models'
            }
        }

        with open(self.config_path, 'w') as f:
            yaml.dump(self.test_config, f)

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_load_config(self):
        """Test basic configuration loading."""
        manager = ConfigurationManager(self.config_path)
        config = manager.load()

        self.assertIsInstance(config, Config)
        self.assertEqual(config.app.name, 'Test App')
        self.assertEqual(config.app.version, '1.0.0')
        self.assertEqual(config.data.min_trading_days, 30)

    def test_nested_config_access(self):
        """Test accessing nested configuration values."""
        manager = ConfigurationManager(self.config_path)
        config = manager.load()

        # Test dot notation access
        self.assertEqual(config.get('app.name'), 'Test App')
        self.assertEqual(config.get('models.xgboost.n_estimators'), 100)
        self.assertEqual(config.get('nonexistent.key', 'default'), 'default')

    def test_env_override(self):
        """Test environment variable overrides."""
        os.environ['RISK_DB_PATH'] = '/override/path.db'
        os.environ['RISK_LOG_LEVEL'] = 'DEBUG'

        manager = ConfigurationManager(self.config_path)
        config = manager.load()

        # Check overrides were applied
        self.assertTrue(config.data.database_path.endswith('/override/path.db'))

        # Clean up env vars
        del os.environ['RISK_DB_PATH']
        del os.environ['RISK_LOG_LEVEL']

    def test_config_merge(self):
        """Test configuration merging."""
        manager = ConfigurationManager(self.config_path)

        # Create override config
        override_config = {
            'app': {
                'version': '2.0.0'  # Override version
            },
            'new_section': {
                'new_key': 'new_value'
            }
        }

        manager._merge_configs(manager._config_dict, override_config)
        config = manager.load()

        # Check merge results
        self.assertEqual(config.app.version, '1.0.0')  # Original loaded first
        self.assertEqual(config.app.name, 'Test App')  # Preserved

    def test_path_resolution(self):
        """Test relative path resolution."""
        manager = ConfigurationManager(self.config_path)
        config = manager.load()

        # Check paths are made absolute
        self.assertTrue(os.path.isabs(config.paths.data_dir))
        self.assertTrue(os.path.isabs(config.paths.models_dir))

    def test_config_to_dict(self):
        """Test converting Config object back to dictionary."""
        manager = ConfigurationManager(self.config_path)
        config = manager.load()

        config_dict = config.to_dict()

        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict['app']['name'], 'Test App')
        self.assertIn('models', config_dict)


class TestConfigHelpers(unittest.TestCase):
    """Test configuration helper functions."""

    def test_set_config_override(self):
        """Test setting configuration overrides."""
        # Get initial config
        config = get_config()
        original_level = config.logging.level if hasattr(config, 'logging') else 'INFO'

        # Set override
        set_config_override({
            'logging': {
                'level': 'DEBUG'
            }
        })

        # Check override applied
        config = get_config()
        self.assertEqual(config.logging.level, 'DEBUG')

        # Reset
        set_config_override({
            'logging': {
                'level': original_level
            }
        })


if __name__ == '__main__':
    unittest.main()
