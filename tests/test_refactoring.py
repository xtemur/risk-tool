"""
Comprehensive test suite for the refactored risk tool architecture.

This test suite validates that all refactored components are working correctly
and following SOLID principles.
"""

import unittest
import tempfile
import os
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.domain import (
    Trader, TradingMetrics, TraderProfile,
    RiskAssessment, RiskLevel, RiskAlert,
    Prediction, PredictionResult, PredictionStatus
)
from src.exceptions import (
    RiskToolException, TraderNotFoundError, ModelNotFoundError,
    InsufficientDataError, DataValidationError
)
from src.config.config_manager import ConfigManager
from src.repositories import TraderRepository, ModelRepository, FillsRepository
from src.container import ServiceContainer, get_container, reset_container
from src.constants import HIGH_RISK_SHARPE_THRESHOLD, HIGH_RISK_WIN_RATE_THRESHOLD


class TestDomainModels(unittest.TestCase):
    """Test domain models and their business logic."""

    def test_trader_model(self):
        """Test Trader domain model."""
        trader = Trader(id=123, name="Test Trader")
        self.assertEqual(trader.display_name, "Test Trader")
        self.assertTrue(trader.is_active())

    def test_trading_metrics_risk_calculation(self):
        """Test risk score calculation in TradingMetrics."""
        # High risk metrics
        high_risk = TradingMetrics(
            bat_30d=25,  # Low win rate
            wl_ratio=0.5,  # Poor win/loss ratio
            sharpe=-0.2,  # Negative Sharpe
            total_trades=50
        )
        self.assertTrue(high_risk.is_high_risk())
        self.assertGreater(high_risk.get_risk_score(), 70)

        # Low risk metrics
        low_risk = TradingMetrics(
            bat_30d=75,  # High win rate
            wl_ratio=2.0,  # Good win/loss ratio
            sharpe=1.5,  # Good Sharpe
            total_trades=100
        )
        self.assertFalse(low_risk.is_high_risk())
        self.assertLess(low_risk.get_risk_score(), 30)

    def test_risk_level_enum(self):
        """Test RiskLevel enum functionality."""
        # Test score-based classification
        self.assertEqual(RiskLevel.from_score(10), RiskLevel.LOW)
        self.assertEqual(RiskLevel.from_score(40), RiskLevel.MEDIUM)
        self.assertEqual(RiskLevel.from_score(65), RiskLevel.HIGH)
        self.assertEqual(RiskLevel.from_score(85), RiskLevel.CRITICAL)

        # Test color mapping
        self.assertEqual(RiskLevel.LOW.get_color(), "green")
        self.assertEqual(RiskLevel.CRITICAL.get_color(), "red")

    def test_risk_assessment(self):
        """Test RiskAssessment model."""
        assessment = RiskAssessment(
            trader_id=123,
            risk_level=RiskLevel.HIGH,
            risk_score=75.5,
            var_95=1000.0,
            expected_shortfall=1200.0,
            max_probable_loss=1100.0,
            assessment_date=datetime.now(),
            confidence=0.85
        )

        self.assertTrue(assessment.is_actionable())
        self.assertIn("monitoring required", assessment.get_recommendation().lower())

        # Test serialization
        dict_repr = assessment.to_dict()
        self.assertEqual(dict_repr['trader_id'], 123)
        self.assertEqual(dict_repr['risk_level'], 'high')

    def test_prediction_result(self):
        """Test PredictionResult model."""
        prediction = PredictionResult(
            trader_id=123,
            prediction_date=datetime.now(),
            var_prediction=1000.0,
            loss_probability=0.75,
            expected_return=-50.0,
            confidence_interval=(-100.0, 50.0)
        )

        self.assertEqual(prediction.get_risk_signal(), "HIGH_RISK")
        self.assertTrue(prediction.should_alert(threshold=0.6))
        self.assertIn("Loss Prob: 75.0%", prediction.format_summary())


class TestExceptions(unittest.TestCase):
    """Test custom exception system."""

    def test_trader_not_found_exception(self):
        """Test TraderNotFoundError."""
        with self.assertRaises(TraderNotFoundError) as context:
            raise TraderNotFoundError(999)

        exception = context.exception
        self.assertEqual(exception.details['trader_id'], 999)
        self.assertIn("999", exception.message)

    def test_model_not_found_exception(self):
        """Test ModelNotFoundError."""
        with self.assertRaises(ModelNotFoundError) as context:
            raise ModelNotFoundError(888, "/path/to/model.pkl")

        exception = context.exception
        self.assertEqual(exception.details['trader_id'], 888)
        self.assertEqual(exception.details['model_path'], "/path/to/model.pkl")

    def test_insufficient_data_error(self):
        """Test InsufficientDataError."""
        with self.assertRaises(InsufficientDataError) as context:
            raise InsufficientDataError(trader_id=777, required=100, actual=50)

        exception = context.exception
        self.assertEqual(exception.details['required_count'], 100)
        self.assertEqual(exception.details['actual_count'], 50)

    def test_data_validation_error(self):
        """Test DataValidationError."""
        with self.assertRaises(DataValidationError) as context:
            raise DataValidationError(
                field='sharpe_ratio',
                reason='Value out of range',
                value=-10.5
            )

        exception = context.exception
        self.assertEqual(exception.details['field'], 'sharpe_ratio')
        self.assertEqual(exception.details['value'], -10.5)


class TestConfiguration(unittest.TestCase):
    """Test configuration management system."""

    def test_config_manager_creation(self):
        """Test ConfigManager initialization."""
        config = ConfigManager(env='test')
        self.assertIsNotNone(config)

        # Test configuration properties
        self.assertIsNotNone(config.database)
        self.assertIsNotNone(config.model)
        self.assertIsNotNone(config.risk)
        self.assertIsInstance(config.active_traders, list)

    def test_config_validation(self):
        """Test configuration validation."""
        config = ConfigManager(env='test')

        # Should validate successfully with defaults
        self.assertTrue(config.validate())

        # Test specific configurations
        self.assertGreater(config.risk.confidence_level, 0)
        self.assertLessEqual(config.risk.confidence_level, 1)
        self.assertGreater(config.model.cv_folds, 1)

    def test_config_get_method(self):
        """Test configuration get method."""
        config = ConfigManager(env='test')

        # Test nested key access
        db_path = config.get('database.path')
        self.assertIsNotNone(db_path)

        # Test with default
        missing_value = config.get('non.existent.key', 'default')
        self.assertEqual(missing_value, 'default')

    def test_environment_override(self):
        """Test environment variable override."""
        # Set environment variable
        test_path = '/tmp/test_db.db'
        os.environ['RISK_TOOL_DB_PATH'] = test_path

        try:
            config = ConfigManager(env='test')
            self.assertEqual(config.database.path, test_path)
        finally:
            # Clean up
            del os.environ['RISK_TOOL_DB_PATH']


class TestRepositories(unittest.TestCase):
    """Test repository pattern implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.db_path = self.temp_db.name
        self.temp_db.close()

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)

    def test_trader_repository_interface(self):
        """Test TraderRepository implements Repository interface."""
        repo = TraderRepository(self.db_path)

        # Test interface methods exist
        self.assertTrue(hasattr(repo, 'find_by_id'))
        self.assertTrue(hasattr(repo, 'find_all'))
        self.assertTrue(hasattr(repo, 'save'))
        self.assertTrue(hasattr(repo, 'delete'))
        self.assertTrue(hasattr(repo, 'exists'))

        # Test additional trader-specific methods
        self.assertTrue(hasattr(repo, 'get_trader_metrics'))
        self.assertTrue(hasattr(repo, 'get_active_traders'))

    def test_model_repository_interface(self):
        """Test ModelRepository implements Repository interface."""
        repo = ModelRepository()

        # Test interface methods
        self.assertTrue(hasattr(repo, 'find_by_id'))
        self.assertTrue(hasattr(repo, 'find_all'))
        self.assertTrue(hasattr(repo, 'save'))
        self.assertTrue(hasattr(repo, 'delete'))
        self.assertTrue(hasattr(repo, 'exists'))

        # Test model-specific methods
        self.assertTrue(hasattr(repo, 'load_model'))
        self.assertTrue(hasattr(repo, 'get_model_info'))
        self.assertTrue(hasattr(repo, 'clear_cache'))

    def test_fills_repository_interface(self):
        """Test FillsRepository implements Repository interface."""
        repo = FillsRepository(self.db_path)

        # Test interface methods
        self.assertTrue(hasattr(repo, 'find_by_id'))
        self.assertTrue(hasattr(repo, 'find_all'))
        self.assertTrue(hasattr(repo, 'save'))
        self.assertTrue(hasattr(repo, 'delete'))
        self.assertTrue(hasattr(repo, 'exists'))

        # Test fills-specific methods
        self.assertTrue(hasattr(repo, 'get_fills_by_trader'))
        self.assertTrue(hasattr(repo, 'get_daily_summary'))
        self.assertTrue(hasattr(repo, 'get_performance_metrics'))


class TestServiceContainer(unittest.TestCase):
    """Test dependency injection container."""

    def setUp(self):
        """Reset container before each test."""
        reset_container()

    def test_container_creation(self):
        """Test ServiceContainer creation."""
        container = ServiceContainer(env='test')
        self.assertIsNotNone(container)
        self.assertIsNotNone(container.config)

    def test_service_resolution(self):
        """Test service resolution from container."""
        container = ServiceContainer(env='test')

        # Test repository resolution
        trader_repo = container.trader_repository
        self.assertIsNotNone(trader_repo)
        self.assertEqual(type(trader_repo).__name__, 'TraderRepository')

        # Test service resolution
        risk_service = container.risk_service
        self.assertIsNotNone(risk_service)
        self.assertEqual(type(risk_service).__name__, 'RiskService')

    def test_singleton_pattern(self):
        """Test that services are singletons."""
        container = ServiceContainer(env='test')

        # Get same service twice
        config1 = container.config
        config2 = container.config
        self.assertIs(config1, config2)

        # Test with repositories
        repo1 = container.trader_repository
        repo2 = container.trader_repository
        self.assertIs(repo1, repo2)

    def test_global_container(self):
        """Test global container functionality."""
        container1 = get_container(env='test')
        container2 = get_container(env='test')

        # Should be same instance
        self.assertIs(container1, container2)

        # Test reset
        reset_container()
        container3 = get_container(env='test')
        self.assertIsNot(container1, container3)

    def test_service_by_name(self):
        """Test getting services by name."""
        container = ServiceContainer(env='test')

        services_to_test = [
            'trader_repository',
            'model_repository',
            'fills_repository',
            'risk_service',
            'trader_service',
            'prediction_service',
            'metrics_service',
            'config'
        ]

        for service_name in services_to_test:
            service = container.get_service(service_name)
            self.assertIsNotNone(service)

        # Test invalid service name
        with self.assertRaises(ValueError):
            container.get_service('non_existent_service')


class TestIntegration(unittest.TestCase):
    """Integration tests for the refactored architecture."""

    def test_end_to_end_flow(self):
        """Test a complete flow through the system."""
        # Create container
        container = ServiceContainer(env='test')

        # Get services
        config = container.config

        # Verify configuration
        self.assertIsNotNone(config.database.path)
        self.assertGreater(len(config.active_traders), 0)

        # Create domain models
        trader = Trader(id=9999, name="Integration Test Trader")
        metrics = TradingMetrics(
            bat_30d=45.5,
            wl_ratio=1.2,
            sharpe=0.8,
            total_trades=100,
            total_pnl=500.0
        )

        # Test risk assessment logic
        risk_score = metrics.get_risk_score()
        risk_level = RiskLevel.from_score(risk_score)

        assessment = RiskAssessment(
            trader_id=trader.id,
            risk_level=risk_level,
            risk_score=risk_score,
            var_95=100.0,
            expected_shortfall=120.0,
            max_probable_loss=110.0,
            assessment_date=datetime.now(),
            confidence=0.9
        )

        # Verify the flow
        self.assertFalse(metrics.is_high_risk())
        self.assertEqual(assessment.trader_id, 9999)
        self.assertIsNotNone(assessment.get_recommendation())

        # Test serialization
        assessment_dict = assessment.to_dict()
        self.assertEqual(assessment_dict['trader_id'], 9999)
        self.assertIn('risk_level', assessment_dict)
        self.assertIn('recommendation', assessment_dict)


def run_all_tests():
    """Run all test suites."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestDomainModels))
    suite.addTests(loader.loadTestsFromTestCase(TestExceptions))
    suite.addTests(loader.loadTestsFromTestCase(TestConfiguration))
    suite.addTests(loader.loadTestsFromTestCase(TestRepositories))
    suite.addTests(loader.loadTestsFromTestCase(TestServiceContainer))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_all_tests()
    if success:
        print("\n🎉 All refactoring tests passed successfully!")
        print("✅ The refactored architecture is solid and working correctly!")
    else:
        print("\n❌ Some tests failed. Please review the output above.")
        sys.exit(1)
