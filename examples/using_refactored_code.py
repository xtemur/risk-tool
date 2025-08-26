"""
Example demonstrating how to use the refactored risk tool architecture.

This script shows best practices for working with the new clean architecture.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime
from src.container import get_container
from src.models.domain import RiskLevel
from src.exceptions import TraderNotFoundError, ModelNotFoundError


def example_1_basic_usage():
    """Example 1: Basic usage with dependency injection."""
    print("\n=== Example 1: Basic Usage ===")

    # Get the global container (singleton)
    container = get_container(env='production')

    # Access configuration
    config = container.config
    print(f"Database path: {config.database.path}")
    print(f"Active traders: {config.active_traders[:3]}...")
    print(f"Risk thresholds - Sharpe: {config.risk.high_risk_sharpe}")


def example_2_working_with_repositories():
    """Example 2: Using repositories for data access."""
    print("\n=== Example 2: Repository Pattern ===")

    container = get_container()

    # Get repository from container
    trader_repo = container.trader_repository

    # Example: Check if a trader exists
    trader_id = 3942
    if trader_repo.exists(trader_id):
        print(f"Trader {trader_id} exists in database")

        # Get trader metrics
        metrics = trader_repo.get_trader_metrics(trader_id, lookback_days=30)
        if metrics:
            print(f"Trader {trader_id} metrics:")
            print(f"  - Win rate: {metrics.bat_30d:.1f}%")
            print(f"  - Sharpe ratio: {metrics.sharpe:.2f}")
            print(f"  - Risk score: {metrics.get_risk_score():.1f}")
            print(f"  - High risk? {metrics.is_high_risk()}")
    else:
        print(f"Trader {trader_id} not found")


def example_3_using_services():
    """Example 3: Using service layer for business logic."""
    print("\n=== Example 3: Service Layer ===")

    container = get_container()

    # Get services from container
    trader_service = container.trader_service

    # Get active traders
    active_traders = trader_service.get_active_traders()
    print(f"Found {len(active_traders)} active traders")

    # Get trader performance (with error handling)
    if active_traders:
        trader = active_traders[0]
        try:
            performance = trader_service.get_trader_performance(
                trader.id,
                days=30
            )
            print(f"\nTrader {trader.id} performance:")
            print(f"  - Total PnL: ${performance['total_pnl']:,.2f}")
            print(f"  - Win rate: {performance['win_rate']:.1f}%")
            print(f"  - Sharpe ratio: {performance['sharpe_ratio']:.2f}")
        except TraderNotFoundError as e:
            print(f"Error: {e.message}")


def example_4_risk_assessment():
    """Example 4: Risk assessment with proper error handling."""
    print("\n=== Example 4: Risk Assessment ===")

    container = get_container()
    risk_service = container.risk_service

    # Assess risk for specific trader
    trader_id = 3942

    try:
        assessment = risk_service.assess_trader_risk(trader_id)

        print(f"Risk Assessment for Trader {trader_id}:")
        print(f"  - Risk Level: {assessment.risk_level.value}")
        print(f"  - Risk Score: {assessment.risk_score:.1f}")
        print(f"  - VaR (95%): ${assessment.var_95:,.2f}")
        print(f"  - Expected Shortfall: ${assessment.expected_shortfall:,.2f}")
        print(f"  - Recommendation: {assessment.get_recommendation()}")
        print(f"  - Action required? {assessment.is_actionable()}")

        # Check if alert needed
        if assessment.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            print(f"  ⚠️ ALERT: High risk detected for trader {trader_id}")

    except TraderNotFoundError as e:
        print(f"Cannot assess risk: {e.message}")
    except ModelNotFoundError as e:
        print(f"Model not available: {e.message}")
    except Exception as e:
        print(f"Risk assessment failed: {str(e)}")


def example_5_generating_predictions():
    """Example 5: Generating predictions with the prediction service."""
    print("\n=== Example 5: Predictions ===")

    container = get_container()
    prediction_service = container.prediction_service

    # Get high risk predictions
    try:
        high_risk_predictions = prediction_service.get_high_risk_predictions(
            threshold=0.6
        )

        print(f"Found {len(high_risk_predictions)} high-risk predictions")

        for pred in high_risk_predictions[:3]:
            print(f"\nTrader {pred.trader_id}:")
            print(f"  - Loss probability: {pred.loss_probability:.1%}")
            print(f"  - VaR prediction: ${pred.var_prediction:,.2f}")
            print(f"  - Risk signal: {pred.get_risk_signal()}")

    except Exception as e:
        print(f"Prediction generation failed: {str(e)}")


def example_6_custom_exception_handling():
    """Example 6: Working with custom exceptions."""
    print("\n=== Example 6: Exception Handling ===")

    from src.exceptions import (
        InsufficientDataError,
        DataValidationError,
        handle_exception
    )

    # Example of handling different exception types
    try:
        # Simulate insufficient data error
        raise InsufficientDataError(
            trader_id=999,
            required=100,
            actual=50
        )
    except InsufficientDataError as e:
        print(f"Handled InsufficientDataError:")
        print(f"  Message: {e.message}")
        print(f"  Details: {e.details}")

    # Example of data validation error
    try:
        raise DataValidationError(
            field='sharpe_ratio',
            reason='Value out of expected range',
            value=-100.5
        )
    except DataValidationError as e:
        print(f"\nHandled DataValidationError:")
        print(f"  Field: {e.details['field']}")
        print(f"  Reason: {e.details['reason']}")
        print(f"  Invalid value: {e.details['value']}")


def example_7_configuration_management():
    """Example 7: Advanced configuration management."""
    print("\n=== Example 7: Configuration Management ===")

    from src.config.config_manager import ConfigManager
    import os

    # Create custom configuration
    config = ConfigManager(env='test')

    # Access nested configuration
    db_path = config.get('database.path')
    print(f"Database path: {db_path}")

    # Access with default value
    custom_setting = config.get('custom.setting', 'default_value')
    print(f"Custom setting: {custom_setting}")

    # Override with environment variable
    os.environ['RISK_TOOL_DB_PATH'] = '/tmp/test.db'
    config.reload()
    print(f"After env override: {config.database.path}")

    # Clean up
    del os.environ['RISK_TOOL_DB_PATH']


def main():
    """Run all examples."""
    print("=" * 60)
    print("REFACTORED RISK TOOL - USAGE EXAMPLES")
    print("=" * 60)

    examples = [
        example_1_basic_usage,
        example_2_working_with_repositories,
        example_3_using_services,
        example_4_risk_assessment,
        example_5_generating_predictions,
        example_6_custom_exception_handling,
        example_7_configuration_management
    ]

    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\n❌ Example failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("✅ All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
