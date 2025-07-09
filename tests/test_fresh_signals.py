#!/usr/bin/env python3
"""
Test script to verify signal generation with fresh data from database.
This script generates signals but does NOT send emails.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference.signal_generator import SignalGenerator
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_fresh_signal_generation():
    """Test signal generation with fresh data from database."""

    logger.info("=" * 80)
    logger.info("TESTING SIGNAL GENERATION WITH FRESH DATA")
    logger.info("=" * 80)

    try:
        # Initialize signal generator
        logger.info("Initializing signal generator...")
        generator = SignalGenerator('configs/main_config.yaml')

        # Generate signals (this will fetch fresh data from database)
        logger.info("\nGenerating daily signals with fresh data...")
        signal_data = generator.generate_daily_signals()

        # Display results
        logger.info("\n" + "=" * 60)
        logger.info("SIGNAL GENERATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"Date: {signal_data['date']}")
        logger.info(f"Total Traders Processed: {len(signal_data['trader_signals'])}")

        # Count risk levels
        risk_counts = {'high': 0, 'medium': 0, 'low': 0}
        for signal in signal_data['trader_signals']:
            risk_counts[signal['risk_level']] += 1

        logger.info(f"\nRisk Distribution:")
        logger.info(f"  High Risk: {risk_counts['high']}")
        logger.info(f"  Medium Risk: {risk_counts['medium']}")
        logger.info(f"  Low Risk: {risk_counts['low']}")

        # Show alerts
        logger.info(f"\nCritical Alerts: {len(signal_data['alerts'])}")
        if signal_data['alerts']:
            for alert in signal_data['alerts']:
                logger.info(f"  - {alert['trader_label']}: {alert['message']}")

        # Show sample trader details
        logger.info("\nSample Trader Signals (first 3):")
        for i, signal in enumerate(signal_data['trader_signals'][:3]):
            logger.info(f"\n  Trader {signal['trader_id']} ({signal['trader_name']}):")
            logger.info(f"    Risk Level: {signal['risk_level']}")
            logger.info(f"    VaR Prediction: ${signal['var_5pct']:,.2f}")
            logger.info(f"    Loss Probability: {signal['loss_probability']:.2%}")
            logger.info(f"    Last Trade Date: {signal['last_trade_date']}")
            logger.info(f"    Model Confidence: {signal['model_confidence']:.2%}")

            # Check if we have fresh data
            if 'last_update' in signal:
                logger.info(f"    Data Last Updated: {signal['last_update']}")

        # Summary statistics
        if signal_data['summary_stats']:
            logger.info(f"\nSummary Statistics:")
            logger.info(f"  Average VaR: ${signal_data['summary_stats']['avg_var']:,.2f}")
            logger.info(f"  Max VaR: ${signal_data['summary_stats']['max_var']:,.2f}")
            logger.info(f"  Average Loss Probability: {signal_data['summary_stats']['avg_loss_prob']:.2%}")
            logger.info(f"  Max Loss Probability: {signal_data['summary_stats']['max_loss_prob']:.2%}")

        logger.info("\n" + "=" * 60)
        logger.info("✅ Signal generation with fresh data completed successfully!")
        logger.info("📧 No emails were sent (test mode)")
        logger.info("=" * 60)

        return True

    except Exception as e:
        logger.error(f"❌ Error during signal generation: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_fresh_signal_generation()
    sys.exit(0 if success else 1)
