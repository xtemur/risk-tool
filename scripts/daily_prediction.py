#!/usr/bin/env python
"""
Daily Prediction Script - Generate risk predictions and send email report
Run this daily before market open
"""

import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.database import Database
from src.data_downloader import DataDownloader
from src.predictor import RiskPredictor
from src.email_service import EmailService


def setup_logging():
    """Configure logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/daily_prediction.log')
        ]
    )


def main():
    """Main prediction function"""
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("Starting Daily Risk Prediction")

    # Step 1: Download recent data (last 7 days)
    logger.info("Downloading recent data...")
    downloader = DataDownloader()
    results = downloader.download_recent(days_back=7)

    if not any(results.values()):
        logger.error("Failed to download recent data")
        return

    # Step 2: Generate predictions
    logger.info("Generating predictions...")
    predictor = RiskPredictor()
    predictions = predictor.predict_all_traders()

    # Step 3: Get summary
    summary = predictor.get_risk_summary(predictions)

    # Log summary
    logger.info(f"\nRisk Summary:")
    logger.info(f"Total Traders: {summary['total_traders']}")
    logger.info(f"High Risk: {summary['high_risk_count']}")
    logger.info(f"Medium Risk: {summary['medium_risk_count']}")
    logger.info(f"Low Risk: {summary['low_risk_count']}")
    logger.info(f"Models Available: {summary['models_available']}")

    # Step 4: Send email report
    logger.info("\nSending email report...")
    email_service = EmailService()
    email_sent = email_service.send_daily_report(predictions, summary)

    if email_sent:
        logger.info("Email report sent successfully")
    else:
        logger.error("Failed to send email report")

    # Save predictions to file as backup
    import pandas as pd
    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_csv(f"data/predictions_{pd.Timestamp.now().date()}.csv", index=False)

    logger.info("\nDaily prediction complete!")

    # Show top risk traders
    if summary['top_risk_traders']:
        logger.info("\nTop Risk Traders:")
        for i, trader in enumerate(summary['top_risk_traders'], 1):
            logger.info(f"{i}. {trader['trader_name']}: {trader['risk_level']} "
                       f"(Predicted P&L: ${trader['predicted_pnl']:.2f})")
            logger.info(f"   Recommendation: {trader['recommendation']}")


if __name__ == "__main__":
    main()
