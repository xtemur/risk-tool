import sys

sys.path.append("src")

import logging
from pathlib import Path

import yaml

from data_loader import DataLoader
from email_service import EmailService
from feature_engineering import FeatureEngineer
from predictor import RiskPredictor

# Import the download function
try:
    from download_totals import download_for_risk_tool
except ImportError:
    logger.warning("download_totals.py not found. Skipping data download.")
    download_for_risk_tool = None


def setup_logging():
    """Setup logging configuration"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("logs/daily_prediction.log"),
            logging.StreamHandler(),
        ],
    )


def main():
    """Main daily prediction pipeline with data download"""
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        logger.info("Starting daily risk prediction pipeline...")

        # Step 1: Download latest data
        if download_for_risk_tool:
            logger.info("Downloading latest data from PropreReports...")
            download_success = download_for_risk_tool()

            if not download_success:
                logger.error("Data download failed. Proceeding with existing data.")
            else:
                logger.info("Data download completed successfully")
        else:
            logger.info("Skipping data download - function not available")

        # Step 2: Load configuration
        with open("config/config.yaml", "r") as f:
            config = yaml.safe_load(f)

        # Step 3: Initialize components
        data_loader = DataLoader()
        feature_engineer = FeatureEngineer(config)
        predictor = RiskPredictor()
        email_service = EmailService(config)

        # Step 4: Load trader data
        logger.info("Loading trader data...")
        all_data = data_loader.load_all_traders_data()

        if not all_data:
            logger.error("No trader data loaded. Check data files and configuration.")
            return

        # Step 5: Engineer features for each trader
        logger.info("Engineering features...")
        for account_id in all_data.keys():
            if not all_data[account_id]["totals"].empty:
                features_df = feature_engineer.engineer_features(
                    all_data[account_id]["totals"]
                )
                all_data[account_id]["features"] = features_df

        # Step 6: Generate predictions
        logger.info("Generating risk predictions...")
        predictions = predictor.predict_all_traders(all_data)

        # Step 7: Send email report
        logger.info("Sending email report...")
        email_sent = email_service.send_email(predictions)

        if email_sent:
            logger.info("Daily risk prediction completed successfully")
        else:
            logger.error("Failed to send email report")

        # Step 8: Save predictions for record keeping
        import pandas as pd

        pred_df = pd.DataFrame(predictions)
        pred_df["prediction_date"] = pd.Timestamp.now().date()

        pred_dir = Path("data/predictions")
        pred_dir.mkdir(exist_ok=True)
        pred_df.to_csv(
            pred_dir / f"predictions_{pd.Timestamp.now().date()}.csv", index=False
        )

        logger.info(f"Predictions saved to data/predictions/")

        # Step 9: Log summary
        high_risk_count = len([p for p in predictions if p["risk_level"] == "High"])
        medium_risk_count = len([p for p in predictions if p["risk_level"] == "Medium"])

        logger.info(
            f"Risk Summary: {high_risk_count} High Risk, {medium_risk_count} Medium Risk traders"
        )

    except Exception as e:
        logger.error(f"Daily prediction pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
