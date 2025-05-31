#!/usr/bin/env python
"""
Setup Script - Initialize database and download all historical data
Run this first to set up the system
"""

import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.database import Database
from src.data_downloader import DataDownloader


def setup_logging():
    """Configure logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/setup.log')
        ]
    )


def main():
    """Main setup function"""
    # Create directories
    directories = ['data', 'data/models', 'logs', 'config']
    for dir_path in directories:
        Path(dir_path).mkdir(exist_ok=True)

    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("Starting Risk Management MVP Setup")

    # Initialize database
    logger.info("Initializing database...")
    db = Database()

    # Download all historical data
    logger.info("Downloading historical data...")
    downloader = DataDownloader()

    # Download last 365 days of data
    results = downloader.download_all_data(days_back=1000)

    # Show summary
    success_count = sum(results.values())
    total_count = len(results)

    logger.info(f"\nSetup Complete!")
    logger.info(f"Downloaded data for {success_count}/{total_count} traders")

    # Show database stats
    stats = db.get_database_stats()
    logger.info("\nDatabase Statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")

    if success_count < total_count:
        logger.warning("\nSome traders failed to download. Check logs for details.")

    logger.info("\nNext steps:")
    logger.info("1. Run 'python scripts/train_models.py' to train models")
    logger.info("2. Run 'python scripts/daily_predict.py' for daily predictions")


if __name__ == "__main__":
    main()
