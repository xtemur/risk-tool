import logging
import subprocess
import time
from datetime import datetime

import pytz
import schedule
import yaml


class RiskScheduler:
    def __init__(self):
        with open("config/config.yaml", "r") as f:
            self.config = yaml.safe_load(f)

        self.timezone = pytz.timezone(self.config["trading"]["timezone"])
        self.email_time = self.config["trading"]["email_time"]

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def run_daily_prediction(self):
        """Run daily prediction script"""
        try:
            self.logger.info("Running daily prediction...")
            result = subprocess.run(
                ["python", "scripts/daily_prediction.py"],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                self.logger.info("Daily prediction completed successfully")
            else:
                self.logger.error(f"Daily prediction failed: {result.stderr}")

        except Exception as e:
            self.logger.error(f"Error running daily prediction: {str(e)}")

    def run_weekly_retrain(self):
        """Run weekly retraining script"""
        try:
            self.logger.info("Running weekly retraining...")
            result = subprocess.run(
                ["python", "scripts/weekly_retrain.py"], capture_output=True, text=True
            )

            if result.returncode == 0:
                self.logger.info("Weekly retraining completed successfully")
            else:
                self.logger.error(f"Weekly retraining failed: {result.stderr}")

        except Exception as e:
            self.logger.error(f"Error running weekly retraining: {str(e)}")

    def start_scheduler(self):
        """Start the scheduler"""
        # Schedule daily predictions
        schedule.every().monday.at(self.email_time).do(self.run_daily_prediction)
        schedule.every().tuesday.at(self.email_time).do(self.run_daily_prediction)
        schedule.every().wednesday.at(self.email_time).do(self.run_daily_prediction)
        schedule.every().thursday.at(self.email_time).do(self.run_daily_prediction)
        schedule.every().friday.at(self.email_time).do(self.run_daily_prediction)

        # Schedule weekly retraining (Sundays at 6 AM)
        schedule.every().sunday.at("06:00").do(self.run_weekly_retrain)

        self.logger.info(
            f"Scheduler started. Daily predictions at {self.email_time} EST"
        )
        self.logger.info("Weekly retraining on Sundays at 6:00 AM")

        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute


if __name__ == "__main__":
    scheduler = RiskScheduler()
    scheduler.start_scheduler()
