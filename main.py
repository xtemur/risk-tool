"""
Main entry point for the Risk Management System
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append("src")


def setup_project():
    """Initial project setup"""
    # Create directory structure
    directories = [
        "data/raw",
        "data/processed",
        "data/features",
        "data/models",
        "data/predictions",
        "logs",
        "config",
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

    print("Project structure created successfully!")
    print("\nNext steps:")
    print("1. Add your trader accounts to config/trader_accounts.yaml")
    print("2. Set up email credentials in .env file")
    print("3. Place your CSV files in data/raw/")
    print("4. Run: python main.py --train to train initial models")
    print("5. Run: python main.py --predict to test predictions")
    print("6. Run: python main.py --schedule to start automated system")


def train_models():
    """Train initial models"""
    from scripts.weekly_retrain import main as retrain_main

    print("Training initial models...")
    retrain_main()
    print("Model training completed!")


def run_prediction():
    """Run single prediction"""
    from scripts.daily_prediction import main as predict_main

    print("Running daily prediction...")
    predict_main()
    print("Prediction completed!")


def start_scheduler():
    """Start the automated scheduler"""
    from src.scheduler import RiskScheduler

    print("Starting automated scheduler...")
    scheduler = RiskScheduler()
    scheduler.start_scheduler()


def main():
    parser = argparse.ArgumentParser(description="Risk Management System")
    parser.add_argument("--setup", action="store_true", help="Setup project structure")
    parser.add_argument("--train", action="store_true", help="Train models")
    parser.add_argument("--predict", action="store_true", help="Run prediction")
    parser.add_argument("--schedule", action="store_true", help="Start scheduler")

    args = parser.parse_args()

    if args.setup:
        setup_project()
    elif args.train:
        train_models()
    elif args.predict:
        run_prediction()
    elif args.schedule:
        start_scheduler()
    else:
        print("Please specify an action: --setup, --train, --predict, or --schedule")
        print("Use --help for more information")


if __name__ == "__main__":
    main()
