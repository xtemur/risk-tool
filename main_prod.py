"""
Risk Management System - Production Main Script
Handles all system operations with proper error handling and logging
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append("src")

# Setup logging
def setup_logging(level=logging.INFO):
    """Setup comprehensive logging"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(log_dir / "system.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)

    return root_logger


def setup_project():
    """Initial project setup"""
    logger = logging.getLogger(__name__)

    try:
        # Create directory structure
        directories = [
            "data/raw",
            "data/processed",
            "data/features",
            "data/models",
            "data/predictions",
            "logs",
            "config",
            "notebooks",
            "scripts/deployment"
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")

        # Create basic config files if they don't exist
        config_files = {
            "config/trader_accounts.yaml": """traders:
  # Add your trader accounts here
  # - account_id: "12345"
  #   name: "Trader Name"
  #   strategy: "Day Trading"
  #   active: true
""",
            ".env.template": """EMAIL_FROM=your-email@gmail.com
EMAIL_PASSWORD=your-gmail-app-password
API_TOKEN=your-propreports-token
""",
            ".gitignore": """.env
*.log
__pycache__/
*.pyc
.ipynb_checkpoints/
data/raw/*.csv
data/models/*.pkl
data/predictions/*.csv
"""
        }

        for file_path, content in config_files.items():
            if not Path(file_path).exists():
                with open(file_path, 'w') as f:
                    f.write(content)
                logger.info(f"Created config file: {file_path}")

        logger.info("Project structure created successfully!")
        print("\n🎉 Setup Complete!")
        print("\nNext steps:")
        print("1. Copy .env.template to .env and add your credentials")
        print("2. Configure trader accounts in config/trader_accounts.yaml")
        print("3. Run: python main.py --download to get data")
        print("4. Run: python main.py --train to train models")
        print("5. Run: python main.py --predict to test predictions")

        return True

    except Exception as e:
        logger.error(f"Setup failed: {str(e)}")
        return False


def download_data():
    """Download data from PropreReports"""
    logger = logging.getLogger(__name__)

    try:
        logger.info("Starting data download...")

        # Check if environment is configured
        from dotenv import load_dotenv
        load_dotenv()

        if not os.getenv('API_TOKEN'):
            logger.error("API_TOKEN not found in environment variables")
            print("❌ Please configure your .env file with API_TOKEN")
            return False

        from propreports_downloader import download_for_risk_tool
        success = download_for_risk_tool()

        if success:
            logger.info("Data download completed successfully")
            print("✅ Data download completed!")
        else:
            logger.error("Data download failed")
            print("❌ Data download failed - check logs")

        return success

    except Exception as e:
        logger.error(f"Data download failed: {str(e)}")
        print(f"❌ Data download error: {str(e)}")
        return False


def train_models():
    """Train ML models"""
    logger = logging.getLogger(__name__)

    try:
        logger.info("Starting model training...")

        # Import training components
        from data_loader import DataLoader
        from simple_feature_engineer import SimpleFeatureEngineer
        from model_trainer import ModelTrainer
        import yaml

        # Load configuration
        with open("config/config.yaml", "r") as f:
            config = yaml.safe_load(f)

        # Initialize components
        data_loader = DataLoader()
        feature_engineer = SimpleFeatureEngineer()
        model_trainer = ModelTrainer()

        # Load and process data
        logger.info("Loading trader data...")
        all_data = data_loader.load_all_traders_data()

        if not all_data:
            logger.error("No trader data loaded")
            print("❌ No trader data found - run: python main.py --download first")
            return False

        logger.info(f"Loaded data for {len(all_data)} traders")

        # Create master dataset
        master_totals, master_fills = data_loader.create_master_dataset(all_data)

        # Engineer features
        logger.info("Engineering features...")
        features_df = feature_engineer.engineer_features(master_totals, master_fills)
        feature_cols = feature_engineer.get_feature_columns()

        logger.info(f"Generated {len(feature_cols)} features for {len(features_df)} samples")

        # Create time-based splits
        train_df, val_df, test_df = model_trainer.create_time_splits(features_df)

        # Train models
        logger.info("Training global model...")
        global_model = model_trainer.train_global_model(train_df, val_df, feature_cols)

        logger.info("Training personal models...")
        personal_models = model_trainer.train_personal_models(train_df, val_df, feature_cols)

        logger.info("Training ARIMA baseline...")
        arima_models = model_trainer.train_arima_baseline(train_df, val_df)

        # Evaluate models
        logger.info("Evaluating models...")
        results = model_trainer.evaluate_models(
            test_df, global_model, personal_models, arima_models, feature_cols
        )

        # Save metadata
        model_trainer.save_model_metadata(feature_cols, results)

        logger.info("Model training completed successfully")
        print("✅ Model training completed!")
        print(f"📊 Global Model RMSE: {results['global']['rmse']:.4f}")
        print(f"📊 Personal Models RMSE: {results['personal']['rmse_mean']:.4f}")
        print(f"🤖 Trained {len(personal_models)} personal models")

        return True

    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        print(f"❌ Model training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def run_prediction():
    """Generate predictions and send email report"""
    logger = logging.getLogger(__name__)

    try:
        logger.info("Starting prediction pipeline...")

        # Import components
        from data_loader import DataLoader
        from simple_feature_engineer import SimpleFeatureEngineer
        from predictor import RiskPredictor
        from email_service import EmailService
        import yaml
        import pandas as pd

        # Load configuration
        with open("config/config.yaml", "r") as f:
            config = yaml.safe_load(f)

        # Initialize components
        data_loader = DataLoader()
        feature_engineer = SimpleFeatureEngineer()
        predictor = RiskPredictor()
        email_service = EmailService(config)

        # Load and process data
        logger.info("Loading trader data...")
        all_data = data_loader.load_all_traders_data()

        if not all_data:
            logger.error("No trader data loaded")
            print("❌ No trader data found")
            return False

        # Engineer features for each trader
        logger.info("Engineering features...")
        prediction_data = {}

        master_totals, master_fills = data_loader.create_master_dataset(all_data)
        features_df = feature_engineer.engineer_features(master_totals, master_fills)

        for account_id in all_data.keys():
            if account_id in features_df['account_id'].values:
                trader_features = features_df[features_df['account_id'] == account_id]
                prediction_data[account_id] = {
                    'features': trader_features,
                    'name': all_data[account_id]['name']
                }

        # Generate predictions
        logger.info("Generating predictions...")
        predictions = predictor.predict_all_traders(prediction_data)

        if not predictions:
            logger.error("No predictions generated")
            print("❌ No predictions generated")
            return False

        logger.info(f"Generated predictions for {len(predictions)} traders")

        # Save predictions
        pred_df = pd.DataFrame(predictions)
        pred_df["prediction_date"] = pd.Timestamp.now().date()

        pred_dir = Path("data/predictions")
        pred_dir.mkdir(exist_ok=True)
        pred_file = pred_dir / f"predictions_{pd.Timestamp.now().strftime('%Y-%m-%d')}.csv"
        pred_df.to_csv(pred_file, index=False)

        logger.info(f"Predictions saved to {pred_file}")

        # Send email report
        logger.info("Sending email report...")
        email_sent = email_service.send_email(predictions, dry_run=False)

        if email_sent:
            logger.info("Email report sent successfully")
            print("✅ Prediction pipeline completed!")
            print("📧 Email report sent")
        else:
            logger.warning("Email sending failed, but predictions completed")
            print("✅ Predictions completed")
            print("⚠️  Email sending failed - check email configuration")

        # Display summary
        risk_counts = {'High': 0, 'Medium': 0, 'Low': 0, 'Unknown': 0}
        for pred in predictions:
            risk_level = pred.get('risk_level', 'Unknown')
            risk_counts[risk_level] += 1

        print(f"📊 Risk Summary:")
        for level, count in risk_counts.items():
            if count > 0:
                print(f"   {level}: {count} traders")

        return True

    except Exception as e:
        logger.error(f"Prediction pipeline failed: {str(e)}")
        print(f"❌ Prediction failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_system():
    """Run comprehensive system test"""
    logger = logging.getLogger(__name__)

    try:
        logger.info("Starting system integration test...")
        print("🧪 Running system integration test...")

        # Test environment
        from dotenv import load_dotenv
        load_dotenv()

        required_vars = ['EMAIL_FROM', 'EMAIL_PASSWORD', 'API_TOKEN']
        env_status = {}

        for var in required_vars:
            value = os.getenv(var)
            env_status[var] = bool(value)
            status = "✅" if value else "❌"
            print(f"   {var}: {status}")

        if not all(env_status.values()):
            print("❌ Environment variables not properly configured")
            return False

        # Test imports
        try:
            from data_loader import DataLoader
            from simple_feature_engineer import SimpleFeatureEngineer
            from model_trainer import ModelTrainer
            from predictor import RiskPredictor
            from email_service import EmailService
            print("   Module Imports: ✅")
        except Exception as e:
            print(f"   Module Imports: ❌ ({e})")
            return False

        # Test file structure
        required_dirs = ['data/raw', 'data/models', 'config', 'logs']
        for directory in required_dirs:
            exists = Path(directory).exists()
            status = "✅" if exists else "❌"
            print(f"   {directory}: {status}")
            if not exists:
                return False

        # Test model files
        model_files = list(Path('data/models').glob('*.pkl'))
        models_ok = len(model_files) > 0
        status = "✅" if models_ok else "❌"
        print(f"   Model Files: {status} ({len(model_files)} found)")

        if not models_ok:
            print("   Run: python main.py --train to create models")
            return False

        # Test email configuration
        try:
            import yaml
            with open('config/config.yaml', 'r') as f:
                config = yaml.safe_load(f)
            email_service = EmailService(config)
            email_ok = email_service.test_email_config()
            status = "✅" if email_ok else "❌"
            print(f"   Email Service: {status}")
        except Exception as e:
            print(f"   Email Service: ❌ ({e})")
            email_ok = False

        overall_ok = all(env_status.values()) and models_ok and email_ok

        print(f"\n🎯 System Status: {'✅ READY' if overall_ok else '❌ NEEDS FIXES'}")

        if overall_ok:
            print("🚀 System ready for production!")
            print("   Next: python main.py --schedule")
        else:
            print("🔧 Fix the issues above before proceeding")

        return overall_ok

    except Exception as e:
        logger.error(f"System test failed: {str(e)}")
        print(f"❌ System test failed: {str(e)}")
        return False


def start_scheduler():
    """Start the automated scheduler"""
    logger = logging.getLogger(__name__)

    try:
        logger.info("Starting automated scheduler...")
        print("🔄 Starting automated scheduler...")
        print("   Daily predictions: 6:00 AM EST (weekdays)")
        print("   Weekly retraining: Sunday 6:00 AM EST")
        print("   Press Ctrl+C to stop")

        from scheduler import RiskScheduler
        scheduler = RiskScheduler()
        scheduler.start_scheduler()

    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user")
        print("\n⏹️  Scheduler stopped")
    except Exception as e:
        logger.error(f"Scheduler failed: {str(e)}")
        print(f"❌ Scheduler failed: {str(e)}")
        return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Risk Management System - Production Version",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --setup          # Initial project setup
  python main.py --download       # Download data from PropreReports
  python main.py --train          # Train ML models
  python main.py --predict        # Generate predictions and send email
  python main.py --test           # Run system integration test
  python main.py --schedule       # Start automated scheduler

For detailed documentation, see README.md
        """
    )

    parser.add_argument("--setup", action="store_true",
                       help="Setup project structure and config files")
    parser.add_argument("--download", action="store_true",
                       help="Download data from PropreReports API")
    parser.add_argument("--train", action="store_true",
                       help="Train ML models")
    parser.add_argument("--predict", action="store_true",
                       help="Generate predictions and send email report")
    parser.add_argument("--test", action="store_true",
                       help="Run comprehensive system test")
    parser.add_argument("--schedule", action="store_true",
                       help="Start automated scheduler")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)

    logger = logging.getLogger(__name__)
    logger.info(f"Risk Management System starting - {datetime.now()}")

    # Execute requested operation
    success = False

    if args.setup:
        success = setup_project()
    elif args.download:
        success = download_data()
    elif args.train:
        success = train_models()
    elif args.predict:
        success = run_prediction()
    elif args.test:
        success = test_system()
    elif args.schedule:
        success = start_scheduler()
    else:
        parser.print_help()
        print("\n💡 Start with: python main.py --setup")
        return

    # Exit with appropriate code
    if success:
        logger.info("Operation completed successfully")
        print("✅ Operation completed successfully")
    else:
        logger.error("Operation failed")
        print("❌ Operation failed - check logs for details")
        sys.exit(1)


if __name__ == "__main__":
    main()
