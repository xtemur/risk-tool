import sys

sys.path.append("src")

import logging
from pathlib import Path

import yaml

from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from model_trainer import ModelTrainer


def setup_logging():
    """Setup logging configuration"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("logs/weekly_retrain.log"),
            logging.StreamHandler(),
        ],
    )


def main():
    """Weekly model retraining pipeline"""
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        logger.info("Starting weekly model retraining...")

        # Load configuration
        with open("config/config.yaml", "r") as f:
            config = yaml.safe_load(f)

        # Initialize components
        data_loader = DataLoader()
        feature_engineer = FeatureEngineer(config)
        model_trainer = ModelTrainer()

        # Load all data
        logger.info("Loading all trader data...")
        all_data = data_loader.load_all_traders_data()

        # Create master dataset
        master_df = data_loader.create_master_dataset(all_data)

        # Engineer features
        logger.info("Engineering features...")
        features_df = feature_engineer.engineer_features(master_df)
        feature_cols = feature_engineer.get_feature_columns()

        # Create time-based splits
        train_df, val_df, test_df = model_trainer.create_time_splits(features_df)

        # Train models
        logger.info("Training global model...")
        global_model = model_trainer.train_global_model(train_df, val_df, feature_cols)

        logger.info("Training personal models...")
        personal_models = model_trainer.train_personal_models(
            train_df, val_df, feature_cols
        )

        # Evaluate models
        logger.info("Evaluating models...")
        results = model_trainer.evaluate_models(
            test_df, global_model, personal_models, feature_cols
        )

        # Save metadata
        model_trainer.save_model_metadata(feature_cols, results)

        logger.info("Weekly retraining completed successfully")
        logger.info(f"Global AUC: {results['global_auc']:.4f}")
        logger.info(f"Personal AUC: {results['personal_auc_mean']:.4f}")

    except Exception as e:
        logger.error(f"Weekly retraining failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
