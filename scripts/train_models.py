#!/usr/bin/env python
"""
Train Models Script - Train personal models for all traders
Run this after setup to create models
"""

import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.database import Database
from src.feature_engineer import FeatureEngineer
from src.model_trainer import ModelTrainer


def setup_logging():
    """Configure logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/training.log')
        ]
    )


def main():
    """Main training function"""
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("Starting Model Training")

    # Initialize components
    db = Database()
    feature_engineer = FeatureEngineer()
    model_trainer = ModelTrainer()

    # Get all traders
    traders_df = db.get_all_traders()
    logger.info(f"Found {len(traders_df)} traders")

    # Process each trader
    all_features = {}

    for _, trader in traders_df.iterrows():
        account_id = trader['account_id']
        trader_name = trader['trader_name']

        logger.info(f"Processing {trader_name} ({account_id})...")

        # Get trader data
        totals_df, fills_df = db.get_trader_data(account_id)

        if totals_df.empty:
            logger.warning(f"No data found for {trader_name}")
            continue

        # Create features
        features_df = feature_engineer.create_features(totals_df, fills_df)

        if not features_df.empty:
            all_features[account_id] = features_df
            logger.info(f"Created {len(features_df)} feature rows for {trader_name}")

    # Get feature columns
    feature_columns = feature_engineer.get_feature_columns()
    logger.info(f"Using {len(feature_columns)} features for training")

    # Train models
    logger.info("\nTraining personal models...")
    results = model_trainer.train_all_models(all_features, feature_columns)

    # Summary
    logger.info(f"\nTraining Complete!")
    logger.info(f"Successfully trained {len(results)} models")

    # Show top models by performance
    if results:
        import pandas as pd
        results_df = pd.DataFrame(results).T
        results_df = results_df.sort_values('rmse')

        logger.info("\nTop 5 Models by RMSE:")
        for idx, (account_id, row) in enumerate(results_df.head().iterrows()):
            logger.info(f"{idx+1}. {account_id}: RMSE={row['rmse']:.2f}, MAE={row['mae']:.2f}")

        # Show feature importance from best model
        best_account = results_df.index[0]
        logger.info(f"\nTop features from best model ({best_account}):")
        for i, feature in enumerate(results[best_account]['top_features']):
            logger.info(f"  {i+1}. {feature}")

    logger.info("\nModels saved to data/models/")
    logger.info("Run 'python scripts/daily_predict.py' for predictions")


if __name__ == "__main__":
    main()
