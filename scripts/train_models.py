# scripts/train_models.py
#!/usr/bin/env python3
"""
Main training pipeline for risk prediction models
Usage: python scripts/train_models.py --config config/config.yaml
"""

import argparse
import logging

from src.data.data_loader import DataLoader
from src.data.feature_engineer import FeatureEngineer
from src.models.ensemble import EnsembleRiskModel
from src.validation.time_series_cv import TimeSeriesCV


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Config file path")
    parser.add_argument("--traders", nargs="+", help="Specific traders to train")
    parser.add_argument(
        "--retrain", action="store_true", help="Retrain existing models"
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Initialize components
    data_loader = DataLoader(config)
    feature_engineer = FeatureEngineer(config["features"])

    # Load and prepare data
    logging.info("Loading training data...")
    all_data = data_loader.load_all_traders()

    # Feature engineering
    logging.info("Engineering features...")
    features_df = feature_engineer.transform(all_data)

    # Train models
    logging.info("Training models...")
    ensemble = train_ensemble_models(features_df, config)

    # Validate models
    logging.info("Validating models...")
    validation_results = validate_models(ensemble, features_df, config)

    # Save models and results
    save_models(ensemble, validation_results, config)

    logging.info("Training pipeline completed successfully!")


if __name__ == "__main__":
    main()
