#!/usr/bin/env python3
"""
Train production models with _prod suffix
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from src.trader_specific_training import TraderSpecificTrainer
from src.threshold_optimization import ThresholdOptimizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/train_prod_models_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main function to train production models"""
    logger.info("Starting production model training pipeline...")

    # Initialize trainer
    trainer = TraderSpecificTrainer()

    # Train models for all traders with _prod suffix
    logger.info("Training production models with _prod suffix...")
    train_results = trainer.train_all_traders()

    if not train_results:
        logger.error("No models were trained successfully!")
        return

    # Rename models to have _prod suffix
    import shutil
    models_dir = Path('models/trader_specific')
    for trader_id in train_results.keys():
        src_path = models_dir / f"{trader_id}_tuned_validated.pkl"
        dst_path = models_dir / f"{trader_id}_tuned_validated_prod.pkl"
        if src_path.exists():
            shutil.copy2(src_path, dst_path)
            logger.info(f"Copied model for trader {trader_id} to {dst_path}")

    # Log training summary
    successful_traders = list(train_results.keys())
    logger.info(f"Successfully trained models for {len(successful_traders)} traders: {successful_traders}")

    # Initialize threshold optimizer
    logger.info("Optimizing thresholds for production models...")
    optimizer = ThresholdOptimizer(model_suffix="_prod")

    # Optimize thresholds
    threshold_results = optimizer.optimize_all_traders(trader_ids=successful_traders)

    # Create production configuration
    production_config = {
        "pipeline_metadata": {
            "generated_at": datetime.now().isoformat(),
            "pipeline_timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "pipeline_version": "1.0.0"
        },
        "model_configuration": {
            "model_suffix": "_prod",
            "models_directory": "models/trader_specific",
            "data_directory": "data/processed/trader_splits"
        },
        "threshold_configuration": {
            "optimization_method": "percentile_based",
            "target_intervention_rate": 0.4,
            "thresholds": threshold_results
        },
        "deployment_info": {
            "ready_for_production": True,
            "trained_models": len(successful_traders),
            "evaluation_completed": True
        }
    }

    # Save production configuration
    config_path = Path("configs/production_config.json")
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Backup existing config if it exists
    if config_path.exists():
        backup_path = config_path.parent / f"production_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        config_path.rename(backup_path)
        logger.info(f"Backed up existing config to {backup_path}")

    with open(config_path, 'w') as f:
        json.dump(production_config, f, indent=2, default=str)

    logger.info(f"Production configuration saved to {config_path}")

    # Save pipeline state
    pipeline_state = {
        "timestamp": datetime.now().isoformat(),
        "training_results": train_results,
        "threshold_results": threshold_results,
        "successful_traders": successful_traders,
        "config_path": str(config_path)
    }

    state_path = Path(f"results/pipeline_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    state_path.parent.mkdir(parents=True, exist_ok=True)

    with open(state_path, 'w') as f:
        json.dump(pipeline_state, f, indent=2, default=str)

    logger.info(f"Pipeline state saved to {state_path}")
    logger.info("Production model training pipeline completed successfully!")

    return {
        "production_config": production_config,
        "pipeline_state": pipeline_state,
        "successful_traders": successful_traders
    }

if __name__ == "__main__":
    results = main()
