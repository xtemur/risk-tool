#!/usr/bin/env python3
"""
Copy trained models to _prod suffix and run threshold optimization
"""

import logging
import json
import shutil
from datetime import datetime
from pathlib import Path
from src.threshold_optimization import ThresholdOptimizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Copy models and optimize thresholds"""
    logger.info("Copying models with _prod suffix...")

    # Get list of traders
    models_dir = Path('models/trader_specific')
    trader_ids = []

    # Copy models to _prod suffix
    for model_file in models_dir.glob("*_tuned_validated.pkl"):
        if "_prod" not in model_file.name and "_80pct" not in model_file.name:
            trader_id = model_file.name.split("_")[0]
            trader_ids.append(trader_id)

            src_path = model_file
            dst_path = models_dir / f"{trader_id}_tuned_validated_prod.pkl"
            shutil.copy2(src_path, dst_path)
            logger.info(f"Copied model for trader {trader_id} to {dst_path}")

    logger.info(f"Copied models for {len(trader_ids)} traders: {trader_ids}")

    # Initialize threshold optimizer
    logger.info("Optimizing thresholds for production models...")
    optimizer = ThresholdOptimizer(model_suffix="_prod")

    # Optimize thresholds
    results = optimizer.run_threshold_optimization()

    # Extract threshold results for production config
    threshold_results = {}
    if 'optimization_results' in results:
        for trader_id, trader_result in results['optimization_results'].items():
            if 'optimal_var_threshold' in trader_result:
                threshold_results[trader_id] = {
                    "var_threshold": trader_result['optimal_var_threshold'],
                    "loss_prob_threshold": trader_result['optimal_loss_prob_threshold']
                }

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
            "trained_models": len(trader_ids),
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
    logger.info("Production model setup completed successfully!")

    return production_config

if __name__ == "__main__":
    results = main()
