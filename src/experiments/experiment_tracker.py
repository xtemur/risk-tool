# src/experiments/experiment_tracker.py
"""
Experiment Tracker
Tracks all model experiments and their results
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import json
import logging
from pathlib import Path
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """
    Tracks all experiments with their configurations and results
    """

    def __init__(self, experiments_dir: str = "experiments"):
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(exist_ok=True, parents=True)

        # Current experiment
        self.current_experiment = None

    def create_experiment(self,
                         name: str,
                         model_type: str,
                         features: List[str],
                         params: Dict[str, Any],
                         description: str = "") -> str:
        """
        Create new experiment

        Returns:
            Experiment ID
        """
        # Generate unique ID
        exp_id = self._generate_experiment_id(name, model_type, features, params)

        # Create experiment directory
        exp_dir = self.experiments_dir / exp_id
        exp_dir.mkdir(exist_ok=True, parents=True)

        # Create experiment metadata
        experiment = {
            'id': exp_id,
            'name': name,
            'model_type': model_type,
            'features': features,
            'n_features': len(features),
            'params': params,
            'description': description,
            'created_at': datetime.now().isoformat(),
            'status': 'created',
            'results': {}
        }

        # Save metadata
        metadata_path = exp_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(experiment, f, indent=2)

        self.current_experiment = experiment
        logger.info(f"Created experiment: {exp_id}")

        return exp_id

    def log_training_results(self,
                           exp_id: str,
                           train_metrics: Dict[str, float],
                           val_metrics: Dict[str, float],
                           training_time: float):
        """Log training results"""

        exp_dir = self.experiments_dir / exp_id
        metadata_path = exp_dir / "metadata.json"

        # Load experiment
        with open(metadata_path, 'r') as f:
            experiment = json.load(f)

        # Update results
        experiment['results']['training'] = {
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'training_time': training_time,
            'completed_at': datetime.now().isoformat()
        }
        experiment['status'] = 'trained'

        # Save updated metadata
        with open(metadata_path, 'w') as f:
            json.dump(experiment, f, indent=2)

        logger.info(f"Logged training results for {exp_id}")

    def log_test_results(self,
                        exp_id: str,
                        test_metrics: Dict[str, float],
                        predictions: pd.DataFrame):
        """Log test results (final evaluation)"""

        exp_dir = self.experiments_dir / exp_id
        metadata_path = exp_dir / "metadata.json"

        # Load experiment
        with open(metadata_path, 'r') as f:
            experiment = json.load(f)

        # Update results
        experiment['results']['test'] = {
            'test_metrics': test_metrics,
            'evaluated_at': datetime.now().isoformat()
        }
        experiment['status'] = 'evaluated'

        # Save updated metadata
        with open(metadata_path, 'w') as f:
            json.dump(experiment, f, indent=2)

        # Save predictions
        predictions_path = exp_dir / "test_predictions.csv"
        predictions.to_csv(predictions_path, index=False)

        logger.info(f"Logged test results for {exp_id}")

    def get_experiment_summary(self) -> pd.DataFrame:
        """Get summary of all experiments"""

        experiments = []

        for exp_dir in self.experiments_dir.iterdir():
            if exp_dir.is_dir():
                metadata_path = exp_dir / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        exp = json.load(f)

                    # Extract key metrics
                    summary = {
                        'id': exp['id'],
                        'name': exp['name'],
                        'model_type': exp['model_type'],
                        'n_features': exp['n_features'],
                        'status': exp['status'],
                        'created_at': exp['created_at']
                    }

                    # Add results if available
                    if 'training' in exp.get('results', {}):
                        summary.update({
                            f'val_{k}': v
                            for k, v in exp['results']['training']['val_metrics'].items()
                        })

                    if 'test' in exp.get('results', {}):
                        summary.update({
                            f'test_{k}': v
                            for k, v in exp['results']['test']['test_metrics'].items()
                        })

                    experiments.append(summary)

        return pd.DataFrame(experiments)

    def _generate_experiment_id(self, name: str, model_type: str,
                               features: List[str], params: Dict[str, Any]) -> str:
        """Generate unique experiment ID"""

        # Create hash from configuration
        config_str = f"{name}_{model_type}_{sorted(features)}_{sorted(params.items())}"
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]

        # Create readable ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_id = f"{name}_{model_type}_{timestamp}_{config_hash}"

        return exp_id
