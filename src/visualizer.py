"""
Training Visualization Module for Risk Management MVP
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class TrainingVisualizer:
    """Visualize model training progress and performance"""

    def __init__(self, save_dir: str = "data/plots"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

    def plot_training_history(self, eval_results: Dict,
                            account_id: str,
                            metric: str = 'rmse',
                            save: bool = True,
                            show: bool = True) -> Optional[plt.Figure]:
        """Plot train vs validation loss curves"""

        if not eval_results or 'train' not in eval_results or 'valid' not in eval_results:
            logger.warning(f"No evaluation results available for {account_id}")
            return None

        # Extract metrics
        train_metric = eval_results['train'].get(metric, [])
        valid_metric = eval_results['valid'].get(metric, [])

        if not train_metric or not valid_metric:
            logger.warning(f"Metric {metric} not found in results")
            return None

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot curves
        epochs = range(1, len(train_metric) + 1)
        ax.plot(epochs, train_metric, 'b-', label=f'Train {metric.upper()}', linewidth=2)
        ax.plot(epochs, valid_metric, 'r-', label=f'Validation {metric.upper()}', linewidth=2)

        # Find best epoch (early stopping point)
        best_epoch = np.argmin(valid_metric) + 1
        best_val = min(valid_metric)
        ax.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7,
                  label=f'Best Epoch: {best_epoch}')
        ax.plot(best_epoch, best_val, 'go', markersize=10)

        # Formatting
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(f'{metric.upper()}', fontsize=12)
        ax.set_title(f'Training History - {account_id}\nBest Val {metric.upper()}: {best_val:.4f} @ Epoch {best_epoch}',
                    fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Add shaded region showing overfitting
        if len(train_metric) > best_epoch + 10:
            ax.axvspan(best_epoch, len(train_metric), alpha=0.2, color='red',
                      label='Overfitting Region')

        plt.tight_layout()

        # Save if requested
        if save:
            filename = self.save_dir / f"training_history_{account_id}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {filename}")

        # Show if requested
        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_all_models_comparison(self, model_trainer,
                                 metric: str = 'rmse',
                                 save: bool = True) -> Optional[plt.Figure]:
        """Compare training curves across all models"""

        models = model_trainer.get_all_models()

        if not models:
            logger.warning("No models found")
            return None

        # Create subplots grid
        n_models = len(models)
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_models > 1 else [axes]

        # Plot each model
        for idx, (account_id, model_data) in enumerate(models.items()):
            ax = axes[idx]

            eval_results = model_data.get('eval_results', {})
            if not eval_results:
                ax.text(0.5, 0.5, 'No training history',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{account_id}')
                continue

            # Plot on subplot
            train_metric = eval_results.get('train', {}).get(metric, [])
            valid_metric = eval_results.get('valid', {}).get(metric, [])

            if train_metric and valid_metric:
                epochs = range(1, len(train_metric) + 1)
                ax.plot(epochs, train_metric, 'b-', label='Train', alpha=0.7)
                ax.plot(epochs, valid_metric, 'r-', label='Valid', alpha=0.7)

                best_epoch = np.argmin(valid_metric) + 1
                ax.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.5)

                ax.set_title(f'{account_id}\nBest: {min(valid_metric):.3f}')
                ax.set_xlabel('Epoch')
                ax.set_ylabel(metric.upper())
                ax.legend(loc='upper right', fontsize=8)
                ax.grid(True, alpha=0.3)

        # Hide empty subplots
        for idx in range(n_models, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle(f'Training Curves Comparison - {metric.upper()}', fontsize=16)
        plt.tight_layout()

        if save:
            filename = self.save_dir / f"all_models_training_comparison.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Saved comparison plot to {filename}")

        return fig

    def plot_learning_curves(self, model_trainer, account_id: str,
                           feature_columns: List[str]) -> Optional[plt.Figure]:
        """Plot learning curves with different training set sizes"""

        from sklearn.model_selection import learning_curve

        # This would require modifications to use sklearn-compatible wrapper
        # Keeping it simple for now
        pass
