"""
Walk-forward validator for time-series cross-validation.

This module implements time-series-aware cross-validation to prevent
look-ahead bias in trading risk models.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


class WalkForwardValidator:
    """
    Time-series aware cross-validation for trading risk models.

    Ensures that:
    1. Training data always precedes test data
    2. No data leakage between folds
    3. Proper evaluation of model performance over time
    """

    def __init__(
        self,
        min_train_days: int = 90,
        test_days: int = 30,
        step_days: int = 7,
        purge_days: int = 2,
    ):
        """
        Initialize walk-forward validator.

        Args:
            min_train_days: Minimum days of training data required
            test_days: Number of days in each test period
            step_days: Days to step forward between folds
            purge_days: Gap days between train and test to prevent leakage
        """
        self.min_train_days = min_train_days
        self.test_days = test_days
        self.step_days = step_days
        self.purge_days = purge_days

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dates: pd.Series,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices for walk-forward validation.

        Args:
            X: Feature dataframe
            y: Target series
            dates: Date series for temporal ordering

        Returns:
            List of (train_indices, test_indices) tuples
        """
        if len(X) != len(y) or len(X) != len(dates):
            raise ValueError("X, y, and dates must have same length")

        # Convert dates to pandas datetime if needed
        dates = pd.to_datetime(dates)

        # Sort by date to ensure temporal order
        sort_idx = dates.argsort()
        dates_sorted = dates.iloc[sort_idx]

        # Get date range
        min_date = dates_sorted.min()
        max_date = dates_sorted.max()

        # Calculate first test start date
        first_test_start = min_date + timedelta(days=self.min_train_days + self.purge_days)

        splits = []
        test_start = first_test_start

        while test_start + timedelta(days=self.test_days) <= max_date:
            # Define train period
            train_end = test_start - timedelta(days=self.purge_days + 1)
            train_mask = (dates >= min_date) & (dates <= train_end)

            # Define test period
            test_end = test_start + timedelta(days=self.test_days - 1)
            test_mask = (dates >= test_start) & (dates <= test_end)

            # Get indices
            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]

            # Only add if we have sufficient data
            if len(train_idx) >= self.min_train_days and len(test_idx) > 0:
                splits.append((train_idx, test_idx))
                logger.info(
                    f"Fold: train {train_end.date()} ({len(train_idx)} samples), "
                    f"test {test_start.date()}-{test_end.date()} ({len(test_idx)} samples)"
                )

            # Move to next fold
            test_start += timedelta(days=self.step_days)

        logger.info(f"Generated {len(splits)} walk-forward splits")
        return splits

    def validate(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        dates: pd.Series,
        sample_weights: Optional[pd.Series] = None,
    ) -> Dict[str, Union[float, List[float], pd.DataFrame]]:
        """
        Perform walk-forward validation of a model.

        Args:
            model: Scikit-learn compatible model
            X: Feature dataframe
            y: Target series (binary: 0=normal, 1=high-risk)
            dates: Date series
            sample_weights: Optional sample weights

        Returns:
            Dictionary with validation metrics and predictions
        """
        splits = self.split(X, y, dates)

        if not splits:
            raise ValueError("No valid splits generated")

        # Store results
        all_predictions = []
        all_probabilities = []
        all_actuals = []
        all_dates = []
        fold_metrics = []

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            logger.info(f"Processing fold {fold_idx + 1}/{len(splits)}")

            # Get train/test data
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Handle sample weights
            if sample_weights is not None:
                w_train = sample_weights.iloc[train_idx]
                model.fit(X_train, y_train, sample_weight=w_train)
            else:
                model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            # Store results
            all_predictions.extend(y_pred)
            all_probabilities.extend(y_prob)
            all_actuals.extend(y_test)
            all_dates.extend(dates.iloc[test_idx])

            # Calculate fold metrics
            fold_metric = {
                'fold': fold_idx,
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
            }

            # Add AUC if we have both classes
            if len(np.unique(y_test)) > 1:
                fold_metric['auc'] = roc_auc_score(y_test, y_prob)
            else:
                fold_metric['auc'] = np.nan

            fold_metrics.append(fold_metric)

        # Convert to arrays
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        all_actuals = np.array(all_actuals)
        all_dates = pd.to_datetime(all_dates)

        # Calculate overall metrics
        overall_metrics = {
            'accuracy': accuracy_score(all_actuals, all_predictions),
            'precision': precision_score(all_actuals, all_predictions, zero_division=0),
            'recall': recall_score(all_actuals, all_predictions, zero_division=0),
            'f1': f1_score(all_actuals, all_predictions, zero_division=0),
        }

        # Add AUC if we have both classes
        if len(np.unique(all_actuals)) > 1:
            overall_metrics['auc'] = roc_auc_score(all_actuals, all_probabilities)
            fpr, tpr, thresholds = roc_curve(all_actuals, all_probabilities)
            overall_metrics['roc_curve'] = {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}

        # Create results dataframe
        results_df = pd.DataFrame({
            'date': all_dates,
            'actual': all_actuals,
            'predicted': all_predictions,
            'probability': all_probabilities,
        })

        # Add classification report
        overall_metrics['classification_report'] = classification_report(
            all_actuals, all_predictions, output_dict=True
        )

        return {
            'overall_metrics': overall_metrics,
            'fold_metrics': pd.DataFrame(fold_metrics),
            'predictions': results_df,
            'n_folds': len(splits),
        }

    def plot_results(
        self,
        results: Dict,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot validation results over time.

        Args:
            results: Results dictionary from validate()
            save_path: Optional path to save plots
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Set style
        sns.set_style('whitegrid')

        # Create figure
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))

        # 1. Metrics over folds
        fold_metrics = results['fold_metrics']
        ax = axes[0, 0]
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
            if metric in fold_metrics.columns:
                ax.plot(fold_metrics['fold'], fold_metrics[metric], marker='o', label=metric)
        ax.set_xlabel('Fold')
        ax.set_ylabel('Score')
        ax.set_title('Metrics Over Time (Walk-Forward Folds)')
        ax.legend()
        ax.grid(True)

        # 2. ROC Curve
        ax = axes[0, 1]
        if 'roc_curve' in results['overall_metrics']:
            roc = results['overall_metrics']['roc_curve']
            auc_score = results['overall_metrics'].get('auc', 0)
            ax.plot(roc['fpr'], roc['tpr'], label=f'ROC (AUC = {auc_score:.3f})')
            ax.plot([0, 1], [0, 1], 'k--', label='Random')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve (All Folds Combined)')
            ax.legend()
            ax.grid(True)

        # 3. Predictions over time
        predictions = results['predictions']
        ax = axes[1, 0]

        # Group by date and calculate daily accuracy
        daily_acc = predictions.groupby(predictions['date'].dt.date).apply(
            lambda x: (x['actual'] == x['predicted']).mean()
        )
        ax.plot(daily_acc.index, daily_acc.values, alpha=0.7)
        ax.axhline(y=results['overall_metrics']['accuracy'], color='r',
                   linestyle='--', label=f"Overall Acc: {results['overall_metrics']['accuracy']:.3f}")
        ax.set_xlabel('Date')
        ax.set_ylabel('Daily Accuracy')
        ax.set_title('Model Accuracy Over Time')
        ax.legend()
        ax.tick_params(axis='x', rotation=45)

        # 4. Risk probability distribution
        ax = axes[1, 1]
        ax.hist(predictions[predictions['actual'] == 0]['probability'],
                bins=30, alpha=0.5, label='Normal Days', density=True)
        ax.hist(predictions[predictions['actual'] == 1]['probability'],
                bins=30, alpha=0.5, label='High-Risk Days', density=True)
        ax.set_xlabel('Predicted Risk Probability')
        ax.set_ylabel('Density')
        ax.set_title('Risk Probability Distribution by Actual Class')
        ax.legend()

        # 5. Confusion matrix over time
        ax = axes[2, 0]
        # Calculate monthly confusion matrix values
        predictions['month'] = predictions['date'].dt.to_period('M')
        monthly_stats = predictions.groupby('month').apply(
            lambda x: pd.Series({
                'true_positives': ((x['actual'] == 1) & (x['predicted'] == 1)).sum(),
                'false_positives': ((x['actual'] == 0) & (x['predicted'] == 1)).sum(),
                'true_negatives': ((x['actual'] == 0) & (x['predicted'] == 0)).sum(),
                'false_negatives': ((x['actual'] == 1) & (x['predicted'] == 0)).sum(),
            })
        )

        if len(monthly_stats) > 0:
            monthly_stats.index = monthly_stats.index.to_timestamp()
            monthly_stats[['true_positives', 'false_negatives']].plot(ax=ax, kind='bar', stacked=True)
            ax.set_xlabel('Month')
            ax.set_ylabel('Count')
            ax.set_title('High-Risk Day Detection Over Time')
            ax.tick_params(axis='x', rotation=45)

        # 6. Feature importance (if available)
        ax = axes[2, 1]
        ax.text(0.5, 0.5, 'Feature Importance\n(Run model.feature_importances_\nafter validation)',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Top Features')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plots saved to {save_path}")

        plt.show()
