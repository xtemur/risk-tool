"""
Performance Evaluator

Comprehensive evaluation of model performance using both statistical
and financial metrics for trading applications.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Scikit-learn metrics
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error
)

from .config import ModelConfig

logger = logging.getLogger(__name__)


class PerformanceEvaluator:
    """
    Comprehensive performance evaluation for trading models
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize the evaluator

        Args:
            config: Model configuration object
        """
        self.config = config or ModelConfig()

    def calculate_statistical_metrics(self, y_true: np.ndarray,
                                    y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate statistical performance metrics

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary of statistical metrics
        """
        metrics = {}

        # Basic regression metrics
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics['r2'] = r2_score(y_true, y_pred)

        # MAPE (handle division by zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1e-8))) * 100
            metrics['mape'] = mape if np.isfinite(mape) else np.inf

        # Additional metrics
        residuals = y_true - y_pred
        metrics['mean_residual'] = np.mean(residuals)
        metrics['std_residual'] = np.std(residuals)

        # Correlation
        if len(y_true) > 1:
            correlation = np.corrcoef(y_true, y_pred)[0, 1]
            metrics['correlation'] = correlation if np.isfinite(correlation) else 0.0
        else:
            metrics['correlation'] = 0.0

        return metrics

    def calculate_financial_metrics(self, y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   dates: Optional[pd.DatetimeIndex] = None) -> Dict[str, float]:
        """
        Calculate financial performance metrics

        Args:
            y_true: True PnL values
            y_pred: Predicted PnL values
            dates: Optional dates for time-based calculations

        Returns:
            Dictionary of financial metrics
        """
        metrics = {}

        # Hit rate (directional accuracy)
        true_direction = np.sign(y_true)
        pred_direction = np.sign(y_pred)
        metrics['hit_rate'] = np.mean(true_direction == pred_direction)

        # Sharpe ratio of predictions vs actuals
        if np.std(y_true) > 0:
            metrics['actual_sharpe'] = np.mean(y_true) / np.std(y_true) * np.sqrt(252)
        else:
            metrics['actual_sharpe'] = 0.0

        if np.std(y_pred) > 0:
            metrics['predicted_sharpe'] = np.mean(y_pred) / np.std(y_pred) * np.sqrt(252)
        else:
            metrics['predicted_sharpe'] = 0.0

        # Maximum drawdown
        cumulative_true = np.cumsum(y_true)
        cumulative_pred = np.cumsum(y_pred)

        running_max_true = np.maximum.accumulate(cumulative_true)
        running_max_pred = np.maximum.accumulate(cumulative_pred)

        drawdown_true = cumulative_true - running_max_true
        drawdown_pred = cumulative_pred - running_max_pred

        metrics['max_drawdown_actual'] = np.min(drawdown_true)
        metrics['max_drawdown_predicted'] = np.min(drawdown_pred)

        # Profit factor (total profits / total losses)
        profits_true = y_true[y_true > 0]
        losses_true = y_true[y_true < 0]

        profits_pred = y_pred[y_pred > 0]
        losses_pred = y_pred[y_pred < 0]

        if len(losses_true) > 0 and np.sum(np.abs(losses_true)) > 0:
            metrics['profit_factor_actual'] = np.sum(profits_true) / np.sum(np.abs(losses_true))
        else:
            metrics['profit_factor_actual'] = np.inf if len(profits_true) > 0 else 0.0

        if len(losses_pred) > 0 and np.sum(np.abs(losses_pred)) > 0:
            metrics['profit_factor_predicted'] = np.sum(profits_pred) / np.sum(np.abs(losses_pred))
        else:
            metrics['profit_factor_predicted'] = np.inf if len(profits_pred) > 0 else 0.0

        # Win/loss ratios
        metrics['win_rate_actual'] = len(profits_true) / len(y_true) if len(y_true) > 0 else 0.0
        metrics['win_rate_predicted'] = len(profits_pred) / len(y_pred) if len(y_pred) > 0 else 0.0

        # Average win/loss
        metrics['avg_win_actual'] = np.mean(profits_true) if len(profits_true) > 0 else 0.0
        metrics['avg_loss_actual'] = np.mean(losses_true) if len(losses_true) > 0 else 0.0

        metrics['avg_win_predicted'] = np.mean(profits_pred) if len(profits_pred) > 0 else 0.0
        metrics['avg_loss_predicted'] = np.mean(losses_pred) if len(losses_pred) > 0 else 0.0

        return metrics

    def calculate_prediction_quality_metrics(self, y_true: np.ndarray,
                                           y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate prediction quality specific metrics

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary of prediction quality metrics
        """
        metrics = {}

        # Information coefficient (IC)
        if len(y_true) > 1:
            ic = np.corrcoef(y_true, y_pred)[0, 1]
            metrics['information_coefficient'] = ic if np.isfinite(ic) else 0.0
        else:
            metrics['information_coefficient'] = 0.0

        # Rank correlation (Spearman)
        if len(y_true) > 1:
            rank_corr, _ = stats.spearmanr(y_true, y_pred)
            metrics['rank_correlation'] = rank_corr if np.isfinite(rank_corr) else 0.0
        else:
            metrics['rank_correlation'] = 0.0

        # Prediction strength in different market conditions
        # Up markets (positive actual returns)
        up_mask = y_true > 0
        if np.sum(up_mask) > 1:
            up_corr = np.corrcoef(y_true[up_mask], y_pred[up_mask])[0, 1]
            metrics['up_market_correlation'] = up_corr if np.isfinite(up_corr) else 0.0
        else:
            metrics['up_market_correlation'] = 0.0

        # Down markets (negative actual returns)
        down_mask = y_true < 0
        if np.sum(down_mask) > 1:
            down_corr = np.corrcoef(y_true[down_mask], y_pred[down_mask])[0, 1]
            metrics['down_market_correlation'] = down_corr if np.isfinite(down_corr) else 0.0
        else:
            metrics['down_market_correlation'] = 0.0

        # Prediction bias
        metrics['prediction_bias'] = np.mean(y_pred - y_true)

        # Prediction consistency (lower is better)
        residuals = y_true - y_pred
        metrics['prediction_consistency'] = np.std(residuals)

        return metrics

    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray,
                      dates: Optional[pd.DatetimeIndex] = None,
                      model_name: str = 'model') -> Dict[str, Any]:
        """
        Comprehensive model evaluation

        Args:
            y_true: True values
            y_pred: Predicted values
            dates: Optional dates for time-based analysis
            model_name: Name of the model being evaluated

        Returns:
            Complete evaluation results
        """
        results = {
            'model_name': model_name,
            'num_samples': len(y_true),
            'date_range': {
                'start': dates.min() if dates is not None else None,
                'end': dates.max() if dates is not None else None
            }
        }

        # Calculate all metrics
        results['statistical_metrics'] = self.calculate_statistical_metrics(y_true, y_pred)
        results['financial_metrics'] = self.calculate_financial_metrics(y_true, y_pred, dates)
        results['prediction_quality'] = self.calculate_prediction_quality_metrics(y_true, y_pred)

        # Overall score (weighted combination of key metrics)
        overall_score = self._calculate_overall_score(results)
        results['overall_score'] = overall_score

        logger.info(f"Model {model_name} evaluation completed:")
        logger.info(f"  MAE: {results['statistical_metrics']['mae']:.4f}")
        logger.info(f"  R²: {results['statistical_metrics']['r2']:.4f}")
        logger.info(f"  Hit Rate: {results['financial_metrics']['hit_rate']:.4f}")
        logger.info(f"  Overall Score: {overall_score:.4f}")

        return results

    def _calculate_overall_score(self, results: Dict[str, Any]) -> float:
        """
        Calculate an overall performance score

        Args:
            results: Evaluation results dictionary

        Returns:
            Overall score (0-1, higher is better)
        """
        # Weights for different metrics
        weights = {
            'r2': 0.3,
            'hit_rate': 0.3,
            'information_coefficient': 0.2,
            'rank_correlation': 0.2
        }

        # Normalize metrics to 0-1 scale
        r2 = max(0, results['statistical_metrics']['r2'])
        hit_rate = results['financial_metrics']['hit_rate']
        ic = abs(results['prediction_quality']['information_coefficient'])
        rank_corr = abs(results['prediction_quality']['rank_correlation'])

        # Calculate weighted score
        score = (
            weights['r2'] * r2 +
            weights['hit_rate'] * hit_rate +
            weights['information_coefficient'] * ic +
            weights['rank_correlation'] * rank_corr
        )

        return min(1.0, score)  # Cap at 1.0

    def compare_models(self, evaluation_results: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Compare multiple model evaluation results

        Args:
            evaluation_results: List of evaluation result dictionaries

        Returns:
            Comparison dataframe
        """
        if not evaluation_results:
            return pd.DataFrame()

        # Extract key metrics for comparison
        comparison_data = []

        for result in evaluation_results:
            row = {
                'model_name': result['model_name'],
                'num_samples': result['num_samples'],
                'overall_score': result['overall_score']
            }

            # Add statistical metrics
            for metric, value in result['statistical_metrics'].items():
                row[f'stat_{metric}'] = value

            # Add key financial metrics
            key_financial = ['hit_rate', 'actual_sharpe', 'profit_factor_actual']
            for metric in key_financial:
                if metric in result['financial_metrics']:
                    row[f'fin_{metric}'] = result['financial_metrics'][metric]

            # Add key prediction quality metrics
            key_quality = ['information_coefficient', 'rank_correlation']
            for metric in key_quality:
                if metric in result['prediction_quality']:
                    row[f'qual_{metric}'] = result['prediction_quality'][metric]

            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)

        # Sort by overall score
        df = df.sort_values('overall_score', ascending=False)

        return df

    def plot_evaluation_results(self, y_true: np.ndarray, y_pred: np.ndarray,
                               dates: Optional[pd.DatetimeIndex] = None,
                               model_name: str = 'model',
                               save_path: Optional[str] = None) -> None:
        """
        Create evaluation plots

        Args:
            y_true: True values
            y_pred: Predicted values
            dates: Optional dates for time series plots
            model_name: Name of the model
            save_path: Optional path to save plots
        """
        if not self.config.PLOT_RESULTS:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Model Evaluation: {model_name}', fontsize=16)

        # 1. Predicted vs Actual scatter plot
        axes[0, 0].scatter(y_true, y_pred, alpha=0.6)
        axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual PnL')
        axes[0, 0].set_ylabel('Predicted PnL')
        axes[0, 0].set_title('Predicted vs Actual')

        # Add correlation text
        corr = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0
        axes[0, 0].text(0.05, 0.95, f'Correlation: {corr:.3f}',
                       transform=axes[0, 0].transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))

        # 2. Residuals plot
        residuals = y_true - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted PnL')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residual Plot')

        # 3. Time series plot (if dates available)
        if dates is not None:
            axes[1, 0].plot(dates, y_true, label='Actual', alpha=0.7)
            axes[1, 0].plot(dates, y_pred, label='Predicted', alpha=0.7)
            axes[1, 0].set_xlabel('Date')
            axes[1, 0].set_ylabel('PnL')
            axes[1, 0].set_title('Time Series Comparison')
            axes[1, 0].legend()
            axes[1, 0].tick_params(axis='x', rotation=45)
        else:
            # Histogram of residuals
            axes[1, 0].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
            axes[1, 0].set_xlabel('Residuals')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Residual Distribution')

        # 4. Cumulative PnL comparison
        cum_actual = np.cumsum(y_true)
        cum_predicted = np.cumsum(y_pred)

        if dates is not None:
            axes[1, 1].plot(dates, cum_actual, label='Actual Cumulative', linewidth=2)
            axes[1, 1].plot(dates, cum_predicted, label='Predicted Cumulative', linewidth=2)
            axes[1, 1].tick_params(axis='x', rotation=45)
        else:
            axes[1, 1].plot(cum_actual, label='Actual Cumulative', linewidth=2)
            axes[1, 1].plot(cum_predicted, label='Predicted Cumulative', linewidth=2)

        axes[1, 1].set_xlabel('Time' if dates is not None else 'Sample')
        axes[1, 1].set_ylabel('Cumulative PnL')
        axes[1, 1].set_title('Cumulative PnL Comparison')
        axes[1, 1].legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Evaluation plots saved to {save_path}")

        plt.show()

    def generate_evaluation_report(self, evaluation_results: Dict[str, Any],
                                 save_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive evaluation report

        Args:
            evaluation_results: Evaluation results dictionary
            save_path: Optional path to save the report

        Returns:
            Report text
        """
        report = f"""
MODEL EVALUATION REPORT
========================

Model: {evaluation_results['model_name']}
Samples: {evaluation_results['num_samples']}
Overall Score: {evaluation_results['overall_score']:.4f}

STATISTICAL METRICS
-------------------
MAE: {evaluation_results['statistical_metrics']['mae']:.4f}
RMSE: {evaluation_results['statistical_metrics']['rmse']:.4f}
R²: {evaluation_results['statistical_metrics']['r2']:.4f}
MAPE: {evaluation_results['statistical_metrics']['mape']:.2f}%
Correlation: {evaluation_results['statistical_metrics']['correlation']:.4f}

FINANCIAL METRICS
-----------------
Hit Rate: {evaluation_results['financial_metrics']['hit_rate']:.1%}
Actual Sharpe Ratio: {evaluation_results['financial_metrics']['actual_sharpe']:.4f}
Predicted Sharpe Ratio: {evaluation_results['financial_metrics']['predicted_sharpe']:.4f}
Max Drawdown (Actual): {evaluation_results['financial_metrics']['max_drawdown_actual']:.2f}
Max Drawdown (Predicted): {evaluation_results['financial_metrics']['max_drawdown_predicted']:.2f}
Profit Factor (Actual): {evaluation_results['financial_metrics']['profit_factor_actual']:.4f}
Win Rate (Actual): {evaluation_results['financial_metrics']['win_rate_actual']:.1%}

PREDICTION QUALITY
------------------
Information Coefficient: {evaluation_results['prediction_quality']['information_coefficient']:.4f}
Rank Correlation: {evaluation_results['prediction_quality']['rank_correlation']:.4f}
Up Market Correlation: {evaluation_results['prediction_quality']['up_market_correlation']:.4f}
Down Market Correlation: {evaluation_results['prediction_quality']['down_market_correlation']:.4f}
Prediction Bias: {evaluation_results['prediction_quality']['prediction_bias']:.4f}
Prediction Consistency: {evaluation_results['prediction_quality']['prediction_consistency']:.4f}
"""

        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            logger.info(f"Evaluation report saved to {save_path}")

        return report
