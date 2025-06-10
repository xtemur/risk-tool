# src/experiments/model_comparison.py
"""
Model Comparison Framework
Compare multiple models and approaches systematically
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelComparison:
    """
    Compare multiple models systematically
    """

    def __init__(self, comparison_dir: str = "comparisons"):
        self.comparison_dir = Path(comparison_dir)
        self.comparison_dir.mkdir(exist_ok=True, parents=True)

    def create_comparison_report(self,
                               evaluations: List[Dict[str, Any]],
                               test_data: pd.DataFrame,
                               save_plots: bool = True) -> Dict[str, Any]:
        """
        Create comprehensive comparison report

        Args:
            evaluations: List of model evaluation results
            test_data: Test dataset for additional analysis
            save_plots: Whether to save visualization plots

        Returns:
            Comparison report dictionary
        """
        report = {
            'summary': self._create_summary_table(evaluations),
            'rankings': self._rank_models(evaluations),
            'trade_offs': self._analyze_trade_offs(evaluations),
            'stability': self._analyze_stability(evaluations, test_data),
            'recommendations': self._generate_recommendations(evaluations)
        }

        if save_plots:
            self._create_visualizations(evaluations, report)

        return report

    def _create_summary_table(self, evaluations: List[Dict[str, Any]]) -> pd.DataFrame:
        """Create summary comparison table"""

        summary_data = []

        for eval_result in evaluations:
            row = {
                'Model': eval_result['model_name'],
                'RMSE': eval_result['statistical_metrics']['rmse'],
                'Direction Acc': eval_result['statistical_metrics']['directional_accuracy'],
                'Sharpe': eval_result['risk_metrics']['sharpe_ratio'],
                'Max DD': eval_result['risk_metrics']['max_drawdown'],
                'Win Rate': eval_result['trading_metrics']['win_rate'],
                'Profit Factor': eval_result['trading_metrics']['profit_factor'],
                'Total Return': eval_result['trading_metrics']['total_return']
            }
            summary_data.append(row)

        return pd.DataFrame(summary_data).round(3)

    def _rank_models(self, evaluations: List[Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
        """Rank models by different criteria"""

        rankings = {}

        # Define ranking criteria
        criteria = {
            'accuracy': lambda e: e['statistical_metrics']['directional_accuracy'],
            'risk_adjusted': lambda e: e['risk_metrics']['sharpe_ratio'],
            'consistency': lambda e: -e['risk_metrics']['max_drawdown'],
            'profitability': lambda e: e['trading_metrics']['total_return'],
            'robustness': lambda e: e['risk_metrics']['sortino_ratio']
        }

        for criterion_name, criterion_func in criteria.items():
            sorted_evals = sorted(evaluations,
                                key=criterion_func,
                                reverse=True)

            ranking_data = []
            for rank, eval_result in enumerate(sorted_evals, 1):
                ranking_data.append({
                    'Rank': rank,
                    'Model': eval_result['model_name'],
                    'Score': criterion_func(eval_result)
                })

            rankings[criterion_name] = pd.DataFrame(ranking_data)

        return rankings

    def _analyze_trade_offs(self, evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trade-offs between different metrics"""

        # Extract key metrics
        metrics_df = pd.DataFrame([
            {
                'model': e['model_name'],
                'return': e['trading_metrics']['total_return'],
                'risk': e['risk_metrics']['volatility'],
                'accuracy': e['statistical_metrics']['directional_accuracy'],
                'drawdown': abs(e['risk_metrics']['max_drawdown'])
            }
            for e in evaluations
        ])

        # Calculate trade-off scores
        trade_offs = {
            'return_vs_risk': metrics_df['return'] / (metrics_df['risk'] + 0.01),
            'return_vs_drawdown': metrics_df['return'] / (metrics_df['drawdown'] + 0.01),
            'accuracy_vs_complexity': None  # Would need model complexity metric
        }

        # Find Pareto optimal models
        pareto_optimal = []
        for i, model1 in metrics_df.iterrows():
            is_pareto = True
            for j, model2 in metrics_df.iterrows():
                if i != j:
                    # Check if model2 dominates model1
                    if (model2['return'] > model1['return'] and
                        model2['risk'] < model1['risk'] and
                        model2['accuracy'] > model1['accuracy']):
                        is_pareto = False
                        break
            if is_pareto:
                pareto_optimal.append(model1['model'])

        return {
            'trade_off_scores': trade_offs,
            'pareto_optimal_models': pareto_optimal
        }

    def _analyze_stability(self,
                         evaluations: List[Dict[str, Any]],
                         test_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze model stability across different conditions"""

        stability_metrics = {}

        for eval_result in evaluations:
            model_name = eval_result['model_name']

            # Temporal stability
            temporal = eval_result.get('temporal_metrics', {})

            # Per-trader consistency
            trader_metrics = eval_result.get('trader_metrics', {})
            if trader_metrics and 'individual' in trader_metrics:
                trader_performances = [
                    t['directional_accuracy']
                    for t in trader_metrics['individual'].values()
                ]
                trader_consistency = 1 - np.std(trader_performances) if trader_performances else 0
            else:
                trader_consistency = 0

            stability_metrics[model_name] = {
                'performance_trend': temporal.get('performance_trend', 0),
                'error_autocorrelation': abs(temporal.get('error_autocorrelation', 0)),
                'trader_consistency': trader_consistency,
                'monthly_stability': 1 - temporal.get('monthly_mae_std', 1)
            }

        return stability_metrics

    def _generate_recommendations(self, evaluations: List[Dict[str, Any]]) -> List[str]:
        """Generate model selection recommendations"""

        recommendations = []

        # Find best models for different objectives
        best_accuracy = max(evaluations,
                          key=lambda e: e['statistical_metrics']['directional_accuracy'])
        best_risk_adjusted = max(evaluations,
                               key=lambda e: e['risk_metrics']['sharpe_ratio'])
        best_consistent = min(evaluations,
                            key=lambda e: abs(e['risk_metrics']['max_drawdown']))

        recommendations.append(
            f"For maximum accuracy: {best_accuracy['model_name']} "
            f"({best_accuracy['statistical_metrics']['directional_accuracy']:.1%} accuracy)"
        )

        recommendations.append(
            f"For best risk-adjusted returns: {best_risk_adjusted['model_name']} "
            f"(Sharpe: {best_risk_adjusted['risk_metrics']['sharpe_ratio']:.2f})"
        )

        recommendations.append(
            f"For most consistent performance: {best_consistent['model_name']} "
            f"(Max DD: {best_consistent['risk_metrics']['max_drawdown']:.1%})"
        )

        # Overall recommendation based on balanced score
        for eval_result in evaluations:
            # Calculate balanced score
            accuracy = eval_result['statistical_metrics']['directional_accuracy']
            sharpe = eval_result['risk_metrics']['sharpe_ratio']
            drawdown = abs(eval_result['risk_metrics']['max_drawdown'])

            # Normalize and combine (you can adjust weights)
            balanced_score = (
                0.3 * accuracy +
                0.4 * (sharpe / 2) +  # Normalize assuming Sharpe ~2 is good
                0.3 * (1 - drawdown)  # Lower drawdown is better
            )
            eval_result['balanced_score'] = balanced_score

        best_overall = max(evaluations, key=lambda e: e['balanced_score'])
        recommendations.append(
            f"\nOVERALL RECOMMENDATION: {best_overall['model_name']} "
            f"(Balanced score: {best_overall['balanced_score']:.3f})"
        )

        return recommendations

    def _create_visualizations(self,
                             evaluations: List[Dict[str, Any]],
                             report: Dict[str, Any]):
        """Create comparison visualizations"""

        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')

        # 1. Performance Overview
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Accuracy comparison
        models = [e['model_name'] for e in evaluations]
        accuracies = [e['statistical_metrics']['directional_accuracy'] for e in evaluations]

        axes[0, 0].bar(models, accuracies)
        axes[0, 0].set_title('Directional Accuracy')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Risk-adjusted returns
        sharpes =
