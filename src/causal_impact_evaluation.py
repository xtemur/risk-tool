import pandas as pd
import numpy as np
import json
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# from src.utils import load_model  # Not needed for this evaluation


class CausalImpactEvaluator:
    """
    Evaluates the causal impact of model-based trading interventions on unseen test data.

    Enhanced to support:
    - Position sizing optimization (0% to 150%)
    - Multiple PnL reduction levels (0%, 10%, 30%, 60%)
    - Position-based risk management
    """

    def __init__(self, base_path: str = "/Users/temurbekkhujaev/Repos/risk-tool"):
        self.base_path = Path(base_path)
        self.models_path = self.base_path / "models" / "trader_specific"
        self.data_path = self.base_path / "data" / "processed" / "trader_splits"
        self.thresholds_path = self.base_path / "configs" / "optimal_thresholds" / "optimal_thresholds.json"

        # Load optimal thresholds
        with open(self.thresholds_path, 'r') as f:
            thresholds_data = json.load(f)

        self.thresholds = {
            str(thresh["trader_id"]): {
                "position_threshold": thresh.get("position_threshold", 0.5),  # Low position threshold
                "loss_prob_threshold": thresh["loss_prob_threshold"]
            }
            for thresh in thresholds_data["thresholds"]
        }

        # PnL reduction levels to test
        self.reduction_levels = [0.0, 0.1, 0.3, 0.6]  # 0%, 10%, 30%, 60%
        self.default_reduction = 0.5  # Current system default

        self.results = {}

    def load_trader_data(self, trader_id: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Load test data and trained model for a specific trader."""
        # Load test data
        test_data_path = self.data_path / trader_id / "test_data.parquet"
        test_data = pd.read_parquet(test_data_path)

        # Load trained model
        model_path = self.models_path / f"{trader_id}_tuned_validated.pkl"
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        return test_data, model_data

    def generate_predictions(self, test_data: pd.DataFrame, model_data: Dict[str, Any]) -> pd.DataFrame:
        """Generate position sizing and loss probability predictions for test data."""
        # Use the feature names from the model
        feature_names = model_data['feature_names']

        # Align features with model expectations
        X_test = test_data[feature_names]

        # Get models
        var_model = model_data['var_model']
        classification_model = model_data['classification_model']

        # Generate predictions
        var_predictions = var_model.predict(X_test)
        loss_probabilities = classification_model.predict_proba(X_test)[:, 1]  # Probability of loss

        # Create predictions dataframe
        predictions_df = test_data[['trader_id', 'date', 'daily_pnl']].copy()
        predictions_df['var_prediction'] = var_predictions
        predictions_df['loss_probability'] = loss_probabilities

        return predictions_df

    def calculate_weighted_risk_score(self, predictions_df: pd.DataFrame,
                                    alpha: float = 0.6, beta: float = 0.4) -> pd.DataFrame:
        """
        Calculate weighted risk score: alpha * normalized_VaR + beta * normalized_LossProb
        """
        # Normalize VaR to 0-1 scale (lower VaR = higher risk)
        var_min = predictions_df['var_prediction'].min()
        var_max = predictions_df['var_prediction'].max()

        if var_max != var_min:
            # Invert normalization so higher score = higher risk
            normalized_var = (var_max - predictions_df['var_prediction']) / (var_max - var_min)
        else:
            normalized_var = 0.5  # Neutral if no variance

        # Loss probability is already 0-1 scale
        normalized_loss_prob = predictions_df['loss_probability']

        # Calculate weighted risk score
        risk_score = alpha * normalized_var + beta * normalized_loss_prob

        predictions_df['normalized_var'] = normalized_var
        predictions_df['normalized_loss_prob'] = normalized_loss_prob
        predictions_df['risk_score'] = risk_score

        return predictions_df

    def classify_risk_level_weighted(self, risk_score: pd.Series,
                                   thresholds: Dict[str, float] = None) -> pd.Series:
        """
        Classify risk level using weighted risk score with 4 levels.
        """
        if thresholds is None:
            thresholds = {
                'high_threshold': 0.7,
                'medium_threshold': 0.5,
                'low_threshold': 0.3
            }

        def classify_single_score(score):
            if score >= thresholds['high_threshold']:
                return 'High Risk'
            elif score >= thresholds['medium_threshold']:
                return 'Medium Risk'
            elif score >= thresholds['low_threshold']:
                return 'Low Risk'
            else:
                return 'Neutral'

        return risk_score.apply(classify_single_score)

    def apply_intervention_logic(self, predictions_df: pd.DataFrame, trader_id: str,
                               reduction_level: float = None,
                               use_weighted_formula: bool = False,
                               alpha: float = 0.6, beta: float = 0.4,
                               risk_thresholds: Dict[str, float] = None) -> pd.DataFrame:
        """
        Apply intervention logic with configurable PnL reduction level.

        Args:
            predictions_df: DataFrame with predictions
            trader_id: Trader ID
            reduction_level: PnL reduction factor (0.0 = no reduction, 0.6 = 60% reduction)
            use_weighted_formula: Whether to use weighted risk formula
            alpha, beta: Coefficients for weighted formula
            risk_thresholds: Thresholds for weighted risk classification
        """
        results_df = predictions_df.copy()

        # Use default reduction if not specified
        if reduction_level is None:
            reduction_level = self.default_reduction

        if use_weighted_formula:
            # Calculate weighted risk score
            results_df = self.calculate_weighted_risk_score(results_df, alpha, beta)

            # Classify risk levels
            results_df['risk_level'] = self.classify_risk_level_weighted(
                results_df['risk_score'], risk_thresholds
            )

            # Intervention logic based on risk level
            if risk_thresholds is None:
                risk_thresholds = {
                    'high_threshold': 0.7,
                    'medium_threshold': 0.5,
                    'low_threshold': 0.3
                }

            # Define intervention strategy
            should_intervene = results_df['risk_score'] >= risk_thresholds['medium_threshold']

        else:
            # Use original binary threshold logic
            trader_thresholds = self.thresholds[trader_id]
            var_threshold = trader_thresholds["var_threshold"]
            loss_prob_threshold = trader_thresholds["loss_prob_threshold"]

            # Original intervention logic
            should_intervene = (
                (results_df['var_prediction'] <= var_threshold) |
                (results_df['loss_probability'] >= loss_prob_threshold)
            )

            results_df['var_threshold'] = var_threshold
            results_df['loss_prob_threshold'] = loss_prob_threshold

        # Calculate adjusted PnL based on reduction level
        # reduction_level: 0.0 = no reduction, 0.6 = 60% reduction
        reduction_factor = 1.0 - reduction_level

        adjusted_pnl = np.where(should_intervene,
                               results_df['daily_pnl'] * reduction_factor,
                               results_df['daily_pnl'])

        # Add results to dataframe
        results_df['should_intervene'] = should_intervene
        results_df['adjusted_pnl'] = adjusted_pnl
        results_df['reduction_level'] = reduction_level
        results_df['reduction_factor'] = reduction_factor

        return results_df

    def calculate_performance_metrics(self, results_df: pd.DataFrame, trader_id: str) -> Dict[str, float]:
        """Calculate performance metrics for a trader."""
        actual_pnl = results_df['daily_pnl'].sum()
        adjusted_pnl = results_df['adjusted_pnl'].sum()

        # Calculate avoided losses and missed gains
        intervention_days = results_df[results_df['should_intervene']]

        # Get reduction level from the results
        reduction_level = results_df['reduction_level'].iloc[0] if len(results_df) > 0 else self.default_reduction

        # Avoided losses: days where we intervened and actual PnL was negative
        avoided_losses = intervention_days[intervention_days['daily_pnl'] < 0]['daily_pnl'].sum() * reduction_level

        # Missed gains: days where we intervened and actual PnL was positive
        missed_gains = intervention_days[intervention_days['daily_pnl'] > 0]['daily_pnl'].sum() * reduction_level

        # Intervention rate
        intervention_rate = results_df['should_intervene'].mean()

        # Net benefit
        net_benefit = adjusted_pnl - actual_pnl

        # Performance improvement
        improvement_pct = ((adjusted_pnl - actual_pnl) / abs(actual_pnl) * 100) if actual_pnl != 0 else 0

        # Risk-adjusted metrics
        sharpe_actual = self.calculate_sharpe_ratio(results_df['daily_pnl'])
        sharpe_adjusted = self.calculate_sharpe_ratio(results_df['adjusted_pnl'])

        return {
            'trader_id': trader_id,
            'reduction_level': reduction_level,
            'actual_pnl': actual_pnl,
            'adjusted_pnl': adjusted_pnl,
            'net_benefit': net_benefit,
            'improvement_pct': improvement_pct,
            'avoided_losses': abs(avoided_losses),  # Make positive for clarity
            'missed_gains': abs(missed_gains),  # Make positive for clarity
            'intervention_rate': intervention_rate,
            'total_days': len(results_df),
            'intervention_days': intervention_days.shape[0],
            'sharpe_actual': sharpe_actual,
            'sharpe_adjusted': sharpe_adjusted,
            'sharpe_improvement': sharpe_adjusted - sharpe_actual
        }

    def calculate_sharpe_ratio(self, pnl_series: pd.Series) -> float:
        """Calculate Sharpe ratio for a PnL series."""
        if len(pnl_series) == 0 or pnl_series.std() == 0:
            return 0.0
        return pnl_series.mean() / pnl_series.std() * np.sqrt(252)  # Annualized

    def evaluate_trader(self, trader_id: str,
                      reduction_level: float = None,
                      use_weighted_formula: bool = False,
                      alpha: float = 0.6, beta: float = 0.4,
                      risk_thresholds: Dict[str, float] = None) -> Dict[str, Any]:
        """Evaluate a single trader and return comprehensive results."""
        print(f"Evaluating trader {trader_id}...")

        # Load data and model
        test_data, model_data = self.load_trader_data(trader_id)

        # Generate predictions
        predictions_df = self.generate_predictions(test_data, model_data)

        # Apply intervention logic
        results_df = self.apply_intervention_logic(
            predictions_df, trader_id, reduction_level,
            use_weighted_formula, alpha, beta, risk_thresholds
        )

        # Calculate metrics
        metrics = self.calculate_performance_metrics(results_df, trader_id)

        # Store results
        trader_results = {
            'metrics': metrics,
            'daily_results': results_df,
            'model_info': {
                'var_model_features': model_data.get('feature_names', []),
                'model_performance': model_data.get('test_metrics', {})
            }
        }

        self.results[trader_id] = trader_results
        return trader_results

    def evaluate_trader_multilevel(self, trader_id: str,
                                 use_weighted_formula: bool = False,
                                 alpha: float = 0.6, beta: float = 0.4,
                                 risk_thresholds: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Evaluate a single trader across all reduction levels.
        """
        print(f"Evaluating trader {trader_id} across all reduction levels...")

        # Load data and model
        test_data, model_data = self.load_trader_data(trader_id)

        # Generate predictions
        predictions_df = self.generate_predictions(test_data, model_data)

        trader_results = {
            'trader_id': trader_id,
            'reduction_level_results': {},
            'model_info': {
                'var_model_features': model_data.get('feature_names', []),
                'model_performance': model_data.get('test_metrics', {})
            }
        }

        # Test each reduction level
        for reduction_level in self.reduction_levels:
            print(f"  Testing {reduction_level*100:.0f}% reduction...")

            # Apply intervention logic
            results_df = self.apply_intervention_logic(
                predictions_df, trader_id, reduction_level,
                use_weighted_formula, alpha, beta, risk_thresholds
            )

            # Calculate metrics
            metrics = self.calculate_performance_metrics(results_df, trader_id)

            # Store results
            trader_results['reduction_level_results'][reduction_level] = {
                'metrics': metrics,
                'daily_results': results_df
            }

        return trader_results

    def evaluate_all_traders(self, reduction_level: float = None,
                           use_weighted_formula: bool = False,
                           alpha: float = 0.6, beta: float = 0.4,
                           risk_thresholds: Dict[str, float] = None) -> Dict[str, Any]:
        """Evaluate all traders with available models."""
        # Get list of available traders
        available_traders = [f.stem.split('_')[0] for f in self.models_path.glob("*_tuned_validated.pkl")]
        available_traders = [t for t in available_traders if t in self.thresholds.keys()]

        print(f"Found {len(available_traders)} traders to evaluate: {available_traders}")

        # Evaluate each trader
        for trader_id in available_traders:
            try:
                self.evaluate_trader(trader_id, reduction_level, use_weighted_formula, alpha, beta, risk_thresholds)
            except Exception as e:
                print(f"Error evaluating trader {trader_id}: {e}")
                continue

        # Generate aggregate results
        aggregate_results = self.calculate_aggregate_metrics()

        return {
            'individual_results': self.results,
            'aggregate_results': aggregate_results,
            'evaluation_summary': {
                'total_traders': len(available_traders),
                'successful_evaluations': len(self.results),
                'evaluation_date': pd.Timestamp.now().isoformat(),
                'reduction_level': reduction_level or self.default_reduction,
                'used_weighted_formula': use_weighted_formula,
                'alpha': alpha,
                'beta': beta,
                'risk_thresholds': risk_thresholds
            }
        }

    def evaluate_all_traders_multilevel(self, use_weighted_formula: bool = False,
                                      alpha: float = 0.6, beta: float = 0.4,
                                      risk_thresholds: Dict[str, float] = None) -> Dict[str, Any]:
        """Evaluate all traders across all reduction levels."""
        # Get list of available traders
        available_traders = [f.stem.split('_')[0] for f in self.models_path.glob("*_tuned_validated.pkl")]
        available_traders = [t for t in available_traders if t in self.thresholds.keys()]

        print(f"Found {len(available_traders)} traders to evaluate: {available_traders}")

        all_results = {}

        # Evaluate each trader
        for trader_id in available_traders:
            try:
                trader_results = self.evaluate_trader_multilevel(
                    trader_id, use_weighted_formula, alpha, beta, risk_thresholds
                )
                all_results[trader_id] = trader_results
            except Exception as e:
                print(f"Error evaluating trader {trader_id}: {e}")
                continue

        # Generate aggregate results
        aggregate_results = self.calculate_aggregate_metrics_multilevel(all_results)

        return {
            'individual_results': all_results,
            'aggregate_results': aggregate_results,
            'evaluation_summary': {
                'total_traders': len(available_traders),
                'successful_evaluations': len(all_results),
                'evaluation_date': pd.Timestamp.now().isoformat(),
                'reduction_levels_tested': self.reduction_levels,
                'used_weighted_formula': use_weighted_formula,
                'alpha': alpha,
                'beta': beta,
                'risk_thresholds': risk_thresholds
            }
        }

    def calculate_aggregate_metrics(self) -> Dict[str, float]:
        """Calculate aggregate metrics across all evaluated traders."""
        if not self.results:
            return {}

        metrics_list = [result['metrics'] for result in self.results.values()]

        aggregate = {
            'total_actual_pnl': sum(m['actual_pnl'] for m in metrics_list),
            'total_adjusted_pnl': sum(m['adjusted_pnl'] for m in metrics_list),
            'total_net_benefit': sum(m['net_benefit'] for m in metrics_list),
            'total_avoided_losses': sum(m['avoided_losses'] for m in metrics_list),
            'total_missed_gains': sum(m['missed_gains'] for m in metrics_list),
            'mean_intervention_rate': np.mean([m['intervention_rate'] for m in metrics_list]),
            'mean_improvement_pct': np.mean([m['improvement_pct'] for m in metrics_list]),
            'positive_improvements': sum(1 for m in metrics_list if m['net_benefit'] > 0),
            'total_traders': len(metrics_list)
        }

        # Calculate overall improvement percentage
        if aggregate['total_actual_pnl'] != 0:
            aggregate['overall_improvement_pct'] = (
                (aggregate['total_adjusted_pnl'] - aggregate['total_actual_pnl']) /
                abs(aggregate['total_actual_pnl']) * 100
            )
        else:
            aggregate['overall_improvement_pct'] = 0

        return aggregate

    def calculate_aggregate_metrics_multilevel(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate aggregate metrics across all traders and reduction levels."""
        if not all_results:
            return {}

        aggregate_by_reduction = {}

        # Aggregate by reduction level
        for reduction_level in self.reduction_levels:
            metrics_list = []

            for trader_id, trader_results in all_results.items():
                if reduction_level in trader_results['reduction_level_results']:
                    metrics_list.append(trader_results['reduction_level_results'][reduction_level]['metrics'])

            if metrics_list:
                aggregate_by_reduction[reduction_level] = {
                    'total_actual_pnl': sum(m['actual_pnl'] for m in metrics_list),
                    'total_adjusted_pnl': sum(m['adjusted_pnl'] for m in metrics_list),
                    'total_net_benefit': sum(m['net_benefit'] for m in metrics_list),
                    'total_avoided_losses': sum(m['avoided_losses'] for m in metrics_list),
                    'total_missed_gains': sum(m['missed_gains'] for m in metrics_list),
                    'mean_intervention_rate': np.mean([m['intervention_rate'] for m in metrics_list]),
                    'mean_improvement_pct': np.mean([m['improvement_pct'] for m in metrics_list]),
                    'mean_sharpe_improvement': np.mean([m['sharpe_improvement'] for m in metrics_list]),
                    'positive_improvements': sum(1 for m in metrics_list if m['net_benefit'] > 0),
                    'total_traders': len(metrics_list)
                }

                # Calculate overall improvement percentage
                total_actual = aggregate_by_reduction[reduction_level]['total_actual_pnl']
                total_adjusted = aggregate_by_reduction[reduction_level]['total_adjusted_pnl']

                if total_actual != 0:
                    aggregate_by_reduction[reduction_level]['overall_improvement_pct'] = (
                        (total_adjusted - total_actual) / abs(total_actual) * 100
                    )
                else:
                    aggregate_by_reduction[reduction_level]['overall_improvement_pct'] = 0

        return aggregate_by_reduction

    def find_optimal_reduction_level(self, aggregate_results: Dict[str, Any]) -> Dict[str, Any]:
        """Find the optimal reduction level based on various metrics."""
        if not aggregate_results:
            return {}

        best_by_metric = {}

        # Find best reduction level by different metrics
        metrics_to_optimize = [
            ('total_net_benefit', 'max'),
            ('overall_improvement_pct', 'max'),
            ('mean_sharpe_improvement', 'max'),
            ('positive_improvements', 'max')
        ]

        for metric, direction in metrics_to_optimize:
            values = [(level, results[metric]) for level, results in aggregate_results.items()]

            if direction == 'max':
                optimal_level, optimal_value = max(values, key=lambda x: x[1])
            else:
                optimal_level, optimal_value = min(values, key=lambda x: x[1])

            best_by_metric[metric] = {
                'optimal_reduction_level': optimal_level,
                'optimal_value': optimal_value
            }

        # Calculate overall recommendation based on weighted score
        recommendation_scores = {}
        for level in self.reduction_levels:
            if level in aggregate_results:
                # Weighted score: 40% net benefit, 30% improvement %, 20% sharpe, 10% positive count
                score = (
                    0.4 * self.normalize_metric(aggregate_results[level]['total_net_benefit'],
                                               [r['total_net_benefit'] for r in aggregate_results.values()]) +
                    0.3 * self.normalize_metric(aggregate_results[level]['overall_improvement_pct'],
                                               [r['overall_improvement_pct'] for r in aggregate_results.values()]) +
                    0.2 * self.normalize_metric(aggregate_results[level]['mean_sharpe_improvement'],
                                               [r['mean_sharpe_improvement'] for r in aggregate_results.values()]) +
                    0.1 * self.normalize_metric(aggregate_results[level]['positive_improvements'],
                                               [r['positive_improvements'] for r in aggregate_results.values()])
                )
                recommendation_scores[level] = score

        if recommendation_scores:
            overall_optimal = max(recommendation_scores.items(), key=lambda x: x[1])

            return {
                'best_by_metric': best_by_metric,
                'overall_recommendation': {
                    'optimal_reduction_level': overall_optimal[0],
                    'score': overall_optimal[1]
                },
                'all_scores': recommendation_scores
            }

        return {}

    def normalize_metric(self, value: float, all_values: List[float]) -> float:
        """Normalize a metric value to 0-1 scale."""
        if not all_values or max(all_values) == min(all_values):
            return 0.5
        return (value - min(all_values)) / (max(all_values) - min(all_values))

    def create_pnl_comparison_plot(self, trader_id: str, save_path: str = None) -> plt.Figure:
        """Create PnL comparison plot for a specific trader."""
        if trader_id not in self.results:
            raise ValueError(f"No results found for trader {trader_id}")

        daily_results = self.results[trader_id]['daily_results']

        # Calculate cumulative PnL
        daily_results['cumulative_actual'] = daily_results['daily_pnl'].cumsum()
        daily_results['cumulative_adjusted'] = daily_results['adjusted_pnl'].cumsum()

        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plot 1: Daily PnL comparison
        ax1.plot(daily_results['date'], daily_results['daily_pnl'],
                label='Actual Daily PnL', alpha=0.7, color='blue')
        ax1.plot(daily_results['date'], daily_results['adjusted_pnl'],
                label='Model-Based Daily PnL', alpha=0.7, color='red')

        # Highlight intervention days
        intervention_days = daily_results[daily_results['should_intervene']]
        ax1.scatter(intervention_days['date'], intervention_days['daily_pnl'],
                   color='orange', s=20, alpha=0.6, label='Intervention Days')

        ax1.set_title(f'Trader {trader_id}: Daily PnL Comparison')
        ax1.set_ylabel('Daily PnL')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Cumulative PnL comparison
        ax2.plot(daily_results['date'], daily_results['cumulative_actual'],
                label='Actual Cumulative PnL', linewidth=2, color='blue')
        ax2.plot(daily_results['date'], daily_results['cumulative_adjusted'],
                label='Model-Based Cumulative PnL', linewidth=2, color='red')

        ax2.set_title(f'Trader {trader_id}: Cumulative PnL Comparison')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Cumulative PnL')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def create_summary_dashboard(self, save_path: str = None) -> plt.Figure:
        """Create summary dashboard showing results across all traders."""
        if not self.results:
            raise ValueError("No evaluation results available")

        # Prepare data for visualization
        metrics_df = pd.DataFrame([result['metrics'] for result in self.results.values()])

        # Create dashboard
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: Net Benefit by Trader
        axes[0, 0].bar(metrics_df['trader_id'], metrics_df['net_benefit'])
        axes[0, 0].set_title('Net Benefit by Trader')
        axes[0, 0].set_ylabel('Net Benefit ($)')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Plot 2: Avoided Losses vs Missed Gains
        x = np.arange(len(metrics_df))
        width = 0.35
        axes[0, 1].bar(x - width/2, metrics_df['avoided_losses'], width,
                      label='Avoided Losses', color='green', alpha=0.7)
        axes[0, 1].bar(x + width/2, metrics_df['missed_gains'], width,
                      label='Missed Gains', color='red', alpha=0.7)
        axes[0, 1].set_title('Avoided Losses vs Missed Gains')
        axes[0, 1].set_ylabel('Amount ($)')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(metrics_df['trader_id'])
        axes[0, 1].legend()
        axes[0, 1].tick_params(axis='x', rotation=45)

        # Plot 3: Intervention Rate by Trader
        axes[1, 0].bar(metrics_df['trader_id'], metrics_df['intervention_rate'] * 100)
        axes[1, 0].set_title('Intervention Rate by Trader')
        axes[1, 0].set_ylabel('Intervention Rate (%)')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # Plot 4: Improvement Percentage by Trader
        colors = ['green' if x > 0 else 'red' for x in metrics_df['improvement_pct']]
        axes[1, 1].bar(metrics_df['trader_id'], metrics_df['improvement_pct'], color=colors, alpha=0.7)
        axes[1, 1].set_title('Performance Improvement by Trader')
        axes[1, 1].set_ylabel('Improvement (%)')
        axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def generate_report(self) -> str:
        """Generate a comprehensive text report of the evaluation results."""
        if not self.results:
            return "No evaluation results available."

        aggregate = self.calculate_aggregate_metrics()

        report = []
        report.append("="*80)
        report.append("CAUSAL IMPACT EVALUATION REPORT")
        report.append("="*80)
        report.append("")

        # Aggregate Summary
        report.append("AGGREGATE SUMMARY")
        report.append("-"*40)
        report.append(f"Total Traders Evaluated: {aggregate['total_traders']}")
        report.append(f"Total Actual PnL: ${aggregate['total_actual_pnl']:,.2f}")
        report.append(f"Total Model-Based PnL: ${aggregate['total_adjusted_pnl']:,.2f}")
        report.append(f"Total Net Benefit: ${aggregate['total_net_benefit']:,.2f}")
        report.append(f"Overall Improvement: {aggregate['overall_improvement_pct']:.2f}%")
        report.append(f"Total Avoided Losses: ${aggregate['total_avoided_losses']:,.2f}")
        report.append(f"Total Missed Gains: ${aggregate['total_missed_gains']:,.2f}")
        report.append(f"Mean Intervention Rate: {aggregate['mean_intervention_rate']*100:.1f}%")
        report.append(f"Positive Improvements: {aggregate['positive_improvements']}/{aggregate['total_traders']}")
        report.append("")

        # Individual Trader Results
        report.append("INDIVIDUAL TRADER RESULTS")
        report.append("-"*40)

        for trader_id, result in self.results.items():
            metrics = result['metrics']
            report.append(f"\nTrader {trader_id}:")
            report.append(f"  Actual PnL: ${metrics['actual_pnl']:,.2f}")
            report.append(f"  Model-Based PnL: ${metrics['adjusted_pnl']:,.2f}")
            report.append(f"  Net Benefit: ${metrics['net_benefit']:,.2f}")
            report.append(f"  Improvement: {metrics['improvement_pct']:.2f}%")
            report.append(f"  Avoided Losses: ${metrics['avoided_losses']:,.2f}")
            report.append(f"  Missed Gains: ${metrics['missed_gains']:,.2f}")
            report.append(f"  Intervention Rate: {metrics['intervention_rate']*100:.1f}%")
            report.append(f"  Days Evaluated: {metrics['total_days']}")

        return "\n".join(report)


def main():
    """Main execution function for causal impact evaluation."""
    # Initialize evaluator
    evaluator = CausalImpactEvaluator()

    # Create output directory
    output_dir = Path("/Users/temurbekkhujaev/Repos/risk-tool/results/causal_impact_evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Starting enhanced causal impact evaluation...")

    # Test 1: Original system (50% reduction, binary thresholds)
    print("\n=== Testing Original System (50% reduction, binary thresholds) ===")
    original_results = evaluator.evaluate_all_traders(reduction_level=0.5, use_weighted_formula=False)

    # Test 2: Multilevel evaluation with binary system
    print("\n=== Testing Multiple Reduction Levels (0%, 10%, 30%, 60%) with Binary System ===")
    binary_multilevel_results = evaluator.evaluate_all_traders_multilevel(use_weighted_formula=False)

    # Test 3: Multilevel evaluation with weighted formula
    print("\n=== Testing Multiple Reduction Levels with Weighted Formula (α=0.6, β=0.4) ===")
    weighted_multilevel_results = evaluator.evaluate_all_traders_multilevel(
        use_weighted_formula=True, alpha=0.6, beta=0.4
    )

    # Save results
    with open(output_dir / "original_system_results.pkl", 'wb') as f:
        pickle.dump(original_results, f)

    with open(output_dir / "binary_multilevel_results.pkl", 'wb') as f:
        pickle.dump(binary_multilevel_results, f)

    with open(output_dir / "weighted_multilevel_results.pkl", 'wb') as f:
        pickle.dump(weighted_multilevel_results, f)

    # Generate reports
    original_report = evaluator.generate_report()
    with open(output_dir / "original_system_report.txt", 'w') as f:
        f.write(original_report)

    # Generate multilevel reports
    binary_report = evaluator.generate_multilevel_report(binary_multilevel_results)
    with open(output_dir / "binary_multilevel_report.txt", 'w') as f:
        f.write(binary_report)

    weighted_report = evaluator.generate_multilevel_report(weighted_multilevel_results)
    with open(output_dir / "weighted_multilevel_report.txt", 'w') as f:
        f.write(weighted_report)

    # Create summary dashboard (using original results for compatibility)
    dashboard_fig = evaluator.create_summary_dashboard(str(output_dir / "original_system_dashboard.png"))
    plt.close(dashboard_fig)

    # Create individual trader plots
    trader_plots_dir = output_dir / "trader_plots"
    trader_plots_dir.mkdir(exist_ok=True)

    for trader_id in evaluator.results.keys():
        plot_fig = evaluator.create_pnl_comparison_plot(trader_id,
                                                       str(trader_plots_dir / f"trader_{trader_id}_pnl_comparison.png"))
        plt.close(plot_fig)

    print(f"\nEvaluation complete! Results saved to: {output_dir}")
    print("\nORIGINAL SYSTEM SUMMARY:")
    print(original_report.split("INDIVIDUAL TRADER RESULTS")[0])

    # Print optimal reduction level findings
    if binary_multilevel_results.get('aggregate_results'):
        binary_optimal = evaluator.find_optimal_reduction_level(binary_multilevel_results['aggregate_results'])
        if binary_optimal:
            print(f"\nBINARY SYSTEM OPTIMAL REDUCTION: {binary_optimal['overall_recommendation']['optimal_reduction_level']*100:.0f}%")

    if weighted_multilevel_results.get('aggregate_results'):
        weighted_optimal = evaluator.find_optimal_reduction_level(weighted_multilevel_results['aggregate_results'])
        if weighted_optimal:
            print(f"WEIGHTED FORMULA OPTIMAL REDUCTION: {weighted_optimal['overall_recommendation']['optimal_reduction_level']*100:.0f}%")

    return {
        'original_results': original_results,
        'binary_multilevel_results': binary_multilevel_results,
        'weighted_multilevel_results': weighted_multilevel_results
    }

def generate_multilevel_report(self, results: Dict[str, Any]) -> str:
    """Generate comprehensive report for multilevel evaluation."""
    if not results:
        return "No evaluation results available."

    report = []
    report.append("="*80)
    report.append("MULTI-LEVEL CAUSAL IMPACT EVALUATION REPORT")
    report.append("="*80)
    report.append("")

    # Summary
    summary = results['evaluation_summary']
    report.append("EVALUATION SUMMARY")
    report.append("-"*40)
    report.append(f"Total Traders Evaluated: {summary['successful_evaluations']}")
    report.append(f"Reduction Levels Tested: {[f'{level*100:.0f}%' for level in summary['reduction_levels_tested']]}")
    report.append(f"Used Weighted Formula: {summary['used_weighted_formula']}")
    if summary['used_weighted_formula']:
        report.append(f"Alpha (VaR Weight): {summary['alpha']}")
        report.append(f"Beta (LossProb Weight): {summary['beta']}")
    report.append("")

    # Aggregate results by reduction level
    aggregate_results = results['aggregate_results']
    report.append("AGGREGATE RESULTS BY REDUCTION LEVEL")
    report.append("-"*40)

    for level, metrics in aggregate_results.items():
        report.append(f"\n{level*100:.0f}% PnL Reduction:")
        report.append(f"  Total Actual PnL: ${metrics['total_actual_pnl']:,.2f}")
        report.append(f"  Total Adjusted PnL: ${metrics['total_adjusted_pnl']:,.2f}")
        report.append(f"  Total Net Benefit: ${metrics['total_net_benefit']:,.2f}")
        report.append(f"  Overall Improvement: {metrics['overall_improvement_pct']:.2f}%")
        report.append(f"  Mean Intervention Rate: {metrics['mean_intervention_rate']*100:.1f}%")
        report.append(f"  Mean Sharpe Improvement: {metrics['mean_sharpe_improvement']:.3f}")
        report.append(f"  Positive Improvements: {metrics['positive_improvements']}/{metrics['total_traders']}")

    # Optimal reduction level analysis
    optimal_analysis = self.find_optimal_reduction_level(aggregate_results)
    if optimal_analysis:
        report.append("\n")
        report.append("OPTIMAL REDUCTION LEVEL ANALYSIS")
        report.append("-"*40)

        overall_rec = optimal_analysis['overall_recommendation']
        report.append(f"Overall Recommendation: {overall_rec['optimal_reduction_level']*100:.0f}% reduction")
        report.append(f"Recommendation Score: {overall_rec['score']:.3f}")
        report.append("")

        report.append("Best by Individual Metrics:")
        for metric, info in optimal_analysis['best_by_metric'].items():
            report.append(f"  {metric}: {info['optimal_reduction_level']*100:.0f}% (value: {info['optimal_value']:.2f})")

    return "\n".join(report)

# Add method to the class
CausalImpactEvaluator.generate_multilevel_report = generate_multilevel_report


if __name__ == "__main__":
    results = main()
