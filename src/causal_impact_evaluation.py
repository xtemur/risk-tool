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

from src.utils import load_model


class CausalImpactEvaluator:
    """
    Evaluates the causal impact of model-based trading interventions on unseen test data.

    When model thresholds indicate 'don't trade', actual PnL is multiplied by 0.5:
    - Negative PnL: avoid half the loss
    - Positive PnL: miss half the gain
    """

    def __init__(self, base_path: str = "/Users/temurbekkhujaev/Repos/risk-tool"):
        self.base_path = Path(base_path)
        self.models_path = self.base_path / "models" / "trader_specific_80pct"
        self.data_path = self.base_path / "data" / "processed" / "trader_splits"
        self.thresholds_path = self.base_path / "configs" / "optimal_thresholds" / "optimal_thresholds.json"

        # Load optimal thresholds
        with open(self.thresholds_path, 'r') as f:
            thresholds_data = json.load(f)

        self.thresholds = {
            str(thresh["trader_id"]): {
                "var_threshold": thresh["var_threshold"],
                "loss_prob_threshold": thresh["loss_prob_threshold"]
            }
            for thresh in thresholds_data["thresholds"]
        }

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
        """Generate VaR and loss probability predictions for test data."""
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

    def apply_intervention_logic(self, predictions_df: pd.DataFrame, trader_id: str) -> pd.DataFrame:
        """Apply intervention logic based on model thresholds."""
        trader_thresholds = self.thresholds[trader_id]
        var_threshold = trader_thresholds["var_threshold"]
        loss_prob_threshold = trader_thresholds["loss_prob_threshold"]

        # Determine intervention decisions
        # Intervene if VaR prediction is below threshold OR loss probability is above threshold
        should_intervene = (
            (predictions_df['var_prediction'] <= var_threshold) |
            (predictions_df['loss_probability'] >= loss_prob_threshold)
        )

        # Calculate adjusted PnL
        # If we intervene (don't trade), multiply actual PnL by 0.5
        adjusted_pnl = np.where(should_intervene,
                               predictions_df['daily_pnl'] * 0.5,
                               predictions_df['daily_pnl'])

        # Add results to dataframe
        results_df = predictions_df.copy()
        results_df['should_intervene'] = should_intervene
        results_df['adjusted_pnl'] = adjusted_pnl
        results_df['var_threshold'] = var_threshold
        results_df['loss_prob_threshold'] = loss_prob_threshold

        return results_df

    def calculate_performance_metrics(self, results_df: pd.DataFrame, trader_id: str) -> Dict[str, float]:
        """Calculate performance metrics for a trader."""
        actual_pnl = results_df['daily_pnl'].sum()
        adjusted_pnl = results_df['adjusted_pnl'].sum()

        # Calculate avoided losses and missed gains
        intervention_days = results_df[results_df['should_intervene']]

        # Avoided losses: days where we intervened and actual PnL was negative
        avoided_losses = intervention_days[intervention_days['daily_pnl'] < 0]['daily_pnl'].sum() * 0.5

        # Missed gains: days where we intervened and actual PnL was positive
        missed_gains = intervention_days[intervention_days['daily_pnl'] > 0]['daily_pnl'].sum() * 0.5

        # Intervention rate
        intervention_rate = results_df['should_intervene'].mean()

        # Net benefit
        net_benefit = adjusted_pnl - actual_pnl

        # Performance improvement
        improvement_pct = ((adjusted_pnl - actual_pnl) / abs(actual_pnl) * 100) if actual_pnl != 0 else 0

        return {
            'trader_id': trader_id,
            'actual_pnl': actual_pnl,
            'adjusted_pnl': adjusted_pnl,
            'net_benefit': net_benefit,
            'improvement_pct': improvement_pct,
            'avoided_losses': abs(avoided_losses),  # Make positive for clarity
            'missed_gains': abs(missed_gains),  # Make positive for clarity
            'intervention_rate': intervention_rate,
            'total_days': len(results_df),
            'intervention_days': intervention_days.shape[0]
        }

    def evaluate_trader(self, trader_id: str) -> Dict[str, Any]:
        """Evaluate a single trader and return comprehensive results."""
        print(f"Evaluating trader {trader_id}...")

        # Load data and model
        test_data, model_data = self.load_trader_data(trader_id)

        # Generate predictions
        predictions_df = self.generate_predictions(test_data, model_data)

        # Apply intervention logic
        results_df = self.apply_intervention_logic(predictions_df, trader_id)

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

    def evaluate_all_traders(self) -> Dict[str, Any]:
        """Evaluate all traders with available models."""
        # Get list of available traders
        available_traders = [f.stem.split('_')[0] for f in self.models_path.glob("*_tuned_validated.pkl")]
        available_traders = [t for t in available_traders if t in self.thresholds.keys()]

        print(f"Found {len(available_traders)} traders to evaluate: {available_traders}")

        # Evaluate each trader
        for trader_id in available_traders:
            try:
                self.evaluate_trader(trader_id)
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
                'evaluation_date': pd.Timestamp.now().isoformat()
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

    # Run evaluation for all traders
    print("Starting causal impact evaluation...")
    results = evaluator.evaluate_all_traders()

    # Create output directory
    output_dir = Path("/Users/temurbekkhujaev/Repos/risk-tool/results/causal_impact_evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate and save report
    report = evaluator.generate_report()
    with open(output_dir / "evaluation_report.txt", 'w') as f:
        f.write(report)

    # Create summary dashboard
    dashboard_fig = evaluator.create_summary_dashboard(str(output_dir / "summary_dashboard.png"))
    plt.close(dashboard_fig)

    # Create individual trader plots
    trader_plots_dir = output_dir / "trader_plots"
    trader_plots_dir.mkdir(exist_ok=True)

    for trader_id in evaluator.results.keys():
        plot_fig = evaluator.create_pnl_comparison_plot(trader_id,
                                                       str(trader_plots_dir / f"trader_{trader_id}_pnl_comparison.png"))
        plt.close(plot_fig)

    # Save detailed results
    with open(output_dir / "detailed_results.pkl", 'wb') as f:
        pickle.dump(results, f)

    print(f"\nEvaluation complete! Results saved to: {output_dir}")
    print("\nSUMMARY:")
    print(report.split("INDIVIDUAL TRADER RESULTS")[0])

    return results


if __name__ == "__main__":
    results = main()
