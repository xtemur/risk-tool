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


class AdjustedPnLCausalImpactEvaluator:
    """
    Evaluates causal impact where interventions adjust PnL values, which then affect subsequent predictions.

    When model indicates 'don't trade', the actual PnL is adjusted by 0.5, and this adjusted value
    is used for all subsequent feature calculations and predictions.
    """

    def __init__(self, base_path: str = "/Users/temurbekkhujaev/Repos/risk-tool", pnl_reduction_pct: float = 50.0):
        self.base_path = Path(base_path)
        self.models_path = self.base_path / "models" / "trader_specific"
        self.data_path = self.base_path / "data" / "processed" / "trader_splits"
        self.thresholds_path = self.base_path / "configs" / "optimal_thresholds" / "optimal_thresholds.json"
        self.pnl_reduction_pct = pnl_reduction_pct  # Percentage reduction when intervening

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

    def recalculate_features(self, df: pd.DataFrame, current_idx: int) -> pd.Series:
        """
        Recalculate features for current day based on updated historical data.
        Uses the same feature engineering logic as the original model training.
        """
        # Get data up to current index (excluding current day)
        historical_data = df.iloc[:current_idx].copy()

        if len(historical_data) == 0:
            # For first day, return existing features
            return df.iloc[current_idx].copy()

        # Initialize features with current day base values
        features = df.iloc[current_idx].copy()

        # Update rolling statistics based on adjusted PnL history
        if len(historical_data) >= 7:
            # 7-day rolling features
            recent_7d = historical_data.tail(7)
            features['pnl_mean_7d'] = recent_7d['daily_pnl'].mean()
            features['pnl_std_7d'] = recent_7d['daily_pnl'].std()
            features['volume_mean_7d'] = recent_7d['daily_volume'].mean()
            features['trades_mean_7d'] = recent_7d['n_trades'].mean()

        if len(historical_data) >= 21:
            # 21-day rolling features
            recent_21d = historical_data.tail(21)
            features['pnl_mean_21d'] = recent_21d['daily_pnl'].mean()
            features['pnl_std_21d'] = recent_21d['daily_pnl'].std()
            features['volume_mean_21d'] = recent_21d['daily_volume'].mean()
            features['trades_mean_21d'] = recent_21d['n_trades'].mean()

        # Update cumulative features
        features['cumulative_pnl'] = historical_data['daily_pnl'].sum()
        features['cumulative_volume'] = historical_data['daily_volume'].sum()
        features['cumulative_trades'] = historical_data['n_trades'].sum()

        # Update lag features
        if len(historical_data) >= 1:
            features['pnl_lag_1'] = historical_data.iloc[-1]['daily_pnl']
            features['volume_lag_1'] = historical_data.iloc[-1]['daily_volume']
            features['trades_lag_1'] = historical_data.iloc[-1]['n_trades']

        if len(historical_data) >= 2:
            features['pnl_lag_2'] = historical_data.iloc[-2]['daily_pnl']
            features['volume_lag_2'] = historical_data.iloc[-2]['daily_volume']
            features['trades_lag_2'] = historical_data.iloc[-2]['n_trades']

        if len(historical_data) >= 3:
            features['pnl_lag_3'] = historical_data.iloc[-3]['daily_pnl']
            features['volume_lag_3'] = historical_data.iloc[-3]['daily_volume']
            features['trades_lag_3'] = historical_data.iloc[-3]['n_trades']

        # Update derived features
        if features['daily_volume'] != 0:
            features['pnl_to_volume'] = features['daily_pnl'] / features['daily_volume']

        if features['daily_fees'] != 0:
            features['gross_to_fees'] = features['daily_gross'] / features['daily_fees']

        return features

    def generate_sequential_predictions(self, test_data: pd.DataFrame, model_data: Dict[str, Any], trader_id: str) -> pd.DataFrame:
        """
        Generate predictions sequentially, adjusting PnL when interventions occur and using
        adjusted values for subsequent feature calculations.
        """
        # Create a copy of test data for adjustments
        adjusted_data = test_data.copy()

        # Get model components
        feature_names = model_data['feature_names']
        var_model = model_data['var_model']
        classification_model = model_data['classification_model']

        # Get thresholds
        trader_thresholds = self.thresholds[trader_id]
        var_threshold = trader_thresholds["var_threshold"]
        loss_prob_threshold = trader_thresholds["loss_prob_threshold"]

        # Results storage
        results = []

        # Process each day sequentially
        for current_idx in range(len(test_data)):
            current_day = test_data.iloc[current_idx]

            # Recalculate features based on adjusted history
            if current_idx > 0:
                updated_features = self.recalculate_features(adjusted_data, current_idx)
                # Update the adjusted data with recalculated features
                for feature in feature_names:
                    if feature in updated_features:
                        adjusted_data.iloc[current_idx, adjusted_data.columns.get_loc(feature)] = updated_features[feature]

            # Get features for prediction
            current_features = adjusted_data.iloc[current_idx]
            X_current = current_features[feature_names].values.reshape(1, -1)

            # Generate predictions
            var_prediction = var_model.predict(X_current)[0]
            loss_probability = classification_model.predict_proba(X_current)[0, 1]

            # Determine if intervention should occur
            should_intervene = (
                (var_prediction <= var_threshold) |
                (loss_probability >= loss_prob_threshold)
            )

            # Calculate adjusted PnL
            actual_pnl = current_day['daily_pnl']
            if should_intervene:
                reduction_factor = (100 - self.pnl_reduction_pct) / 100  # Convert percentage to multiplier
                adjusted_pnl = actual_pnl * reduction_factor
                # Update the adjusted data for future calculations
                adjusted_data.iloc[current_idx, adjusted_data.columns.get_loc('daily_pnl')] = adjusted_pnl
            else:
                adjusted_pnl = actual_pnl

            # Store results
            results.append({
                'trader_id': trader_id,
                'date': current_day['date'],
                'day_index': current_idx,
                'actual_pnl': actual_pnl,
                'adjusted_pnl': adjusted_pnl,
                'var_prediction': var_prediction,
                'loss_probability': loss_probability,
                'should_intervene': should_intervene,
                'var_threshold': var_threshold,
                'loss_prob_threshold': loss_prob_threshold,
                'intervention_effect': adjusted_pnl - actual_pnl
            })

        return pd.DataFrame(results)

    def calculate_performance_metrics(self, results_df: pd.DataFrame, trader_id: str) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        total_actual_pnl = results_df['actual_pnl'].sum()
        total_adjusted_pnl = results_df['adjusted_pnl'].sum()

        # Intervention analysis
        intervention_days = results_df[results_df['should_intervene']]
        no_intervention_days = results_df[~results_df['should_intervene']]

        # Avoided losses and missed gains
        avoided_losses = intervention_days[intervention_days['actual_pnl'] < 0]['intervention_effect'].sum()
        missed_gains = abs(intervention_days[intervention_days['actual_pnl'] > 0]['intervention_effect'].sum())

        # Intervention statistics
        intervention_rate = results_df['should_intervene'].mean()
        total_intervention_effect = results_df['intervention_effect'].sum()

        # Performance improvement
        net_benefit = total_adjusted_pnl - total_actual_pnl
        improvement_pct = ((net_benefit) / abs(total_actual_pnl) * 100) if total_actual_pnl != 0 else 0

        # Sequential impact analysis
        cumulative_actual = results_df['actual_pnl'].cumsum()
        cumulative_adjusted = results_df['adjusted_pnl'].cumsum()
        final_difference = cumulative_adjusted.iloc[-1] - cumulative_actual.iloc[-1]

        # Volatility impact
        actual_volatility = results_df['actual_pnl'].std()
        adjusted_volatility = results_df['adjusted_pnl'].std()
        volatility_reduction = ((actual_volatility - adjusted_volatility) / actual_volatility * 100) if actual_volatility != 0 else 0

        return {
            'trader_id': trader_id,
            'total_actual_pnl': total_actual_pnl,
            'total_adjusted_pnl': total_adjusted_pnl,
            'net_benefit': net_benefit,
            'improvement_pct': improvement_pct,
            'avoided_losses': abs(avoided_losses),
            'missed_gains': missed_gains,
            'intervention_rate': intervention_rate,
            'total_days': len(results_df),
            'intervention_days': len(intervention_days),
            'total_intervention_effect': total_intervention_effect,
            'final_difference': final_difference,
            'actual_volatility': actual_volatility,
            'adjusted_volatility': adjusted_volatility,
            'volatility_reduction_pct': volatility_reduction,
            'avg_daily_actual': results_df['actual_pnl'].mean(),
            'avg_daily_adjusted': results_df['adjusted_pnl'].mean()
        }

    def evaluate_trader(self, trader_id: str) -> Dict[str, Any]:
        """Evaluate a single trader with sequential adjustment logic."""
        print(f"Evaluating trader {trader_id} with sequential PnL adjustments...")

        # Load data and model
        test_data, model_data = self.load_trader_data(trader_id)

        # Generate sequential predictions with adjustments
        results_df = self.generate_sequential_predictions(test_data, model_data, trader_id)

        # Calculate metrics
        metrics = self.calculate_performance_metrics(results_df, trader_id)

        # Store results
        trader_results = {
            'metrics': metrics,
            'daily_results': results_df,
            'model_info': {
                'feature_names': model_data.get('feature_names', []),
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
                'evaluation_date': pd.Timestamp.now().isoformat(),
                'method': 'sequential_pnl_adjustment'
            }
        }

    def calculate_aggregate_metrics(self) -> Dict[str, float]:
        """Calculate aggregate metrics across all evaluated traders."""
        if not self.results:
            return {}

        metrics_list = [result['metrics'] for result in self.results.values()]

        aggregate = {
            'total_actual_pnl': sum(m['total_actual_pnl'] for m in metrics_list),
            'total_adjusted_pnl': sum(m['total_adjusted_pnl'] for m in metrics_list),
            'total_net_benefit': sum(m['net_benefit'] for m in metrics_list),
            'total_avoided_losses': sum(m['avoided_losses'] for m in metrics_list),
            'total_missed_gains': sum(m['missed_gains'] for m in metrics_list),
            'mean_intervention_rate': np.mean([m['intervention_rate'] for m in metrics_list]),
            'mean_improvement_pct': np.mean([m['improvement_pct'] for m in metrics_list]),
            'positive_improvements': sum(1 for m in metrics_list if m['net_benefit'] > 0),
            'total_traders': len(metrics_list),
            'mean_volatility_reduction': np.mean([m['volatility_reduction_pct'] for m in metrics_list]),
            'total_intervention_effect': sum(m['total_intervention_effect'] for m in metrics_list)
        }

        # Calculate overall improvement percentage
        if aggregate['total_actual_pnl'] != 0:
            aggregate['overall_improvement_pct'] = (
                aggregate['total_net_benefit'] / abs(aggregate['total_actual_pnl']) * 100
            )
        else:
            aggregate['overall_improvement_pct'] = 0

        return aggregate

    def create_sequential_comparison_plot(self, trader_id: str, save_path: str = None) -> plt.Figure:
        """Create comparison plot showing sequential impact of adjustments."""
        if trader_id not in self.results:
            raise ValueError(f"No results found for trader {trader_id}")

        daily_results = self.results[trader_id]['daily_results']

        # Calculate cumulative effects
        daily_results['cumulative_actual'] = daily_results['actual_pnl'].cumsum()
        daily_results['cumulative_adjusted'] = daily_results['adjusted_pnl'].cumsum()
        daily_results['cumulative_effect'] = daily_results['intervention_effect'].cumsum()

        # Create plot
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))

        # Plot 1: Daily PnL comparison with interventions
        axes[0].plot(daily_results['date'], daily_results['actual_pnl'],
                    label='Original Daily PnL', alpha=0.8, color='blue', linewidth=1.5)
        axes[0].plot(daily_results['date'], daily_results['adjusted_pnl'],
                    label='Adjusted Daily PnL', alpha=0.8, color='red', linewidth=1.5)

        # Highlight intervention days
        intervention_days = daily_results[daily_results['should_intervene']]
        axes[0].scatter(intervention_days['date'], intervention_days['actual_pnl'],
                       color='orange', s=25, alpha=0.7, label='Intervention Days', zorder=5)

        axes[0].set_title(f'Trader {trader_id}: Sequential PnL Adjustment Impact')
        axes[0].set_ylabel('Daily PnL ($)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Cumulative PnL comparison
        axes[1].plot(daily_results['date'], daily_results['cumulative_actual'],
                    label='Original Cumulative PnL', linewidth=2.5, color='blue')
        axes[1].plot(daily_results['date'], daily_results['cumulative_adjusted'],
                    label='Adjusted Cumulative PnL', linewidth=2.5, color='red')

        axes[1].set_title(f'Trader {trader_id}: Cumulative PnL Evolution')
        axes[1].set_ylabel('Cumulative PnL ($)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Plot 3: Cumulative intervention effects
        axes[2].plot(daily_results['date'], daily_results['cumulative_effect'],
                    label='Cumulative Intervention Effect', linewidth=2, color='green')
        axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.5)

        axes[2].set_title(f'Trader {trader_id}: Cumulative Intervention Impact')
        axes[2].set_xlabel('Date')
        axes[2].set_ylabel('Cumulative Effect ($)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def create_comprehensive_dashboard(self, save_path: str = None) -> plt.Figure:
        """Create comprehensive dashboard for sequential adjustment results."""
        if not self.results:
            raise ValueError("No evaluation results available")

        metrics_df = pd.DataFrame([result['metrics'] for result in self.results.values()])

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Plot 1: Net Benefit by Trader
        colors = ['green' if x > 0 else 'red' for x in metrics_df['net_benefit']]
        axes[0, 0].bar(metrics_df['trader_id'], metrics_df['net_benefit'], color=colors, alpha=0.7)
        axes[0, 0].set_title('Net Benefit by Trader\n(Sequential Adjustment)')
        axes[0, 0].set_ylabel('Net Benefit ($)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)

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

        # Plot 3: Intervention Rate vs Improvement
        axes[0, 2].scatter(metrics_df['intervention_rate'] * 100, metrics_df['improvement_pct'],
                          s=80, alpha=0.7, c=metrics_df['net_benefit'], cmap='RdYlGn')
        axes[0, 2].set_xlabel('Intervention Rate (%)')
        axes[0, 2].set_ylabel('Improvement (%)')
        axes[0, 2].set_title('Intervention Rate vs Performance')
        axes[0, 2].grid(True, alpha=0.3)

        # Plot 4: Volatility Reduction
        axes[1, 0].bar(metrics_df['trader_id'], metrics_df['volatility_reduction_pct'], alpha=0.7)
        axes[1, 0].set_title('Volatility Reduction by Trader')
        axes[1, 0].set_ylabel('Volatility Reduction (%)')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # Plot 5: Cumulative effects over time (sample trader)
        sample_trader = list(self.results.keys())[0]
        sample_results = self.results[sample_trader]['daily_results']
        sample_results['cumulative_effect'] = sample_results['intervention_effect'].cumsum()

        axes[1, 1].plot(sample_results['date'], sample_results['cumulative_effect'],
                       linewidth=2, color='purple')
        axes[1, 1].set_title(f'Sequential Impact Example\n(Trader {sample_trader})')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Cumulative Effect ($)')
        axes[1, 1].grid(True, alpha=0.3)

        # Plot 6: Before vs After PnL Distribution
        all_actual = []
        all_adjusted = []
        for result in self.results.values():
            all_actual.extend(result['daily_results']['actual_pnl'].tolist())
            all_adjusted.extend(result['daily_results']['adjusted_pnl'].tolist())

        axes[1, 2].hist(all_actual, bins=30, alpha=0.6, label='Original PnL', color='blue', density=True)
        axes[1, 2].hist(all_adjusted, bins=30, alpha=0.6, label='Adjusted PnL', color='red', density=True)
        axes[1, 2].set_title('PnL Distribution Comparison')
        axes[1, 2].set_xlabel('Daily PnL ($)')
        axes[1, 2].set_ylabel('Density')
        axes[1, 2].legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive report of sequential adjustment evaluation."""
        if not self.results:
            return "No evaluation results available."

        aggregate = self.calculate_aggregate_metrics()

        report = []
        report.append("="*80)
        report.append("SEQUENTIAL PnL ADJUSTMENT CAUSAL IMPACT EVALUATION")
        report.append("="*80)
        report.append("")
        report.append("METHODOLOGY:")
        report.append(f"- When model indicates 'don't trade', actual PnL is reduced by {self.pnl_reduction_pct}%")
        report.append("- Adjusted PnL values are used for subsequent feature calculations")
        report.append("- Each prediction considers the cumulative effect of prior adjustments")
        report.append("")

        # Aggregate Summary
        report.append("AGGREGATE SUMMARY")
        report.append("-"*50)
        report.append(f"Total Traders Evaluated: {aggregate['total_traders']}")
        report.append(f"Total Original PnL: ${aggregate['total_actual_pnl']:,.2f}")
        report.append(f"Total Adjusted PnL: ${aggregate['total_adjusted_pnl']:,.2f}")
        report.append(f"Total Net Benefit: ${aggregate['total_net_benefit']:,.2f}")
        report.append(f"Overall Improvement: {aggregate['overall_improvement_pct']:.2f}%")
        report.append(f"Total Avoided Losses: ${aggregate['total_avoided_losses']:,.2f}")
        report.append(f"Total Missed Gains: ${aggregate['total_missed_gains']:,.2f}")
        report.append(f"Mean Intervention Rate: {aggregate['mean_intervention_rate']*100:.1f}%")
        report.append(f"Mean Volatility Reduction: {aggregate['mean_volatility_reduction']:.1f}%")
        report.append(f"Positive Improvements: {aggregate['positive_improvements']}/{aggregate['total_traders']}")
        report.append("")

        # Individual Results
        report.append("INDIVIDUAL TRADER RESULTS")
        report.append("-"*50)

        for trader_id, result in self.results.items():
            metrics = result['metrics']
            report.append(f"\nTrader {trader_id}:")
            report.append(f"  Original PnL: ${metrics['total_actual_pnl']:,.2f}")
            report.append(f"  Adjusted PnL: ${metrics['total_adjusted_pnl']:,.2f}")
            report.append(f"  Net Benefit: ${metrics['net_benefit']:,.2f}")
            report.append(f"  Improvement: {metrics['improvement_pct']:.2f}%")
            report.append(f"  Avoided Losses: ${metrics['avoided_losses']:,.2f}")
            report.append(f"  Missed Gains: ${metrics['missed_gains']:,.2f}")
            report.append(f"  Intervention Rate: {metrics['intervention_rate']*100:.1f}%")
            report.append(f"  Volatility Reduction: {metrics['volatility_reduction_pct']:.1f}%")
            report.append(f"  Days Evaluated: {metrics['total_days']}")

        return "\n".join(report)


def run_multiple_evaluations(reduction_percentages: List[float] = [25, 50, 70, 90]):
    """Run evaluations for multiple PnL reduction percentages and create comparison table."""

    all_results = {}
    comparison_data = []

    base_output_dir = Path("/Users/temurbekkhujaev/Repos/risk-tool/results/causal_impact_comparison")
    base_output_dir.mkdir(parents=True, exist_ok=True)

    for reduction_pct in reduction_percentages:
        print(f"\n{'='*60}")
        print(f"Running evaluation with {reduction_pct}% PnL reduction...")
        print(f"{'='*60}")

        # Initialize evaluator with specific reduction percentage
        evaluator = AdjustedPnLCausalImpactEvaluator(pnl_reduction_pct=reduction_pct)

        # Run evaluation
        results = evaluator.evaluate_all_traders()
        all_results[reduction_pct] = results

        # Create output directory for this reduction percentage
        output_dir = base_output_dir / f"reduction_{reduction_pct}pct"
        output_dir.mkdir(exist_ok=True)

        # Save detailed results
        with open(output_dir / "detailed_results.pkl", 'wb') as f:
            pickle.dump(results, f)

        # Generate report
        report = evaluator.generate_comprehensive_report()
        with open(output_dir / "evaluation_report.txt", 'w') as f:
            f.write(report)

        # Create comprehensive dashboard
        dashboard_path = output_dir / f"comprehensive_dashboard_{reduction_pct}pct.png"
        try:
            dashboard_fig = evaluator.create_comprehensive_dashboard(save_path=str(dashboard_path))
            plt.close(dashboard_fig)
        except Exception as e:
            print(f"Error creating dashboard for {reduction_pct}%: {e}")

        # Create individual trader plots
        trader_plots_dir = output_dir / "trader_plots"
        trader_plots_dir.mkdir(exist_ok=True)

        for trader_id in evaluator.results.keys():
            try:
                plot_path = trader_plots_dir / f"trader_{trader_id}_sequential_comparison.png"
                plot_fig = evaluator.create_sequential_comparison_plot(trader_id, save_path=str(plot_path))
                plt.close(plot_fig)
            except Exception as e:
                print(f"Error creating plot for trader {trader_id} at {reduction_pct}%: {e}")

        # Extract aggregate metrics for comparison
        aggregate = results['aggregate_results']
        comparison_data.append({
            'Reduction %': f"{reduction_pct}%",
            'Total Original PnL': aggregate['total_actual_pnl'],
            'Total Adjusted PnL': aggregate['total_adjusted_pnl'],
            'Net Benefit': aggregate['total_net_benefit'],
            'Overall Improvement %': aggregate['overall_improvement_pct'],
            'Avoided Losses': aggregate['total_avoided_losses'],
            'Missed Gains': aggregate['total_missed_gains'],
            'Mean Intervention Rate %': aggregate['mean_intervention_rate'] * 100,
            'Mean Volatility Reduction %': aggregate['mean_volatility_reduction'],
            'Positive Improvements': f"{aggregate['positive_improvements']}/{aggregate['total_traders']}"
        })

    # Create comparison table
    comparison_df = pd.DataFrame(comparison_data)

    # Save comparison table
    comparison_df.to_csv(base_output_dir / "comparison_table.csv", index=False)

    # Create comparison table with nice formatting
    create_formatted_comparison_table(comparison_df, base_output_dir)

    # Create comparison plots
    create_comparison_plots(comparison_df, base_output_dir)

    print(f"\n{'='*60}")
    print("COMPARISON TABLE - PnL Reduction Sensitivity Analysis")
    print(f"{'='*60}")
    print(comparison_df.to_string(index=False, float_format='%.2f'))
    print(f"\nAll results saved to: {base_output_dir}")

    return all_results, comparison_df

def create_formatted_comparison_table(comparison_df: pd.DataFrame, output_dir: Path):
    """Create a nicely formatted comparison table."""

    # Create formatted table text
    formatted_table = []
    formatted_table.append("="*120)
    formatted_table.append("PnL REDUCTION SENSITIVITY ANALYSIS - COMPARISON TABLE")
    formatted_table.append("="*120)
    formatted_table.append("")

    # Header
    header = f"{'Reduction %':<12} {'Net Benefit':<15} {'Improvement %':<15} {'Avoided Losses':<18} {'Missed Gains':<15} {'Intervention %':<15} {'Vol Reduction %':<15} {'Positive':<10}"
    formatted_table.append(header)
    formatted_table.append("-" * 120)

    # Data rows
    for _, row in comparison_df.iterrows():
        data_row = f"{row['Reduction %']:<12} ${row['Net Benefit']:<14,.0f} {row['Overall Improvement %']:<15.1f} ${row['Avoided Losses']:<17,.0f} ${row['Missed Gains']:<14,.0f} {row['Mean Intervention Rate %']:<15.1f} {row['Mean Volatility Reduction %']:<15.1f} {row['Positive Improvements']:<10}"
        formatted_table.append(data_row)

    formatted_table.append("")
    formatted_table.append("KEY INSIGHTS:")
    formatted_table.append("- Higher reduction % = More conservative intervention (lower PnL when intervening)")
    formatted_table.append("- Net Benefit shows total $ improvement vs original strategy")
    formatted_table.append("- Intervention Rate shows how often model triggers 'don't trade' signal")
    formatted_table.append("- Positive Improvements shows how many traders benefit from strategy")

    # Save formatted table
    with open(output_dir / "formatted_comparison_table.txt", 'w') as f:
        f.write("\n".join(formatted_table))

def create_comparison_plots(comparison_df: pd.DataFrame, output_dir: Path):
    """Create comparison plots for different reduction percentages."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Convert reduction % to numeric for plotting
    reduction_pcts = [float(x.rstrip('%')) for x in comparison_df['Reduction %']]

    # Plot 1: Net Benefit vs Reduction %
    axes[0, 0].plot(reduction_pcts, comparison_df['Net Benefit'], 'bo-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('PnL Reduction Percentage (%)')
    axes[0, 0].set_ylabel('Net Benefit ($)')
    axes[0, 0].set_title('Net Benefit vs PnL Reduction Percentage')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)

    # Plot 2: Overall Improvement % vs Reduction %
    axes[0, 1].plot(reduction_pcts, comparison_df['Overall Improvement %'], 'go-', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('PnL Reduction Percentage (%)')
    axes[0, 1].set_ylabel('Overall Improvement (%)')
    axes[0, 1].set_title('Performance Improvement vs PnL Reduction')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)

    # Plot 3: Avoided Losses vs Missed Gains
    x = np.arange(len(reduction_pcts))
    width = 0.35
    axes[1, 0].bar(x - width/2, comparison_df['Avoided Losses'], width,
                   label='Avoided Losses', color='green', alpha=0.7)
    axes[1, 0].bar(x + width/2, comparison_df['Missed Gains'], width,
                   label='Missed Gains', color='red', alpha=0.7)
    axes[1, 0].set_xlabel('PnL Reduction Percentage')
    axes[1, 0].set_ylabel('Amount ($)')
    axes[1, 0].set_title('Avoided Losses vs Missed Gains')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels([f"{pct}%" for pct in reduction_pcts])
    axes[1, 0].legend()

    # Plot 4: Intervention Rate vs Volatility Reduction
    axes[1, 1].scatter(comparison_df['Mean Intervention Rate %'],
                      comparison_df['Mean Volatility Reduction %'],
                      s=100, alpha=0.7, c=reduction_pcts, cmap='viridis')
    axes[1, 1].set_xlabel('Mean Intervention Rate (%)')
    axes[1, 1].set_ylabel('Mean Volatility Reduction (%)')
    axes[1, 1].set_title('Intervention Rate vs Volatility Reduction')
    axes[1, 1].grid(True, alpha=0.3)

    # Add colorbar for the scatter plot
    cbar = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1], label='Reduction %')

    plt.tight_layout()
    plt.savefig(output_dir / "comparison_plots.png", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main execution function."""
    # Run multiple evaluations with different reduction percentages
    all_results, comparison_df = run_multiple_evaluations([25, 50, 70, 90])
    return all_results, comparison_df


if __name__ == "__main__":
    results = main()
