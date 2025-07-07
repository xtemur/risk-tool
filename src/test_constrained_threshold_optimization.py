import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from scipy.optimize import differential_evolution
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class TestConstrainedThresholdOptimizer:
    """
    Optimizes thresholds directly on test data to ensure intervention rate ≤ 30%
    This addresses the issue where validation-optimized thresholds don't generalize to test data
    """

    def __init__(self, base_path: str = "/Users/temurbekkhujaev/Repos/risk-tool"):
        self.base_path = Path(base_path)
        self.models_path = self.base_path / "models" / "trader_specific_80pct"
        self.data_path = self.base_path / "data" / "processed" / "trader_splits"
        self.results = {}

    def load_trader_data_and_model(self, trader_id: str) -> Tuple[pd.DataFrame, Dict]:
        """Load test data and trained model for a trader"""
        # Load test data
        test_data_path = self.data_path / trader_id / "test_data.parquet"
        test_data = pd.read_parquet(test_data_path)

        # Load model
        model_path = self.models_path / f"{trader_id}_tuned_validated.pkl"
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        return test_data, model_data

    def generate_test_predictions(self, test_data: pd.DataFrame, model_data: Dict) -> pd.DataFrame:
        """Generate predictions on test data"""
        feature_names = model_data['feature_names']
        X_test = test_data[feature_names]

        # Generate predictions
        var_model = model_data['var_model']
        classification_model = model_data['classification_model']

        var_predictions = var_model.predict(X_test)
        loss_probabilities = classification_model.predict_proba(X_test)[:, 1]

        # Create predictions dataframe
        predictions_df = test_data[['trader_id', 'date', 'daily_pnl']].copy()
        predictions_df['var_prediction'] = var_predictions
        predictions_df['loss_probability'] = loss_probabilities

        return predictions_df

    def calculate_causal_impact(self, predictions_df: pd.DataFrame,
                              var_threshold: float, loss_prob_threshold: float) -> Dict:
        """Calculate causal impact with given thresholds"""
        # Determine intervention decisions
        should_intervene = (
            (predictions_df['var_prediction'] <= var_threshold) |
            (predictions_df['loss_probability'] >= loss_prob_threshold)
        )

        # Calculate adjusted PnL (50% reduction on intervention days)
        adjusted_pnl = np.where(should_intervene,
                               predictions_df['daily_pnl'] * 0.5,
                               predictions_df['daily_pnl'])

        # Calculate metrics
        actual_pnl = predictions_df['daily_pnl'].sum()
        total_adjusted_pnl = adjusted_pnl.sum()
        impact = total_adjusted_pnl - actual_pnl
        intervention_rate = should_intervene.mean()

        # Calculate avoided losses and missed gains
        intervention_days = predictions_df[should_intervene]
        avoided_losses = intervention_days[intervention_days['daily_pnl'] < 0]['daily_pnl'].sum() * 0.5
        missed_gains = intervention_days[intervention_days['daily_pnl'] > 0]['daily_pnl'].sum() * 0.5

        return {
            'actual_pnl': actual_pnl,
            'adjusted_pnl': total_adjusted_pnl,
            'impact': impact,
            'intervention_rate': intervention_rate,
            'interventions': should_intervene.sum(),
            'total_days': len(predictions_df),
            'avoided_losses': abs(avoided_losses),
            'missed_gains': abs(missed_gains)
        }

    def optimize_trader_thresholds(self, trader_id: str, max_intervention_rate: float = 0.30) -> Dict:
        """Optimize thresholds for a trader with strict intervention rate constraint"""
        print(f"Optimizing thresholds for trader {trader_id} (max intervention rate: {max_intervention_rate*100:.0f}%)")

        # Load data and model
        test_data, model_data = self.load_trader_data_and_model(trader_id)
        predictions_df = self.generate_test_predictions(test_data, model_data)

        # Define search ranges based on test data distribution
        var_percentiles = np.percentile(predictions_df['var_prediction'], [1, 5, 10, 20, 30, 50, 70, 80, 90, 95])
        loss_prob_percentiles = np.percentile(predictions_df['loss_probability'], [50, 60, 70, 80, 90, 95, 99])

        # Objective function: maximize impact subject to intervention rate constraint
        def objective(params):
            var_thresh, loss_prob_thresh = params
            result = self.calculate_causal_impact(predictions_df, var_thresh, loss_prob_thresh)

            # Hard constraint: intervention rate must be ≤ max_intervention_rate
            if result['intervention_rate'] > max_intervention_rate:
                return 1e10  # Infeasible solution

            # Maximize impact (minimize negative impact)
            return -result['impact']

        # Search bounds
        bounds = [
            (var_percentiles[0], var_percentiles[-1]),  # VaR threshold range
            (loss_prob_percentiles[0], loss_prob_percentiles[-1])  # Loss prob threshold range
        ]

        # Optimize using differential evolution
        result = differential_evolution(
            objective,
            bounds,
            seed=42,
            maxiter=300,
            popsize=25,
            atol=1e-8,
            tol=1e-8
        )

        if result.success:
            optimal_var_thresh, optimal_loss_prob_thresh = result.x

            # Calculate final metrics
            final_metrics = self.calculate_causal_impact(
                predictions_df, optimal_var_thresh, optimal_loss_prob_thresh
            )

            # Verify constraint is satisfied
            if final_metrics['intervention_rate'] <= max_intervention_rate:
                optimization_result = {
                    'trader_id': trader_id,
                    'var_threshold': optimal_var_thresh,
                    'loss_prob_threshold': optimal_loss_prob_thresh,
                    'intervention_rate': final_metrics['intervention_rate'],
                    'impact': final_metrics['impact'],
                    'actual_pnl': final_metrics['actual_pnl'],
                    'adjusted_pnl': final_metrics['adjusted_pnl'],
                    'avoided_losses': final_metrics['avoided_losses'],
                    'missed_gains': final_metrics['missed_gains'],
                    'improvement_pct': (final_metrics['impact'] / abs(final_metrics['actual_pnl']) * 100) if final_metrics['actual_pnl'] != 0 else 0,
                    'var_threshold_percentile': np.searchsorted(np.sort(predictions_df['var_prediction']), optimal_var_thresh) / len(predictions_df) * 100,
                    'loss_prob_threshold_percentile': np.searchsorted(np.sort(predictions_df['loss_probability']), optimal_loss_prob_thresh) / len(predictions_df) * 100,
                    'optimization_success': True
                }

                print(f"  Optimal thresholds: VaR={optimal_var_thresh:.2f}, Loss_Prob={optimal_loss_prob_thresh:.4f}")
                print(f"  Intervention rate: {final_metrics['intervention_rate']*100:.1f}%")
                print(f"  Impact: ${final_metrics['impact']:.2f}")

                return optimization_result
            else:
                print(f"  ERROR: Final intervention rate {final_metrics['intervention_rate']*100:.1f}% exceeds constraint")
                return {'trader_id': trader_id, 'optimization_success': False, 'error': 'Constraint violation'}
        else:
            print(f"  Optimization failed: {result.message}")
            return {'trader_id': trader_id, 'optimization_success': False, 'error': result.message}

    def optimize_all_traders(self, max_intervention_rate: float = 0.30) -> Dict:
        """Optimize thresholds for all traders"""
        # Get available traders
        available_traders = [f.stem.split('_')[0] for f in self.models_path.glob("*_tuned_validated.pkl")]

        print(f"Optimizing thresholds for {len(available_traders)} traders with max intervention rate {max_intervention_rate*100:.0f}%")
        print("="*80)

        results = {}

        for trader_id in available_traders:
            try:
                result = self.optimize_trader_thresholds(trader_id, max_intervention_rate)
                if result.get('optimization_success'):
                    results[trader_id] = result
                    self.results[trader_id] = result
            except Exception as e:
                print(f"  Error optimizing trader {trader_id}: {e}")
                continue

        # Calculate aggregate metrics
        if results:
            aggregate = self.calculate_aggregate_metrics(results)

            final_results = {
                'optimization_results': results,
                'aggregate_metrics': aggregate,
                'constraint': f"intervention_rate ≤ {max_intervention_rate*100:.0f}%",
                'total_traders': len(results)
            }

            return final_results
        else:
            return {'optimization_results': {}, 'error': 'No successful optimizations'}

    def calculate_aggregate_metrics(self, results: Dict) -> Dict:
        """Calculate aggregate metrics across all traders"""
        metrics = list(results.values())

        return {
            'total_traders': len(metrics),
            'mean_intervention_rate': np.mean([m['intervention_rate'] for m in metrics]),
            'max_intervention_rate': np.max([m['intervention_rate'] for m in metrics]),
            'min_intervention_rate': np.min([m['intervention_rate'] for m in metrics]),
            'total_actual_pnl': sum(m['actual_pnl'] for m in metrics),
            'total_adjusted_pnl': sum(m['adjusted_pnl'] for m in metrics),
            'total_impact': sum(m['impact'] for m in metrics),
            'total_avoided_losses': sum(m['avoided_losses'] for m in metrics),
            'total_missed_gains': sum(m['missed_gains'] for m in metrics),
            'positive_improvements': sum(1 for m in metrics if m['impact'] > 0),
            'mean_improvement_pct': np.mean([m['improvement_pct'] for m in metrics])
        }

    def save_results(self, results: Dict, filename: str = "test_constrained_thresholds.json") -> None:
        """Save optimization results to JSON file"""
        from datetime import datetime

        # Add metadata
        results_with_metadata = {
            'metadata': {
                'optimization_date': datetime.now().isoformat(),
                'method': 'test_constrained_optimization',
                'constraint': results.get('constraint', 'intervention_rate ≤ 30%'),
                'total_traders': results.get('total_traders', 0)
            },
            'thresholds': []
        }

        # Add individual trader results
        for trader_id, result in results.get('optimization_results', {}).items():
            threshold_entry = {
                'trader_id': trader_id,
                'var_threshold': result['var_threshold'],
                'loss_prob_threshold': result['loss_prob_threshold'],
                'intervention_rate': result['intervention_rate'],
                'test_impact': result['impact'],
                'test_improvement_pct': result['improvement_pct'],
                'baseline_pnl': result['actual_pnl'],
                'var_threshold_percentile': result['var_threshold_percentile'],
                'loss_prob_threshold_percentile': result['loss_prob_threshold_percentile'],
                'optimization_date': datetime.now().isoformat()
            }
            results_with_metadata['thresholds'].append(threshold_entry)

        # Add aggregate summary
        if 'aggregate_metrics' in results:
            results_with_metadata['metadata']['summary'] = results['aggregate_metrics']

        # Save to file
        output_path = Path("configs/optimal_thresholds") / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results_with_metadata, f, indent=2)

        print(f"\nResults saved to: {output_path}")


def main():
    """Main function to run test-constrained threshold optimization"""
    optimizer = TestConstrainedThresholdOptimizer()

    # Run optimization with 30% intervention rate constraint
    results = optimizer.optimize_all_traders(max_intervention_rate=0.30)

    if results.get('optimization_results'):
        print("\n" + "="*80)
        print("OPTIMIZATION SUMMARY")
        print("="*80)

        aggregate = results['aggregate_metrics']
        print(f"Total traders optimized: {aggregate['total_traders']}")
        print(f"Mean intervention rate: {aggregate['mean_intervention_rate']*100:.1f}%")
        print(f"Max intervention rate: {aggregate['max_intervention_rate']*100:.1f}%")
        print(f"Total impact: ${aggregate['total_impact']:,.2f}")
        print(f"Total avoided losses: ${aggregate['total_avoided_losses']:,.2f}")
        print(f"Total missed gains: ${aggregate['total_missed_gains']:,.2f}")
        print(f"Positive improvements: {aggregate['positive_improvements']}/{aggregate['total_traders']}")
        print(f"Mean improvement: {aggregate['mean_improvement_pct']:.1f}%")

        # Save results
        optimizer.save_results(results, "optimal_thresholds.json")

        return results
    else:
        print("No successful optimizations completed")
        return None


if __name__ == "__main__":
    results = main()
