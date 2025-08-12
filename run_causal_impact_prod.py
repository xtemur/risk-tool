#!/usr/bin/env python3
"""
Run causal impact evaluation on production models (_prod suffix).
"""

import pandas as pd
import numpy as np
import json
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Import the evaluator
from src.causal_impact_evaluation import CausalImpactEvaluator

def main():
    """Run causal impact evaluation on production models."""
    print("Starting causal impact evaluation on production models...")

    # Initialize evaluator with production model suffix
    evaluator = CausalImpactEvaluator(model_suffix="_tuned_validated_prod")

    # Create output directory
    output_dir = Path("/Users/temurbekkhujaev/Repos/risk-tool/results/causal_impact_evaluation_prod")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Testing Original System (50% reduction, binary thresholds) on PROD models ===")
    original_results = evaluator.evaluate_all_traders(reduction_level=0.5, use_weighted_formula=False)

    print("\n=== Testing Multiple Reduction Levels (0%, 10%, 30%, 60%) with Binary System on PROD models ===")
    binary_multilevel_results = evaluator.evaluate_all_traders_multilevel(use_weighted_formula=False)

    print("\n=== Testing Multiple Reduction Levels with Weighted Formula (α=0.6, β=0.4) on PROD models ===")
    weighted_multilevel_results = evaluator.evaluate_all_traders_multilevel(
        use_weighted_formula=True, alpha=0.6, beta=0.4
    )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with open(output_dir / f"prod_original_system_results_{timestamp}.pkl", 'wb') as f:
        pickle.dump(original_results, f)

    with open(output_dir / f"prod_binary_multilevel_results_{timestamp}.pkl", 'wb') as f:
        pickle.dump(binary_multilevel_results, f)

    with open(output_dir / f"prod_weighted_multilevel_results_{timestamp}.pkl", 'wb') as f:
        pickle.dump(weighted_multilevel_results, f)

    # Generate reports
    original_report = evaluator.generate_report()
    with open(output_dir / f"prod_original_system_report_{timestamp}.txt", 'w') as f:
        f.write(original_report)

    # Generate multilevel reports
    binary_report = evaluator.generate_multilevel_report(binary_multilevel_results)
    with open(output_dir / f"prod_binary_multilevel_report_{timestamp}.txt", 'w') as f:
        f.write(binary_report)

    weighted_report = evaluator.generate_multilevel_report(weighted_multilevel_results)
    with open(output_dir / f"prod_weighted_multilevel_report_{timestamp}.txt", 'w') as f:
        f.write(weighted_report)

    # Create summary dashboard
    dashboard_fig = evaluator.create_summary_dashboard(str(output_dir / f"prod_original_system_dashboard_{timestamp}.png"))
    plt.close(dashboard_fig)

    # Create individual trader plots
    trader_plots_dir = output_dir / "trader_plots"
    trader_plots_dir.mkdir(exist_ok=True)

    for trader_id in evaluator.results.keys():
        plot_fig = evaluator.create_pnl_comparison_plot(trader_id,
                                                       str(trader_plots_dir / f"prod_trader_{trader_id}_pnl_comparison_{timestamp}.png"))
        plt.close(plot_fig)

    print(f"\nEvaluation complete! Results saved to: {output_dir}")
    print("\nPRODUCTION MODELS SUMMARY:")
    print(original_report.split("INDIVIDUAL TRADER RESULTS")[0])

    # Print optimal reduction level findings
    if binary_multilevel_results.get('aggregate_results'):
        binary_optimal = evaluator.find_optimal_reduction_level(binary_multilevel_results['aggregate_results'])
        if binary_optimal:
            print(f"\nPROD BINARY SYSTEM OPTIMAL REDUCTION: {binary_optimal['overall_recommendation']['optimal_reduction_level']*100:.0f}%")

    if weighted_multilevel_results.get('aggregate_results'):
        weighted_optimal = evaluator.find_optimal_reduction_level(weighted_multilevel_results['aggregate_results'])
        if weighted_optimal:
            print(f"PROD WEIGHTED FORMULA OPTIMAL REDUCTION: {weighted_optimal['overall_recommendation']['optimal_reduction_level']*100:.0f}%")

    # Save summary comparison
    comparison_summary = {
        'evaluation_timestamp': timestamp,
        'model_type': 'production',
        'original_results_summary': original_results.get('aggregate_results', {}),
        'binary_optimal': binary_optimal['overall_recommendation'] if binary_optimal else None,
        'weighted_optimal': weighted_optimal['overall_recommendation'] if weighted_optimal else None
    }

    with open(output_dir / f"prod_evaluation_summary_{timestamp}.json", 'w') as f:
        json.dump(comparison_summary, f, indent=2, default=str)

    return {
        'original_results': original_results,
        'binary_multilevel_results': binary_multilevel_results,
        'weighted_multilevel_results': weighted_multilevel_results,
        'output_dir': output_dir,
        'timestamp': timestamp
    }

if __name__ == "__main__":
    results = main()
