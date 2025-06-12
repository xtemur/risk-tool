"""
Modeling Pipeline Demo

Demonstrates the complete time series validation and modeling workflow
for trader PnL prediction with proper holdout testing and walk-forward validation.
"""

import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add the src directory to the Python path and change working directory
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))
os.chdir(project_root)

import pandas as pd
import numpy as np
from modeling import PredictionPipeline, ModelConfig
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """
    Main demo function
    """
    print("=" * 80)
    print("TIME SERIES MODELING PIPELINE DEMO")
    print("=" * 80)

    # Load processed feature data
    print("\n1. Loading Processed Feature Data...")
    try:
        data = pd.read_csv("data/processed/features_demo.csv")
        print(f"✓ Loaded {len(data)} samples from {data['account_id'].nunique()} traders")
        print(f"✓ Date range: {data['date'].min()} to {data['date'].max()}")
        print(f"✓ Features: {len([col for col in data.columns if col not in ['account_id', 'date', 'target_next_pnl']])}")
    except FileNotFoundError:
        print("✗ Feature data not found. Please run the feature engineering demo first.")
        print("  Command: python examples/feature_engineering_demo.py")
        return

    # Configuration
    print("\n2. Setting Up Configuration...")

    # Use quick testing config for demo (faster execution)
    config = ModelConfig.for_quick_testing()
    config.PLOT_RESULTS = False  # Disable plots for demo
    config.SAVE_MODELS = True
    config.FEATURE_SELECTION = False  # Disable feature selection to avoid NaN issues
    config.HANDLE_MISSING = 'median'  # Ensure missing value handling

    print(f"✓ Holdout period: Last {config.HOLDOUT_MONTHS} months")
    print(f"✓ Training window: {config.TRAINING_WINDOW_MONTHS} months")
    print(f"✓ Validation step: {config.VALIDATION_STEP_MONTHS} month(s)")
    print(f"✓ Model types available: {list(config.MODEL_TYPES.keys())}")

    # Initialize pipeline
    pipeline = PredictionPipeline(config)

    # Run validation summary
    print("\n3. Validation Setup Analysis...")
    validation_summary = pipeline.validator.get_validation_summary(data)

    print(f"✓ Total samples: {validation_summary['total_samples']}")
    print(f"✓ Valid traders: {validation_summary.get('per_trader', {}).get('valid_traders', 0)}")
    print(f"✓ Training samples: {validation_summary['holdout_split']['training_samples']}")
    print(f"✓ Holdout samples: {validation_summary['holdout_split']['holdout_samples']}")
    print(f"✓ Walk-forward splits: {validation_summary.get('walk_forward', {}).get('num_splits', 0)}")

    # Run single trader demo (faster)
    print("\n4. Single Trader Demo...")
    try:
        # Pick a trader with good amount of data
        trader_counts = data['account_id'].value_counts()
        demo_trader = trader_counts.index[0]  # Trader with most data
        demo_data = data[data['account_id'] == demo_trader].copy()

        print(f"✓ Demo trader: {demo_trader} ({len(demo_data)} samples)")

        # Run pipeline for single trader
        model_types = ['linear', 'ridge']  # Start with simple baselines

        print(f"✓ Training models: {model_types}")

        results = pipeline.run_full_pipeline(
            demo_data,
            model_types=model_types,
            individual_traders=False  # Treat as single entity for demo
        )

        print("✓ Pipeline completed successfully!")

        # Show results
        print("\n5. Results Summary...")

        # Walk-forward results
        if 'global' in results['walk_forward_results']:
            wf_results = results['walk_forward_results']['global']
            print(f"✓ Walk-forward validation: {len(wf_results.get('folds', []))} folds")

            # Show aggregate results
            if 'aggregate' in wf_results:
                for model_type, agg_results in wf_results['aggregate'].items():
                    avg_metrics = agg_results.get('average_metrics', {})
                    print(f"  {model_type.upper()}:")
                    print(f"    - Avg R²: {avg_metrics.get('avg_r2', 0):.4f} (±{avg_metrics.get('std_r2', 0):.4f})")
                    print(f"    - Avg Hit Rate: {avg_metrics.get('avg_hit_rate', 0):.1%} (±{avg_metrics.get('std_hit_rate', 0):.1%})")
                    print(f"    - Avg Overall Score: {avg_metrics.get('avg_overall_score', 0):.4f}")

        # Holdout results
        if 'global' in results['holdout_results']:
            holdout_results = results['holdout_results']['global']
            print(f"\n✓ Holdout Test Results:")

            for model_type, result in holdout_results.items():
                if 'error' in result:
                    print(f"  {model_type.upper()}: ✗ Error - {result['error']}")
                    continue

                stat_metrics = result['statistical_metrics']
                fin_metrics = result['financial_metrics']

                print(f"  {model_type.upper()}:")
                print(f"    - MAE: {stat_metrics['mae']:.2f}")
                print(f"    - R²: {stat_metrics['r2']:.4f}")
                print(f"    - Hit Rate: {fin_metrics['hit_rate']:.1%}")
                print(f"    - Sharpe Ratio: {fin_metrics['actual_sharpe']:.4f}")
                print(f"    - Overall Score: {result['overall_score']:.4f}")

        # Best model
        best_models = pipeline.get_best_models()
        if best_models:
            best_model = list(best_models.values())[0]
            print(f"\n✓ Best Model: {best_model['model_type'].upper()} (Score: {best_model['score']:.4f})")

        # Causal Impact Analysis
        print("\n6. Causal Impact Analysis...")
        try:
            if 'causal_impact_results' in results:
                causal_results = results['causal_impact_results']

                # Show key causal impact metrics
                if 'global' in causal_results:
                    for model_type, causal_data in causal_results['global'].items():
                        if 'error' not in causal_data:
                            scenarios = causal_data['causal_impact_scenarios']
                            perfect = scenarios['perfect_following']
                            directional = scenarios['directional_trading']

                            print(f"  {model_type.upper()} Causal Impact:")
                            print(f"    - Perfect Following: ${perfect['pnl_improvement']:,.2f} ({perfect['pnl_improvement_pct']:+.1f}%)")
                            print(f"    - Directional Trading: ${directional['pnl_improvement']:,.2f} ({directional['pnl_improvement_pct']:+.1f}%)")
                            print(f"    - Trading Frequency: {directional['trading_frequency']:.1%}")

                # Generate and show sample report
                reports = pipeline.generate_causal_impact_reports()
                if reports:
                    print(f"\n✓ Generated {len(reports)} causal impact reports")

                    # Show first few lines of first report
                    first_report = list(reports.values())[0]
                    report_preview = '\n'.join(first_report.split('\n')[:10])
                    print(f"\n  Sample Report Preview:")
                    print(f"  {'-' * 40}")
                    for line in report_preview.split('\n'):
                        if line.strip():
                            print(f"  {line}")
                    print(f"  {'-' * 40}")
            else:
                print("✗ No causal impact results found")

        except Exception as e:
            print(f"✗ Error in causal impact analysis: {e}")
            logger.error(f"Causal impact analysis failed: {e}", exc_info=True)

    except Exception as e:
        print(f"✗ Error in single trader demo: {e}")
        logger.error(f"Single trader demo failed: {e}", exc_info=True)

    # Multi-trader demo (if data permits)
    print("\n7. Multi-Trader Demo...")
    try:
        # Filter traders with enough data
        trader_counts = data['account_id'].value_counts()
        valid_traders = trader_counts[trader_counts >= config.MIN_SAMPLES_PER_TRADER].index[:3]  # Top 3 traders

        if len(valid_traders) >= 2:
            multi_trader_data = data[data['account_id'].isin(valid_traders)].copy()
            print(f"✓ Multi-trader data: {len(valid_traders)} traders, {len(multi_trader_data)} samples")

            # Quick multi-trader pipeline
            multi_results = pipeline.run_full_pipeline(
                multi_trader_data,
                model_types=['ridge'],  # Just one model for speed
                individual_traders=True
            )

            print("✓ Multi-trader pipeline completed!")

            # Show trader-specific results
            trader_holdout = multi_results.get('holdout_results', {})
            for trader, results_dict in trader_holdout.items():
                if trader == 'global':
                    continue

                if 'ridge' in results_dict and 'error' not in results_dict['ridge']:
                    result = results_dict['ridge']
                    print(f"  Trader {trader}: R² = {result['statistical_metrics']['r2']:.4f}, "
                          f"Hit Rate = {result['financial_metrics']['hit_rate']:.1%}")

            # Show aggregate causal impact if available
            if 'causal_impact_results' in multi_results:
                causal_results = multi_results['causal_impact_results']
                if 'aggregate_analysis' in causal_results and 'error' not in causal_results['aggregate_analysis']:
                    agg = causal_results['aggregate_analysis']['aggregate_metrics']
                    print(f"\n  Aggregate Causal Impact:")
                    print(f"    - Total Improvement: ${agg['total_improvement']:,.2f} ({agg['total_improvement_pct']:+.1f}%)")
                    print(f"    - Success Rate: {agg['improvement_success_rate']:.1%} of traders")

        else:
            print(f"✗ Not enough traders with sufficient data (need ≥{config.MIN_SAMPLES_PER_TRADER} samples each)")

    except Exception as e:
        print(f"✗ Error in multi-trader demo: {e}")
        logger.error(f"Multi-trader demo failed: {e}", exc_info=True)

    # Save results
    print("\n8. Saving Results...")
    try:
        save_path = pipeline.save_results("results/modeling_demo")
        print(f"✓ Results saved to: {save_path}")

        # Save causal impact reports
        if hasattr(pipeline, 'causal_impact_results') and pipeline.causal_impact_results:
            reports = pipeline.generate_causal_impact_reports("results/modeling_demo/causal_impact_reports")
            print(f"✓ Causal impact reports saved: {len(reports)} reports")
    except Exception as e:
        print(f"✗ Error saving results: {e}")

    # XGBoost readiness check
    print("\n9. XGBoost Integration Check...")
    try:
        import xgboost
        print("✓ XGBoost is available and ready for integration")
        print(f"  Version: {xgboost.__version__}")
        print("  To use XGBoost, add 'xgboost' to model_types in run_full_pipeline()")
    except ImportError:
        print("ℹ XGBoost not installed. To add XGBoost support:")
        print("  pip install xgboost")

    print("\n" + "=" * 80)
    print("MODELING DEMO COMPLETED")
    print("=" * 80)

    # Show usage summary
    print("\nUSAGE SUMMARY:")
    print("1. Load processed feature data")
    print("2. Configure pipeline: ModelConfig.for_production()")
    print("3. Initialize: PredictionPipeline(config)")
    print("4. Run pipeline: pipeline.run_full_pipeline(data, model_types=['ridge', 'xgboost'])")
    print("5. Get best models: pipeline.get_best_models()")
    print("6. Save results: pipeline.save_results()")

    print("\nKEY FEATURES DEMONSTRATED:")
    print("✓ Proper temporal data splitting (no future leakage)")
    print("✓ Walk-forward validation for model selection")
    print("✓ Holdout testing on unseen last 2 months")
    print("✓ Individual trader model training")
    print("✓ Comprehensive evaluation (statistical + financial metrics)")
    print("✓ Causal impact analysis (what-if scenarios)")
    print("✓ Automated report generation for business insights")
    print("✓ Flexible model architecture (Linear → XGBoost ready)")
    print("✓ Production-ready pipeline with proper validation")


if __name__ == "__main__":
    main()
