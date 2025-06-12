"""
Feature Engineering Demo

Demonstrates how to use the feature engineering pipeline
to process trading data for PnL prediction.
"""

import sys
import os
from pathlib import Path

# Add the src directory to the Python path and change working directory
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))
os.chdir(project_root)

import pandas as pd
import numpy as np
from feature_engineering import FeatureEngineeringPipeline
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """
    Main demo function
    """
    print("=" * 60)
    print("FEATURE ENGINEERING PIPELINE DEMO")
    print("=" * 60)

    # Initialize pipeline
    pipeline = FeatureEngineeringPipeline(
        db_path="data/trading_risk.db",
        lookback_window=10,
        n_features=10
    )

    # Run diagnostic
    print("\n1. Running Pipeline Diagnostic...")
    diagnostic = pipeline.run_diagnostic()

    print(f"Database Status: {diagnostic.get('pipeline_status', 'unknown')}")
    if 'database_stats' in diagnostic:
        db_stats = diagnostic['database_stats']
        print(f"Total Accounts: {db_stats.get('accounts_count', 0)}")
        print(f"Daily Summary Records: {db_stats.get('account_daily_summary_count', 0)}")
        print(f"Fills Records: {db_stats.get('fills_count', 0)}")
        print(f"Date Range: {db_stats.get('date_range', 'Unknown')}")

    # Process single trader
    print("\n2. Processing Single Trader...")
    try:
        # Get first available account
        accounts = pipeline.db_manager.get_accounts()
        if accounts.empty:
            print("No accounts found in database!")
            return

        sample_account = accounts.iloc[0]['account_id']
        print(f"Processing trader: {sample_account}")

        trader_data, trader_summary = pipeline.process_trader_data(sample_account)

        if not trader_data.empty:
            print(f"✓ Generated {len(trader_data)} samples")
            print(f"✓ Selected {len(trader_summary.get('selected_features', []))} features")
            print(f"✓ Date range: {trader_data['date'].min()} to {trader_data['date'].max()}")

            # Show feature categories
            if 'feature_importance' in trader_summary:
                feature_cats = trader_summary['feature_importance'].get('feature_categories', {})
                print(f"✓ Feature categories: {list(feature_cats.keys())}")

            # Show top features
            selected_features = trader_summary.get('selected_features', [])
            print(f"✓ Top 10 features: {selected_features}")

            # Show target statistics
            target_stats = trader_summary.get('data_summary', {}).get('target_stats', {})
            if target_stats:
                print(f"✓ Target mean: {target_stats.get('mean', 0):.4f}")
                print(f"✓ Target std: {target_stats.get('std', 0):.4f}")
                print(f"✓ Positive days: {target_stats.get('positive_days', 0)}")
                print(f"✓ Negative days: {target_stats.get('negative_days', 0)}")

        else:
            print("✗ No data generated for trader")

    except Exception as e:
        print(f"✗ Error processing single trader: {e}")

    # Process all traders
    print("\n3. Processing All Traders...")
    try:
        all_data, overall_summary = pipeline.process_all_traders(min_samples=20)

        if not all_data.empty:
            print(f"✓ Generated {len(all_data)} total samples")
            print(f"✓ From {all_data['account_id'].nunique()} traders")

            overall_stats = overall_summary.get('overall_stats', {})
            target_dist = overall_stats.get('target_distribution', {})

            print(f"✓ Target mean: {target_dist.get('mean', 0):.4f}")
            print(f"✓ Target std: {target_dist.get('std', 0):.4f}")
            print(f"✓ Positive ratio: {target_dist.get('positive_ratio', 0):.2%}")

            # Save processed data
            output_path = "data/processed/features_demo.csv"
            pipeline.save_processed_data(all_data, output_path)
            print(f"✓ Saved processed data to {output_path}")

        else:
            print("✗ No data generated for any traders")

    except Exception as e:
        print(f"✗ Error processing all traders: {e}")

    # Get ML-ready data
    print("\n4. Preparing ML-Ready Data...")
    try:
        X, y, feature_names = pipeline.get_feature_ready_data()

        if len(X) > 0:
            print(f"✓ Feature matrix shape: {X.shape}")
            print(f"✓ Target vector shape: {y.shape}")
            print(f"✓ Feature names: {feature_names}")

            # Basic statistics
            print(f"✓ Features mean: {np.mean(X, axis=0)[:5]}")  # First 5 features
            print(f"✓ Target mean: {np.mean(y):.4f}")
            print(f"✓ Target std: {np.std(y):.4f}")

        else:
            print("✗ No ML-ready data generated")

    except Exception as e:
        print(f"✗ Error preparing ML data: {e}")

    print("\n" + "=" * 60)
    print("DEMO COMPLETED")
    print("=" * 60)

    # Show usage summary
    print("\nUSAGE SUMMARY:")
    print("1. Initialize pipeline: FeatureEngineeringPipeline()")
    print("2. Process single trader: pipeline.process_trader_data(account_id)")
    print("3. Process all traders: pipeline.process_all_traders()")
    print("4. Get ML-ready data: pipeline.get_feature_ready_data()")
    print("5. Save/load data: pipeline.save_processed_data() / load_processed_data()")


if __name__ == "__main__":
    main()
