#!/usr/bin/env python
"""
Debug Feature Engineering - Find why target is still zeros
"""

import sys
import pandas as pd
import numpy as np

# Add src to path
sys.path.append('src')

from database import Database
from feature_engineer import FeatureEngineer

def debug_feature_engineering():
    """Debug the feature engineering process step by step"""

    print("="*60)
    print("FEATURE ENGINEERING DEBUG")
    print("="*60)

    # Initialize components
    db = Database()
    feature_engineer = FeatureEngineer()

    # Get a trader with recent data
    traders_df = db.get_all_traders()
    trader = traders_df[traders_df['trading_days'] > 60].iloc[0]
    account_id = str(trader['account_id'])
    trader_name = trader['trader_name']

    print(f"Debugging trader: {trader_name} ({account_id})")

    # Step 1: Check raw data
    print(f"\n1. RAW DATA CHECK")
    print("-" * 30)

    totals_df, fills_df = db.get_trader_data(account_id)
    print(f"Total records: {len(totals_df)}")
    print(f"Date range: {totals_df['date'].min()} to {totals_df['date'].max()}")

    # Check P&L in raw data
    print(f"\nRaw P&L Statistics:")
    print(f"  net_pnl mean: ${totals_df['net_pnl'].mean():.2f}")
    print(f"  net_pnl std: ${totals_df['net_pnl'].std():.2f}")
    print(f"  net_pnl min: ${totals_df['net_pnl'].min():.2f}")
    print(f"  net_pnl max: ${totals_df['net_pnl'].max():.2f}")
    print(f"  Zero days: {(totals_df['net_pnl'] == 0).sum()}/{len(totals_df)}")

    # Show sample data
    print(f"\nSample raw data (last 10 days):")
    sample_data = totals_df[['date', 'net_pnl', 'gross_pnl', 'orders_count']].tail(10)
    for _, row in sample_data.iterrows():
        print(f"  {row['date']}: P&L=${row['net_pnl']:.2f}, Orders={row['orders_count']}")

    # Step 2: Debug feature creation step by step
    print(f"\n2. FEATURE CREATION DEBUG")
    print("-" * 30)

    # Create features and examine each step
    features_df = totals_df.copy()
    features_df = features_df.sort_values('date')

    print(f"After copying totals_df: {len(features_df)} rows")
    print(f"  net_pnl still varies: std={features_df['net_pnl'].std():.2f}")

    # Check what happens in target creation
    print(f"\n3. TARGET CREATION DEBUG")
    print("-" * 30)

    # This is the critical step - create target (next day's P&L)
    print("Creating target variable...")
    features_df['target'] = features_df['net_pnl'].shift(-1)

    print(f"Target statistics:")
    print(f"  mean: ${features_df['target'].mean():.2f}")
    print(f"  std: ${features_df['target'].std():.2f}")
    print(f"  min: ${features_df['target'].min():.2f}")
    print(f"  max: ${features_df['target'].max():.2f}")
    print(f"  NaN count: {features_df['target'].isna().sum()}")
    print(f"  Zero count: {(features_df['target'] == 0).sum()}")

    # Show target vs current P&L
    print(f"\nTarget vs Current P&L comparison (last 10 valid days):")
    comparison = features_df[['date', 'net_pnl', 'target']].dropna().tail(10)
    for _, row in comparison.iterrows():
        print(f"  {row['date']}: Today=${row['net_pnl']:.2f} → Tomorrow=${row['target']:.2f}")

    # Step 4: Check if the issue is in the feature engineering function
    print(f"\n4. FULL FEATURE ENGINEERING TEST")
    print("-" * 30)

    # Run the actual feature engineering function
    full_features_df = feature_engineer.create_features(totals_df, fills_df)

    print(f"Full feature engineering result:")
    print(f"  Rows: {len(full_features_df)}")
    print(f"  Columns: {len(full_features_df.columns)}")

    if 'target' in full_features_df.columns:
        print(f"  Target mean: ${full_features_df['target'].mean():.2f}")
        print(f"  Target std: ${full_features_df['target'].std():.2f}")
        print(f"  Target zeros: {(full_features_df['target'] == 0).sum()}/{len(full_features_df)}")

        # Check if target is just all the same value
        unique_targets = full_features_df['target'].nunique()
        print(f"  Unique target values: {unique_targets}")

        if unique_targets <= 2:
            print(f"  🚨 TARGET HAS NO VARIANCE!")
            print(f"  Unique values: {sorted(full_features_df['target'].unique())}")

    # Step 5: Check what happens during feature engineering
    print(f"\n5. DETAILED FEATURE ENGINEERING ANALYSIS")
    print("-" * 30)

    # Let's manually step through the feature engineering
    print("Stepping through feature_engineer.create_features()...")

    # Start with totals data
    features = totals_df.copy()
    features = features.sort_values('date')
    print(f"Step 1 - After sorting: net_pnl std = {features['net_pnl'].std():.2f}")

    # The issue might be in the cleaning steps
    # Check what happens with replace and fillna

    # Check for infinite values
    inf_count_before = np.isinf(features.select_dtypes(include=[np.number])).sum().sum()
    print(f"Step 2 - Infinite values before cleaning: {inf_count_before}")

    # Apply the cleaning steps from feature engineering
    features = features.replace([np.inf, -np.inf], 0)
    features = features.fillna(0)

    print(f"Step 3 - After cleaning: net_pnl std = {features['net_pnl'].std():.2f}")

    # Create target
    features['target'] = features['net_pnl'].shift(-1)
    print(f"Step 4 - After target creation: target std = {features['target'].std():.2f}")

    # Remove last row (no target)
    features = features[:-1]
    print(f"Step 5 - After removing last row: target std = {features['target'].std():.2f}")

    # Final check
    print(f"\nFINAL DIAGNOSIS:")
    if features['target'].std() < 0.01:
        print("🚨 ISSUE CONFIRMED: Target has no variance after feature engineering")
        print("This could be due to:")
        print("1. Data cleaning (replace/fillna) affecting target")
        print("2. All P&L values being the same")
        print("3. Issue in shift operation")

        # Check the actual values
        print(f"\nActual target values (first 20):")
        print(features['target'].head(20).tolist())

    else:
        print("✅ Target looks good - issue might be elsewhere")

    return features

if __name__ == "__main__":
    debug_features = debug_feature_engineering()
