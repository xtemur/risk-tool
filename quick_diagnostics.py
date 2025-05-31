#!/usr/bin/env python
"""
Quick Diagnosis Script - Identify why models are producing zeros
Run this to understand the data and model issues
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append('src')

from database import Database
from feature_engineer import FeatureEngineer
from model_trainer import ModelTrainer

def diagnose_data_and_models():
    """Quick diagnosis of data and model issues"""

    print("="*60)
    print("QUICK DIAGNOSIS - Risk Management MVP")
    print("="*60)

    # Initialize components
    db = Database()
    feature_engineer = FeatureEngineer()
    model_trainer = ModelTrainer()

    # 1. Check database stats
    print("\n1. DATABASE HEALTH CHECK")
    print("-" * 30)
    stats = db.get_database_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # 2. Check recent data availability
    print("\n2. RECENT DATA AVAILABILITY")
    print("-" * 30)
    traders_df = db.get_all_traders()
    print(f"Total traders in DB: {len(traders_df)}")

    # Check last 7 days of data
    from datetime import date, timedelta
    yesterday = date.today() - timedelta(days=1)
    week_ago = yesterday - timedelta(days=7)

    recent_data_count = 0
    for _, trader in traders_df.iterrows():
        account_id = str(trader['account_id'])
        totals_df, _ = db.get_trader_data(account_id,
                                        start_date=week_ago.strftime('%Y-%m-%d'),
                                        end_date=yesterday.strftime('%Y-%m-%d'))
        if not totals_df.empty:
            recent_data_count += 1
            print(f"  ✓ {trader['trader_name']}: {len(totals_df)} days")
        else:
            print(f"  ✗ {trader['trader_name']}: No recent data")

    print(f"\nTraders with recent data: {recent_data_count}/{len(traders_df)}")

    # 3. Check model quality for a sample trader
    print("\n3. MODEL QUALITY CHECK")
    print("-" * 30)

    # Find a trader with good data
    sample_trader = None
    for _, trader in traders_df.iterrows():
        if trader['trading_days'] > 60:
            sample_trader = trader
            break

    if sample_trader is None:
        print("  ✗ No trader with sufficient data found")
        return

    account_id = str(sample_trader['account_id'])
    trader_name = sample_trader['trader_name']
    print(f"  Analyzing: {trader_name} ({account_id})")

    # Get all data for this trader
    totals_df, fills_df = db.get_trader_data(account_id)
    print(f"  Total trading days: {len(totals_df)}")
    print(f"  Date range: {totals_df['date'].min()} to {totals_df['date'].max()}")
    print(f"  P&L range: ${totals_df['net_pnl'].min():.2f} to ${totals_df['net_pnl'].max():.2f}")
    print(f"  P&L std: ${totals_df['net_pnl'].std():.2f}")

    # Create features
    features_df = feature_engineer.create_features(totals_df, fills_df)
    print(f"  Features created: {len(features_df)} rows, {len(features_df.columns)} columns")

    # Check target variable
    if 'target' in features_df.columns:
        target_stats = features_df['target'].describe()
        print(f"  Target stats:")
        print(f"    Mean: ${target_stats['mean']:.2f}")
        print(f"    Std: ${target_stats['std']:.2f}")
        print(f"    Min: ${target_stats['min']:.2f}")
        print(f"    Max: ${target_stats['max']:.2f}")
        print(f"    Zeros: {(features_df['target'] == 0).sum()}/{len(features_df)}")

        # Check for data leakage
        if target_stats['std'] < 0.01:
            print("  🚨 WARNING: Target has very low variance - possible data leakage!")

        # Check correlation between features and target
        feature_cols = feature_engineer.get_feature_columns()
        available_features = [col for col in feature_cols if col in features_df.columns]

        correlations = []
        for col in available_features[:10]:  # Check first 10 features
            try:
                corr = features_df[col].corr(features_df['target'])
                if not np.isnan(corr):
                    correlations.append((col, abs(corr)))
            except:
                pass

        correlations.sort(key=lambda x: x[1], reverse=True)
        print(f"  Top correlated features:")
        for feat, corr in correlations[:5]:
            print(f"    {feat}: {corr:.3f}")

    # 4. Check if model exists and load it
    print("\n4. MODEL CHECK")
    print("-" * 30)

    model_data = model_trainer.load_model(account_id)
    if model_data is None:
        print(f"  ✗ No model found for {trader_name}")
        return

    print(f"  ✓ Model found for {trader_name}")
    print(f"  Training date: {model_data.get('training_date', 'Unknown')}")
    print(f"  Validation RMSE: {model_data.get('validation_metrics', {}).get('rmse', 'Unknown')}")
    print(f"  Test RMSE: {model_data.get('test_metrics', {}).get('test_rmse', 'Unknown')}")
    print(f"  Threshold: {model_data.get('threshold', 'Unknown')}")

    # Test prediction on recent data
    feature_columns = model_data['feature_columns']
    if len(features_df) > 0:
        latest_features = features_df[feature_columns].iloc[-1:].values
        model = model_data['model']

        try:
            prediction = model.predict(latest_features, num_iteration=model.best_iteration)[0]
            print(f"  Sample prediction: ${prediction:.2f}")

            if abs(prediction) < 0.01:
                print("  🚨 WARNING: Model predicting near-zero values!")
        except Exception as e:
            print(f"  ✗ Prediction failed: {str(e)}")

    # 5. Check feature engineering issues
    print("\n5. FEATURE ENGINEERING CHECK")
    print("-" * 30)

    # Check for NaN or infinite values
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    nan_counts = features_df[numeric_cols].isnull().sum()
    inf_counts = np.isinf(features_df[numeric_cols]).sum()

    problematic_features = []
    for col in numeric_cols:
        if nan_counts[col] > 0 or inf_counts[col] > 0:
            problematic_features.append(col)

    if problematic_features:
        print(f"  ⚠️ Features with NaN/Inf values: {len(problematic_features)}")
        for feat in problematic_features[:5]:
            print(f"    {feat}: {nan_counts.get(feat, 0)} NaN, {inf_counts.get(feat, 0)} Inf")
    else:
        print("  ✓ No NaN/Inf values found")

    # Check feature variance
    low_variance_features = []
    for col in numeric_cols:
        if features_df[col].std() < 0.01:
            low_variance_features.append(col)

    if low_variance_features:
        print(f"  ⚠️ Low variance features: {len(low_variance_features)}")
        for feat in low_variance_features[:5]:
            print(f"    {feat}: std={features_df[feat].std():.6f}")

    print("\n" + "="*60)
    print("DIAGNOSIS COMPLETE")
    print("="*60)

    # Summary recommendations
    print("\nRECOMMENDATIONS:")
    if recent_data_count < len(traders_df) * 0.5:
        print("1. 🔴 DATA ISSUE: Less than 50% of traders have recent data")
        print("   - Check data download process")
        print("   - Verify API connectivity")

    if 'target' in features_df.columns and features_df['target'].std() < 0.01:
        print("2. 🔴 MODEL ISSUE: Target variable has no variance")
        print("   - Check feature engineering logic")
        print("   - Verify target creation (next day P&L)")

    print("3. 💡 Next steps:")
    print("   - Run: python scripts/setup_database.py (if data is missing)")
    print("   - Check: Feature engineering in src/feature_engineer.py")
    print("   - Review: Model training splits in src/model_trainer.py")

if __name__ == "__main__":
    diagnose_data_and_models()
