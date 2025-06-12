# scripts/test_features.py
"""
Test script to verify feature engineering works with actual data
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from src.data.database_manager import DatabaseManager
from src.features.feature_pipeline import FeaturePipeline


def test_feature_generation():
    """Test feature generation with real data"""

    # Initialize
    db = DatabaseManager()
    pipeline = FeaturePipeline()

    # Get accounts
    accounts = db.get_accounts()
    print(f"Found {len(accounts)} accounts")

    # Test with first account
    if not accounts.empty:
        account_id = accounts.iloc[0]['account_id']
        account_name = accounts.iloc[0]['account_name']

        print(f"\nTesting with account: {account_name} ({account_id})")

        # Get data
        daily_summary = db.get_account_daily_summary(account_id=account_id)
        fills = db.get_fills(account_id=account_id)

        print(f"Daily summary records: {len(daily_summary)}")
        print(f"Fills records: {len(fills)}")

        if len(daily_summary) >= 20:  # Need minimum data
            # Generate features
            features = pipeline.generate_features(daily_summary, fills, account_id)

            print(f"\nGenerated features shape: {features.shape}")
            print(f"Features: {features.columns.tolist()[:10]}...")  # First 10

            # Show sample
            print("\nSample features:")
            print(features.head())

            # Check for NaN
            nan_counts = features.isnull().sum()
            high_nan = nan_counts[nan_counts > len(features) * 0.5]
            if len(high_nan) > 0:
                print(f"\nWarning: {len(high_nan)} features have >50% missing values")

            # Feature selection test
            if 'net' in daily_summary.columns:
                # Create target (next day's P&L)
                target = daily_summary.set_index('date')['net'].shift(-1)
                target = target.dropna()

                # Select features
                selected = pipeline.select_features(
                    features.set_index('date'),
                    target,
                    method='correlation',
                    top_k=20
                )

                print(f"\nTop 20 features by correlation:")
                for i, feat in enumerate(selected[:10], 1):
                    print(f"  {i}. {feat}")

            return features
        else:
            print(f"Not enough data for account {account_id}")

    return None


if __name__ == "__main__":
    features = test_feature_generation()

    if features is not None:
        print("\n✅ Feature engineering test passed!")
    else:
        print("\n❌ Feature engineering test failed!")
