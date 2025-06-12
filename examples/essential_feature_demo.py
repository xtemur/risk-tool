"""
Essential Feature Engineering Demo

Demonstrates the new essential feature extraction with data leakage prevention.
Creates 15-20 high-quality features from the existing dataset.
"""

import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add the src directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))
os.chdir(project_root)

import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Import our essential feature extractor
from feature_engineering.essential_features import EssentialFeatureExtractor
from data.database_manager import DatabaseManager
from feature_engineering.feature_processor import FeatureProcessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_existing_data():
    """Load existing processed data"""
    logger.info("Loading existing processed data...")

    try:
        # Try to load existing processed features
        data = pd.read_csv("data/processed/features_demo.csv")
        logger.info(f"Loaded {len(data)} samples from existing features")

        # Convert date column
        data['date'] = pd.to_datetime(data['date'])

        return data

    except FileNotFoundError:
        logger.error("Existing feature data not found. Please run feature engineering first.")
        logger.info("Command: python examples/feature_engineering_demo.py")
        return None


def create_essential_features():
    """Create essential features from raw data"""
    logger.info("Creating essential features from raw database...")

    # Initialize components
    db_manager = DatabaseManager("data/trading_risk.db")

    # Get accounts
    accounts = db_manager.get_accounts()
    if accounts.empty:
        logger.error("No accounts found in database!")
        return None

    all_data = []

    for _, account in accounts.iterrows():
        account_id = account['account_id']
        logger.info(f"Processing account {account_id}...")

        # Get daily summary data (use raw data without heavy processing)
        daily_summary = db_manager.get_account_daily_summary(account_id=account_id)
        if daily_summary.empty:
            continue

        # Basic cleaning only
        daily_summary = daily_summary.copy()
        daily_summary['date'] = pd.to_datetime(daily_summary['date'])

        # Keep essential columns and remove unnecessary ones
        keep_cols = ['account_id', 'date', 'fills', 'qty', 'gross', 'net', 'unrealized']
        available_cols = [col for col in keep_cols if col in daily_summary.columns]
        daily_summary = daily_summary[available_cols]

        # Create target variable (next day's PnL)
        daily_summary = daily_summary.sort_values(['account_id', 'date']).reset_index(drop=True)
        daily_summary['target_next_pnl'] = daily_summary.groupby('account_id')['net'].shift(-1)

        # Remove last row per account (no target)
        daily_summary = daily_summary.dropna(subset=['target_next_pnl'])

        if not daily_summary.empty:
            all_data.append(daily_summary)

    if not all_data:
        logger.error("No data processed from any accounts!")
        return None

    # Combine all data
    combined_data = pd.concat(all_data, ignore_index=True)

    # Sort by account_id and date for temporal consistency
    combined_data = combined_data.sort_values(['account_id', 'date']).reset_index(drop=True)

    logger.info(f"Created dataset with {len(combined_data)} samples from {combined_data['account_id'].nunique()} accounts")

    return combined_data


def demo_essential_features():
    """
    Main demo function for essential feature engineering
    """
    print("=" * 80)
    print("ESSENTIAL FEATURE ENGINEERING DEMO")
    print("=" * 80)

    # Create fresh features from raw database to fix data quality issues
    logger.info("Creating fresh features from raw database data...")
    data = create_essential_features()

    if data is None:
        logger.info("Falling back to existing processed data...")
        data = load_existing_data()

    if data is None:
        logger.error("Could not load or create data!")
        return

    # Initialize essential feature extractor
    logger.info("\n1. Initializing Essential Feature Extractor...")
    extractor = EssentialFeatureExtractor(
        lookback_window=20,
        max_features=20,
        temporal_validation=True
    )

    # Extract essential features
    logger.info("\n2. Extracting Essential Features...")
    logger.info("This process focuses on 15-20 high-quality features with data leakage prevention")

    try:
        essential_data = extractor.extract_all_essential_features(data)

        logger.info(f"✓ Successfully extracted essential features")
        logger.info(f"✓ Original features: {len(data.columns)}")
        logger.info(f"✓ Essential features: {len(essential_data.columns) - 3}")  # Exclude account_id, date, target

    except Exception as e:
        logger.error(f"✗ Feature extraction failed: {e}")
        return

    # Feature validation
    logger.info("\n3. Validating Features for Data Leakage...")
    validation_report = extractor.validate_features_temporal(essential_data)

    if validation_report['temporal_ordering_valid']:
        logger.info("✓ Temporal ordering validation passed")
    else:
        logger.error("✗ Temporal ordering validation failed")

    if not validation_report['data_leakage_detected']:
        logger.info("✓ No data leakage detected")
    else:
        logger.warning("⚠️ Potential data leakage detected")

    # Missing data analysis
    logger.info("\n4. Missing Data Analysis...")
    missing_stats = validation_report['missing_data_stats']
    high_missing = {k: v for k, v in missing_stats.items() if v > 0.1}

    if high_missing:
        logger.warning(f"Features with >10% missing data: {high_missing}")
    else:
        logger.info("✓ All features have <10% missing data")

    # Feature summary
    logger.info("\n5. Feature Summary...")
    summary = extractor.get_feature_summary()

    logger.info(f"Total selected features: {summary['total_selected_features']}")
    logger.info("Features by category:")
    for category, features in summary['features_by_category'].items():
        if features:
            logger.info(f"  {category}: {len(features)} features - {features}")

    # Feature selection demonstration
    logger.info("\n6. Feature Selection Demonstration...")
    if 'target_next_pnl' in essential_data.columns:
        # Prepare data for feature selection
        feature_cols = [col for col in essential_data.columns
                       if col not in ['account_id', 'date', 'target_next_pnl']]

        X = essential_data[feature_cols]
        y = essential_data['target_next_pnl']

        # Remove rows with missing targets
        valid_idx = y.notna()
        X_clean = X[valid_idx]
        y_clean = y[valid_idx]

        if len(X_clean) > 100:  # Minimum samples for feature selection
            try:
                # Select top features using mutual information
                top_features = extractor.select_best_features(
                    X_clean, y_clean,
                    method='mutual_info',
                    n_features=15
                )

                logger.info(f"Top features by mutual information: {top_features}")

                # Statistical summary of selected features
                logger.info("\n7. Selected Features Statistics...")
                if top_features:
                    selected_data = X_clean[top_features]

                    stats_summary = {
                        'mean_correlation_with_target': abs(selected_data.corrwith(y_clean)).mean(),
                        'mean_feature_correlation': abs(selected_data.corr()).mean().mean(),
                        'missing_data_rate': selected_data.isnull().mean().mean()
                    }

                    for stat, value in stats_summary.items():
                        logger.info(f"  {stat}: {value:.4f}")
                else:
                    logger.warning("No features selected after filtering")

            except Exception as e:
                logger.warning(f"Feature selection failed: {e}")
                logger.info("This is expected with high missing data rates")

        else:
            logger.warning("Insufficient data for feature selection demonstration")

    # Save results
    logger.info("\n8. Saving Results...")
    output_dir = Path("data/processed")
    output_dir.mkdir(exist_ok=True, parents=True)

    # Save essential features
    essential_output_path = output_dir / "essential_features.csv"
    essential_data.to_csv(essential_output_path, index=False)
    logger.info(f"✓ Essential features saved to {essential_output_path}")

    # Save feature summary
    summary_output_path = output_dir / "essential_features_summary.json"
    import json
    with open(summary_output_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"✓ Feature summary saved to {summary_output_path}")

    # Data quality report
    logger.info("\n9. Data Quality Report...")
    logger.info(f"Total samples: {len(essential_data)}")
    logger.info(f"Date range: {essential_data['date'].min()} to {essential_data['date'].max()}")
    logger.info(f"Accounts: {essential_data['account_id'].nunique()}")

    if 'target_next_pnl' in essential_data.columns:
        target_stats = essential_data['target_next_pnl'].describe()
        logger.info(f"Target statistics: mean={target_stats['mean']:.2f}, std={target_stats['std']:.2f}")
        logger.info(f"Target missing rate: {essential_data['target_next_pnl'].isnull().mean():.1%}")

    # Next steps
    logger.info("\n" + "=" * 80)
    logger.info("ESSENTIAL FEATURE ENGINEERING COMPLETED!")
    logger.info("=" * 80)
    logger.info("\nKey Improvements:")
    logger.info(f"  ✓ Reduced features from ~56 to {summary['total_selected_features']} essential ones")
    logger.info("  ✓ Implemented data leakage prevention")
    logger.info("  ✓ Added advanced position sizing features")
    logger.info("  ✓ Added loss management discipline features")
    logger.info("  ✓ Added risk control adherence features")
    logger.info("  ✓ Added market adaptation features")

    logger.info("\nNext Steps:")
    logger.info("  1. Train models with: python train_xgboost_model.py")
    logger.info("  2. Compare performance vs original features")
    logger.info("  3. Run temporal validation tests")
    logger.info("  4. Validate with walk-forward testing")

    return essential_data, summary


if __name__ == "__main__":
    try:
        essential_data, summary = demo_essential_features()
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
