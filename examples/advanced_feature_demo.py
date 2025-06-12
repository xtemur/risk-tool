"""
Advanced Feature Engineering Demo

Demonstrates Phase 1 improvements:
- Alternative target variables with better signal-to-noise ratios
- Cross-sectional and behavioral features
- Improved validation framework
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

# Import our advanced feature engineering
from feature_engineering.advanced_features import AdvancedFeatureEngineer
from data.database_manager import DatabaseManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_clean_data():
    """Load clean data directly from database"""
    logger.info("Loading clean data from database...")

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

        # Get daily summary data (use raw data)
        daily_summary = db_manager.get_account_daily_summary(account_id=account_id)
        if daily_summary.empty:
            continue

        # Basic cleaning only
        daily_summary = daily_summary.copy()
        daily_summary['date'] = pd.to_datetime(daily_summary['date'])

        # Keep essential columns
        keep_cols = ['account_id', 'date', 'fills', 'qty', 'gross', 'net', 'unrealized']
        available_cols = [col for col in keep_cols if col in daily_summary.columns]
        daily_summary = daily_summary[available_cols]

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


def analyze_target_improvements(original_data, advanced_data):
    """Analyze improvements in target variables"""
    logger.info("\n" + "="*60)
    logger.info("TARGET VARIABLE ANALYSIS")
    logger.info("="*60)

    # Original target analysis
    if 'target_next_pnl' in original_data.columns:
        orig_target = original_data['target_next_pnl']
        orig_snr = abs(orig_target.mean()) / (orig_target.std() + 1e-6)

        logger.info("Original Daily PnL Target:")
        logger.info(f"  Signal-to-Noise Ratio: {orig_snr:.4f}")
        logger.info(f"  Mean: {orig_target.mean():.2f}")
        logger.info(f"  Std: {orig_target.std():.2f}")
        logger.info(f"  Kurtosis: {orig_target.kurtosis():.2f}")

    # Advanced target analysis
    target_cols = [col for col in advanced_data.columns if col.startswith('target_')]

    logger.info(f"\nAdvanced Target Variables ({len(target_cols)} created):")
    logger.info("-" * 60)

    improvements = []

    for col in target_cols:
        if advanced_data[col].notna().sum() > 100:
            target_data = advanced_data[col].dropna()

            if 'direction' in col or 'above_median' in col or 'quartile' in col:
                # Classification target
                class_balance = target_data.mean()
                # For classification, "signal" is deviation from 50%
                signal_strength = abs(class_balance - 0.5) * 2

                logger.info(f"  {col}:")
                logger.info(f"    Type: Classification")
                logger.info(f"    Class Balance: {class_balance:.3f} (positive class)")
                logger.info(f"    Signal Strength: {signal_strength:.3f}")

                improvements.append({
                    'target': col,
                    'type': 'classification',
                    'signal_metric': signal_strength,
                    'baseline_comparison': signal_strength > 0.1
                })

            else:
                # Regression target
                snr = abs(target_data.mean()) / (target_data.std() + 1e-6)
                kurtosis = target_data.kurtosis()

                logger.info(f"  {col}:")
                logger.info(f"    Type: Regression")
                logger.info(f"    Signal-to-Noise: {snr:.4f}")
                logger.info(f"    Mean: {target_data.mean():.2f}")
                logger.info(f"    Std: {target_data.std():.2f}")
                logger.info(f"    Kurtosis: {kurtosis:.2f}")

                # Compare to original
                if 'target_next_pnl' in original_data.columns:
                    improvement = snr / orig_snr
                    logger.info(f"    SNR Improvement: {improvement:.2f}x")

                improvements.append({
                    'target': col,
                    'type': 'regression',
                    'signal_metric': snr,
                    'baseline_comparison': snr > orig_snr if 'target_next_pnl' in original_data.columns else True
                })

    # Summary of improvements
    logger.info("\nTarget Improvement Summary:")
    logger.info("-" * 40)

    classification_targets = [x for x in improvements if x['type'] == 'classification']
    regression_targets = [x for x in improvements if x['type'] == 'regression']

    logger.info(f"Classification Targets: {len(classification_targets)}")
    good_classification = sum(1 for x in classification_targets if x['signal_metric'] > 0.1)
    logger.info(f"  Strong Signal (>0.1): {good_classification}/{len(classification_targets)}")

    logger.info(f"Regression Targets: {len(regression_targets)}")
    if regression_targets:
        avg_snr = np.mean([x['signal_metric'] for x in regression_targets])
        logger.info(f"  Average SNR: {avg_snr:.4f}")
        better_than_baseline = sum(1 for x in regression_targets if x['baseline_comparison'])
        logger.info(f"  Better than baseline: {better_than_baseline}/{len(regression_targets)}")

    return improvements


def analyze_feature_improvements(advanced_data):
    """Analyze new features created"""
    logger.info("\n" + "="*60)
    logger.info("FEATURE ENGINEERING ANALYSIS")
    logger.info("="*60)

    # Categorize features
    feature_categories = {
        'cross_sectional': [col for col in advanced_data.columns if any(x in col for x in ['market_', 'rank', 'relative'])],
        'behavioral': [col for col in advanced_data.columns if any(x in col for x in ['win_streak', 'loss', 'momentum', 'consistency'])],
        'market_regime': [col for col in advanced_data.columns if any(x in col for x in ['regime', 'vol_', 'stress', 'monday', 'friday'])],
        'risk_adjusted': [col for col in advanced_data.columns if any(x in col for x in ['vol_adjusted', 'winsorized'])]
    }

    logger.info("Feature Categories Created:")
    for category, features in feature_categories.items():
        if features:
            logger.info(f"  {category}: {len(features)} features")
            # Show some examples
            examples = features[:3]
            logger.info(f"    Examples: {', '.join(examples)}")

    # Data quality analysis
    logger.info("\nData Quality Metrics:")
    total_features = len([col for col in advanced_data.columns if col not in ['account_id', 'date']])
    logger.info(f"  Total Features: {total_features}")

    # Missing data analysis
    missing_rates = advanced_data.isnull().mean()
    high_missing = missing_rates[missing_rates > 0.1]

    if len(high_missing) > 0:
        logger.warning(f"  Features with >10% missing: {len(high_missing)}")
    else:
        logger.info("  ✓ All features have <10% missing data")

    # Feature correlations (sample)
    numeric_features = advanced_data.select_dtypes(include=[np.number]).columns
    feature_subset = [col for col in numeric_features if not col.startswith('target_')][:20]

    if len(feature_subset) > 1:
        corr_matrix = advanced_data[feature_subset].corr()
        high_corr_pairs = []

        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = abs(corr_matrix.iloc[i, j])
                if corr_val > 0.8:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))

        if high_corr_pairs:
            logger.warning(f"  High correlation pairs (>0.8): {len(high_corr_pairs)}")
        else:
            logger.info("  ✓ No excessive feature correlations detected")


def demo_validation_improvements(advanced_data):
    """Demonstrate improved validation framework"""
    logger.info("\n" + "="*60)
    logger.info("VALIDATION FRAMEWORK IMPROVEMENTS")
    logger.info("="*60)

    # Walk-forward validation setup
    dates = pd.to_datetime(advanced_data['date']).unique()
    dates = np.sort(dates)

    n_periods = len(dates)
    train_size = int(0.7 * n_periods)

    logger.info("Walk-Forward Validation Setup:")
    logger.info(f"  Total time periods: {n_periods}")
    logger.info(f"  Training periods: {train_size}")
    logger.info(f"  Test periods: {n_periods - train_size}")
    logger.info(f"  Date range: {pd.to_datetime(dates[0]).strftime('%Y-%m-%d')} to {pd.to_datetime(dates[-1]).strftime('%Y-%m-%d')}")

    # Cross-trader validation
    traders = advanced_data['account_id'].unique()
    logger.info(f"\nCross-Trader Validation:")
    logger.info(f"  Total traders: {len(traders)}")
    logger.info(f"  Suggested train/test split: {int(0.7 * len(traders))}/{len(traders) - int(0.7 * len(traders))}")

    # Regime-based validation
    if 'high_vol_regime' in advanced_data.columns:
        regime_dist = advanced_data['high_vol_regime'].value_counts()
        logger.info(f"\nRegime Distribution:")
        logger.info(f"  Low volatility periods: {regime_dist.get(0, 0)} ({regime_dist.get(0, 0)/len(advanced_data):.1%})")
        logger.info(f"  High volatility periods: {regime_dist.get(1, 0)} ({regime_dist.get(1, 0)/len(advanced_data):.1%})")


def main():
    """Main demo function"""
    print("=" * 80)
    print("ADVANCED FEATURE ENGINEERING DEMO")
    print("=" * 80)

    # Load clean data
    data = load_clean_data()
    if data is None:
        logger.error("Could not load data!")
        return

    # Create original target for comparison
    data = data.sort_values(['account_id', 'date']).reset_index(drop=True)
    data['target_next_pnl'] = data.groupby('account_id')['net'].shift(-1)
    data = data.dropna(subset=['target_next_pnl'])

    logger.info(f"Starting with {len(data)} samples")

    # Initialize advanced feature engineer
    logger.info("\n1. Initializing Advanced Feature Engineer...")
    engineer = AdvancedFeatureEngineer(
        target_smoothing_windows=[3, 5, 7],
        outlier_percentile=0.05,
        volatility_window=20
    )

    # Process advanced features
    logger.info("\n2. Processing Advanced Features...")
    try:
        advanced_data = engineer.process_all_advanced_features(data)
        logger.info("✓ Advanced feature engineering completed successfully")
    except Exception as e:
        logger.error(f"✗ Feature engineering failed: {e}")
        return

    # Analyze improvements
    logger.info("\n3. Analyzing Target Variable Improvements...")
    target_improvements = analyze_target_improvements(data, advanced_data)

    logger.info("\n4. Analyzing Feature Engineering Improvements...")
    analyze_feature_improvements(advanced_data)

    logger.info("\n5. Demonstrating Validation Improvements...")
    demo_validation_improvements(advanced_data)

    # Save results
    logger.info("\n6. Saving Advanced Features...")
    output_dir = Path("data/processed")
    output_dir.mkdir(exist_ok=True, parents=True)

    # Save advanced features
    advanced_output_path = output_dir / "advanced_features.csv"
    advanced_data.to_csv(advanced_output_path, index=False)
    logger.info(f"✓ Advanced features saved to {advanced_output_path}")

    # Save target summary
    target_summary = engineer.get_target_summary(advanced_data)
    import json
    summary_path = output_dir / "advanced_targets_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(target_summary, f, indent=2, default=str)
    logger.info(f"✓ Target summary saved to {summary_path}")

    # Final recommendations
    logger.info("\n" + "="*80)
    logger.info("RECOMMENDATIONS FOR NEXT STEPS")
    logger.info("="*80)

    # Count promising targets
    classification_strong = sum(1 for x in target_improvements
                               if x['type'] == 'classification' and x['signal_metric'] > 0.1)
    regression_improved = sum(1 for x in target_improvements
                             if x['type'] == 'regression' and x['baseline_comparison'])

    logger.info("Target Variable Recommendations:")
    if classification_strong > 0:
        logger.info(f"  ✓ Try classification models with {classification_strong} strong direction targets")
    if regression_improved > 0:
        logger.info(f"  ✓ Try regression models with {regression_improved} improved SNR targets")

    logger.info("\nModeling Recommendations:")
    logger.info("  1. Start with classification (direction prediction) - easier than exact PnL")
    logger.info("  2. Use multi-day smoothed targets to reduce noise")
    logger.info("  3. Implement walk-forward validation for realistic performance estimates")
    logger.info("  4. Try quantile regression for robust predictions")
    logger.info("  5. Consider regime-specific models (high/low volatility)")

    logger.info("\nExpected Improvements:")
    logger.info("  • Classification accuracy: 55-65% (vs current 45% hit rate)")
    logger.info("  • Reduced overfitting with cross-sectional features")
    logger.info("  • More stable performance across market regimes")
    logger.info("  • Better risk-adjusted predictions")

    return advanced_data, target_improvements


if __name__ == "__main__":
    try:
        advanced_data, improvements = main()
        logger.info("\n✓ Advanced feature engineering demo completed successfully!")
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
