"""
Train Models with Improved Target Variables

Tests the advanced features and alternative targets to demonstrate improvements
over the original approach.
"""

import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add the src directory to the Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root / "src"))
os.chdir(project_root)

import pandas as pd
import numpy as np
from datetime import datetime
import logging
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_advanced_features():
    """Load the advanced features dataset"""
    try:
        df = pd.read_csv("data/processed/advanced_features.csv")
        logger.info(f"Loaded {len(df)} samples with {df.shape[1]} features")
        return df
    except FileNotFoundError:
        logger.error("Advanced features not found. Please run: python examples/advanced_feature_demo.py")
        return None


def prepare_features_and_targets(df):
    """Prepare feature matrix and target variables"""
    # Basic features (exclude target and metadata columns)
    feature_cols = [col for col in df.columns
                   if not col.startswith('target_')
                   and col not in ['account_id', 'date']]

    X = df[feature_cols].copy()

    # Handle missing values
    X = X.fillna(X.median())

    # Target variables to test
    targets = {
        'original_pnl': 'target_next_pnl',
        'smoothed_3d': 'target_pnl_3d_winsorized',
        'smoothed_7d': 'target_pnl_7d_winsorized',
        'direction_1d': 'target_direction_1d',
        'direction_5d': 'target_direction_5d',
        'top_quartile': 'target_top_quartile_1d'
    }

    # Keep only available targets
    available_targets = {k: v for k, v in targets.items() if v in df.columns}

    logger.info(f"Features: {len(feature_cols)}")
    logger.info(f"Available targets: {list(available_targets.keys())}")

    return X, available_targets, df


def walk_forward_validation(X, y, test_ratio=0.3):
    """
    Implement walk-forward validation for time series data
    """
    n_samples = len(X)
    train_size = int((1 - test_ratio) * n_samples)

    X_train = X.iloc[:train_size]
    X_test = X.iloc[train_size:]
    y_train = y.iloc[:train_size]
    y_test = y.iloc[train_size:]

    return X_train, X_test, y_train, y_test


def train_classification_models(X_train, X_test, y_train, y_test, target_name):
    """Train and evaluate classification models"""
    logger.info(f"\nTraining Classification Models for {target_name}")
    logger.info("-" * 50)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    }

    results = {}

    for model_name, model in models.items():
        # Train model
        model.fit(X_train_scaled, y_train)

        # Predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        class_balance = y_test.mean()

        logger.info(f"{model_name}:")
        logger.info(f"  Accuracy: {accuracy:.3f}")
        logger.info(f"  Class Balance: {class_balance:.3f}")
        logger.info(f"  Improvement vs Random: {(accuracy - 0.5) * 100:+.1f} percentage points")

        # Feature importance for Random Forest
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

            logger.info(f"  Top 5 Features: {feature_importance.head(5)['feature'].tolist()}")

        results[model_name] = {
            'accuracy': accuracy,
            'class_balance': class_balance,
            'improvement_vs_random': (accuracy - 0.5) * 100,
            'predictions': y_pred.tolist(),
            'probabilities': y_pred_proba.tolist() if y_pred_proba is not None else None
        }

    return results


def train_regression_models(X_train, X_test, y_train, y_test, target_name):
    """Train and evaluate regression models"""
    logger.info(f"\nTraining Regression Models for {target_name}")
    logger.info("-" * 50)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'Ridge Regression': Ridge(alpha=1.0, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    }

    results = {}

    for model_name, model in models.items():
        # Train model
        model.fit(X_train_scaled, y_train)

        # Predictions
        y_pred = model.predict(X_test_scaled)

        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Signal-to-noise ratios
        actual_snr = abs(y_test.mean()) / (y_test.std() + 1e-6)
        pred_snr = abs(y_pred.mean()) / (y_pred.std() + 1e-6)

        logger.info(f"{model_name}:")
        logger.info(f"  MAE: {mae:.2f}")
        logger.info(f"  R²: {r2:.4f}")
        logger.info(f"  Actual SNR: {actual_snr:.4f}")
        logger.info(f"  Predicted SNR: {pred_snr:.4f}")

        # Feature importance for Random Forest
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

            logger.info(f"  Top 5 Features: {feature_importance.head(5)['feature'].tolist()}")

        results[model_name] = {
            'mae': mae,
            'r2': r2,
            'actual_snr': actual_snr,
            'predicted_snr': pred_snr,
            'predictions': y_pred.tolist()
        }

    return results


def compare_target_performance(all_results):
    """Compare performance across different target variables"""
    logger.info("\n" + "="*60)
    logger.info("TARGET PERFORMANCE COMPARISON")
    logger.info("="*60)

    # Classification targets
    classification_targets = []
    regression_targets = []

    for target_name, models in all_results.items():
        if 'direction' in target_name or 'quartile' in target_name:
            classification_targets.append(target_name)
        else:
            regression_targets.append(target_name)

    # Classification comparison
    if classification_targets:
        logger.info("\nClassification Targets (Accuracy):")
        logger.info("-" * 40)

        for target in classification_targets:
            if target in all_results:
                rf_acc = all_results[target]['Random Forest']['accuracy']
                improvement = all_results[target]['Random Forest']['improvement_vs_random']
                logger.info(f"  {target}: {rf_acc:.3f} ({improvement:+.1f}pp vs random)")

    # Regression comparison
    if regression_targets:
        logger.info("\nRegression Targets (R²):")
        logger.info("-" * 40)

        baseline_r2 = None
        if 'original_pnl' in all_results:
            baseline_r2 = all_results['original_pnl']['Random Forest']['r2']
            logger.info(f"  original_pnl (baseline): {baseline_r2:.4f}")

        for target in regression_targets:
            if target != 'original_pnl' and target in all_results:
                r2 = all_results[target]['Random Forest']['r2']
                improvement = ""
                if baseline_r2 is not None:
                    if r2 > baseline_r2:
                        improvement = f" ({(r2/baseline_r2 - 1)*100:+.1f}% vs baseline)"
                    else:
                        improvement = f" ({(r2/baseline_r2 - 1)*100:.1f}% vs baseline)"

                logger.info(f"  {target}: {r2:.4f}{improvement}")


def main():
    """Main training function"""
    print("=" * 80)
    print("IMPROVED MODEL TRAINING WITH ADVANCED TARGETS")
    print("=" * 80)

    # Load data
    df = load_advanced_features()
    if df is None:
        return

    # Prepare features and targets
    X, targets, df = prepare_features_and_targets(df)

    # Results storage
    all_results = {}

    # Train models for each target
    for target_name, target_col in targets.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"PROCESSING TARGET: {target_name}")
        logger.info(f"{'='*60}")

        # Get target data
        y = df[target_col].copy()

        # Remove rows with missing targets
        valid_mask = y.notna()
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]

        logger.info(f"Valid samples: {len(y_valid)}")

        if len(y_valid) < 100:
            logger.warning(f"Insufficient data for {target_name}, skipping...")
            continue

        # Split data using walk-forward validation
        X_train, X_test, y_train, y_test = walk_forward_validation(X_valid, y_valid)

        logger.info(f"Train samples: {len(X_train)}")
        logger.info(f"Test samples: {len(X_test)}")

        # Determine if classification or regression
        if target_col in ['target_direction_1d', 'target_direction_5d', 'target_top_quartile_1d']:
            # Classification
            results = train_classification_models(X_train, X_test, y_train, y_test, target_name)
        else:
            # Regression
            results = train_regression_models(X_train, X_test, y_train, y_test, target_name)

        all_results[target_name] = results

    # Compare performance across targets
    compare_target_performance(all_results)

    # Save results
    output_dir = Path("results/improved_models")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "target_comparison_results.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info(f"\nResults saved to {output_dir / 'target_comparison_results.json'}")

    # Final recommendations
    logger.info("\n" + "="*80)
    logger.info("IMPROVEMENT RECOMMENDATIONS")
    logger.info("="*80)

    # Find best performers
    best_classification = None
    best_classification_acc = 0

    best_regression = None
    best_regression_r2 = -float('inf')

    for target_name, models in all_results.items():
        if 'Random Forest' in models:
            if 'accuracy' in models['Random Forest']:
                acc = models['Random Forest']['accuracy']
                if acc > best_classification_acc:
                    best_classification_acc = acc
                    best_classification = target_name

            if 'r2' in models['Random Forest']:
                r2 = models['Random Forest']['r2']
                if r2 > best_regression_r2:
                    best_regression_r2 = r2
                    best_regression = target_name

    logger.info("Key Findings:")
    if best_classification:
        improvement = (best_classification_acc - 0.5) * 100
        logger.info(f"  • Best Classification Target: {best_classification} ({best_classification_acc:.3f} accuracy, {improvement:+.1f}pp vs random)")

    if best_regression:
        logger.info(f"  • Best Regression Target: {best_regression} (R² = {best_regression_r2:.4f})")

    logger.info("\nNext Steps:")
    logger.info("  1. Focus on classification targets for more reliable predictions")
    logger.info("  2. Use multi-day smoothed targets to reduce noise")
    logger.info("  3. Consider ensemble methods combining multiple targets")
    logger.info("  4. Implement proper temporal validation in production")

    return all_results


if __name__ == "__main__":
    try:
        results = main()
        logger.info("\n✓ Improved model training completed successfully!")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
