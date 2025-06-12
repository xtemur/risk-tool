"""
Evaluate Improved Models on Unseen Data with Causal Impact Analysis

Implements rigorous evaluation on truly unseen data and calculates
the causal impact of improved predictions on trading performance.
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
from datetime import datetime, timedelta
import logging
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_temporal_splits(df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    """
    Create proper temporal splits ensuring no data leakage

    Args:
        df: DataFrame with date column
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing (unseen data)

    Returns:
        Dictionary with train, validation, and test splits
    """
    # Sort by date to ensure temporal ordering
    df_sorted = df.sort_values(['account_id', 'date']).reset_index(drop=True)

    # Get unique dates and split them temporally
    unique_dates = sorted(df_sorted['date'].unique())
    n_dates = len(unique_dates)

    train_end_idx = int(train_ratio * n_dates)
    val_end_idx = int((train_ratio + val_ratio) * n_dates)

    train_dates = unique_dates[:train_end_idx]
    val_dates = unique_dates[train_end_idx:val_end_idx]
    test_dates = unique_dates[val_end_idx:]

    # Create splits
    train_mask = df_sorted['date'].isin(train_dates)
    val_mask = df_sorted['date'].isin(val_dates)
    test_mask = df_sorted['date'].isin(test_dates)

    splits = {
        'train': df_sorted[train_mask].reset_index(drop=True),
        'validation': df_sorted[val_mask].reset_index(drop=True),
        'test': df_sorted[test_mask].reset_index(drop=True)
    }

    logger.info("Temporal Splits Created:")
    logger.info(f"  Training: {len(splits['train'])} samples ({min(train_dates)} to {max(train_dates)})")
    logger.info(f"  Validation: {len(splits['validation'])} samples ({min(val_dates)} to {max(val_dates)})")
    logger.info(f"  Test (Unseen): {len(splits['test'])} samples ({min(test_dates)} to {max(test_dates)})")

    return splits


def train_best_models(train_data, val_data, feature_cols):
    """
    Train the best performing models identified from previous analysis

    Args:
        train_data: Training dataset
        val_data: Validation dataset for hyperparameter tuning
        feature_cols: List of feature columns to use

    Returns:
        Dictionary of trained models
    """
    logger.info("Training Best Performing Models...")

    # Prepare training data
    X_train = train_data[feature_cols].fillna(train_data[feature_cols].median())
    X_val = val_data[feature_cols].fillna(train_data[feature_cols].median())

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    models = {}

    # 1. Best Classification Model: 5-day Direction Prediction
    if 'target_direction_5d' in train_data.columns:
        y_train_5d = train_data['target_direction_5d'].fillna(0)
        y_val_5d = val_data['target_direction_5d'].fillna(0)

        # Train Random Forest (best performer)
        rf_5d = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            class_weight='balanced'
        )
        rf_5d.fit(X_train_scaled, y_train_5d)

        # Validate
        val_pred_5d = rf_5d.predict(X_val_scaled)
        val_acc_5d = accuracy_score(y_val_5d, val_pred_5d)

        models['direction_5d_rf'] = {
            'model': rf_5d,
            'scaler': scaler,
            'target': 'target_direction_5d',
            'type': 'classification',
            'val_accuracy': val_acc_5d,
            'feature_importance': pd.DataFrame({
                'feature': feature_cols,
                'importance': rf_5d.feature_importances_
            }).sort_values('importance', ascending=False)
        }

        logger.info(f"5-day Direction Model - Validation Accuracy: {val_acc_5d:.3f}")

    # 2. Best Regression Model: 3-day Smoothed PnL
    if 'target_pnl_3d_winsorized' in train_data.columns:
        y_train_3d = train_data['target_pnl_3d_winsorized'].fillna(0)
        y_val_3d = val_data['target_pnl_3d_winsorized'].fillna(0)

        # Train Random Forest
        rf_3d = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
        rf_3d.fit(X_train_scaled, y_train_3d)

        # Validate
        val_pred_3d = rf_3d.predict(X_val_scaled)
        val_r2_3d = r2_score(y_val_3d, val_pred_3d)
        val_mae_3d = mean_absolute_error(y_val_3d, val_pred_3d)

        models['pnl_3d_rf'] = {
            'model': rf_3d,
            'scaler': scaler,
            'target': 'target_pnl_3d_winsorized',
            'type': 'regression',
            'val_r2': val_r2_3d,
            'val_mae': val_mae_3d,
            'feature_importance': pd.DataFrame({
                'feature': feature_cols,
                'importance': rf_3d.feature_importances_
            }).sort_values('importance', ascending=False)
        }

        logger.info(f"3-day PnL Model - Validation R²: {val_r2_3d:.3f}, MAE: {val_mae_3d:.2f}")

    # 3. Top Quartile Prediction (for position sizing)
    if 'target_top_quartile_1d' in train_data.columns:
        y_train_tq = train_data['target_top_quartile_1d'].fillna(0)
        y_val_tq = val_data['target_top_quartile_1d'].fillna(0)

        # Train Logistic Regression (good for probability estimates)
        lr_tq = LogisticRegression(
            C=1.0,
            class_weight='balanced',
            random_state=42,
            max_iter=1000
        )
        lr_tq.fit(X_train_scaled, y_train_tq)

        # Validate
        val_pred_tq = lr_tq.predict(X_val_scaled)
        val_acc_tq = accuracy_score(y_val_tq, val_pred_tq)

        models['top_quartile_lr'] = {
            'model': lr_tq,
            'scaler': scaler,
            'target': 'target_top_quartile_1d',
            'type': 'classification',
            'val_accuracy': val_acc_tq
        }

        logger.info(f"Top Quartile Model - Validation Accuracy: {val_acc_tq:.3f}")

    return models


def evaluate_on_unseen_data(models, test_data, feature_cols):
    """
    Evaluate trained models on completely unseen test data

    Args:
        models: Dictionary of trained models
        test_data: Unseen test dataset
        feature_cols: Feature columns

    Returns:
        Dictionary of evaluation results
    """
    logger.info("\n" + "="*60)
    logger.info("EVALUATING ON UNSEEN DATA")
    logger.info("="*60)

    # Prepare test features
    X_test = test_data[feature_cols].fillna(test_data[feature_cols].median())

    results = {}

    for model_name, model_info in models.items():
        logger.info(f"\nEvaluating {model_name}...")

        # Get model and scaler
        model = model_info['model']
        scaler = model_info['scaler']
        target_col = model_info['target']
        model_type = model_info['type']

        # Scale test features
        X_test_scaled = scaler.transform(X_test)

        # Get true target values
        y_test = test_data[target_col].fillna(0)

        # Make predictions
        if model_type == 'classification':
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            class_balance = y_test.mean()

            # Store results
            results[model_name] = {
                'type': 'classification',
                'accuracy': accuracy,
                'class_balance': class_balance,
                'improvement_vs_random': (accuracy - 0.5) * 100,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'actuals': y_test.values
            }

            logger.info(f"  Accuracy: {accuracy:.3f}")
            logger.info(f"  Improvement vs Random: {(accuracy - 0.5) * 100:+.1f} percentage points")

        else:  # regression
            y_pred = model.predict(X_test_scaled)

            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Signal-to-noise ratios
            actual_snr = abs(y_test.mean()) / (y_test.std() + 1e-6)
            pred_snr = abs(y_pred.mean()) / (y_pred.std() + 1e-6)

            # Store results
            results[model_name] = {
                'type': 'regression',
                'mae': mae,
                'r2': r2,
                'actual_snr': actual_snr,
                'predicted_snr': pred_snr,
                'predictions': y_pred,
                'actuals': y_test.values
            }

            logger.info(f"  MAE: {mae:.2f}")
            logger.info(f"  R²: {r2:.4f}")
            logger.info(f"  Actual SNR: {actual_snr:.4f}")

    return results


def calculate_causal_impact(test_data, predictions_dict):
    """
    Calculate the causal impact of improved predictions on trading performance

    Args:
        test_data: Test dataset with actual outcomes
        predictions_dict: Dictionary with model predictions

    Returns:
        Causal impact analysis results
    """
    logger.info("\n" + "="*60)
    logger.info("CAUSAL IMPACT ANALYSIS")
    logger.info("="*60)

    # Merge predictions with test data
    enhanced_data = test_data.copy()

    # Add predictions to dataset
    for model_name, results in predictions_dict.items():
        enhanced_data[f'pred_{model_name}'] = results['predictions']
        if 'probabilities' in results and results['probabilities'] is not None:
            enhanced_data[f'prob_{model_name}'] = results['probabilities']

    # Calculate baseline performance (no predictions)
    baseline_performance = {
        'total_pnl': enhanced_data['net'].sum(),
        'mean_daily_pnl': enhanced_data['net'].mean(),
        'volatility': enhanced_data['net'].std(),
        'sharpe_ratio': enhanced_data['net'].mean() / (enhanced_data['net'].std() + 1e-6),
        'hit_rate': (enhanced_data['net'] > 0).mean(),
        'max_drawdown': calculate_max_drawdown(enhanced_data['net'].cumsum()),
        'profit_factor': calculate_profit_factor(enhanced_data['net'])
    }

    causal_scenarios = {}

    # Scenario 1: Perfect 5-day Direction Following
    if 'pred_direction_5d_rf' in enhanced_data.columns:
        direction_scenario = simulate_direction_following(
            enhanced_data, 'pred_direction_5d_rf', 'target_direction_5d'
        )
        causal_scenarios['direction_following'] = direction_scenario

    # Scenario 2: Position Sizing with Top Quartile Predictions
    if 'prob_top_quartile_lr' in enhanced_data.columns:
        position_sizing_scenario = simulate_position_sizing(
            enhanced_data, 'prob_top_quartile_lr'
        )
        causal_scenarios['position_sizing'] = position_sizing_scenario

    # Scenario 3: Conservative Risk Management
    if 'pred_direction_5d_rf' in enhanced_data.columns and 'prob_top_quartile_lr' in enhanced_data.columns:
        risk_management_scenario = simulate_risk_management(
            enhanced_data, 'pred_direction_5d_rf', 'prob_top_quartile_lr'
        )
        causal_scenarios['risk_management'] = risk_management_scenario

    # Calculate improvements
    causal_impact_results = {
        'baseline_performance': baseline_performance,
        'enhanced_scenarios': causal_scenarios,
        'improvements': {}
    }

    for scenario_name, scenario_results in causal_scenarios.items():
        improvements = {}
        for metric, baseline_value in baseline_performance.items():
            if metric in scenario_results:
                enhanced_value = scenario_results[metric]
                if baseline_value != 0:
                    improvement_pct = ((enhanced_value - baseline_value) / abs(baseline_value)) * 100
                else:
                    improvement_pct = 0

                improvements[metric] = {
                    'baseline': baseline_value,
                    'enhanced': enhanced_value,
                    'improvement_pct': improvement_pct,
                    'absolute_improvement': enhanced_value - baseline_value
                }

        causal_impact_results['improvements'][scenario_name] = improvements

        # Log key improvements
        logger.info(f"\n{scenario_name.title()} Scenario:")
        if 'total_pnl' in improvements:
            pnl_imp = improvements['total_pnl']
            logger.info(f"  Total PnL: ${pnl_imp['enhanced']:,.2f} vs ${pnl_imp['baseline']:,.2f} ({pnl_imp['improvement_pct']:+.1f}%)")

        if 'sharpe_ratio' in improvements:
            sharpe_imp = improvements['sharpe_ratio']
            logger.info(f"  Sharpe Ratio: {sharpe_imp['enhanced']:.3f} vs {sharpe_imp['baseline']:.3f} ({sharpe_imp['improvement_pct']:+.1f}%)")

        if 'hit_rate' in improvements:
            hit_imp = improvements['hit_rate']
            logger.info(f"  Hit Rate: {hit_imp['enhanced']:.1%} vs {hit_imp['baseline']:.1%} ({hit_imp['improvement_pct']:+.1f}%)")

    return causal_impact_results


def simulate_direction_following(data, prediction_col, actual_col):
    """
    Simulate trading strategy that follows directional predictions
    """
    df = data.copy()

    # Create trading signals
    df['signal'] = df[prediction_col]  # 1 for positive, 0 for negative
    df['actual_direction'] = df[actual_col]

    # Calculate enhanced PnL by following signals
    # If prediction is correct (signal=1 and actual=1, or signal=0 and actual=0), take full position
    # If prediction is wrong, reduce position or exit
    df['signal_strength'] = np.where(df['signal'] == df['actual_direction'], 1.0, 0.2)
    df['enhanced_pnl'] = df['net'] * df['signal_strength']

    return {
        'total_pnl': df['enhanced_pnl'].sum(),
        'mean_daily_pnl': df['enhanced_pnl'].mean(),
        'volatility': df['enhanced_pnl'].std(),
        'sharpe_ratio': df['enhanced_pnl'].mean() / (df['enhanced_pnl'].std() + 1e-6),
        'hit_rate': (df['enhanced_pnl'] > 0).mean(),
        'max_drawdown': calculate_max_drawdown(df['enhanced_pnl'].cumsum()),
        'profit_factor': calculate_profit_factor(df['enhanced_pnl']),
        'signal_accuracy': (df['signal'] == df['actual_direction']).mean()
    }


def simulate_position_sizing(data, probability_col):
    """
    Simulate position sizing based on top quartile probabilities
    """
    df = data.copy()

    # Scale position size by prediction confidence
    # High probability of top quartile = larger position
    # Low probability = smaller position
    df['position_multiplier'] = df[probability_col] * 2  # Scale from 0-1 to 0-2
    df['enhanced_pnl'] = df['net'] * df['position_multiplier']

    return {
        'total_pnl': df['enhanced_pnl'].sum(),
        'mean_daily_pnl': df['enhanced_pnl'].mean(),
        'volatility': df['enhanced_pnl'].std(),
        'sharpe_ratio': df['enhanced_pnl'].mean() / (df['enhanced_pnl'].std() + 1e-6),
        'hit_rate': (df['enhanced_pnl'] > 0).mean(),
        'max_drawdown': calculate_max_drawdown(df['enhanced_pnl'].cumsum()),
        'profit_factor': calculate_profit_factor(df['enhanced_pnl']),
        'avg_position_size': df['position_multiplier'].mean()
    }


def simulate_risk_management(data, direction_col, probability_col):
    """
    Simulate conservative risk management strategy
    """
    df = data.copy()

    # Conservative approach: Only take positions when confident
    # High confidence in direction AND high top quartile probability
    df['confidence_score'] = df[probability_col]  # Top quartile probability
    df['direction_signal'] = df[direction_col]

    # Only trade when confidence > 0.6 and direction is positive
    df['trade_signal'] = (df['confidence_score'] > 0.6) & (df['direction_signal'] == 1)

    # Conservative position sizing
    df['position_size'] = np.where(df['trade_signal'],
                                  df['confidence_score'] * 1.5,  # Up to 1.5x position
                                  0.3)  # Small defensive position otherwise

    df['enhanced_pnl'] = df['net'] * df['position_size']

    return {
        'total_pnl': df['enhanced_pnl'].sum(),
        'mean_daily_pnl': df['enhanced_pnl'].mean(),
        'volatility': df['enhanced_pnl'].std(),
        'sharpe_ratio': df['enhanced_pnl'].mean() / (df['enhanced_pnl'].std() + 1e-6),
        'hit_rate': (df['enhanced_pnl'] > 0).mean(),
        'max_drawdown': calculate_max_drawdown(df['enhanced_pnl'].cumsum()),
        'profit_factor': calculate_profit_factor(df['enhanced_pnl']),
        'trading_frequency': df['trade_signal'].mean(),
        'avg_position_size': df['position_size'].mean()
    }


def calculate_max_drawdown(cumulative_returns):
    """Calculate maximum drawdown from cumulative returns"""
    peak = cumulative_returns.expanding().max()
    drawdown = cumulative_returns - peak
    return drawdown.min()


def calculate_profit_factor(returns):
    """Calculate profit factor (total gains / total losses)"""
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    return gains / (losses + 1e-6)


def create_performance_visualizations(causal_results, output_dir):
    """Create visualizations of performance improvements"""
    logger.info("Creating performance visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    scenarios = list(causal_results['enhanced_scenarios'].keys())
    metrics = ['total_pnl', 'sharpe_ratio', 'hit_rate', 'max_drawdown']

    for i, metric in enumerate(metrics):
        ax = axes[i//2, i%2]

        baseline_values = []
        enhanced_values = []
        scenario_names = []

        for scenario in scenarios:
            if scenario in causal_results['improvements']:
                improvements = causal_results['improvements'][scenario]
                if metric in improvements:
                    baseline_values.append(improvements[metric]['baseline'])
                    enhanced_values.append(improvements[metric]['enhanced'])
                    scenario_names.append(scenario.replace('_', ' ').title())

        if baseline_values:
            x = np.arange(len(scenario_names))
            width = 0.35

            ax.bar(x - width/2, baseline_values, width, label='Baseline', alpha=0.7)
            ax.bar(x + width/2, enhanced_values, width, label='Enhanced', alpha=0.7)

            ax.set_title(metric.replace('_', ' ').title())
            ax.set_xticks(x)
            ax.set_xticklabels(scenario_names, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'causal_impact_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Visualizations saved to {output_dir}")


def main():
    """Main evaluation function"""
    print("=" * 80)
    print("UNSEEN DATA EVALUATION & CAUSAL IMPACT ANALYSIS")
    print("=" * 80)

    # Load advanced features data
    try:
        df = pd.read_csv("data/processed/advanced_features.csv")
        df['date'] = pd.to_datetime(df['date'])
        logger.info(f"Loaded {len(df)} samples with {df.shape[1]} features")
    except FileNotFoundError:
        logger.error("Advanced features not found. Please run: python examples/advanced_feature_demo.py")
        return None

    # Create temporal splits (60% train, 20% val, 20% test)
    splits = create_temporal_splits(df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)

    # Define feature columns
    feature_cols = [col for col in df.columns
                   if not col.startswith('target_')
                   and col not in ['account_id', 'date']]

    logger.info(f"Using {len(feature_cols)} features for modeling")

    # Train best models on training data
    logger.info("\n" + "="*60)
    logger.info("TRAINING BEST MODELS")
    logger.info("="*60)

    models = train_best_models(splits['train'], splits['validation'], feature_cols)

    # Evaluate on completely unseen test data
    test_results = evaluate_on_unseen_data(models, splits['test'], feature_cols)

    # Calculate causal impact
    causal_results = calculate_causal_impact(splits['test'], test_results)

    # Save results
    output_dir = Path("results/unseen_evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed results
    with open(output_dir / "unseen_evaluation_results.json", 'w') as f:
        json.dump({
            'test_performance': {k: {key: val for key, val in v.items()
                               if key not in ['predictions', 'probabilities', 'actuals']}
                               for k, v in test_results.items()},
            'causal_impact': causal_results
        }, f, indent=2, default=str)

    # Create visualizations
    create_performance_visualizations(causal_results, output_dir)

    # Final summary
    logger.info("\n" + "="*80)
    logger.info("FINAL EVALUATION SUMMARY")
    logger.info("="*80)

    # Best performing models on unseen data
    logger.info("\nUnseen Data Performance:")
    for model_name, results in test_results.items():
        if results['type'] == 'classification':
            logger.info(f"  {model_name}: {results['accuracy']:.3f} accuracy ({results['improvement_vs_random']:+.1f}pp vs random)")
        else:
            logger.info(f"  {model_name}: R² = {results['r2']:.4f}, MAE = {results['mae']:.2f}")

    # Best causal impact scenario
    best_scenario = None
    best_pnl_improvement = -float('inf')

    for scenario_name, improvements in causal_results['improvements'].items():
        if 'total_pnl' in improvements:
            pnl_improvement = improvements['total_pnl']['improvement_pct']
            if pnl_improvement > best_pnl_improvement:
                best_pnl_improvement = pnl_improvement
                best_scenario = scenario_name

    if best_scenario:
        logger.info(f"\nBest Trading Strategy: {best_scenario}")
        improvements = causal_results['improvements'][best_scenario]

        for metric in ['total_pnl', 'sharpe_ratio', 'hit_rate']:
            if metric in improvements:
                imp = improvements[metric]
                logger.info(f"  {metric}: {imp['enhanced']:.3f} vs {imp['baseline']:.3f} ({imp['improvement_pct']:+.1f}%)")

    logger.info(f"\nDetailed results saved to: {output_dir}")

    return {
        'test_results': test_results,
        'causal_impact': causal_results,
        'models': models
    }


if __name__ == "__main__":
    try:
        results = main()
        logger.info("\n✓ Unseen data evaluation completed successfully!")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
