#!/usr/bin/env python
"""
Train Models Script
Trains risk prediction models for all traders
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.database import Database
from src.pipeline.data_validator import DataValidator
from src.pipeline.feature_pipeline import FeaturePipeline
from src.pipeline.model_pipeline import ModelPipeline
from src.models.risk_model import RiskModel
from src.utils.time_series_cv import TimeSeriesSplit, WalkForwardAnalysis
from src.monitoring.drift_detector import DriftDetector
from src.backtesting.performance_metrics import PerformanceMetrics
from src.core.constants import TradingConstants as TC


def setup_logging():
    """Configure logging"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'logs/train_models_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )


def validate_data_quality(db: Database) -> bool:
    """Validate data quality before training"""
    logger = logging.getLogger(__name__)
    logger.info("Validating data quality")

    validator = DataValidator()
    traders = db.get_all_traders()

    valid_traders = 0
    total_traders = len(traders)

    for _, trader in traders.iterrows():
        account_id = trader['account_id']

        # Get all historical data
        totals_df, fills_df = db.get_trader_data(account_id)

        if totals_df.empty:
            logger.warning(f"No data for trader {trader['trader_name']}")
            continue

        # Validate
        result = validator.validate_combined(totals_df, fills_df)

        if result.is_valid:
            valid_traders += 1
        else:
            logger.warning(f"Validation failed for {trader['trader_name']}: {result.errors}")

    logger.info(f"Valid traders: {valid_traders}/{total_traders}")

    # Require at least 50% valid traders
    return valid_traders >= total_traders * 0.5


def prepare_training_data(db: Database,
                         feature_pipeline: FeaturePipeline,
                         min_days: int = TC.MIN_TRAINING_DAYS) -> pd.DataFrame:
    """Prepare consolidated training data from all traders"""
    logger = logging.getLogger(__name__)
    logger.info("Preparing training data")

    all_features = []
    trader_stats = []

    traders = db.get_all_traders()

    for _, trader in traders.iterrows():
        account_id = trader['account_id']
        trader_name = trader['trader_name']

        try:
            # Get historical data
            totals_df, fills_df = db.get_trader_data(account_id)

            if totals_df.empty or len(totals_df) < min_days:
                logger.warning(f"Insufficient data for {trader_name}: {len(totals_df)} days")
                continue

            # Generate features
            features = feature_pipeline.generate_features(totals_df, fills_df)

            if features.empty:
                logger.warning(f"No features generated for {trader_name}")
                continue

            # Add trader identifier
            features['trader_id'] = account_id

            all_features.append(features)

            # Collect trader statistics
            trader_stats.append({
                'trader_id': account_id,
                'trader_name': trader_name,
                'n_days': len(totals_df),
                'total_pnl': totals_df['net_pnl'].sum(),
                'avg_daily_pnl': totals_df['net_pnl'].mean(),
                'volatility': totals_df['net_pnl'].std(),
                'sharpe': totals_df['net_pnl'].mean() / (totals_df['net_pnl'].std() + TC.MIN_VARIANCE) * np.sqrt(TC.TRADING_DAYS_PER_YEAR)
            })

        except Exception as e:
            logger.error(f"Error processing {trader_name}: {e}")
            continue

    if not all_features:
        raise ValueError("No valid training data found")

    # Combine all features
    combined_features = pd.concat(all_features, ignore_index=True)

    # Log statistics
    logger.info(f"Training data prepared:")
    logger.info(f"  Total samples: {len(combined_features)}")
    logger.info(f"  Total traders: {len(trader_stats)}")
    logger.info(f"  Date range: {combined_features.index.min()} to {combined_features.index.max()}")

    # Save trader statistics
    stats_df = pd.DataFrame(trader_stats)
    stats_path = Path("data") / "trader_statistics.csv"
    stats_df.to_csv(stats_path, index=False)
    logger.info(f"Trader statistics saved to {stats_path}")

    return combined_features


def analyze_feature_importance(model: RiskModel,
                              feature_names: List[str],
                              save_path: Optional[Path] = None) -> pd.DataFrame:
    """Analyze and visualize feature importance"""
    logger = logging.getLogger(__name__)

    # Get feature importance
    importance_df = model.get_feature_importance()

    if importance_df.empty:
        logger.warning("No feature importance available")
        return importance_df

    # Add feature categories
    importance_df['category'] = importance_df['feature'].apply(lambda x: x.split('_')[0])

    # Log top features
    logger.info("Top 20 most important features:")
    for idx, row in importance_df.head(20).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")

    # Category summary
    category_importance = importance_df.groupby('category')['importance'].agg(['sum', 'mean', 'count'])
    logger.info("\nImportance by category:")
    for cat, stats in category_importance.iterrows():
        logger.info(f"  {cat}: sum={stats['sum']:.4f}, mean={stats['mean']:.4f}, count={stats['count']}")

    # Save if requested
    if save_path:
        importance_df.to_csv(save_path, index=False)
        logger.info(f"Feature importance saved to {save_path}")

    return importance_df


def evaluate_model_performance(model: RiskModel,
                              X_test: pd.DataFrame,
                              y_test: pd.Series,
                              trader_ids: Optional[pd.Series] = None) -> Dict[str, float]:
    """Comprehensive model evaluation"""
    logger = logging.getLogger(__name__)
    logger.info("Evaluating model performance")

    # Make predictions
    predictions = model.predict(X_test)

    # Calculate metrics
    perf_calc = PerformanceMetrics()

    # Overall metrics
    overall_metrics = {
        'rmse': np.sqrt(np.mean((y_test - predictions) ** 2)),
        'mae': np.mean(np.abs(y_test - predictions)),
        'r2': model.score(X_test, y_test),
        'directional_accuracy': ((predictions > 0) == (y_test > 0)).mean()
    }

    # If we have trader IDs, calculate per-trader metrics
    if trader_ids is not None:
        trader_metrics = []

        for trader_id in trader_ids.unique():
            mask = trader_ids == trader_id
            if mask.sum() < 10:  # Need minimum samples
                continue

            trader_pred = predictions[mask]
            trader_actual = y_test[mask]

            trader_metrics.append({
                'trader_id': trader_id,
                'rmse': np.sqrt(np.mean((trader_actual - trader_pred) ** 2)),
                'directional_accuracy': ((trader_pred > 0) == (trader_actual > 0)).mean(),
                'n_samples': mask.sum()
            })

        # Add aggregated trader metrics
        if trader_metrics:
            trader_df = pd.DataFrame(trader_metrics)
            overall_metrics['avg_trader_rmse'] = trader_df['rmse'].mean()
            overall_metrics['std_trader_rmse'] = trader_df['rmse'].std()
            overall_metrics['pct_profitable_models'] = (trader_df['directional_accuracy'] > 0.5).mean()

    # Risk-based evaluation
    risk_predictions = model.predict_risk(X_test)

    # Evaluate risk categories
    high_risk = risk_predictions['risk_score'] > TC.HIGH_RISK_SCORE
    actual_losses = y_test < 0

    overall_metrics['risk_precision'] = (actual_losses[high_risk]).mean() if high_risk.any() else 0
    overall_metrics['risk_recall'] = (high_risk[actual_losses]).mean() if actual_losses.any() else 0

    # Log metrics
    logger.info("Model Performance Metrics:")
    for metric, value in overall_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")

    return overall_metrics


def train_ensemble_models(combined_features: pd.DataFrame,
                         model_pipeline: ModelPipeline) -> Dict[str, RiskModel]:
    """Train multiple models for ensemble"""
    logger = logging.getLogger(__name__)
    logger.info("Training ensemble models")

    models = {}

    # Prepare target
    if 'net_pnl' in combined_features.columns:
        # Create forward-looking target
        combined_features = combined_features.sort_values(['trader_id', 'date'])
        combined_features['target'] = combined_features.groupby('trader_id')['net_pnl'].shift(-1)
        combined_features = combined_features.dropna(subset=['target'])
    else:
        raise ValueError("No target column found")

    # 1. Global model (all traders)
    logger.info("Training global model")
    model_pipeline.experiment_name = "global_risk_model"
    global_model = model_pipeline.run_pipeline(
        combined_features,
        target_col='target',
        target_type='regression',
        optimize_hyperparams=True,
        n_trials=30,
        cv_splits=3
    )
    models['global'] = global_model

    # 2. Volatility regime models
    logger.info("Training regime-specific models")

    # Calculate volatility regime
    volatility = combined_features.groupby('trader_id')['net_pnl'].transform(
        lambda x: x.rolling(20, min_periods=5).std()
    )
    vol_percentile = volatility.rank(pct=True)

    # High volatility model
    high_vol_data = combined_features[vol_percentile > 0.7]
    if len(high_vol_data) > 1000:
        model_pipeline.experiment_name = "high_volatility_model"
        high_vol_model = model_pipeline.run_pipeline(
            high_vol_data,
            target_col='target',
            target_type='regression',
            optimize_hyperparams=False  # Use default params
        )
        models['high_volatility'] = high_vol_model

    # Low volatility model
    low_vol_data = combined_features[vol_percentile < 0.3]
    if len(low_vol_data) > 1000:
        model_pipeline.experiment_name = "low_volatility_model"
        low_vol_model = model_pipeline.run_pipeline(
            low_vol_data,
            target_col='target',
            target_type='regression',
            optimize_hyperparams=False
        )
        models['low_volatility'] = low_vol_model

    # 3. Classification model (profit/loss)
    logger.info("Training classification model")
    model_pipeline.experiment_name = "classification_risk_model"

    # Create binary target
    combined_features['target_binary'] = (combined_features['target'] > 0).astype(int)

    classification_model = model_pipeline.run_pipeline(
        combined_features,
        target_col='target_binary',
        target_type='classification',
        optimize_hyperparams=True,
        n_trials=20,
        cv_splits=3
    )
    models['classification'] = classification_model

    return models


def create_training_report(models: Dict[str, RiskModel],
                          performance_metrics: Dict[str, Dict[str, float]],
                          feature_importance: pd.DataFrame) -> str:
    """Create comprehensive training report"""
    logger = logging.getLogger(__name__)

    report = []
    report.append("=" * 80)
    report.append("MODEL TRAINING REPORT")
    report.append("=" * 80)
    report.append(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Models Trained: {len(models)}")
    report.append("")

    # Model Performance Summary
    report.append("MODEL PERFORMANCE SUMMARY:")
    report.append("-" * 80)
    report.append(f"{'Model':<20} {'RMSE':<10} {'MAE':<10} {'R²':<10} {'Dir Acc':<10} {'Risk Prec':<10}")
    report.append("-" * 80)

    for model_name, metrics in performance_metrics.items():
        report.append(
            f"{model_name:<20} "
            f"{metrics.get('rmse', 0):<10.4f} "
            f"{metrics.get('mae', 0):<10.4f} "
            f"{metrics.get('r2', 0):<10.4f} "
            f"{metrics.get('directional_accuracy', 0):<10.4f} "
            f"{metrics.get('risk_precision', 0):<10.4f}"
        )

    # Feature Importance
    report.append("")
    report.append("TOP 20 FEATURES:")
    report.append("-" * 80)

    for idx, row in feature_importance.head(20).iterrows():
        report.append(f"{row['feature']:<40} {row['importance']:>10.4f}")

    # Feature Categories
    report.append("")
    report.append("FEATURE IMPORTANCE BY CATEGORY:")
    report.append("-" * 80)

    category_importance = feature_importance.groupby('category')['importance'].agg(['sum', 'mean', 'count'])
    for cat, stats in category_importance.iterrows():
        report.append(f"{cat:<20} Sum: {stats['sum']:>8.4f}  Mean: {stats['mean']:>8.4f}  Count: {stats['count']:>4.0f}")

    # Model Selection
    report.append("")
    report.append("MODEL SELECTION:")

    # Find best model based on directional accuracy
    best_model = max(performance_metrics.items(), key=lambda x: x[1].get('directional_accuracy', 0))
    report.append(f"Best Model: {best_model[0]} (Directional Accuracy: {best_model[1]['directional_accuracy']:.4f})")

    # Recommendations
    report.append("")
    report.append("RECOMMENDATIONS:")

    if best_model[1]['directional_accuracy'] < 0.55:
        report.append("  ⚠️  Model accuracy is below target - consider feature engineering improvements")

    if best_model[1]['rmse'] > 0.05:
        report.append("  ⚠️  High RMSE detected - model may need regularization tuning")

    if len(models) > 1:
        report.append("  ✓  Multiple models trained - ensemble approach recommended")

    report.append("")
    report.append("NEXT STEPS:")
    report.append("  1. Run daily_predict.py for daily risk predictions")
    report.append("  2. Monitor model performance with monitoring dashboard")
    report.append("  3. Retrain models weekly or when drift is detected")

    report.append("")
    report.append("=" * 80)

    return "\n".join(report)


def main():
    """Main training function"""
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("Starting model training process")

    try:
        # Initialize components
        db = Database()
        feature_pipeline = FeaturePipeline()
        model_pipeline = ModelPipeline()
        drift_detector = DriftDetector()

        # Step 1: Validate data quality
        logger.info("Step 1: Validating data quality")
        if not validate_data_quality(db):
            logger.error("Data quality validation failed")
            return

        # Step 2: Prepare training data
        logger.info("Step 2: Preparing training data")
        combined_features = prepare_training_data(db, feature_pipeline)

        # Step 3: Set drift detection baseline
        logger.info("Step 3: Setting drift detection baseline")
        drift_detector.set_reference(combined_features)

        # Save drift detector reference
        drift_ref_path = Path("data/models") / "drift_reference.pkl"
        import joblib
        joblib.dump(drift_detector, drift_ref_path)
        logger.info(f"Drift detector reference saved to {drift_ref_path}")

        # Step 4: Train models
        logger.info("Step 4: Training models")
        models = train_ensemble_models(combined_features, model_pipeline)

        # Step 5: Evaluate models
        logger.info("Step 5: Evaluating models")

        # Prepare test data (last 20% of data)
        test_split = int(len(combined_features) * 0.8)
        test_features = combined_features.iloc[test_split:]

        # Separate features and target
        feature_cols = [col for col in test_features.columns
                       if col not in ['target', 'target_binary', 'trader_id', 'date', 'net_pnl']]

        X_test = test_features[feature_cols]
        y_test = test_features['target']
        trader_ids = test_features.get('trader_id', None)

        # Evaluate each model
        performance_metrics = {}
        for model_name, model in models.items():
            logger.info(f"Evaluating {model_name}")

            # Use appropriate target for classification
            if model_name == 'classification':
                y_test_model = test_features['target_binary']
            else:
                y_test_model = y_test

            metrics = evaluate_model_performance(model, X_test, y_test_model, trader_ids)
            performance_metrics[model_name] = metrics

        # Step 6: Analyze feature importance
        logger.info("Step 6: Analyzing feature importance")

        # Use global model for feature importance
        global_model = models.get('global')
        if global_model:
            feature_importance = analyze_feature_importance(
                global_model,
                feature_cols,
                save_path=Path("data/models") / "feature_importance.csv"
            )
        else:
            feature_importance = pd.DataFrame()

        # Step 7: Generate training report
        logger.info("Step 7: Generating training report")
        report = create_training_report(models, performance_metrics, feature_importance)

        # Save report
        report_path = Path("reports") / f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        report_path.parent.mkdir(exist_ok=True)

        with open(report_path, 'w') as f:
            f.write(report)

        logger.info(f"Training report saved to {report_path}")

        # Print summary
        print("\n" + report)

        # Save training metadata
        metadata = {
            'training_date': datetime.now().isoformat(),
            'n_samples': len(combined_features),
            'n_features': len(feature_cols),
            'n_traders': combined_features['trader_id'].nunique() if 'trader_id' in combined_features.columns else 1,
            'models_trained': list(models.keys()),
            'best_model': max(performance_metrics.items(), key=lambda x: x[1].get('directional_accuracy', 0))[0],
            'performance_summary': {k: v.get('directional_accuracy', 0) for k, v in performance_metrics.items()}
        }

        metadata_path = Path("data/models") / "training_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info("Model training completed successfully")

    except Exception as e:
        logger.error(f"Error in model training: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
