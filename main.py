# main.py

import argparse
import logging
import sys
import os
from datetime import datetime, timedelta

# Add src to path
sys.path.append('src')

from src.data_processing import create_trader_day_panel
from src.feature_engineering import build_features
from src.modeling import run_walk_forward_backtest, train_production_model, run_purged_walk_forward_backtest, run_strict_walk_forward_backtest
# from src.model_monitoring import generate_feature_drift_report, generate_model_stability_report
from src.utils import load_config, load_model, ensure_directory_exists, create_performance_summary

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_backtest_mode(config):
    """Run the backtest mode: process data, engineer features, backtest, and train models."""
    logger.info("Starting backtest mode...")

    # Ensure directories exist
    ensure_directory_exists(config['paths']['model_dir'])
    ensure_directory_exists(config['paths']['report_dir'])
    ensure_directory_exists('data/processed')

    # Step 1: Create trader-day panel
    logger.info("Step 1: Creating trader-day panel...")
    panel_df = create_trader_day_panel(config)

    # Step 2: Build features
    logger.info("Step 2: Engineering features...")
    feature_df = build_features(panel_df, config)

    # Step 3: Run enhanced walk-forward backtest
    logger.info("Step 3: Running enhanced walk-forward backtest...")

    # Use strict walk-forward CV to eliminate temporal leakage
    use_strict_cv = config.get('model_quality', {}).get('use_strict_cv', True)
    use_purged_cv = config.get('model_quality', {}).get('use_purged_cv', False)

    if use_strict_cv:
        logger.info("Using STRICT walk-forward CV with in-fold feature selection...")
        backtest_results = run_strict_walk_forward_backtest(feature_df, config)
    elif use_purged_cv:
        logger.info("Using purged cross-validation for enhanced model validation...")
        backtest_results = run_purged_walk_forward_backtest(feature_df, config)
    else:
        logger.info("Using standard time series cross-validation...")
        backtest_results = run_walk_forward_backtest(feature_df, config)

    # Create performance summary
    performance_summary = create_performance_summary(backtest_results, config)
    performance_summary.to_csv(
        os.path.join(config['paths']['report_dir'], 'performance_summary.csv'),
        index=False
    )
    logger.info("Performance summary:")
    logger.info(performance_summary)

    # Step 4: Train production models
    logger.info("Step 4: Training production models...")
    var_model, loss_model = train_production_model(feature_df, config)

    logger.info("Backtest mode completed successfully!")
    logger.info(f"Models saved to: {config['paths']['model_dir']}")
    logger.info(f"Features saved to: {config['paths']['processed_features']}")


def run_validation_mode(config):
    """Run comprehensive model validation with enhanced quality checks."""
    logger.info("Starting comprehensive model validation mode...")

    from src.model_monitoring import ModelPerformanceMonitor, ABTestingFramework, generate_monitoring_dashboard_data

    # Initialize monitoring framework
    monitor = ModelPerformanceMonitor(config)
    ab_framework = ABTestingFramework(config)

    # Load existing models and data
    var_model_path = os.path.join(config['paths']['model_dir'], 'lgbm_var_model.joblib')
    loss_model_path = os.path.join(config['paths']['model_dir'], 'lgbm_loss_model.joblib')

    logger.info("Loading models and data...")
    var_model = load_model(var_model_path)
    loss_model = load_model(loss_model_path)

    # Load feature data
    import pandas as pd
    import pickle
    pickle_path = config['paths']['processed_features'].replace('.parquet', '.pkl')
    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as f:
            feature_df = pickle.load(f)
    else:
        feature_df = pd.read_parquet(config['paths']['processed_features'])

    # Add date_idx for consistency with training (same as monitor mode)
    if 'date_idx' not in feature_df.columns:
        unique_dates = feature_df['trade_date'].unique()
        date_to_idx = {date: idx for idx, date in enumerate(unique_dates)}
        feature_df['date_idx'] = feature_df['trade_date'].map(date_to_idx)

    # Load recent backtest results for validation
    # Check which CV method was used and load the appropriate results file
    if config['model_quality'].get('use_strict_cv', False):
        backtest_path = os.path.join(config['paths']['model_dir'], 'strict_walk_forward_results.csv')
    elif config['model_quality'].get('use_purged_cv', False):
        backtest_path = os.path.join(config['paths']['model_dir'], 'purged_backtest_results.csv')
    else:
        backtest_path = os.path.join(config['paths']['model_dir'], 'backtest_results.csv')

    if not os.path.exists(backtest_path):
        raise FileNotFoundError(f"Backtest results not found at {backtest_path}. "
                               f"Make sure the model was trained with the same CV settings.")

    backtest_results = pd.read_csv(backtest_path)
    backtest_results['trade_date'] = pd.to_datetime(backtest_results['trade_date'])

    # Comprehensive model validation
    logger.info("Performing comprehensive model validation...")

    # 1. Load and validate model metadata
    metadata_path = os.path.join(config['paths']['model_dir'], 'model_metadata.json')
    if os.path.exists(metadata_path):
        import json
        with open(metadata_path, 'r') as f:
            model_metadata = json.load(f)

        logger.info("Model Training Summary:")
        logger.info(f"  Training Date: {model_metadata.get('training_date', 'Unknown')}")
        logger.info(f"  Features Used: {model_metadata.get('n_features', 'Unknown')}")
        logger.info(f"  Data Quality Issues: {model_metadata.get('data_quality_issues', 'Unknown')}")

        # Check model age
        training_date = datetime.fromisoformat(model_metadata['training_date'].replace('Z', '+00:00'))
        model_age_days = (datetime.now(training_date.tzinfo) - training_date).days

        if model_age_days > 30:
            logger.warning(f"Model is {model_age_days} days old - consider retraining")

    # 2. Validate current model performance
    logger.info("Validating current model performance...")

    # Create mock recent predictions for validation
    recent_data = feature_df.tail(100)  # Last 100 observations

    # Load selected features from model metadata (same as signal_generator.py fix)
    if 'model_metadata' in locals() and model_metadata and 'selected_features' in model_metadata:
        selected_features = model_metadata['selected_features']
        logger.info(f"Using {len(selected_features)} selected features from model training")
    else:
        # Fallback to all features if metadata not available
        logger.warning("Using all available features for validation")
        selected_features = [col for col in recent_data.columns if col not in [
            'account_id', 'trade_date', 'target_pnl', 'target_large_loss',
            'daily_pnl', 'large_loss_threshold'
        ]]

    # Check if all selected features are available
    missing_features = [f for f in selected_features if f not in recent_data.columns]
    if missing_features:
        logger.error(f"Missing required features: {missing_features}")
        raise ValueError(f"Missing features required for validation: {missing_features}")

    X_recent = recent_data[selected_features]
    recent_var_pred = var_model.predict(X_recent)
    recent_loss_pred = loss_model.predict_proba(X_recent)[:, 1]

    # Create predictions DataFrame
    predictions_df = pd.DataFrame({
        'account_id': recent_data['account_id'].values,
        'trade_date': recent_data['trade_date'].values,
        'pred_var': recent_var_pred,
        'pred_loss_proba': recent_loss_pred
    })

    # Create actuals DataFrame (using actual target values for validation)
    actuals_df = pd.DataFrame({
        'account_id': recent_data['account_id'].values,
        'trade_date': recent_data['trade_date'].values,
        'actual_pnl': recent_data['target_pnl'].values,
        'actual_large_loss': recent_data['target_large_loss'].values
    })

    # Track performance
    performance_metrics = monitor.track_daily_performance(predictions_df, actuals_df)

    # 3. Generate comprehensive validation report
    logger.info("Generating comprehensive validation report...")

    validation_report = {
        'validation_timestamp': datetime.now().isoformat(),
        'model_metadata': model_metadata if 'model_metadata' in locals() else {},
        'performance_metrics': performance_metrics,
        'data_quality_summary': {
            'total_features': len(selected_features),
            'total_samples': len(feature_df),
            'date_range': [
                feature_df['trade_date'].min().isoformat(),
                feature_df['trade_date'].max().isoformat()
            ]
        }
    }

    # Load advanced metrics if available
    advanced_metrics_path = config['paths']['model_dir'] + '/advanced_risk_metrics.json'
    if os.path.exists(advanced_metrics_path):
        with open(advanced_metrics_path, 'r') as f:
            advanced_metrics = json.load(f)
        validation_report['advanced_metrics'] = advanced_metrics

    # Load validation metrics if available
    validation_metrics_path = config['paths']['model_dir'] + '/purged_validation_metrics.json'
    if os.path.exists(validation_metrics_path):
        with open(validation_metrics_path, 'r') as f:
            purged_validation = json.load(f)
        validation_report['purged_validation'] = purged_validation

    # 4. Generate dashboard data
    dashboard_data = generate_monitoring_dashboard_data(monitor, ab_framework)

    # 5. Save comprehensive validation report
    ensure_directory_exists(config['paths']['report_dir'])

    validation_report_path = os.path.join(config['paths']['report_dir'], 'comprehensive_validation_report.json')
    with open(validation_report_path, 'w') as f:
        json.dump(validation_report, f, indent=2, default=str)

    dashboard_path = os.path.join(config['paths']['report_dir'], 'monitoring_dashboard_data.json')
    with open(dashboard_path, 'w') as f:
        json.dump(dashboard_data, f, indent=2, default=str)

    # 6. Print validation summary
    logger.info("\n" + "="*60)
    logger.info("COMPREHENSIVE MODEL VALIDATION SUMMARY")
    logger.info("="*60)

    # Performance summary
    if 'var_performance' in performance_metrics:
        var_perf = performance_metrics['var_performance']
        logger.info(f"VaR Model Performance:")
        logger.info(f"  Violation Rate: {var_perf['violation_rate']:.1%} (expected: {var_perf['expected_rate']:.1%})")
        logger.info(f"  Violation Deviation: {var_perf['violation_deviation']:.1%}")
        logger.info(f"  Violations: {var_perf['n_violations']}/{var_perf['n_samples']}")

    if 'loss_performance' in performance_metrics:
        loss_perf = performance_metrics['loss_performance']
        logger.info(f"Loss Model Performance:")
        logger.info(f"  AUC: {loss_perf['auc']:.3f}")
        logger.info(f"  Calibration Error: {loss_perf['calibration_error']:.3f}")

    # Alerts summary
    alerts = performance_metrics.get('alerts', [])
    if alerts:
        logger.warning(f"ALERTS DETECTED ({len(alerts)}):")
        for alert in alerts:
            logger.warning(f"  ⚠️  {alert}")
    else:
        logger.info("✅ No performance alerts detected")

    # Model quality summary
    if 'model_metadata' in locals() and model_metadata:
        conv_metrics = model_metadata.get('var_convergence', {})
        if conv_metrics.get('likely_overfitting', False):
            logger.warning("⚠️  VaR model shows signs of overfitting")

        conv_metrics = model_metadata.get('loss_convergence', {})
        if conv_metrics.get('likely_overfitting', False):
            logger.warning("⚠️  Loss model shows signs of overfitting")

    logger.info("="*60)
    logger.info("Validation mode completed successfully!")
    logger.info(f"Comprehensive report saved to: {validation_report_path}")
    logger.info(f"Dashboard data saved to: {dashboard_path}")

    return validation_report


def main():
    """Main entry point for the risk modeling system."""
    parser = argparse.ArgumentParser(
        description='Quantitative Risk Model for Active Traders'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['backtest', 'monitor', 'validate'],
        required=True,
        help='Execution mode: backtest (train models), monitor (drift reports), or validate (comprehensive validation)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/main_config.yaml',
        help='Path to configuration file (default: configs/main_config.yaml)'
    )

    args = parser.parse_args()

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Execute based on mode
    if args.mode == 'backtest':
        run_backtest_mode(config)
    elif args.mode == 'validate':
        run_validation_mode(config)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == '__main__':
    main()
