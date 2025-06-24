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
from src.modeling import run_walk_forward_backtest, train_production_model
from src.monitoring import generate_feature_drift_report, generate_model_stability_report
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

    # Step 3: Run walk-forward backtest
    logger.info("Step 3: Running walk-forward backtest...")
    backtest_results = run_walk_forward_backtest(feature_df, config)

    # Create performance summary
    performance_summary = create_performance_summary(backtest_results, config)
    performance_summary.to_csv(
        f"{config['paths']['report_dir']}/performance_summary.csv",
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


def run_monitor_mode(config):
    """Run the monitor mode: generate drift and stability reports."""
    logger.info("Starting monitor mode...")

    # Load models
    var_model_path = f"{config['paths']['model_dir']}/lgbm_var_model.joblib"
    loss_model_path = f"{config['paths']['model_dir']}/lgbm_loss_model.joblib"

    logger.info("Loading models...")
    var_model = load_model(var_model_path)
    loss_model = load_model(loss_model_path)

    # Load historical feature set
    logger.info("Loading feature data...")
    import pandas as pd
    import pickle
    # Try pickle first, fallback to parquet
    pickle_path = config['paths']['processed_features'].replace('.parquet', '.pkl')
    if os.path.exists(pickle_path):
        logger.info(f"Loading features from pickle: {pickle_path}")
        with open(pickle_path, 'rb') as f:
            full_features = pickle.load(f)
    else:
        full_features = pd.read_parquet(config['paths']['processed_features'])

    # Add date_idx for consistency with training
    unique_dates = full_features['trade_date'].unique()
    date_to_idx = {date: idx for idx, date in enumerate(unique_dates)}
    full_features['date_idx'] = full_features['trade_date'].map(date_to_idx)

    # Define recent period (last 30 days of data)
    recent_cutoff = full_features['trade_date'].max() - timedelta(days=30)

    # Split into training and recent data
    training_df = full_features[full_features['trade_date'] < recent_cutoff]
    recent_df = full_features[full_features['trade_date'] >= recent_cutoff]

    logger.info(f"Training data: {len(training_df)} samples")
    logger.info(f"Recent data: {len(recent_df)} samples")

    # Generate drift report
    logger.info("Generating feature drift report...")
    generate_feature_drift_report(training_df, recent_df, config)

    # Generate stability reports
    logger.info("Generating model stability reports...")
    generate_model_stability_report(var_model, training_df, recent_df, config, model_type='var')
    generate_model_stability_report(loss_model, training_df, recent_df, config, model_type='loss')

    # Generate causal impact analysis
    logger.info("Performing causal impact analysis on unseen data...")
    from src.causal_impact import analyze_causal_impact, generate_causal_impact_report

    # Use recent data as "unseen" data for causal impact analysis
    var_impact = analyze_causal_impact(var_model, recent_df, config, model_type='var')
    loss_impact = analyze_causal_impact(loss_model, recent_df, config, model_type='loss')

    # Generate comprehensive report
    generate_causal_impact_report(var_impact, loss_impact, config)

    logger.info("Monitor mode completed successfully!")
    logger.info(f"Reports saved to: {config['paths']['report_dir']}")

    # Print key causal impact findings
    logger.info("\nKey Causal Impact Findings:")
    logger.info(f"- VaR violation rate: {var_impact['violation_metrics']['actual_violation_rate']:.3f} "
                f"(expected: {var_impact['violation_metrics']['expected_violation_rate']:.3f})")
    logger.info(f"- Economic net benefit: ${var_impact['economic_impact']['net_benefit']:,.2f}")

    best_threshold = loss_impact['threshold_analysis'].loc[
        loss_impact['threshold_analysis']['net_benefit'].idxmax()
    ]
    logger.info(f"- Optimal loss threshold: {best_threshold['threshold']:.1f} "
                f"(net benefit: ${best_threshold['net_benefit']:,.2f})")


def main():
    """Main entry point for the risk modeling system."""
    parser = argparse.ArgumentParser(
        description='Quantitative Risk Model for Active Traders'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['backtest', 'monitor'],
        required=True,
        help='Execution mode: backtest (train models) or monitor (generate reports)'
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
    elif args.mode == 'monitor':
        run_monitor_mode(config)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == '__main__':
    main()
