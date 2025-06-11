#!/usr/bin/env python
"""
Daily Prediction Script
Generates daily risk predictions for all traders
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.database import Database
from src.pipeline.data_validator import DataValidator
from src.pipeline.feature_pipeline import FeaturePipeline
from src.pipeline.model_pipeline import ModelPipeline
from src.models.risk_model import RiskModel
from src.monitoring.model_monitor import ModelMonitor
from src.monitoring.drift_detector import DriftDetector
from src.monitoring.alert_system import AlertSystem
from src.monitoring.dashboard_generator import DashboardGenerator
from src.data_downloader import DataDownloader


def setup_logging():
    """Configure logging"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'logs/daily_predict_{datetime.now().strftime("%Y%m%d")}.log')
        ]
    )


def download_recent_data(db: Database, days_back: int = 3) -> bool:
    """Download recent data for all traders"""
    logger = logging.getLogger(__name__)
    logger.info(f"Downloading last {days_back} days of data")

    try:
        downloader = DataDownloader()
        results = downloader.download_recent(days_back=days_back)

        success_count = sum(results.values())
        logger.info(f"Downloaded data for {success_count}/{len(results)} traders")

        return success_count > 0
    except Exception as e:
        logger.error(f"Error downloading data: {e}")
        return False


def load_latest_model(model_pipeline: ModelPipeline) -> RiskModel:
    """Load the latest trained model"""
    logger = logging.getLogger(__name__)

    model = model_pipeline.load_best_model()
    if model is None:
        raise ValueError("No trained model found. Run train_models.py first.")

    logger.info(f"Loaded model: {model.model_name}")
    return model


def generate_predictions(db: Database,
                        model: RiskModel,
                        feature_pipeline: FeaturePipeline) -> pd.DataFrame:
    """Generate predictions for all active traders"""
    logger = logging.getLogger(__name__)

    # Get all active traders
    traders = db.get_all_traders()
    active_traders = traders[traders['trading_days'] > 0]

    logger.info(f"Generating predictions for {len(active_traders)} active traders")

    all_predictions = []

    for _, trader in active_traders.iterrows():
        account_id = trader['account_id']
        trader_name = trader['trader_name']

        try:
            # Get recent data
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=90)  # 90 days for feature generation

            totals_df, fills_df = db.get_trader_data(
                account_id,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )

            if totals_df.empty or len(totals_df) < 20:
                logger.warning(f"Insufficient data for {trader_name}")
                continue

            # Generate features
            features = feature_pipeline.generate_features(totals_df, fills_df)

            if features.empty:
                logger.warning(f"No features generated for {trader_name}")
                continue

            # Get latest features (for next day prediction)
            latest_features = features.iloc[-1:].copy()

            # Make prediction
            risk_results = model.predict_risk(latest_features)

            # Add trader info
            risk_results['account_id'] = account_id
            risk_results['trader_name'] = trader_name
            risk_results['prediction_date'] = datetime.now().date()

            # Add recent performance context
            recent_pnl = totals_df['net_pnl'].tail(5).mean()
            risk_results['recent_avg_pnl'] = recent_pnl

            all_predictions.append(risk_results)

        except Exception as e:
            logger.error(f"Error predicting for {trader_name}: {e}")
            continue

    if all_predictions:
        predictions_df = pd.concat(all_predictions, ignore_index=True)
        logger.info(f"Generated {len(predictions_df)} predictions")
        return predictions_df
    else:
        logger.warning("No predictions generated")
        return pd.DataFrame()


def monitor_and_alert(predictions: pd.DataFrame,
                     model: RiskModel,
                     monitor: ModelMonitor,
                     detector: DriftDetector,
                     alert_system: AlertSystem) -> Dict[str, Any]:
    """Monitor model performance and check for alerts"""
    logger = logging.getLogger(__name__)

    monitoring_results = {}

    # 1. Log predictions to monitor
    prediction_time = datetime.now()
    monitor.log_predictions(
        predictions,
        prediction_time_ms=(datetime.now() - prediction_time).total_seconds() * 1000
    )

    # 2. Check for data drift
    if detector.reference_data is not None:
        # Get recent features for drift detection
        recent_features = predictions.select_dtypes(include=['float64', 'int64'])
        feature_cols = [col for col in recent_features.columns
                       if col.startswith(('tech_', 'behav_', 'regime_'))]

        if feature_cols:
            drift_results = detector.detect_drift(recent_features[feature_cols])
            monitoring_results['drift_results'] = drift_results

            # Calculate drift summary
            drifted_features = [name for name, result in drift_results.items() if result.is_drifted]
            drift_rate = len(drifted_features) / len(drift_results) if drift_results else 0

            monitoring_results['drift_summary'] = {
                'drift_rate': drift_rate,
                'n_drifted': len(drifted_features),
                'drifted_features': drifted_features[:5]  # Top 5
            }

    # 3. Check alerts
    alert_metrics = {
        'n_high_risk': len(predictions[predictions['risk_score'] > 0.8]),
        'n_critical_risk': len(predictions[predictions['risk_score'] > 0.9]),
        'avg_risk_score': predictions['risk_score'].mean(),
        'max_risk_score': predictions['risk_score'].max(),
        'portfolio_risk_score': predictions['risk_score'].mean(),  # Simplified
        'n_predictions': len(predictions)
    }

    # Add drift metrics
    if 'drift_summary' in monitoring_results:
        alert_metrics.update(monitoring_results['drift_summary'])

    # Check alerts
    triggered_alerts = alert_system.check_alerts(alert_metrics, source='daily_predict')
    monitoring_results['alerts'] = triggered_alerts

    # 4. Get monitoring summary
    monitoring_results['summary'] = monitor.get_monitoring_summary(period_days=7)

    return monitoring_results


def save_predictions(db: Database, predictions: pd.DataFrame):
    """Save predictions to database"""
    logger = logging.getLogger(__name__)

    if predictions.empty:
        logger.warning("No predictions to save")
        return

    # Convert to database format
    db_predictions = []
    for _, row in predictions.iterrows():
        db_predictions.append({
            'account_id': row['account_id'],
            'predicted_pnl': row.get('predicted_pnl', 0),
            'risk_score': row['risk_score'],
            'confidence': str(row.get('confidence', 'Medium'))
        })

    # Save to database
    db.save_predictions(db_predictions)
    logger.info(f"Saved {len(db_predictions)} predictions to database")


def generate_daily_report(predictions: pd.DataFrame,
                         monitoring_results: Dict[str, Any],
                         dashboard_gen: DashboardGenerator) -> str:
    """Generate daily risk report"""
    logger = logging.getLogger(__name__)

    # Sort by risk score
    predictions_sorted = predictions.sort_values('risk_score', ascending=False)

    report = []
    report.append("=" * 80)
    report.append("DAILY RISK MANAGEMENT REPORT")
    report.append("=" * 80)
    report.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Total Traders Analyzed: {len(predictions)}")
    report.append("")

    # Risk Summary
    report.append("RISK SUMMARY:")
    report.append(f"  Average Risk Score: {predictions['risk_score'].mean():.3f}")
    report.append(f"  High Risk Traders: {len(predictions[predictions['risk_score'] > 0.8])}")
    report.append(f"  Critical Risk Traders: {len(predictions[predictions['risk_score'] > 0.9])}")
    report.append("")

    # Top Risk Traders
    report.append("TOP 10 HIGH RISK TRADERS:")
    report.append("-" * 80)
    report.append(f"{'Trader':<20} {'Risk Score':<12} {'Predicted P&L':<15} {'Confidence':<12} {'Action':<20}")
    report.append("-" * 80)

    for _, trader in predictions_sorted.head(10).iterrows():
        action = "REDUCE POSITION" if trader['risk_score'] > 0.9 else "MONITOR CLOSELY"
        report.append(
            f"{trader['trader_name']:<20} "
            f"{trader['risk_score']:<12.3f} "
            f"{trader.get('predicted_pnl', 0):<15.2f} "
            f"{str(trader.get('confidence', 'N/A')):<12} "
            f"{action:<20}"
        )

    # Model Performance
    if 'summary' in monitoring_results:
        summary = monitoring_results['summary']
        report.append("")
        report.append("MODEL PERFORMANCE (7-day):")
        if 'metrics' in summary:
            metrics = summary['metrics']
            report.append(f"  Average Accuracy: {metrics.get('avg_accuracy', 0):.3f}")
            report.append(f"  Average RMSE: {metrics.get('avg_rmse', 0):.4f}")
            report.append(f"  Error Rate: {metrics.get('error_rate', 0):.2%}")

    # Drift Detection
    if 'drift_summary' in monitoring_results:
        drift = monitoring_results['drift_summary']
        report.append("")
        report.append("DATA DRIFT DETECTION:")
        report.append(f"  Drift Rate: {drift['drift_rate']:.1%}")
        report.append(f"  Features with Drift: {drift['n_drifted']}")
        if drift['drifted_features']:
            report.append(f"  Top Drifted Features: {', '.join(drift['drifted_features'])}")

    # Active Alerts
    if 'alerts' in monitoring_results and monitoring_results['alerts']:
        report.append("")
        report.append("ACTIVE ALERTS:")
        for alert in monitoring_results['alerts']:
            report.append(f"  [{alert.severity.value.upper()}] {alert.title}: {alert.message}")

    # Recommendations
    report.append("")
    report.append("RECOMMENDATIONS:")

    high_risk_count = len(predictions[predictions['risk_score'] > 0.8])
    if high_risk_count > len(predictions) * 0.2:
        report.append("  ⚠️  Significant portion of portfolio at high risk - Consider reducing overall exposure")

    if 'drift_summary' in monitoring_results and monitoring_results['drift_summary']['drift_rate'] > 0.2:
        report.append("  ⚠️  Significant data drift detected - Model retraining recommended")

    if not any(predictions['risk_score'] > 0.8):
        report.append("  ✓  No high-risk traders identified - Continue normal operations")

    report.append("")
    report.append("=" * 80)

    # Save report
    report_text = "\n".join(report)
    report_path = Path("reports") / f"daily_report_{datetime.now().strftime('%Y%m%d')}.txt"
    report_path.parent.mkdir(exist_ok=True)

    with open(report_path, 'w') as f:
        f.write(report_text)

    logger.info(f"Daily report saved to {report_path}")

    # Also generate HTML dashboard
    try:
        # Get historical data for dashboard
        db = Database()
        historical = pd.DataFrame()
        for account_id in predictions['account_id'].unique():
            totals, _ = db.get_trader_data(account_id)
            if not totals.empty:
                totals['account_id'] = account_id
                historical = pd.concat([historical, totals])

        dashboard_path = dashboard_gen.create_risk_dashboard(
            predictions=predictions,
            historical_performance=historical,
            feature_importance={},  # Would need to get from model
            monitoring_metrics=monitoring_results.get('summary', {}),
            drift_results=monitoring_results.get('drift_results', {})
        )
        logger.info(f"Dashboard saved to {dashboard_path}")
    except Exception as e:
        logger.error(f"Error generating dashboard: {e}")

    return report_text


def main():
    """Main execution function"""
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("Starting daily prediction process")

    try:
        # Initialize components
        db = Database()
        feature_pipeline = FeaturePipeline()
        model_pipeline = ModelPipeline()
        monitor = ModelMonitor("risk_model")
        detector = DriftDetector()
        alert_system = AlertSystem()
        dashboard_gen = DashboardGenerator()

        # Step 1: Download recent data
        logger.info("Step 1: Downloading recent data")
        if not download_recent_data(db, days_back=3):
            logger.error("Failed to download recent data")
            return

        # Step 2: Load model
        logger.info("Step 2: Loading model")
        model = load_latest_model(model_pipeline)

        # Step 3: Set drift detection baseline if needed
        if detector.reference_data is None:
            logger.info("Setting drift detection baseline")
            # Get historical data for baseline
            end_date = datetime.now().date() - timedelta(days=30)
            start_date = end_date - timedelta(days=90)

            all_features = []
            traders = db.get_all_traders()

            for _, trader in traders.iterrows():
                totals, fills = db.get_trader_data(
                    trader['account_id'],
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d')
                )

                if not totals.empty:
                    features = feature_pipeline.generate_features(totals, fills)
                    if not features.empty:
                        all_features.append(features)

            if all_features:
                baseline_features = pd.concat(all_features)
                detector.set_reference(baseline_features)

        # Step 4: Generate predictions
        logger.info("Step 3: Generating predictions")
        predictions = generate_predictions(db, model, feature_pipeline)

        if predictions.empty:
            logger.error("No predictions generated")
            return

        # Step 5: Monitor and alert
        logger.info("Step 4: Monitoring and alerting")
        monitoring_results = monitor_and_alert(
            predictions, model, monitor, detector, alert_system
        )

        # Step 6: Save predictions
        logger.info("Step 5: Saving predictions")
        save_predictions(db, predictions)

        # Step 7: Generate report
        logger.info("Step 6: Generating daily report")
        report = generate_daily_report(predictions, monitoring_results, dashboard_gen)

        # Print summary
        print("\n" + report)

        # Send email if configured
        if alert_system.email_config:
            try:
                # Email the report
                from src.monitoring.alert_system import Alert, AlertType, AlertSeverity

                report_alert = Alert(
                    alert_id="daily_report",
                    timestamp=datetime.now(),
                    alert_type=AlertType.MODEL_PERFORMANCE,
                    severity=AlertSeverity.INFO,
                    title="Daily Risk Management Report",
                    message=report,
                    metrics={},
                    source="daily_predict"
                )

                alert_system._send_email_alert(report_alert)
                logger.info("Daily report emailed successfully")
            except Exception as e:
                logger.error(f"Failed to email report: {e}")

        logger.info("Daily prediction process completed successfully")

    except Exception as e:
        logger.error(f"Error in daily prediction process: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
