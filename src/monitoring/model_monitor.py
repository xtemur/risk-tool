"""
Model Monitor
Real-time monitoring of model performance and health
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path
from collections import deque

from src.core.constants import TradingConstants as TC
from src.backtesting.performance_metrics import PerformanceMetrics

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Container for model performance metrics"""
    timestamp: datetime
    prediction_accuracy: float
    prediction_mae: float
    prediction_rmse: float
    feature_importance_stability: float
    prediction_latency_ms: float
    n_predictions: int
    n_errors: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertCondition:
    """Alert condition definition"""
    name: str
    metric: str
    threshold: float
    direction: str  # 'above' or 'below'
    window: int  # Number of periods to check
    severity: str  # 'low', 'medium', 'high', 'critical'

    def check(self, value: float) -> bool:
        """Check if condition is triggered"""
        if self.direction == 'above':
            return value > self.threshold
        else:
            return value < self.threshold


class ModelMonitor:
    """
    Monitor model performance in production
    Tracks accuracy, drift, and operational metrics
    """

    def __init__(self,
                 model_name: str,
                 monitoring_window: int = 20,
                 log_dir: str = "logs/monitoring",
                 alert_conditions: Optional[List[AlertCondition]] = None):
        """
        Initialize model monitor

        Args:
            model_name: Name of the model being monitored
            monitoring_window: Days to keep in rolling window
            log_dir: Directory for monitoring logs
            alert_conditions: List of alert conditions to check
        """
        self.model_name = model_name
        self.monitoring_window = monitoring_window
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)

        # Metrics storage
        self.metrics_history = deque(maxlen=monitoring_window)
        self.predictions_log = []
        self.errors_log = []

        # Alert conditions
        self.alert_conditions = alert_conditions or self._get_default_alerts()

        # Performance analyzer
        self.perf_metrics = PerformanceMetrics()

        # Baseline metrics for comparison
        self.baseline_metrics = None
        self.feature_importance_baseline = None

    def _get_default_alerts(self) -> List[AlertCondition]:
        """Get default alert conditions"""
        return [
            AlertCondition(
                name="High Prediction Error",
                metric="prediction_rmse",
                threshold=0.05,  # 5% RMSE
                direction="above",
                window=5,
                severity="high"
            ),
            AlertCondition(
                name="Low Prediction Accuracy",
                metric="prediction_accuracy",
                threshold=0.5,  # 50% accuracy
                direction="below",
                window=5,
                severity="high"
            ),
            AlertCondition(
                name="High Latency",
                metric="prediction_latency_ms",
                threshold=1000,  # 1 second
                direction="above",
                window=10,
                severity="medium"
            ),
            AlertCondition(
                name="Feature Importance Drift",
                metric="feature_importance_stability",
                threshold=0.7,  # 70% stability
                direction="below",
                window=3,
                severity="medium"
            ),
            AlertCondition(
                name="High Error Rate",
                metric="error_rate",
                threshold=0.1,  # 10% errors
                direction="above",
                window=5,
                severity="critical"
            )
        ]

    def set_baseline(self,
                    baseline_metrics: Dict[str, float],
                    feature_importance: Dict[str, float]):
        """
        Set baseline metrics for comparison

        Args:
            baseline_metrics: Baseline performance metrics
            feature_importance: Baseline feature importance
        """
        self.baseline_metrics = baseline_metrics
        self.feature_importance_baseline = feature_importance
        logger.info(f"Baseline metrics set for {self.model_name}")

    def log_predictions(self,
                       predictions: pd.DataFrame,
                       actuals: Optional[pd.Series] = None,
                       features: Optional[pd.DataFrame] = None,
                       prediction_time_ms: float = 0):
        """
        Log predictions and calculate metrics

        Args:
            predictions: DataFrame with predictions
            actuals: Optional actual values for accuracy calculation
            features: Optional features used for prediction
            prediction_time_ms: Time taken for prediction in milliseconds
        """
        timestamp = datetime.now()

        # Store predictions
        self.predictions_log.append({
            'timestamp': timestamp,
            'predictions': predictions.copy(),
            'n_predictions': len(predictions)
        })

        # Calculate metrics
        metrics = ModelMetrics(
            timestamp=timestamp,
            prediction_accuracy=0.0,
            prediction_mae=0.0,
            prediction_rmse=0.0,
            feature_importance_stability=1.0,
            prediction_latency_ms=prediction_time_ms,
            n_predictions=len(predictions),
            n_errors=0
        )

        # If we have actuals, calculate accuracy metrics
        if actuals is not None and len(actuals) > 0:
            metrics.prediction_accuracy = self._calculate_accuracy(predictions, actuals)
            metrics.prediction_mae = self._calculate_mae(predictions, actuals)
            metrics.prediction_rmse = self._calculate_rmse(predictions, actuals)

        # Log features if provided
        if features is not None:
            metrics.metadata['feature_stats'] = self._calculate_feature_stats(features)

        # Add to history
        self.metrics_history.append(metrics)

        # Check alerts
        self._check_alerts()

        # Log to file
        self._save_metrics(metrics)

    def log_error(self, error: Exception, context: Dict[str, Any]):
        """
        Log prediction errors

        Args:
            error: Exception that occurred
            context: Context information about the error
        """
        error_info = {
            'timestamp': datetime.now(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context
        }

        self.errors_log.append(error_info)

        # Update error count in latest metrics
        if self.metrics_history:
            self.metrics_history[-1].n_errors += 1

        logger.error(f"Model error logged: {error_info}")

    def _calculate_accuracy(self, predictions: pd.DataFrame, actuals: pd.Series) -> float:
        """Calculate prediction accuracy"""

        if 'predicted_class' in predictions.columns:
            # Classification accuracy
            return (predictions['predicted_class'] == actuals).mean()
        elif 'predicted_pnl' in predictions.columns:
            # Directional accuracy for regression
            pred_direction = predictions['predicted_pnl'] > 0
            actual_direction = actuals > 0
            return (pred_direction == actual_direction).mean()
        else:
            return 0.0

    def _calculate_mae(self, predictions: pd.DataFrame, actuals: pd.Series) -> float:
        """Calculate mean absolute error"""

        if 'predicted_pnl' in predictions.columns:
            return np.abs(predictions['predicted_pnl'] - actuals).mean()
        else:
            return 0.0

    def _calculate_rmse(self, predictions: pd.DataFrame, actuals: pd.Series) -> float:
        """Calculate root mean squared error"""

        if 'predicted_pnl' in predictions.columns:
            return np.sqrt(((predictions['predicted_pnl'] - actuals) ** 2).mean())
        else:
            return 0.0

    def _calculate_feature_stats(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Calculate feature statistics"""

        stats = {
            'n_features': len(features.columns),
            'missing_pct': features.isnull().sum().sum() / features.size,
            'zero_pct': (features == 0).sum().sum() / features.size
        }

        # Feature value ranges
        numeric_features = features.select_dtypes(include=[np.number])
        if not numeric_features.empty:
            stats['feature_ranges'] = {
                col: {
                    'min': numeric_features[col].min(),
                    'max': numeric_features[col].max(),
                    'mean': numeric_features[col].mean(),
                    'std': numeric_features[col].std()
                }
                for col in numeric_features.columns[:10]  # Top 10 features
            }

        return stats

    def calculate_feature_importance_stability(self,
                                             current_importance: Dict[str, float]) -> float:
        """
        Calculate stability of feature importance

        Args:
            current_importance: Current feature importance scores

        Returns:
            Stability score (0-1, higher is more stable)
        """
        if self.feature_importance_baseline is None:
            return 1.0

        # Get common features
        common_features = set(current_importance.keys()) & set(self.feature_importance_baseline.keys())

        if not common_features:
            return 0.0

        # Calculate correlation of importance scores
        baseline_scores = [self.feature_importance_baseline[f] for f in common_features]
        current_scores = [current_importance[f] for f in common_features]

        if len(baseline_scores) > 1:
            correlation = np.corrcoef(baseline_scores, current_scores)[0, 1]
            return max(0, correlation)  # Ensure non-negative
        else:
            return 1.0

    def _check_alerts(self):
        """Check alert conditions and trigger if necessary"""

        if len(self.metrics_history) < 1:
            return

        # Calculate current metrics
        current_metrics = self._calculate_current_metrics()

        # Check each alert condition
        triggered_alerts = []

        for condition in self.alert_conditions:
            if condition.metric in current_metrics:
                # Get recent values
                recent_values = self._get_recent_metric_values(
                    condition.metric,
                    condition.window
                )

                if recent_values:
                    # Check if condition is met for all recent values
                    if all(condition.check(v) for v in recent_values):
                        triggered_alerts.append({
                            'condition': condition,
                            'current_value': recent_values[-1],
                            'recent_values': recent_values
                        })

        # Log alerts
        for alert in triggered_alerts:
            self._trigger_alert(alert)

    def _calculate_current_metrics(self) -> Dict[str, float]:
        """Calculate current metrics from history"""

        if not self.metrics_history:
            return {}

        # Get recent metrics
        recent_metrics = list(self.metrics_history)[-10:]  # Last 10 entries

        # Calculate aggregates
        current = {
            'prediction_accuracy': np.mean([m.prediction_accuracy for m in recent_metrics]),
            'prediction_mae': np.mean([m.prediction_mae for m in recent_metrics]),
            'prediction_rmse': np.mean([m.prediction_rmse for m in recent_metrics]),
            'prediction_latency_ms': np.mean([m.prediction_latency_ms for m in recent_metrics]),
            'feature_importance_stability': np.mean([m.feature_importance_stability for m in recent_metrics])
        }

        # Calculate error rate
        total_predictions = sum(m.n_predictions for m in recent_metrics)
        total_errors = sum(m.n_errors for m in recent_metrics)
        current['error_rate'] = total_errors / total_predictions if total_predictions > 0 else 0

        return current

    def _get_recent_metric_values(self, metric: str, window: int) -> List[float]:
        """Get recent values for a specific metric"""

        values = []

        for m in list(self.metrics_history)[-window:]:
            if metric == 'error_rate':
                # Calculate error rate
                error_rate = m.n_errors / m.n_predictions if m.n_predictions > 0 else 0
                values.append(error_rate)
            elif hasattr(m, metric):
                values.append(getattr(m, metric))

        return values

    def _trigger_alert(self, alert: Dict[str, Any]):
        """Trigger an alert"""

        condition = alert['condition']

        alert_message = (
            f"ALERT [{condition.severity.upper()}]: {condition.name}\n"
            f"Metric: {condition.metric}\n"
            f"Current Value: {alert['current_value']:.4f}\n"
            f"Threshold: {condition.threshold} ({condition.direction})\n"
            f"Recent Values: {[f'{v:.4f}' for v in alert['recent_values']]}"
        )

        logger.warning(alert_message)

        # Save alert to file
        alert_log = {
            'timestamp': datetime.now().isoformat(),
            'alert_name': condition.name,
            'severity': condition.severity,
            'metric': condition.metric,
            'value': alert['current_value'],
            'threshold': condition.threshold,
            'message': alert_message
        }

        alert_file = self.log_dir / f"alerts_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(alert_file, 'a') as f:
            f.write(json.dumps(alert_log) + '\n')

    def _save_metrics(self, metrics: ModelMetrics):
        """Save metrics to log file"""

        metrics_dict = {
            'timestamp': metrics.timestamp.isoformat(),
            'prediction_accuracy': metrics.prediction_accuracy,
            'prediction_mae': metrics.prediction_mae,
            'prediction_rmse': metrics.prediction_rmse,
            'feature_importance_stability': metrics.feature_importance_stability,
            'prediction_latency_ms': metrics.prediction_latency_ms,
            'n_predictions': metrics.n_predictions,
            'n_errors': metrics.n_errors,
            'metadata': metrics.metadata
        }

        log_file = self.log_dir / f"metrics_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(metrics_dict, default=str) + '\n')

    def get_monitoring_summary(self,
                              period_days: int = 7) -> Dict[str, Any]:
        """
        Get monitoring summary for specified period

        Args:
            period_days: Number of days to summarize

        Returns:
            Summary dictionary
        """
        # Filter metrics for period
        cutoff_date = datetime.now() - timedelta(days=period_days)
        period_metrics = [m for m in self.metrics_history
                         if m.timestamp >= cutoff_date]

        if not period_metrics:
            return {'status': 'No data for period'}

        # Calculate summary statistics
        summary = {
            'model_name': self.model_name,
            'period_days': period_days,
            'period_start': period_metrics[0].timestamp,
            'period_end': period_metrics[-1].timestamp,
            'n_predictions': sum(m.n_predictions for m in period_metrics),
            'n_errors': sum(m.n_errors for m in period_metrics),
            'metrics': {
                'avg_accuracy': np.mean([m.prediction_accuracy for m in period_metrics]),
                'avg_mae': np.mean([m.prediction_mae for m in period_metrics]),
                'avg_rmse': np.mean([m.prediction_rmse for m in period_metrics]),
                'avg_latency_ms': np.mean([m.prediction_latency_ms for m in period_metrics]),
                'max_latency_ms': max(m.prediction_latency_ms for m in period_metrics),
                'error_rate': sum(m.n_errors for m in period_metrics) /
                             sum(m.n_predictions for m in period_metrics)
            }
        }

        # Compare to baseline
        if self.baseline_metrics:
            summary['vs_baseline'] = {
                metric: (summary['metrics'].get(f'avg_{metric}', 0) -
                        self.baseline_metrics.get(metric, 0))
                for metric in ['accuracy', 'mae', 'rmse']
            }

        # Recent alerts
        recent_alerts = []
        alert_file = self.log_dir / f"alerts_{datetime.now().strftime('%Y%m%d')}.jsonl"
        if alert_file.exists():
            with open(alert_file, 'r') as f:
                for line in f:
                    alert = json.loads(line)
                    if datetime.fromisoformat(alert['timestamp']) >= cutoff_date:
                        recent_alerts.append(alert)

        summary['recent_alerts'] = recent_alerts
        summary['n_alerts'] = len(recent_alerts)

        return summary

    def create_monitoring_report(self) -> str:
        """Create formatted monitoring report"""

        summary = self.get_monitoring_summary()

        report = []
        report.append("=" * 60)
        report.append(f"MODEL MONITORING REPORT - {self.model_name}")
        report.append("=" * 60)
        report.append(f"Period: {summary['period_start']} to {summary['period_end']}")
        report.append(f"Total Predictions: {summary['n_predictions']}")
        report.append(f"Total Errors: {summary['n_errors']}")
        report.append("")

        report.append("PERFORMANCE METRICS:")
        for metric, value in summary['metrics'].items():
            report.append(f"  {metric}: {value:.4f}")
        report.append("")

        if 'vs_baseline' in summary:
            report.append("VS BASELINE:")
            for metric, diff in summary['vs_baseline'].items():
                sign = '+' if diff >= 0 else ''
                report.append(f"  {metric}: {sign}{diff:.4f}")
            report.append("")

        if summary['recent_alerts']:
            report.append(f"RECENT ALERTS ({summary['n_alerts']}):")
            for alert in summary['recent_alerts'][:5]:  # Show top 5
                report.append(f"  [{alert['severity']}] {alert['alert_name']}")
                report.append(f"    Value: {alert['value']:.4f}, Threshold: {alert['threshold']}")
        else:
            report.append("No recent alerts")

        report.append("=" * 60)

        return "\n".join(report)
