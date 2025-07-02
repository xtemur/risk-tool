# src/model_monitoring.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import json
import os
from datetime import datetime, timedelta
from scipy import stats
from sklearn.metrics import roc_auc_score
import warnings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelPerformanceMonitor:
    """
    Comprehensive model performance monitoring and alerting system.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.performance_history = []
        self.alert_thresholds = config.get('model_quality', {}).get('thresholds', {})

    def track_daily_performance(self, predictions: pd.DataFrame, actuals: pd.DataFrame) -> Dict:
        """
        Track daily model performance and detect degradation.

        Args:
            predictions: DataFrame with model predictions
            actuals: DataFrame with actual outcomes

        Returns:
            Dict with performance metrics and alerts
        """
        logger.info("Tracking daily model performance...")

        # Merge predictions with actuals
        merged = predictions.merge(actuals, on=['account_id', 'trade_date'], how='inner')

        if len(merged) == 0:
            logger.warning("No matching predictions and actuals found")
            return {'status': 'no_data', 'alerts': ['No matching data found']}

        # Calculate performance metrics
        metrics = self._calculate_performance_metrics(merged)

        # Add timestamp
        metrics['timestamp'] = datetime.now().isoformat()
        metrics['date'] = merged['trade_date'].max().isoformat()

        # Check for alerts
        alerts = self._check_performance_alerts(metrics)
        metrics['alerts'] = alerts

        # Store in performance history
        self.performance_history.append(metrics)

        # Trim history to keep only recent data (30 days)
        cutoff_date = datetime.now() - timedelta(days=30)
        self.performance_history = [
            m for m in self.performance_history
            if datetime.fromisoformat(m['timestamp']) > cutoff_date
        ]

        logger.info(f"Performance tracking completed. Found {len(alerts)} alerts.")

        return metrics

    def _calculate_performance_metrics(self, merged_data: pd.DataFrame) -> Dict:
        """Calculate comprehensive performance metrics."""

        metrics = {}

        # VaR Model Performance
        if 'pred_var' in merged_data.columns and 'actual_pnl' in merged_data.columns:
            violations = merged_data['actual_pnl'] < merged_data['pred_var']
            violation_rate = violations.mean()

            # Expected vs actual violation rate
            expected_rate = self.config['production_model']['var_model']['alpha']
            violation_deviation = abs(violation_rate - expected_rate)

            # Average violation size
            if violations.sum() > 0:
                avg_violation_size = merged_data.loc[violations, 'actual_pnl'].mean()
                avg_predicted_var = merged_data.loc[violations, 'pred_var'].mean()
                violation_severity = avg_violation_size / avg_predicted_var
            else:
                avg_violation_size = 0
                violation_severity = 0

            metrics['var_performance'] = {
                'violation_rate': violation_rate,
                'expected_rate': expected_rate,
                'violation_deviation': violation_deviation,
                'avg_violation_size': avg_violation_size,
                'violation_severity': violation_severity,
                'n_violations': violations.sum(),
                'n_samples': len(merged_data)
            }

        # Loss Model Performance
        if 'pred_loss_proba' in merged_data.columns and 'actual_large_loss' in merged_data.columns:
            try:
                auc = roc_auc_score(merged_data['actual_large_loss'], merged_data['pred_loss_proba'])
            except:
                auc = 0.5

            # Calibration check
            prob_bins = np.linspace(0, 1, 11)
            calibration_error = 0
            for i in range(len(prob_bins) - 1):
                mask = (merged_data['pred_loss_proba'] >= prob_bins[i]) & \
                       (merged_data['pred_loss_proba'] < prob_bins[i+1])
                if mask.sum() > 0:
                    expected_rate = merged_data.loc[mask, 'pred_loss_proba'].mean()
                    actual_rate = merged_data.loc[mask, 'actual_large_loss'].mean()
                    calibration_error += abs(expected_rate - actual_rate) * mask.sum()

            calibration_error /= len(merged_data)

            metrics['loss_performance'] = {
                'auc': auc,
                'calibration_error': calibration_error,
                'positive_rate': merged_data['actual_large_loss'].mean(),
                'avg_predicted_prob': merged_data['pred_loss_proba'].mean()
            }

        # Data Quality Metrics
        metrics['data_quality'] = {
            'missing_predictions': merged_data['pred_var'].isnull().sum() if 'pred_var' in merged_data else 0,
            'missing_actuals': merged_data['actual_pnl'].isnull().sum() if 'actual_pnl' in merged_data else 0,
            'extreme_predictions': (abs(merged_data['pred_var']) > 50000).sum() if 'pred_var' in merged_data else 0,
            'data_freshness_hours': self._calculate_data_freshness(merged_data)
        }

        return metrics

    def _calculate_data_freshness(self, data: pd.DataFrame) -> float:
        """Calculate how fresh the data is."""
        if 'trade_date' in data.columns:
            latest_date = pd.to_datetime(data['trade_date']).max()
            hours_old = (datetime.now() - latest_date).total_seconds() / 3600
            return hours_old
        return 0

    def _check_performance_alerts(self, metrics: Dict) -> List[str]:
        """Check for performance degradation alerts."""

        alerts = []

        # VaR Performance Alerts
        if 'var_performance' in metrics:
            var_perf = metrics['var_performance']

            # Violation rate too high/low
            violation_threshold = self.alert_thresholds.get('var_violation_deviation', 0.02)
            if var_perf['violation_deviation'] > violation_threshold:
                alerts.append(f"VaR violation rate deviation: {var_perf['violation_deviation']:.1%} "
                            f"(threshold: {violation_threshold:.1%})")

            # Too many extreme violations
            if var_perf['violation_severity'] > 2.0:
                alerts.append(f"Severe VaR violations: average {var_perf['violation_severity']:.1f}x larger than predicted")

        # Loss Model Performance Alerts
        if 'loss_performance' in metrics:
            loss_perf = metrics['loss_performance']

            # AUC degradation
            min_auc = self.alert_thresholds.get('min_auc', 0.55)
            if loss_perf['auc'] < min_auc:
                alerts.append(f"Loss model AUC degradation: {loss_perf['auc']:.3f} < {min_auc}")

            # Poor calibration
            max_calibration_error = self.alert_thresholds.get('max_calibration_error', 0.1)
            if loss_perf['calibration_error'] > max_calibration_error:
                alerts.append(f"Poor model calibration: {loss_perf['calibration_error']:.3f} error")

        # Data Quality Alerts
        if 'data_quality' in metrics:
            data_qual = metrics['data_quality']

            # Stale data
            if data_qual['data_freshness_hours'] > 48:
                alerts.append(f"Stale data: {data_qual['data_freshness_hours']:.1f} hours old")

            # Too many missing values
            if data_qual['missing_predictions'] > 0:
                alerts.append(f"Missing predictions: {data_qual['missing_predictions']} records")

        return alerts

    def generate_performance_report(self, lookback_days: int = 7) -> Dict:
        """Generate performance report for recent period."""

        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        recent_metrics = [
            m for m in self.performance_history
            if datetime.fromisoformat(m['timestamp']) > cutoff_date
        ]

        if not recent_metrics:
            return {'status': 'no_data', 'message': 'No recent performance data available'}

        # Aggregate metrics
        report = {
            'period': f'Last {lookback_days} days',
            'n_observations': len(recent_metrics),
            'date_range': [
                min(m['timestamp'] for m in recent_metrics),
                max(m['timestamp'] for m in recent_metrics)
            ]
        }

        # VaR performance trends
        var_violation_rates = [m['var_performance']['violation_rate'] for m in recent_metrics if 'var_performance' in m]
        if var_violation_rates:
            report['var_trends'] = {
                'avg_violation_rate': np.mean(var_violation_rates),
                'violation_rate_trend': self._calculate_trend(var_violation_rates),
                'violation_rate_stability': np.std(var_violation_rates)
            }

        # Loss model performance trends
        loss_aucs = [m['loss_performance']['auc'] for m in recent_metrics if 'loss_performance' in m]
        if loss_aucs:
            report['loss_trends'] = {
                'avg_auc': np.mean(loss_aucs),
                'auc_trend': self._calculate_trend(loss_aucs),
                'auc_stability': np.std(loss_aucs)
            }

        # Alert summary
        all_alerts = []
        for m in recent_metrics:
            all_alerts.extend(m.get('alerts', []))

        report['alert_summary'] = {
            'total_alerts': len(all_alerts),
            'unique_alert_types': len(set(all_alerts)),
            'most_common_alerts': self._get_most_common_alerts(all_alerts)
        }

        return report

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction of a time series."""
        if len(values) < 3:
            return 'insufficient_data'

        x = np.arange(len(values))
        slope, _, r_value, p_value, _ = stats.linregress(x, values)

        if p_value > 0.1:  # Not statistically significant
            return 'stable'
        elif slope > 0:
            return 'improving' if abs(r_value) > 0.5 else 'slightly_improving'
        else:
            return 'degrading' if abs(r_value) > 0.5 else 'slightly_degrading'

    def _get_most_common_alerts(self, alerts: List[str]) -> List[Tuple[str, int]]:
        """Get most common alert types."""
        from collections import Counter
        alert_types = [alert.split(':')[0] for alert in alerts]
        return Counter(alert_types).most_common(3)


class ABTestingFramework:
    """
    A/B testing framework for model comparisons.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.experiments = {}

    def create_experiment(self, name: str, model_a_path: str, model_b_path: str,
                         traffic_split: float = 0.5, duration_days: int = 14) -> str:
        """
        Create a new A/B testing experiment.

        Args:
            name: Experiment name
            model_a_path: Path to champion model
            model_b_path: Path to challenger model
            traffic_split: Fraction of traffic for model B (0.0 to 1.0)
            duration_days: Experiment duration in days

        Returns:
            Experiment ID
        """
        experiment_id = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        experiment = {
            'id': experiment_id,
            'name': name,
            'model_a_path': model_a_path,
            'model_b_path': model_b_path,
            'traffic_split': traffic_split,
            'start_date': datetime.now().isoformat(),
            'end_date': (datetime.now() + timedelta(days=duration_days)).isoformat(),
            'status': 'active',
            'results': [],
            'metadata': {
                'created_by': 'model_monitoring_system',
                'duration_days': duration_days
            }
        }

        self.experiments[experiment_id] = experiment

        logger.info(f"Created A/B test experiment: {experiment_id}")
        logger.info(f"  Champion: {model_a_path}")
        logger.info(f"  Challenger: {model_b_path}")
        logger.info(f"  Traffic split: {traffic_split:.0%} to challenger")

        return experiment_id

    def assign_model(self, experiment_id: str, trader_id: str) -> str:
        """
        Assign a model to a trader for an experiment.

        Args:
            experiment_id: Experiment identifier
            trader_id: Trader identifier

        Returns:
            'model_a' or 'model_b'
        """
        if experiment_id not in self.experiments:
            return 'model_a'  # Default to champion

        experiment = self.experiments[experiment_id]

        # Check if experiment is still active
        if datetime.now() > datetime.fromisoformat(experiment['end_date']):
            experiment['status'] = 'completed'
            return 'model_a'

        # Deterministic assignment based on trader ID hash
        import hashlib
        hash_value = int(hashlib.md5(f"{experiment_id}_{trader_id}".encode()).hexdigest(), 16)
        assignment_ratio = (hash_value % 1000) / 1000.0

        if assignment_ratio < experiment['traffic_split']:
            return 'model_b'
        else:
            return 'model_a'

    def record_experiment_result(self, experiment_id: str, trader_id: str,
                               model_used: str, predictions: Dict, actuals: Dict) -> None:
        """Record experiment results for analysis."""

        if experiment_id not in self.experiments:
            return

        result = {
            'timestamp': datetime.now().isoformat(),
            'trader_id': trader_id,
            'model_used': model_used,
            'predictions': predictions,
            'actuals': actuals
        }

        self.experiments[experiment_id]['results'].append(result)

    def analyze_experiment(self, experiment_id: str) -> Dict:
        """
        Analyze A/B test results with statistical significance testing.

        Args:
            experiment_id: Experiment identifier

        Returns:
            Dict with analysis results
        """
        if experiment_id not in self.experiments:
            return {'error': 'Experiment not found'}

        experiment = self.experiments[experiment_id]
        results = experiment['results']

        if len(results) < 10:
            return {'status': 'insufficient_data', 'n_samples': len(results)}

        # Separate results by model
        model_a_results = [r for r in results if r['model_used'] == 'model_a']
        model_b_results = [r for r in results if r['model_used'] == 'model_b']

        analysis = {
            'experiment_id': experiment_id,
            'experiment_name': experiment['name'],
            'analysis_date': datetime.now().isoformat(),
            'n_samples_a': len(model_a_results),
            'n_samples_b': len(model_b_results),
            'metrics_comparison': {}
        }

        # VaR Performance Comparison
        if model_a_results and model_b_results:
            var_analysis = self._compare_var_performance(model_a_results, model_b_results)
            analysis['metrics_comparison']['var'] = var_analysis

            # Loss Model Performance Comparison
            loss_analysis = self._compare_loss_performance(model_a_results, model_b_results)
            analysis['metrics_comparison']['loss'] = loss_analysis

            # Overall recommendation
            analysis['recommendation'] = self._generate_recommendation(var_analysis, loss_analysis)

        return analysis

    def _compare_var_performance(self, results_a: List[Dict], results_b: List[Dict]) -> Dict:
        """Compare VaR performance between two models."""

        # Extract violation rates
        violations_a = []
        violations_b = []

        for result in results_a:
            if 'pred_var' in result['predictions'] and 'actual_pnl' in result['actuals']:
                violation = result['actuals']['actual_pnl'] < result['predictions']['pred_var']
                violations_a.append(violation)

        for result in results_b:
            if 'pred_var' in result['predictions'] and 'actual_pnl' in result['actuals']:
                violation = result['actuals']['actual_pnl'] < result['predictions']['pred_var']
                violations_b.append(violation)

        if not violations_a or not violations_b:
            return {'status': 'insufficient_data'}

        violation_rate_a = np.mean(violations_a)
        violation_rate_b = np.mean(violations_b)

        # Statistical significance test
        from scipy.stats import chi2_contingency
        contingency_table = [
            [sum(violations_a), len(violations_a) - sum(violations_a)],
            [sum(violations_b), len(violations_b) - sum(violations_b)]
        ]

        chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)

        return {
            'model_a_violation_rate': violation_rate_a,
            'model_b_violation_rate': violation_rate_b,
            'improvement': violation_rate_a - violation_rate_b,  # Negative is better for model B
            'relative_improvement': (violation_rate_a - violation_rate_b) / violation_rate_a if violation_rate_a > 0 else 0,
            'statistical_significance': p_value < 0.05,
            'p_value': p_value,
            'confidence_level': 0.95
        }

    def _compare_loss_performance(self, results_a: List[Dict], results_b: List[Dict]) -> Dict:
        """Compare loss model performance between two models."""

        # Extract AUC scores
        probs_a, actuals_a = [], []
        probs_b, actuals_b = [], []

        for result in results_a:
            if 'pred_loss_proba' in result['predictions'] and 'actual_large_loss' in result['actuals']:
                probs_a.append(result['predictions']['pred_loss_proba'])
                actuals_a.append(result['actuals']['actual_large_loss'])

        for result in results_b:
            if 'pred_loss_proba' in result['predictions'] and 'actual_large_loss' in result['actuals']:
                probs_b.append(result['predictions']['pred_loss_proba'])
                actuals_b.append(result['actuals']['actual_large_loss'])

        if not probs_a or not probs_b:
            return {'status': 'insufficient_data'}

        try:
            auc_a = roc_auc_score(actuals_a, probs_a)
            auc_b = roc_auc_score(actuals_b, probs_b)
        except:
            return {'status': 'calculation_error'}

        # Simple comparison (more sophisticated tests could be implemented)
        return {
            'model_a_auc': auc_a,
            'model_b_auc': auc_b,
            'improvement': auc_b - auc_a,  # Positive is better for model B
            'relative_improvement': (auc_b - auc_a) / auc_a if auc_a > 0 else 0,
            'n_samples_a': len(probs_a),
            'n_samples_b': len(probs_b)
        }

    def _generate_recommendation(self, var_analysis: Dict, loss_analysis: Dict) -> Dict:
        """Generate overall recommendation based on A/B test results."""

        recommendation = {
            'action': 'continue_champion',  # Default
            'confidence': 'low',
            'reasons': []
        }

        # Check VaR performance
        if var_analysis.get('statistical_significance', False):
            var_improvement = var_analysis.get('improvement', 0)
            if var_improvement < -0.01:  # 1% better violation rate
                recommendation['reasons'].append('Challenger has significantly better VaR performance')
                recommendation['action'] = 'deploy_challenger'
            elif var_improvement > 0.01:  # 1% worse violation rate
                recommendation['reasons'].append('Challenger has significantly worse VaR performance')

        # Check loss model performance
        loss_improvement = loss_analysis.get('improvement', 0)
        if loss_improvement > 0.02:  # 2% AUC improvement
            recommendation['reasons'].append('Challenger has better loss prediction performance')
            if recommendation['action'] != 'continue_champion':
                recommendation['action'] = 'deploy_challenger'
        elif loss_improvement < -0.02:  # 2% AUC degradation
            recommendation['reasons'].append('Challenger has worse loss prediction performance')
            recommendation['action'] = 'continue_champion'

        # Set confidence level
        n_total_samples = var_analysis.get('n_samples_a', 0) + var_analysis.get('n_samples_b', 0)
        if n_total_samples > 100:
            recommendation['confidence'] = 'high'
        elif n_total_samples > 50:
            recommendation['confidence'] = 'medium'

        # If no strong signal either way
        if not recommendation['reasons']:
            recommendation['reasons'].append('No significant performance difference detected')
            recommendation['action'] = 'continue_testing'

        return recommendation

    def save_experiments(self, file_path: str) -> None:
        """Save experiments to file."""
        with open(file_path, 'w') as f:
            json.dump(self.experiments, f, indent=2, default=str)

    def load_experiments(self, file_path: str) -> None:
        """Load experiments from file."""
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                self.experiments = json.load(f)


def generate_monitoring_dashboard_data(monitor: ModelPerformanceMonitor,
                                     ab_framework: ABTestingFramework) -> Dict:
    """Generate data for monitoring dashboard."""

    dashboard_data = {
        'timestamp': datetime.now().isoformat(),
        'performance_summary': monitor.generate_performance_report(7),
        'active_experiments': [],
        'recent_alerts': []
    }

    # Add active experiments
    for exp_id, experiment in ab_framework.experiments.items():
        if experiment['status'] == 'active':
            experiment_summary = {
                'id': exp_id,
                'name': experiment['name'],
                'start_date': experiment['start_date'],
                'end_date': experiment['end_date'],
                'traffic_split': experiment['traffic_split'],
                'n_results': len(experiment['results'])
            }
            dashboard_data['active_experiments'].append(experiment_summary)

    # Add recent alerts
    if monitor.performance_history:
        recent_metrics = monitor.performance_history[-5:]  # Last 5 observations
        for metrics in recent_metrics:
            for alert in metrics.get('alerts', []):
                dashboard_data['recent_alerts'].append({
                    'timestamp': metrics['timestamp'],
                    'alert': alert
                })

    return dashboard_data
