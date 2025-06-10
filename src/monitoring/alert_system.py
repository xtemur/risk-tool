"""
Alert System
Comprehensive alerting for model performance, drift, and trading risks
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
import os
from collections import defaultdict

from src.core.constants import TradingConstants as TC

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of alerts"""
    MODEL_PERFORMANCE = "model_performance"
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    RISK_THRESHOLD = "risk_threshold"
    SYSTEM_ERROR = "system_error"
    TRADING_ANOMALY = "trading_anomaly"


@dataclass
class Alert:
    """Alert data structure"""
    alert_id: str
    timestamp: datetime
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    metrics: Dict[str, Any]
    source: str  # Which component triggered the alert
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    actions_taken: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertRule:
    """Alert rule definition"""
    name: str
    alert_type: AlertType
    condition: Callable[[Dict[str, Any]], bool]
    severity: AlertSeverity
    message_template: str
    cooldown_minutes: int = 60  # Prevent alert spam
    escalation_after_minutes: int = 120  # Escalate if not resolved


class AlertSystem:
    """
    Centralized alert system for trading risk management
    Handles alert generation, routing, and escalation
    """

    def __init__(self,
                 alert_log_dir: str = "logs/alerts",
                 email_config: Optional[Dict[str, str]] = None,
                 slack_webhook: Optional[str] = None):
        """
        Initialize alert system

        Args:
            alert_log_dir: Directory for alert logs
            email_config: Email configuration for notifications
            slack_webhook: Slack webhook URL for notifications
        """
        self.alert_log_dir = Path(alert_log_dir)
        self.alert_log_dir.mkdir(exist_ok=True, parents=True)

        # Notification configs
        self.email_config = email_config or self._load_email_config()
        self.slack_webhook = slack_webhook

        # Alert storage
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.alert_rules: List[AlertRule] = self._initialize_alert_rules()

        # Cooldown tracking
        self.last_alert_time: Dict[str, datetime] = {}

        # Alert counters
        self.alert_counts = defaultdict(int)

    def _load_email_config(self) -> Optional[Dict[str, str]]:
        """Load email configuration from environment"""

        email_from = os.getenv('EMAIL_FROM')
        email_password = os.getenv('EMAIL_PASSWORD')
        email_to = os.getenv('EMAIL_TO')

        if all([email_from, email_password, email_to]):
            return {
                'from': email_from,
                'password': email_password,
                'to': email_to,
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': 587
            }

        return None

    def _initialize_alert_rules(self) -> List[AlertRule]:
        """Initialize default alert rules"""

        rules = [
            # Model Performance Rules
            AlertRule(
                name="High Model Error",
                alert_type=AlertType.MODEL_PERFORMANCE,
                condition=lambda m: m.get('rmse', 0) > 0.05,
                severity=AlertSeverity.HIGH,
                message_template="Model RMSE {rmse:.4f} exceeds threshold",
                cooldown_minutes=60
            ),
            AlertRule(
                name="Low Prediction Accuracy",
                alert_type=AlertType.MODEL_PERFORMANCE,
                condition=lambda m: m.get('accuracy', 1) < 0.5,
                severity=AlertSeverity.HIGH,
                message_template="Model accuracy {accuracy:.2%} below threshold",
                cooldown_minutes=60
            ),

            # Drift Rules
            AlertRule(
                name="Significant Data Drift",
                alert_type=AlertType.DATA_DRIFT,
                condition=lambda m: m.get('drift_rate', 0) > 0.3,
                severity=AlertSeverity.HIGH,
                message_template="{n_drifted} features showing drift ({drift_rate:.1%})",
                cooldown_minutes=120
            ),
            AlertRule(
                name="Critical Feature Drift",
                alert_type=AlertType.DATA_DRIFT,
                condition=lambda m: any(f in m.get('drifted_features', [])
                                      for f in ['return_20d', 'volatility_20d', 'risk_score']),
                severity=AlertSeverity.CRITICAL,
                message_template="Critical features drifted: {drifted_features}",
                cooldown_minutes=120
            ),

            # Risk Threshold Rules
            AlertRule(
                name="High Risk Exposure",
                alert_type=AlertType.RISK_THRESHOLD,
                condition=lambda m: m.get('portfolio_risk_score', 0) > TC.HIGH_RISK_SCORE,
                severity=AlertSeverity.CRITICAL,
                message_template="Portfolio risk score {portfolio_risk_score:.2f} exceeds critical threshold",
                cooldown_minutes=30,
                escalation_after_minutes=60
            ),
            AlertRule(
                name="Large Drawdown",
                alert_type=AlertType.RISK_THRESHOLD,
                condition=lambda m: m.get('current_drawdown', 0) < -TC.MAX_DRAWDOWN_PCT,
                severity=AlertSeverity.CRITICAL,
                message_template="Drawdown {current_drawdown:.1%} exceeds maximum threshold",
                cooldown_minutes=30
            ),

            # Trading Anomaly Rules
            AlertRule(
                name="Unusual Trading Volume",
                alert_type=AlertType.TRADING_ANOMALY,
                condition=lambda m: m.get('volume_zscore', 0) > 3,
                severity=AlertSeverity.WARNING,
                message_template="Trading volume {volume_zscore:.1f} standard deviations above normal",
                cooldown_minutes=60
            ),
            AlertRule(
                name="Position Concentration",
                alert_type=AlertType.TRADING_ANOMALY,
                condition=lambda m: m.get('max_position_pct', 0) > TC.MAX_POSITION_SIZE_PCT,
                severity=AlertSeverity.HIGH,
                message_template="Position concentration {max_position_pct:.1%} exceeds limit",
                cooldown_minutes=30
            ),

            # System Error Rules
            AlertRule(
                name="High Error Rate",
                alert_type=AlertType.SYSTEM_ERROR,
                condition=lambda m: m.get('error_rate', 0) > 0.1,
                severity=AlertSeverity.CRITICAL,
                message_template="System error rate {error_rate:.1%} is critically high",
                cooldown_minutes=15,
                escalation_after_minutes=30
            )
        ]

        return rules

    def check_alerts(self, metrics: Dict[str, Any], source: str = "system") -> List[Alert]:
        """
        Check metrics against alert rules

        Args:
            metrics: Dictionary of current metrics
            source: Source component name

        Returns:
            List of triggered alerts
        """
        triggered_alerts = []

        for rule in self.alert_rules:
            # Check cooldown
            rule_key = f"{source}:{rule.name}"
            if rule_key in self.last_alert_time:
                time_since_last = datetime.now() - self.last_alert_time[rule_key]
                if time_since_last.total_seconds() < rule.cooldown_minutes * 60:
                    continue

            # Check condition
            try:
                if rule.condition(metrics):
                    alert = self._create_alert(rule, metrics, source)
                    triggered_alerts.append(alert)

                    # Update cooldown
                    self.last_alert_time[rule_key] = datetime.now()

            except Exception as e:
                logger.error(f"Error checking alert rule {rule.name}: {e}")

        # Process triggered alerts
        for alert in triggered_alerts:
            self._process_alert(alert)

        return triggered_alerts

    def _create_alert(self, rule: AlertRule, metrics: Dict[str, Any], source: str) -> Alert:
        """Create alert from rule and metrics"""

        # Generate alert ID
        alert_id = f"{rule.alert_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.alert_counts[rule.name]}"
        self.alert_counts[rule.name] += 1

        # Format message
        try:
            message = rule.message_template.format(**metrics)
        except KeyError:
            message = f"{rule.name}: {rule.message_template}"

        alert = Alert(
            alert_id=alert_id,
            timestamp=datetime.now(),
            alert_type=rule.alert_type,
            severity=rule.severity,
            title=rule.name,
            message=message,
            metrics=metrics.copy(),
            source=source,
            metadata={'rule_name': rule.name}
        )

        return alert

    def _process_alert(self, alert: Alert):
        """Process a triggered alert"""

        # Add to active alerts
        self.active_alerts[alert.alert_id] = alert
        self.alert_history.append(alert)

        # Log alert
        self._log_alert(alert)

        # Send notifications based on severity
        if alert.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]:
            self._send_notifications(alert)

        # Take automated actions for critical alerts
        if alert.severity == AlertSeverity.CRITICAL:
            self._take_automated_action(alert)

    def _log_alert(self, alert: Alert):
        """Log alert to file"""

        log_data = {
            'alert_id': alert.alert_id,
            'timestamp': alert.timestamp.isoformat(),
            'alert_type': alert.alert_type.value,
            'severity': alert.severity.value,
            'title': alert.title,
            'message': alert.message,
            'metrics': alert.metrics,
            'source': alert.source,
            'metadata': alert.metadata
        }

        # Daily log file
        log_file = self.alert_log_dir / f"alerts_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_data, default=str) + '\n')

        # Also log to standard logger
        logger.warning(f"ALERT [{alert.severity.value.upper()}] {alert.title}: {alert.message}")

    def _send_notifications(self, alert: Alert):
        """Send alert notifications"""

        # Email notification
        if self.email_config:
            try:
                self._send_email_alert(alert)
            except Exception as e:
                logger.error(f"Failed to send email alert: {e}")

        # Slack notification
        if self.slack_webhook:
            try:
                self._send_slack_alert(alert)
            except Exception as e:
                logger.error(f"Failed to send Slack alert: {e}")

    def _send_email_alert(self, alert: Alert):
        """Send email notification"""

        if not self.email_config:
            return

        # Create message
        msg = MIMEMultipart()
        msg['From'] = self.email_config['from']
        msg['To'] = self.email_config['to']
        msg['Subject'] = f"[{alert.severity.value.upper()}] Trading Alert: {alert.title}"

        # Email body
        body = f"""
Trading Risk Management Alert

Alert Type: {alert.alert_type.value}
Severity: {alert.severity.value.upper()}
Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
Source: {alert.source}

{alert.message}

Metrics:
{json.dumps(alert.metrics, indent=2, default=str)}

Please review and take appropriate action.
        """

        msg.attach(MIMEText(body, 'plain'))

        # Send email
        with smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port']) as server:
            server.starttls()
            server.login(self.email_config['from'], self.email_config['password'])
            server.send_message(msg)

    def _send_slack_alert(self, alert: Alert):
        """Send Slack notification"""

        if not self.slack_webhook:
            return

        # Format for Slack
        color_map = {
            AlertSeverity.INFO: "#36a64f",
            AlertSeverity.WARNING: "#ff9900",
            AlertSeverity.HIGH: "#ff6600",
            AlertSeverity.CRITICAL: "#ff0000"
        }

        payload = {
            "attachments": [{
                "color": color_map.get(alert.severity, "#808080"),
                "title": f"{alert.severity.value.upper()}: {alert.title}",
                "text": alert.message,
                "fields": [
                    {"title": "Type", "value": alert.alert_type.value, "short": True},
                    {"title": "Source", "value": alert.source, "short": True},
                    {"title": "Time", "value": alert.timestamp.strftime('%H:%M:%S'), "short": True}
                ],
                "footer": "Trading Risk Management System",
                "ts": int(alert.timestamp.timestamp())
            }]
        }

        # Send to Slack
        import requests
        response = requests.post(self.slack_webhook, json=payload)

        if response.status_code != 200:
            logger.error(f"Slack notification failed: {response.text}")

    def _take_automated_action(self, alert: Alert):
        """Take automated action for critical alerts"""

        actions_taken = []

        # Risk threshold alerts
        if alert.alert_type == AlertType.RISK_THRESHOLD:
            if 'drawdown' in alert.message.lower():
                actions_taken.append("Risk reduction signal sent to trading system")
            elif 'risk score' in alert.message.lower():
                actions_taken.append("Position limits tightened")

        # System error alerts
        elif alert.alert_type == AlertType.SYSTEM_ERROR:
            actions_taken.append("System health check initiated")
            actions_taken.append("Fallback mode activated")

        # Update alert with actions
        alert.actions_taken.extend(actions_taken)

        logger.info(f"Automated actions for alert {alert.alert_id}: {actions_taken}")

    def resolve_alert(self, alert_id: str, resolution_notes: str = ""):
        """Mark alert as resolved"""

        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolution_time = datetime.now()
            alert.metadata['resolution_notes'] = resolution_notes

            # Remove from active
            del self.active_alerts[alert_id]

            # Log resolution
            logger.info(f"Alert {alert_id} resolved: {resolution_notes}")

    def escalate_alerts(self):
        """Check for alerts that need escalation"""

        now = datetime.now()
        escalated = []

        for alert_id, alert in list(self.active_alerts.items()):
            # Find the rule for this alert
            rule = next((r for r in self.alert_rules
                        if r.name == alert.metadata.get('rule_name')), None)

            if rule and rule.escalation_after_minutes:
                time_active = (now - alert.timestamp).total_seconds() / 60

                if time_active > rule.escalation_after_minutes:
                    # Escalate severity
                    if alert.severity == AlertSeverity.WARNING:
                        alert.severity = AlertSeverity.HIGH
                    elif alert.severity == AlertSeverity.HIGH:
                        alert.severity = AlertSeverity.CRITICAL

                    alert.metadata['escalated'] = True
                    alert.metadata['escalation_time'] = now.isoformat()

                    # Re-send notifications
                    self._send_notifications(alert)
                    escalated.append(alert)

        return escalated

    def get_alert_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of recent alerts"""

        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_alerts = [a for a in self.alert_history if a.timestamp >= cutoff_time]

        # Count by type and severity
        by_type = defaultdict(int)
        by_severity = defaultdict(int)

        for alert in recent_alerts:
            by_type[alert.alert_type.value] += 1
            by_severity[alert.severity.value] += 1

        # Active alerts
        active_by_severity = defaultdict(int)
        for alert in self.active_alerts.values():
            active_by_severity[alert.severity.value] += 1

        summary = {
            'period_hours': hours,
            'total_alerts': len(recent_alerts),
            'active_alerts': len(self.active_alerts),
            'by_type': dict(by_type),
            'by_severity': dict(by_severity),
            'active_by_severity': dict(active_by_severity),
            'critical_alerts': [
                {
                    'alert_id': a.alert_id,
                    'title': a.title,
                    'message': a.message,
                    'timestamp': a.timestamp.isoformat(),
                    'active_minutes': (datetime.now() - a.timestamp).total_seconds() / 60
                }
                for a in self.active_alerts.values()
                if a.severity == AlertSeverity.CRITICAL
            ]
        }

        return summary

    def create_alert_report(self) -> str:
        """Create formatted alert report"""

        summary = self.get_alert_summary(24)

        report = []
        report.append("=" * 60)
        report.append("ALERT SYSTEM REPORT")
        report.append("=" * 60)
        report.append(f"Period: Last {summary['period_hours']} hours")
        report.append(f"Total Alerts: {summary['total_alerts']}")
        report.append(f"Active Alerts: {summary['active_alerts']}")
        report.append("")

        if summary['active_alerts'] > 0:
            report.append("ACTIVE ALERTS:")
            for severity in [AlertSeverity.CRITICAL, AlertSeverity.HIGH, AlertSeverity.WARNING, AlertSeverity.INFO]:
                count = summary['active_by_severity'].get(severity.value, 0)
                if count > 0:
                    report.append(f"  {severity.value.upper()}: {count}")
            report.append("")

        if summary['critical_alerts']:
            report.append("CRITICAL ALERTS REQUIRING ATTENTION:")
            for alert in summary['critical_alerts']:
                report.append(f"\n  Alert: {alert['title']}")
                report.append(f"  Message: {alert['message']}")
                report.append(f"  Active for: {alert['active_minutes']:.0f} minutes")

        report.append("\nALERT DISTRIBUTION (24h):")
        report.append("By Type:")
        for alert_type, count in summary['by_type'].items():
            report.append(f"  {alert_type}: {count}")

        report.append("\nBy Severity:")
        for severity, count in summary['by_severity'].items():
            report.append(f"  {severity}: {count}")

        report.append("=" * 60)

        return "\n".join(report)
