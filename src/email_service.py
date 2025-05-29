import logging
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, List

import pandas as pd
from dotenv import load_dotenv
from jinja2 import Template

load_dotenv()


class EmailService:
    def __init__(self, config: Dict):
        self.config = config["email"]
        self.risk_config = config.get("risk", {})
        self.logger = logging.getLogger(__name__)

        # Email credentials from environment
        self.from_email = os.getenv("EMAIL_FROM", self.config.get("from_email", ""))
        self.password = os.getenv("EMAIL_PASSWORD", self.config.get("password", ""))

        if not self.from_email or not self.password:
            self.logger.warning("Email credentials not properly configured")

        # Load email template
        self.template = self._load_template()

    def _load_template(self) -> Template:
        """Load email template with fallback"""
        try:
            with open("config/email_template.html", "r") as f:
                template_content = f.read()
        except FileNotFoundError:
            self.logger.warning("Email template not found, using default")
            template_content = self._get_default_template()

        return Template(template_content)

    def _get_default_template(self) -> str:
        """Default email template if file not found"""
        return """
<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background-color: #2c3e50; color: white; padding: 20px; text-align: center; }
        .summary { background-color: #ecf0f1; padding: 15px; margin: 20px 0; border-radius: 5px; }
        .risk-high { background-color: #e74c3c; color: white; }
        .risk-medium { background-color: #f39c12; color: white; }
        .risk-low { background-color: #27ae60; color: white; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #34495e; color: white; }
        .risk-cell { padding: 8px; border-radius: 4px; text-align: center; font-weight: bold; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Daily Risk Report</h1>
        <p>{{ summary.date }}</p>
    </div>

    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Total Traders:</strong> {{ summary.total_traders }}</p>
        <p><strong>High Risk:</strong> {{ summary.high_risk_count }} |
           <strong>Medium Risk:</strong> {{ summary.medium_risk_count }} |
           <strong>Low Risk:</strong> {{ summary.low_risk_count }}</p>
        <p><strong>Total Recent P&L (5d):</strong> ${{ "{:,.2f}".format(summary.total_recent_pnl) }}</p>
    </div>

    {% if high_risk_traders %}
    <h2 style="color: #e74c3c;">🚨 High Risk Traders</h2>
    <table>
        <tr>
            <th>Trader</th>
            <th>Risk Level</th>
            <th>Predicted P&L</th>
            <th>Recent P&L (5d)</th>
            <th>Confidence</th>
        </tr>
        {% for trader in high_risk_traders %}
        <tr>
            <td>{{ trader.trader_name }}</td>
            <td><span class="risk-cell risk-high">{{ trader.risk_level }}</span></td>
            <td>${{ "{:,.2f}".format(trader.predicted_pnl) }}</td>
            <td>${{ "{:,.2f}".format(trader.recent_pnl) }}</td>
            <td>{{ trader.confidence }}</td>
        </tr>
        {% endfor %}
    </table>
    {% endif %}

    <h2>All Traders Risk Assessment</h2>
    <table>
        <tr>
            <th>Trader</th>
            <th>Risk Level</th>
            <th>Predicted P&L</th>
            <th>Recent P&L (5d)</th>
            <th>Recent Performance (3d)</th>
            <th>Confidence</th>
        </tr>
        {% for trader in predictions %}
        <tr>
            <td>{{ trader.trader_name }}</td>
            <td>
                <span class="risk-cell risk-{{ trader.risk_level.lower() }}">
                    {{ trader.risk_level }}
                </span>
            </td>
            <td>${{ "{:,.2f}".format(trader.predicted_pnl) }}</td>
            <td>${{ "{:,.2f}".format(trader.recent_pnl) }}</td>
            <td>${{ "{:,.2f}".format(trader.recent_performance) }}</td>
            <td>{{ trader.confidence }}</td>
        </tr>
        {% endfor %}
    </table>

    <div style="margin-top: 30px; font-size: 12px; color: #7f8c8d;">
        <p><strong>Risk Level:</strong> Based on predicted P&L for tomorrow</p>
        <p><strong>Confidence:</strong> High = Personal model available, Medium = Global model only</p>
        <p><strong>Generated:</strong> {{ summary.date }} by Risk Management System</p>
    </div>
</body>
</html>
        """

    def create_email_content(self, predictions: List[Dict]) -> str:
        """Create HTML email content from predictions"""
        # Ensure all predictions have required fields with defaults
        for pred in predictions:
            pred.setdefault('trader_name', pred.get('account_id', 'Unknown'))
            pred.setdefault('risk_level', 'Unknown')
            pred.setdefault('predicted_pnl', 0.0)
            pred.setdefault('recent_pnl', 0.0)
            pred.setdefault('recent_performance', 0.0)
            pred.setdefault('confidence', 'Low')

        # Categorize traders by risk level
        high_risk = [p for p in predictions if p["risk_level"] == "High"]
        medium_risk = [p for p in predictions if p["risk_level"] == "Medium"]
        low_risk = [p for p in predictions if p["risk_level"] == "Low"]

        # Calculate summary statistics
        total_recent_pnl = sum(p.get("recent_pnl", 0) for p in predictions)

        # Render template
        content = self.template.render(
            predictions=predictions,
            high_risk_traders=high_risk,
            medium_risk_traders=medium_risk,
            low_risk_traders=low_risk,
            summary={
                "total_traders": len(predictions),
                "high_risk_count": len(high_risk),
                "medium_risk_count": len(medium_risk),
                "low_risk_count": len(low_risk),
                "total_recent_pnl": total_recent_pnl,
                "date": pd.Timestamp.now().strftime("%Y-%m-%d"),
            },
        )

        return content

    def send_email(self, predictions: List[Dict], dry_run: bool = False) -> bool:
        """Send risk report email"""
        try:
            if not self.from_email or not self.password:
                self.logger.error("Email credentials not configured")
                return False

            # Create message
            msg = MIMEMultipart("alternative")
            msg["From"] = self.from_email
            msg["To"] = ", ".join(self.config.get("to_emails", ["admin@company.com"]))
            msg["Subject"] = f"Daily Risk Report - {pd.Timestamp.now().strftime('%Y-%m-%d')}"

            # Create HTML content
            html_content = self.create_email_content(predictions)
            html_part = MIMEText(html_content, "html")
            msg.attach(html_part)

            if dry_run:
                self.logger.info("DRY RUN: Email content generated successfully")
                # Save for review
                with open('email_preview.html', 'w') as f:
                    f.write(html_content)
                self.logger.info("Email preview saved as 'email_preview.html'")
                return True

            # Send email
            with smtplib.SMTP(
                self.config.get("smtp_server", "smtp.gmail.com"),
                self.config.get("smtp_port", 587)
            ) as server:
                server.starttls()
                server.login(self.from_email, self.password)
                server.send_message(msg)

            self.logger.info("Risk report email sent successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to send email: {str(e)}")
            return False

    def test_email_config(self) -> bool:
        """Test email configuration without sending"""
        try:
            # Test template rendering
            test_predictions = [
                {
                    'account_id': 'TEST001',
                    'trader_name': 'Test Trader',
                    'risk_level': 'Medium',
                    'predicted_pnl': -100.0,
                    'recent_pnl': -250.0,
                    'recent_performance': -150.0,
                    'confidence': 'High'
                }
            ]

            html_content = self.create_email_content(test_predictions)

            if len(html_content) < 500:
                self.logger.error("Email content too short")
                return False

            # Test SMTP connection (without sending)
            if self.from_email and self.password:
                with smtplib.SMTP(
                    self.config.get("smtp_server", "smtp.gmail.com"),
                    self.config.get("smtp_port", 587)
                ) as server:
                    server.starttls()
                    server.login(self.from_email, self.password)
                    self.logger.info("SMTP connection test successful")
            else:
                self.logger.warning("Email credentials not available for SMTP test")

            self.logger.info("Email configuration test passed")
            return True

        except Exception as e:
            self.logger.error(f"Email configuration test failed: {str(e)}")
            return False
