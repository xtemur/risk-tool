"""
Simplified Email Service for Risk Management MVP
Sends daily risk reports
"""

import os
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Dict
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class EmailService:
    """Simple email service for risk reports"""

    def __init__(self):
        self.from_email = os.getenv('EMAIL_FROM')
        self.password = os.getenv('EMAIL_PASSWORD')
        self.to_emails = os.getenv('EMAIL_TO', 'admin@company.com').split(',')

        if not self.from_email or not self.password:
            logger.warning("Email credentials not configured")

    def create_html_report(self, predictions: List[Dict], summary: Dict) -> str:
        """Create HTML email report"""

        # Convert to DataFrame for easier formatting
        df = pd.DataFrame(predictions)

        # Split by risk level
        high_risk = df[df['risk_level'] == 'High']
        medium_risk = df[df['risk_level'] == 'Medium']

        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #34495e; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #34495e; color: white; }}
                .high-risk {{ background-color: #e74c3c; color: white; padding: 5px 10px; border-radius: 3px; }}
                .medium-risk {{ background-color: #f39c12; color: white; padding: 5px 10px; border-radius: 3px; }}
                .low-risk {{ background-color: #27ae60; color: white; padding: 5px 10px; border-radius: 3px; }}
                .unknown-risk {{ background-color: #95a5a6; color: white; padding: 5px 10px; border-radius: 3px; }}
                .summary {{ background-color: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                .alert {{ background-color: #ffe6e6; border-left: 5px solid #e74c3c; padding: 10px; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <h1>Daily Risk Report - {pd.Timestamp.now().strftime('%Y-%m-%d')}</h1>

            <div class="summary">
                <h2>Summary</h2>
                <p><strong>Total Traders:</strong> {summary['total_traders']}</p>
                <p><strong>Risk Distribution:</strong>
                   High: {summary['high_risk_count']} |
                   Medium: {summary['medium_risk_count']} |
                   Low: {summary['low_risk_count']} |
                   Unknown: {summary.get('unknown_risk_count', 0)}</p>
                <p><strong>Total Predicted P&L:</strong> ${summary['total_predicted_pnl']:,.2f}</p>
                <p><strong>Total Recent P&L (5d):</strong> ${summary['total_recent_pnl']:,.2f}</p>
                <p><strong>Models Available:</strong> {summary['models_available']} / {summary['total_traders']}</p>
            </div>
        """

        # High risk section
        if not high_risk.empty:
            html += """
            <h2 style="color: #e74c3c;">⚠️ High Risk Traders - Immediate Attention Required</h2>
            <div class="alert">
                <strong>Action Required:</strong> The following traders are predicted to have significant losses.
                Consider reducing position sizes or additional risk controls.
            </div>
            <table>
                <tr>
                    <th>Trader</th>
                    <th>Risk Level</th>
                    <th>Predicted P&L</th>
                    <th>Recent P&L (5d)</th>
                    <th>Volatility</th>
                    <th>Recommendation</th>
                </tr>
            """

            for _, trader in high_risk.iterrows():
                volatility = trader.get('recent_volatility', 0)
                html += f"""
                <tr>
                    <td><strong>{trader['trader_name']}</strong></td>
                    <td><span class="high-risk">HIGH</span></td>
                    <td style="color: #e74c3c; font-weight: bold;">${trader['predicted_pnl']:,.2f}</td>
                    <td>${trader['recent_pnl_5d']:,.2f}</td>
                    <td>${volatility:,.2f}</td>
                    <td>{trader['recommendation']}</td>
                </tr>
                """

            html += "</table>"

        # Medium risk section
        if not medium_risk.empty:
            html += """
            <h2 style="color: #f39c12;">⚡ Medium Risk Traders - Monitor Closely</h2>
            <table>
                <tr>
                    <th>Trader</th>
                    <th>Risk Level</th>
                    <th>Predicted P&L</th>
                    <th>Recent P&L (5d)</th>
                    <th>Recommendation</th>
                </tr>
            """

            for _, trader in medium_risk.iterrows():
                html += f"""
                <tr>
                    <td>{trader['trader_name']}</td>
                    <td><span class="medium-risk">MEDIUM</span></td>
                    <td style="color: #f39c12;">${trader['predicted_pnl']:,.2f}</td>
                    <td>${trader['recent_pnl_5d']:,.2f}</td>
                    <td>{trader['recommendation']}</td>
                </tr>
                """

            html += "</table>"

        # All traders table
        html += """
        <h2>Complete Risk Assessment - All Traders</h2>
        <table>
            <tr>
                <th>Trader</th>
                <th>Risk Level</th>
                <th>Risk Score</th>
                <th>Predicted P&L</th>
                <th>Recent P&L (5d)</th>
                <th>Model Status</th>
            </tr>
        """

        for _, trader in df.iterrows():
            risk_class = trader['risk_level'].lower() + '-risk'

            # Color code predicted P&L
            pnl_color = '#e74c3c' if trader['predicted_pnl'] < -1000 else '#f39c12' if trader['predicted_pnl'] < 0 else '#27ae60'

            html += f"""
            <tr>
                <td>{trader['trader_name']}</td>
                <td><span class="{risk_class}">{trader['risk_level'].upper()}</span></td>
                <td>{trader['risk_score']:.2f}</td>
                <td style="color: {pnl_color}; font-weight: bold;">${trader['predicted_pnl']:,.2f}</td>
                <td>${trader['recent_pnl_5d']:,.2f}</td>
                <td>{trader['confidence']}</td>
            </tr>
            """

        html += """
        </table>

        <div style="margin-top: 40px; padding: 20px; background-color: #f8f9fa; border-radius: 5px;">
            <h3>Risk Level Definitions</h3>
            <ul>
                <li><strong>High Risk:</strong> Predicted loss > $1,000 - Immediate action recommended</li>
                <li><strong>Medium Risk:</strong> Predicted loss $0-$1,000 - Close monitoring required</li>
                <li><strong>Low Risk:</strong> Predicted profit - Normal trading conditions</li>
                <li><strong>Unknown:</strong> Insufficient data for prediction - Manual review needed</li>
            </ul>

            <h3>Recommendations</h3>
            <ul>
                <li><strong>For High Risk:</strong> Consider reducing position sizes by 50% or skip trading</li>
                <li><strong>For Medium Risk:</strong> Implement tighter stop losses and monitor intraday</li>
                <li><strong>For Unknown:</strong> Wait for 30+ days of data to enable model training</li>
            </ul>
        </div>

        <p style="margin-top: 30px; font-size: 12px; color: #7f8c8d; text-align: center;">
        Generated by Risk Management System on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
        This is an automated report based on machine learning predictions. Past performance does not guarantee future results.
        </p>

        </body>
        </html>
        """

        return html

    def create_text_report(self, predictions: List[Dict], summary: Dict) -> str:
        """Create plain text report as fallback"""

        text = f"""
DAILY RISK REPORT - {pd.Timestamp.now().strftime('%Y-%m-%d')}
{'='*60}

SUMMARY
-------
Total Traders: {summary['total_traders']}
High Risk: {summary['high_risk_count']}
Medium Risk: {summary['medium_risk_count']}
Low Risk: {summary['low_risk_count']}
Models Available: {summary['models_available']}

Total Predicted P&L: ${summary['total_predicted_pnl']:,.2f}
Total Recent P&L (5d): ${summary['total_recent_pnl']:,.2f}

HIGH RISK TRADERS
-----------------
"""

        for pred in predictions:
            if pred['risk_level'] == 'High':
                text += f"{pred['trader_name']}: ${pred['predicted_pnl']:,.2f} (Recent: ${pred['recent_pnl_5d']:,.2f})\n"
                text += f"  Recommendation: {pred['recommendation']}\n\n"

        text += "\nFULL RISK ASSESSMENT\n"
        text += "-" * 60 + "\n"
        text += f"{'Trader':<20} {'Risk':<10} {'Predicted P&L':>15} {'Recent P&L':>15}\n"
        text += "-" * 60 + "\n"

        for pred in predictions:
            text += f"{pred['trader_name']:<20} {pred['risk_level']:<10} "
            text += f"${pred['predicted_pnl']:>14,.2f} ${pred['recent_pnl_5d']:>14,.2f}\n"

        return text

    def send_daily_report(self, predictions: List[Dict], summary: Dict) -> bool:
        """Send daily risk report email"""

        if not self.from_email or not self.password:
            logger.error("Email credentials not configured")
            return False

        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['From'] = self.from_email
            msg['To'] = ', '.join(self.to_emails)
            msg['Subject'] = f"Daily Risk Report - {pd.Timestamp.now().strftime('%Y-%m-%d')}"

            # Add high risk count to subject if any
            high_risk_count = summary.get('high_risk_count', 0)
            if high_risk_count > 0:
                msg['Subject'] = f"🚨 ALERT: {high_risk_count} High Risk - " + msg['Subject']

            # Create both HTML and text versions
            text_content = self.create_text_report(predictions, summary)
            html_content = self.create_html_report(predictions, summary)

            # Attach parts
            text_part = MIMEText(text_content, 'plain')
            html_part = MIMEText(html_content, 'html')

            msg.attach(text_part)
            msg.attach(html_part)

            # Send email
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login(self.from_email, self.password)
                server.send_message(msg)

            logger.info(f"Risk report sent to {', '.join(self.to_emails)}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email: {str(e)}")
            return False

    def send_test_email(self) -> bool:
        """Send a test email to verify configuration"""

        test_predictions = [
            {
                'trader_name': 'Test Trader 1',
                'risk_level': 'High',
                'risk_score': 0.9,
                'predicted_pnl': -1500,
                'recent_pnl_5d': -2000,
                'recent_volatility': 500,
                'confidence': 'High',
                'recommendation': 'Reduce position sizes by 50%'
            },
            {
                'trader_name': 'Test Trader 2',
                'risk_level': 'Medium',
                'risk_score': 0.6,
                'predicted_pnl': -300,
                'recent_pnl_5d': 500,
                'recent_volatility': 200,
                'confidence': 'High',
                'recommendation': 'Monitor closely'
            },
            {
                'trader_name': 'Test Trader 3',
                'risk_level': 'Low',
                'risk_score': 0.2,
                'predicted_pnl': 800,
                'recent_pnl_5d': 1200,
                'recent_volatility': 150,
                'confidence': 'High',
                'recommendation': 'Normal trading'
            }
        ]

        test_summary = {
            'total_traders': 3,
            'high_risk_count': 1,
            'medium_risk_count': 1,
            'low_risk_count': 1,
            'unknown_risk_count': 0,
            'total_predicted_pnl': -1000,
            'total_recent_pnl': -300,
            'models_available': 3
        }

        logger.info("Sending test email...")
        success = self.send_daily_report(test_predictions, test_summary)

        if success:
            logger.info("Test email sent successfully! Check your inbox.")
        else:
            logger.error("Test email failed. Check your credentials.")

        return success
