"""
Email Service for Daily Risk Reports
Integrates with the existing email infrastructure (CLAUDE.md requirement)
"""

import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List
from datetime import datetime
from config import config


class EmailService:
    """
    Email service for sending daily risk reports
    Uses the existing email layout as per CLAUDE.md requirements
    """

    def __init__(self):
        # Email configuration - update these with your SMTP settings
        self.smtp_server = "smtp.gmail.com"  # Update for your email provider
        self.port = 587
        self.sender_email = "risk-system@yourfirm.com"  # Update with your email
        self.sender_password = None  # Set via environment variable

    def send_daily_risk_report(self, predictions: Dict[str, Dict],
                             system_type: str = "Rules-based") -> bool:
        """
        Send daily risk report email

        Args:
            predictions: Dict of trader predictions
            system_type: "Rules-based" or "ML model"

        Returns:
            bool: True if sent successfully, False otherwise
        """
        try:
            # Create email content
            subject, body = self._create_risk_report(predictions, system_type)

            # Create message
            message = MIMEMultipart("alternative")
            message["Subject"] = subject
            message["From"] = self.sender_email
            message["To"] = ", ".join(config.RECIPIENTS)

            # Create HTML and text versions
            text_part = MIMEText(body, "plain")
            html_part = MIMEText(self._create_html_report(predictions, system_type), "html")

            message.attach(text_part)
            message.attach(html_part)

            # Send email
            context = ssl.create_default_context()
            with smtplib.SMTP(self.smtp_server, self.port) as server:
                server.starttls(context=context)
                if self.sender_password:
                    server.login(self.sender_email, self.sender_password)

                server.sendmail(
                    self.sender_email,
                    config.RECIPIENTS,
                    message.as_string()
                )

            print(f"✅ Risk report sent to {len(config.RECIPIENTS)} recipients")
            return True

        except Exception as e:
            print(f"❌ Failed to send email: {e}")
            return False

    def _create_risk_report(self, predictions: Dict[str, Dict],
                          system_type: str) -> tuple[str, str]:
        """Create email subject and body"""
        date_str = datetime.now().strftime('%Y-%m-%d')

        # Count risk levels
        high_risk = [(t, p) for t, p in predictions.items()
                     if p.get('reduction_pct', 0) > 40]
        moderate_risk = [(t, p) for t, p in predictions.items()
                        if 20 <= p.get('reduction_pct', 0) <= 40]
        total_restricted = len([p for p in predictions.values()
                               if p.get('reduction_pct', 0) > 0])

        # Subject line
        if high_risk:
            subject = f"🚨 RISK ALERT - {len(high_risk)} High Risk Traders - {date_str}"
        elif moderate_risk:
            subject = f"⚠️  Risk Update - {len(moderate_risk)} Moderate Risk - {date_str}"
        else:
            subject = f"✅ Risk Update - All Clear - {date_str}"

        # Email body
        body = f"""DAILY RISK LIMITS - {date_str}
================================

SYSTEM: {system_type}
TOTAL TRADERS: {len(predictions)}
RESTRICTIONS: {total_restricted}

"""

        # High risk section
        if high_risk:
            body += "🚨 IMMEDIATE ACTION REQUIRED (>40% reduction):\n"
            body += "=" * 50 + "\n"

            for trader_id, pred in sorted(high_risk,
                                        key=lambda x: x[1].get('reduction_pct', 0),
                                        reverse=True):
                reduction = pred.get('reduction_pct', 0)
                reasons = ', '.join(pred.get('reasons', ['Unknown']))
                new_limit = config.DEFAULT_LIMIT * (1 - reduction/100)

                body += f"""
Trader {trader_id}: REDUCE LIMIT BY {reduction:.0f}%
  New limit: ${new_limit:,.0f} (was ${config.DEFAULT_LIMIT:,.0f})
  Reasons: {reasons}
  Confidence: {pred.get('confidence', 'High')}
---"""
        else:
            body += "✅ No high-risk traders requiring immediate action.\n"

        # Moderate risk section
        if moderate_risk:
            body += "\n\n⚠️  MODERATE ADJUSTMENTS (20-40%):\n"
            body += "=" * 40 + "\n"

            for trader_id, pred in moderate_risk:
                reduction = pred.get('reduction_pct', 0)
                reasons = ', '.join(pred.get('reasons', ['Unknown']))
                new_limit = config.DEFAULT_LIMIT * (1 - reduction/100)
                body += f"Trader {trader_id}: {reduction:.0f}% → ${new_limit:,.0f} ({reasons})\n"

        # Footer
        body += f"""

SUMMARY:
- System: {system_type}
- High Risk (>40%): {len(high_risk)}
- Moderate Risk (20-40%): {len(moderate_risk)}
- Total Restricted: {total_restricted}/{len(predictions)}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Risk Management System v2.0
"""

        return subject, body

    def _create_html_report(self, predictions: Dict[str, Dict],
                          system_type: str) -> str:
        """Create HTML version of the report"""
        date_str = datetime.now().strftime('%Y-%m-%d')

        high_risk = [(t, p) for t, p in predictions.items()
                     if p.get('reduction_pct', 0) > 40]
        moderate_risk = [(t, p) for t, p in predictions.items()
                        if 20 <= p.get('reduction_pct', 0) <= 40]

        html = f"""
<html>
<body style="font-family: Arial, sans-serif; line-height: 1.4;">
<h2 style="color: #d32f2f;">Daily Risk Limits - {date_str}</h2>

<div style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin: 10px 0;">
    <strong>System:</strong> {system_type} |
    <strong>Total Traders:</strong> {len(predictions)} |
    <strong>Restrictions:</strong> {len([p for p in predictions.values() if p.get('reduction_pct', 0) > 0])}
</div>
"""

        if high_risk:
            html += """
<h3 style="color: #d32f2f;">🚨 IMMEDIATE ACTION REQUIRED</h3>
<table border="1" style="border-collapse: collapse; width: 100%;">
<tr style="background-color: #ffebee;">
    <th>Trader</th><th>Reduction</th><th>New Limit</th><th>Reasons</th>
</tr>"""

            for trader_id, pred in high_risk:
                reduction = pred.get('reduction_pct', 0)
                reasons = ', '.join(pred.get('reasons', ['Unknown']))
                new_limit = config.DEFAULT_LIMIT * (1 - reduction/100)

                html += f"""
<tr>
    <td><strong>{trader_id}</strong></td>
    <td style="color: #d32f2f;"><strong>{reduction:.0f}%</strong></td>
    <td>${new_limit:,.0f}</td>
    <td>{reasons}</td>
</tr>"""

            html += "</table>"
        else:
            html += "<p style='color: #4caf50;'>✅ No high-risk traders today.</p>"

        if moderate_risk:
            html += """
<h3 style="color: #ff9800;">⚠️ Moderate Adjustments</h3>
<ul>"""
            for trader_id, pred in moderate_risk:
                reduction = pred.get('reduction_pct', 0)
                reasons = ', '.join(pred.get('reasons', ['Unknown']))
                new_limit = config.DEFAULT_LIMIT * (1 - reduction/100)
                html += f"<li><strong>{trader_id}:</strong> {reduction:.0f}% → ${new_limit:,.0f} ({reasons})</li>"
            html += "</ul>"

        html += f"""
<hr>
<p style="font-size: 12px; color: #666;">
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
Risk Management System v2.0 | {system_type}
</p>
</body>
</html>"""

        return html

    def send_error_alert(self, error_message: str) -> bool:
        """Send alert when the system fails"""
        try:
            subject = f"🔥 RISK SYSTEM FAILURE - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            body = f"""RISK SYSTEM FAILURE ALERT
==========================

The morning risk pipeline has failed:

Error: {error_message}

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

URGENT: Manual risk assessment required for today's trading.

Please check:
1. Database connectivity
2. System logs
3. Data quality

System Administrator should investigate immediately.
"""

            message = MIMEText(body)
            message["Subject"] = subject
            message["From"] = self.sender_email
            message["To"] = ", ".join(config.RECIPIENTS)

            context = ssl.create_default_context()
            with smtplib.SMTP(self.smtp_server, self.port) as server:
                server.starttls(context=context)
                if self.sender_password:
                    server.login(self.sender_email, self.sender_password)

                server.sendmail(
                    self.sender_email,
                    config.RECIPIENTS,
                    message.as_string()
                )

            print(f"✅ Error alert sent to {len(config.RECIPIENTS)} recipients")
            return True

        except Exception as e:
            print(f"❌ Failed to send error alert: {e}")
            return False
