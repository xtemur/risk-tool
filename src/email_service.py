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
        self.risk_config = config["risk"]
        self.logger = logging.getLogger(__name__)

        # Email credentials from environment
        self.from_email = os.getenv("EMAIL_FROM", self.config["from_email"])
        self.password = os.getenv("EMAIL_PASSWORD", self.config["password"])

        # Load email template
        with open("config/email_template.html", "r") as f:
            self.template = Template(f.read())

    def create_email_content(self, predictions: List[Dict]) -> str:
        """Create HTML email content from predictions"""
        # Categorize traders by risk level
        high_risk = [p for p in predictions if p["risk_level"] == "High"]
        medium_risk = [p for p in predictions if p["risk_level"] == "Medium"]
        low_risk = [p for p in predictions if p["risk_level"] == "Low"]

        # Calculate summary statistics
        avg_risk = sum(p["risk_probability"] for p in predictions) / len(predictions)
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
                "average_risk": avg_risk,
                "total_recent_pnl": total_recent_pnl,
                "date": pd.Timestamp.now().strftime("%Y-%m-%d"),
            },
        )

        return content

    def send_email(self, predictions: List[Dict]) -> bool:
        """Send risk report email"""
        try:
            # Create message
            msg = MIMEMultipart("alternative")
            msg["From"] = self.from_email
            msg["To"] = ", ".join(self.config["to_emails"])
            msg[
                "Subject"
            ] = f"Daily Risk Report - {pd.Timestamp.now().strftime('%Y-%m-%d')}"

            # Create HTML content
            html_content = self.create_email_content(predictions)
            html_part = MIMEText(html_content, "html")
            msg.attach(html_part)

            # Send email
            with smtplib.SMTP(
                self.config["smtp_server"], self.config["smtp_port"]
            ) as server:
                server.starttls()
                server.login(self.from_email, self.password)
                server.send_message(msg)

            self.logger.info("Risk report email sent successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to send email: {str(e)}")
            return False
