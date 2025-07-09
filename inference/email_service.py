# inference/email_service.py

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime
from typing import List, Dict, Optional
import logging
from jinja2 import Environment, FileSystemLoader
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmailService:
    """Service for sending risk signal emails with Bloomberg-style formatting."""

    def __init__(self, require_credentials=True):
        """Initialize email service with credentials from environment."""
        # Use EMAIL_FROM for both username and from address
        self.smtp_username = os.getenv('EMAIL_FROM')
        self.smtp_password = os.getenv('EMAIL_PASSWORD')
        self.from_email = self.smtp_username

        # SMTP settings (allow override via environment)
        self.smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
        self.use_ssl = os.getenv('SMTP_USE_SSL', 'false').lower() == 'true'
        self.from_name = os.getenv('FROM_NAME', 'Risk Management System')

        # Get default recipients from env
        recipients_str = os.getenv('EMAIL_RECIPIENTS', '')
        self.default_recipients = [email.strip() for email in recipients_str.split(',') if email.strip()]

        # Validate credentials only if required
        if require_credentials and (not self.smtp_username or not self.smtp_password):
            raise ValueError("Email credentials (EMAIL_FROM, EMAIL_PASSWORD) not found in environment variables")

        # Setup Jinja2 template environment
        template_dir = os.path.join(os.path.dirname(__file__), 'templates')
        self.template_env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=True
        )

        # Add custom filters
        self.template_env.filters['format_number'] = self._format_number

    @staticmethod
    def _format_number(value):
        """Format number with thousands separator."""
        try:
            return f"{int(value):,}"
        except:
            return str(value)

    def render_daily_signals(self, signal_data: Dict) -> str:
        """
        Render the daily signals email template with data.

        Args:
            signal_data: Dictionary containing signal data

        Returns:
            Rendered HTML string
        """
        template = self.template_env.get_template('daily_signals.html')

        # Add generated timestamp
        signal_data['generated_time'] = datetime.now().strftime('%H:%M:%S UTC')

        # Calculate summary metrics if not provided
        if 'high_risk_count' not in signal_data:
            signal_data['high_risk_count'] = sum(
                1 for s in signal_data.get('trader_signals', [])
                if s.get('risk_level') == 'high'
            )

        if 'medium_risk_count' not in signal_data:
            signal_data['medium_risk_count'] = sum(
                1 for s in signal_data.get('trader_signals', [])
                if s.get('risk_level') == 'medium'
            )

        if 'low_risk_count' not in signal_data:
            signal_data['low_risk_count'] = sum(
                1 for s in signal_data.get('trader_signals', [])
                if s.get('risk_level') == 'low'
            )

        if 'neutral_risk_count' not in signal_data:
            signal_data['neutral_risk_count'] = sum(
                1 for s in signal_data.get('trader_signals', [])
                if s.get('risk_level') == 'neutral'
            )

        if 'total_traders' not in signal_data:
            signal_data['total_traders'] = len(signal_data.get('trader_signals', []))

        return template.render(**signal_data)

    def send_email(
        self,
        to_emails: List[str],
        subject: str,
        html_content: str,
        attachments: Optional[List[Dict]] = None,
        cc_emails: Optional[List[str]] = None
    ) -> bool:
        """
        Send an email with HTML content.

        Args:
            to_emails: List of recipient email addresses
            subject: Email subject
            html_content: HTML content of the email
            attachments: Optional list of attachments (dict with 'filename' and 'content')
            cc_emails: Optional list of CC email addresses

        Returns:
            True if email sent successfully, False otherwise
        """
        try:
            # Create message
            msg = MIMEMultipart('mixed')
            msg['From'] = f"{self.from_name} <{self.from_email}>"
            msg['To'] = ', '.join(to_emails)
            msg['Subject'] = subject

            if cc_emails:
                msg['Cc'] = ', '.join(cc_emails)

            # Add HTML content
            html_part = MIMEText(html_content, 'html')
            msg.attach(html_part)

            # Add attachments if any
            if attachments:
                for attachment in attachments:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(attachment['content'])
                    encoders.encode_base64(part)
                    part.add_header(
                        'Content-Disposition',
                        f'attachment; filename={attachment["filename"]}'
                    )
                    msg.attach(part)

            # Connect to server and send
            logger.debug(f"Connecting to SMTP server: {self.smtp_server}:{self.smtp_port} (SSL: {self.use_ssl})")

            # Create SMTP connection with timeout
            if self.use_ssl:
                server = smtplib.SMTP_SSL(self.smtp_server, self.smtp_port, timeout=30)
            else:
                server = smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=30)

            try:
                # Enable debug output for SMTP
                if logger.isEnabledFor(logging.DEBUG):
                    server.set_debuglevel(1)

                # EHLO
                server.ehlo()

                # Start TLS if not using SSL
                if not self.use_ssl:
                    logger.debug("Starting TLS...")
                    server.starttls()
                    server.ehlo()

                logger.debug(f"Logging in as: {self.smtp_username}")
                server.login(self.smtp_username, self.smtp_password)

                # Combine all recipients
                all_recipients = to_emails + (cc_emails or [])

                logger.debug(f"Sending message to: {all_recipients}")
                server.send_message(msg, to_addrs=all_recipients)

                logger.info(f"Email sent successfully to {', '.join(to_emails)}")
                return True

            finally:
                try:
                    server.quit()
                except:
                    pass

        except smtplib.SMTPAuthenticationError as e:
            logger.error(f"Authentication failed. Please check EMAIL_FROM and EMAIL_PASSWORD in .env")
            logger.error(f"Error: {str(e)}")
            return False
        except smtplib.SMTPConnectError as e:
            logger.error(f"Failed to connect to SMTP server {self.smtp_server}:{self.smtp_port}")
            logger.error(f"Error: {str(e)}")
            return False
        except smtplib.SMTPServerDisconnected as e:
            logger.error(f"Server unexpectedly disconnected")
            logger.error(f"Error: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Failed to send email: {type(e).__name__}: {str(e)}")
            return False

    def send_daily_signals(
        self,
        signal_data: Dict,
        to_emails: List[str],
        save_to_file: Optional[str] = None
    ) -> bool:
        """
        Send daily risk signals email.

        Args:
            signal_data: Dictionary containing signal data
            to_emails: List of recipient email addresses
            save_to_file: Optional path to save the rendered HTML

        Returns:
            True if email sent successfully, False otherwise
        """
        # Render the email
        html_content = self.render_daily_signals(signal_data)

        # Save to file if requested
        if save_to_file:
            os.makedirs(os.path.dirname(save_to_file), exist_ok=True)
            with open(save_to_file, 'w') as f:
                f.write(html_content)
            logger.info(f"Email content saved to {save_to_file}")

        # Generate subject with date
        date_str = signal_data.get('date', datetime.now().strftime('%Y-%m-%d'))
        subject = f"Risk Signals Daily - {date_str}"

        # Send the email
        return self.send_email(to_emails, subject, html_content)


def test_email_service():
    """Test the email service with sample data."""
    service = EmailService()

    # Sample signal data
    signal_data = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'trader_signals': [
            {
                'trader_id': '3942',
                'risk_level': 'high',
                'var_5pct': -5234.50,
                'loss_probability': 0.82,
                'current_pnl': -1250.00,
                'volatility': 1823.45,
                'warning_signals': ['REVENGE_TRADING', 'HIGH_VOLATILITY']
            },
            {
                'trader_id': '3943',
                'risk_level': 'medium',
                'var_5pct': -2100.00,
                'loss_probability': 0.45,
                'current_pnl': 350.00,
                'volatility': 980.20,
                'warning_signals': ['ELEVATED_RISK']
            },
            {
                'trader_id': '3946',
                'risk_level': 'low',
                'var_5pct': -800.00,
                'loss_probability': 0.15,
                'current_pnl': 1200.00,
                'volatility': 450.30,
                'warning_signals': []
            }
        ],
        'alerts': [
            {
                'trader_id': '3942',
                'message': 'Loss probability exceeds 80% threshold. Immediate attention required.'
            }
        ]
    }

    # Test rendering
    html = service.render_daily_signals(signal_data)
    print("Email rendered successfully!")

    # Save to file for preview
    output_path = 'inference/outputs/test_email.html'
    with open(output_path, 'w') as f:
        f.write(html)
    print(f"Test email saved to {output_path}")


if __name__ == '__main__':
    test_email_service()
