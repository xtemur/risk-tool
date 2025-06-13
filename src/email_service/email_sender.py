"""
Email Sender

Core email sending functionality with retry logic and error handling.
"""

import smtplib
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import List, Optional, Dict, Any, Union
import logging
from pathlib import Path

from .email_config import EmailConfig

logger = logging.getLogger(__name__)


class EmailSender:
    """
    Core email sending service with retry logic and error handling
    """

    def __init__(self, config: Optional[EmailConfig] = None):
        """
        Initialize email sender

        Args:
            config: Email configuration object
        """
        self.config = config or EmailConfig.from_env()

        if not self.config.validate():
            raise ValueError("Invalid email configuration")

    def send_email(self,
                   to_emails: Union[str, List[str]],
                   subject: str,
                   html_content: str,
                   text_content: Optional[str] = None,
                   attachments: Optional[List[str]] = None,
                   cc_emails: Optional[Union[str, List[str]]] = None,
                   bcc_emails: Optional[Union[str, List[str]]] = None) -> bool:
        """
        Send email with retry logic

        Args:
            to_emails: Recipient email address(es)
            subject: Email subject
            html_content: HTML email content
            text_content: Plain text content (optional)
            attachments: List of file paths to attach (optional)
            cc_emails: CC recipients (optional)
            bcc_emails: BCC recipients (optional)

        Returns:
            True if sent successfully, False otherwise
        """
        # Normalize email lists
        to_emails = self._normalize_email_list(to_emails)
        cc_emails = self._normalize_email_list(cc_emails or [])
        bcc_emails = self._normalize_email_list(bcc_emails or [])

        for attempt in range(self.config.max_retries):
            try:
                # Create message
                msg = self._create_message(
                    to_emails=to_emails,
                    cc_emails=cc_emails,
                    bcc_emails=bcc_emails,
                    subject=subject,
                    html_content=html_content,
                    text_content=text_content,
                    attachments=attachments
                )

                # Send email
                success = self._send_message(msg, to_emails + cc_emails + bcc_emails)

                if success:
                    logger.info(f"Email sent successfully to {', '.join(to_emails)}")
                    return True

            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")

                if attempt < self.config.max_retries - 1:
                    logger.info(f"Retrying in {self.config.retry_delay} seconds...")
                    time.sleep(self.config.retry_delay)
                else:
                    logger.error(f"Failed to send email after {self.config.max_retries} attempts")

        return False

    def _normalize_email_list(self, emails: Union[str, List[str]]) -> List[str]:
        """
        Normalize email input to list format

        Args:
            emails: Email string or list

        Returns:
            List of email addresses
        """
        if isinstance(emails, str):
            return [emails] if emails else []
        return emails or []

    def _create_message(self,
                       to_emails: List[str],
                       cc_emails: List[str],
                       bcc_emails: List[str],
                       subject: str,
                       html_content: str,
                       text_content: Optional[str] = None,
                       attachments: Optional[List[str]] = None) -> MIMEMultipart:
        """
        Create email message

        Args:
            to_emails: Recipient emails
            cc_emails: CC emails
            bcc_emails: BCC emails
            subject: Email subject
            html_content: HTML content
            text_content: Plain text content
            attachments: Attachment file paths

        Returns:
            Configured MIMEMultipart message
        """
        msg = MIMEMultipart('alternative')
        msg['From'] = self.config.email_from
        msg['To'] = ', '.join(to_emails)
        msg['Subject'] = subject

        if cc_emails:
            msg['Cc'] = ', '.join(cc_emails)

        # Add text content
        if text_content:
            text_part = MIMEText(text_content, 'plain', 'utf-8')
            msg.attach(text_part)

        # Add HTML content
        html_part = MIMEText(html_content, 'html', 'utf-8')
        msg.attach(html_part)

        # Add attachments
        if attachments:
            for attachment_path in attachments:
                self._add_attachment(msg, attachment_path)

        return msg

    def _add_attachment(self, msg: MIMEMultipart, file_path: str) -> None:
        """
        Add attachment to email message

        Args:
            msg: Email message
            file_path: Path to attachment file
        """
        try:
            path = Path(file_path)
            if not path.exists():
                logger.warning(f"Attachment file not found: {file_path}")
                return

            with open(path, 'rb') as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())

            encoders.encode_base64(part)
            part.add_header(
                'Content-Disposition',
                f'attachment; filename= {path.name}'
            )
            msg.attach(part)

            logger.debug(f"Added attachment: {path.name}")

        except Exception as e:
            logger.error(f"Failed to add attachment {file_path}: {e}")

    def _send_message(self, msg: MIMEMultipart, all_recipients: List[str]) -> bool:
        """
        Send the email message via SMTP

        Args:
            msg: Configured email message
            all_recipients: All recipient email addresses

        Returns:
            True if sent successfully, False otherwise
        """
        try:
            # Create SMTP session
            with smtplib.SMTP(self.config.smtp_server, self.config.smtp_port, timeout=self.config.timeout) as server:
                if self.config.use_tls:
                    server.starttls()

                # Login
                server.login(self.config.email_from, self.config.email_password)

                # Send email
                server.send_message(msg, to_addrs=all_recipients)

                return True

        except smtplib.SMTPAuthenticationError as e:
            logger.error(f"SMTP authentication failed: {e}")
            return False
        except smtplib.SMTPException as e:
            logger.error(f"SMTP error: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending email: {e}")
            return False

    def test_connection(self) -> bool:
        """
        Test SMTP connection and authentication

        Returns:
            True if connection successful, False otherwise
        """
        try:
            with smtplib.SMTP(self.config.smtp_server, self.config.smtp_port, timeout=self.config.timeout) as server:
                if self.config.use_tls:
                    server.starttls()

                server.login(self.config.email_from, self.config.email_password)
                logger.info("SMTP connection test successful")
                return True

        except Exception as e:
            logger.error(f"SMTP connection test failed: {e}")
            return False

    def send_test_email(self, to_email: Optional[str] = None) -> bool:
        """
        Send a test email

        Args:
            to_email: Recipient email (uses default if not provided)

        Returns:
            True if sent successfully, False otherwise
        """
        recipient = to_email or self.config.email_to

        html_content = """
        <html>
        <body>
            <h2>Email Service Test</h2>
            <p>This is a test email from the Risk Tool email service.</p>
            <p>If you received this email, the service is working correctly.</p>
            <hr>
            <p><small>Sent from Risk Tool Email Service</small></p>
        </body>
        </html>
        """

        text_content = """
        Email Service Test

        This is a test email from the Risk Tool email service.
        If you received this email, the service is working correctly.

        ---
        Sent from Risk Tool Email Service
        """

        return self.send_email(
            to_emails=recipient,
            subject="Risk Tool Email Service Test",
            html_content=html_content,
            text_content=text_content
        )
