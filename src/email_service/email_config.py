"""
Email Configuration

Manages email service configuration including SMTP settings and credentials.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class EmailConfig:
    """
    Email service configuration
    """
    # SMTP Configuration
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    use_tls: bool = True

    # Authentication
    email_from: str = ""
    email_password: str = ""

    # Default recipients
    email_to: str = ""

    # Email settings
    timeout: int = 30
    max_retries: int = 3
    retry_delay: int = 5

    # Template settings
    template_dir: str = "src/email_service/templates"

    @classmethod
    def from_env(cls) -> 'EmailConfig':
        """
        Create configuration from environment variables

        Returns:
            EmailConfig instance
        """
        return cls(
            email_from=os.getenv('EMAIL_FROM', ''),
            email_password=os.getenv('EMAIL_PASSWORD', ''),
            email_to=os.getenv('EMAIL_TO', ''),
            smtp_server=os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
            smtp_port=int(os.getenv('SMTP_PORT', '587')),
            use_tls=os.getenv('USE_TLS', 'true').lower() == 'true',
            timeout=int(os.getenv('EMAIL_TIMEOUT', '30')),
            max_retries=int(os.getenv('EMAIL_MAX_RETRIES', '3')),
            retry_delay=int(os.getenv('EMAIL_RETRY_DELAY', '5')),
            template_dir=os.getenv('EMAIL_TEMPLATE_DIR', 'src/email_service/templates')
        )

    def validate(self) -> bool:
        """
        Validate configuration

        Returns:
            True if valid, False otherwise
        """
        if not self.email_from:
            logger.error("EMAIL_FROM is required")
            return False

        if not self.email_password:
            logger.error("EMAIL_PASSWORD is required")
            return False

        if not self.email_to:
            logger.error("EMAIL_TO is required")
            return False

        return True

    def get_smtp_config(self) -> Dict[str, Any]:
        """
        Get SMTP configuration dictionary

        Returns:
            SMTP configuration
        """
        return {
            'server': self.smtp_server,
            'port': self.smtp_port,
            'use_tls': self.use_tls,
            'timeout': self.timeout
        }

    def get_auth_config(self) -> Dict[str, str]:
        """
        Get authentication configuration

        Returns:
            Authentication configuration
        """
        return {
            'username': self.email_from,
            'password': self.email_password
        }
