"""
Email Service Module

This module provides email functionality for sending trading signals,
predictions, and other automated reports to stakeholders.
"""

from .email_sender import EmailSender
from .signal_email import SignalEmailService
from .email_config import EmailConfig

__all__ = [
    'EmailSender',
    'SignalEmailService',
    'EmailConfig'
]
