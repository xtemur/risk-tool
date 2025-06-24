# inference/__init__.py
"""Risk signal inference and email service."""

from .signal_generator import SignalGenerator
from .email_service import EmailService

__all__ = ['SignalGenerator', 'EmailService']
