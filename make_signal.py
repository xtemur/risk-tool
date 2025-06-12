#!/usr/bin/env python3
"""
Make Signal Command

Command-line interface for generating daily trading signals and sending them via email.

Usage:
    python make_signal.py [options]

Examples:
    # Send signals to default email
    python make_signal.py

    # Send to specific emails with attachments
    python make_signal.py --to "trader1@example.com,trader2@example.com" --attachments

    # Dry run (generate predictions but don't send email)
    python make_signal.py --dry-run

    # Use specific data and model files
    python make_signal.py --data data/processed/features_latest.csv --models results/models_latest

    # Test email functionality
    python make_signal.py --test

Environment Variables Required:
    EMAIL_FROM: Sender email address
    EMAIL_PASSWORD: Email password (app password for Gmail)
    EMAIL_TO: Default recipient email address

Optional Environment Variables:
    SMTP_SERVER: SMTP server (default: smtp.gmail.com)
    SMTP_PORT: SMTP port (default: 587)
"""

import argparse
import sys
import os
from pathlib import Path
import logging

# Add the src directory to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root / "src"))

from email_service.signal_command import SignalCommand


def setup_logging(verbose: bool = False):
    """
    Setup logging configuration

    Args:
        verbose: Enable verbose logging
    """
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def check_environment():
    """
    Check required environment variables

    Returns:
        True if all required variables are set, False otherwise
    """
    required_vars = ['EMAIL_FROM', 'EMAIL_PASSWORD', 'EMAIL_TO']
    missing_vars = []

    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("\nPlease set the following variables:")
        for var in missing_vars:
            print(f"  export {var}=your_value")
        print("\nOr create a .env file with these variables.")
        return False

    return True


def main():
    """
    Main command-line interface
    """
    parser = argparse.ArgumentParser(
        description="Generate and send daily trading signals via email",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Action options
    parser.add_argument(
        '--test',
        action='store_true',
        help='Send test email to verify email service is working'
    )

    parser.add_argument(
        '--test-connection',
        action='store_true',
        help='Test SMTP connection without sending email'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Generate predictions but do not send email'
    )

    # Data and model options
    parser.add_argument(
        '--data',
        type=str,
        help='Path to feature data file (uses latest if not specified)'
    )

    parser.add_argument(
        '--models',
        type=str,
        help='Path to saved model directory (trains new models if not specified)'
    )

    # Email options
    parser.add_argument(
        '--to',
        type=str,
        help='Comma-separated list of recipient emails (uses EMAIL_TO if not specified)'
    )

    parser.add_argument(
        '--no-performance',
        action='store_true',
        help='Exclude model performance metrics from email'
    )

    parser.add_argument(
        '--attachments',
        action='store_true',
        help='Include prediction files as email attachments'
    )

    # Output options
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress all output except errors'
    )

    args = parser.parse_args()

    # Setup logging
    if not args.quiet:
        setup_logging(args.verbose)
    else:
        logging.basicConfig(level=logging.ERROR)

    logger = logging.getLogger(__name__)

    # Check environment
    if not check_environment():
        sys.exit(1)

    try:
        # Initialize signal command
        signal_cmd = SignalCommand()

        # Handle test actions
        if args.test_connection:
            print("🔍 Testing SMTP connection...")
            if signal_cmd.test_email_connection():
                print("✅ SMTP connection successful!")
                sys.exit(0)
            else:
                print("❌ SMTP connection failed!")
                sys.exit(1)

        if args.test:
            print("📧 Sending test email...")
            if signal_cmd.send_test_email():
                print("✅ Test email sent successfully!")
                sys.exit(0)
            else:
                print("❌ Failed to send test email!")
                sys.exit(1)

        # Main signal generation
        print("🔮 Starting signal generation...")

        success = signal_cmd.make_signal(
            data_path=args.data,
            model_path=args.models,
            to_emails=args.to,
            include_performance=not args.no_performance,
            include_attachments=args.attachments,
            dry_run=args.dry_run
        )

        if success:
            if args.dry_run:
                print("✅ Signal generation completed (dry run)")
            else:
                print("✅ Signal email sent successfully!")
            sys.exit(0)
        else:
            print("❌ Signal generation failed!")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
