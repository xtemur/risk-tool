#!/usr/bin/env python3
"""
Send daily risk signals email report.

Usage:
    python send_daily_signals.py [--email recipient@example.com] [--save-only]

Options:
    --email: Email address to send report to (can be used multiple times)
    --save-only: Only save HTML file, don't send email
    --date: Specific date for report (YYYY-MM-DD), defaults to today
"""

import argparse
import logging
from datetime import datetime
import sys
import os

# Add directories to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference.signal_generator import SignalGenerator
from inference.email_service import EmailService

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main function to generate and send daily signals."""
    parser = argparse.ArgumentParser(description='Send daily risk signals email')
    parser.add_argument(
        '--email',
        action='append',
        help='Email address to send report to (can be used multiple times)'
    )
    parser.add_argument(
        '--save-only',
        action='store_true',
        help='Only save HTML file without sending email'
    )
    parser.add_argument(
        '--date',
        type=str,
        default=None,
        help='Specific date for report (YYYY-MM-DD), defaults to today'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/main_config.yaml',
        help='Path to configuration file'
    )

    args = parser.parse_args()

    try:
        # Initialize components
        logger.info("Initializing signal generator...")
        generator = SignalGenerator(args.config)

        # Generate signals
        logger.info("Generating daily signals...")
        signal_data = generator.generate_daily_signals(args.date)

        # Initialize email service (don't require credentials if save-only)
        email_service = EmailService(require_credentials=not args.save_only)

        # Prepare output filename
        date_str = signal_data['date']
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"inference/outputs/risk_signals_{date_str}_{timestamp}.html"

        # Determine recipients
        recipients = args.email if args.email else email_service.default_recipients

        if args.save_only:
            # Just save the file
            logger.info("Saving signals to file only (--save-only flag)...")
            html_content = email_service.render_daily_signals(signal_data)

            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as f:
                f.write(html_content)

            logger.info(f"Risk signals saved to: {output_file}")

        elif recipients:
            # Send email
            logger.info(f"Sending signals to: {', '.join(recipients)}")

            success = email_service.send_daily_signals(
                signal_data=signal_data,
                to_emails=recipients,
                save_to_file=output_file
            )

            if success:
                logger.info("Daily signals sent successfully!")
            else:
                logger.error("Failed to send daily signals email")
                sys.exit(1)

        else:
            # No recipients and not save-only
            logger.warning("No email recipients found. Set EMAIL_RECIPIENTS in .env or use --email flag.")
            logger.info("Saving signals to file instead...")

            html_content = email_service.render_daily_signals(signal_data)
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as f:
                f.write(html_content)

            logger.info(f"Risk signals saved to: {output_file}")

        # Print summary
        print("\n" + "="*60)
        print("DAILY RISK SIGNALS SUMMARY")
        print("="*60)
        print(f"Date: {signal_data['date']}")
        print(f"Total Traders: {len(signal_data['trader_signals'])}")

        position_counts = {'reduce': 0, 'conservative': 0, 'normal': 0, 'aggressive': 0}
        for signal in signal_data['trader_signals']:
            position_level = signal.get('position_level', 'normal')
            # Map old risk levels to position levels for backward compatibility
            if position_level in ['high', 'medium', 'low', 'neutral']:
                position_map = {'high': 'reduce', 'medium': 'conservative', 'low': 'normal', 'neutral': 'normal'}
                position_level = position_map.get(position_level, 'normal')
            if position_level in position_counts:
                position_counts[position_level] += 1

        print(f"Reduce Position: {position_counts['reduce']}")
        print(f"Conservative: {position_counts['conservative']}")
        print(f"Normal: {position_counts['normal']}")
        print(f"Aggressive: {position_counts['aggressive']}")
        # Position counts already printed above
        print(f"Critical Alerts: {len(signal_data['alerts'])}")

        if signal_data['alerts']:
            print("\nCRITICAL ALERTS:")
            for alert in signal_data['alerts']:
                trader_label = alert.get('trader_label')
                if not trader_label:
                    trader_label = f"Trader {alert['trader_id']}"
                print(f"  - {trader_label}: {alert['message']}")

        print("="*60)

    except Exception as e:
        logger.error(f"Error generating/sending signals: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
