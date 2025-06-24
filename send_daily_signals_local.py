#!/usr/bin/env python3
"""
Send daily risk signals - Local file version.

This version saves emails as HTML files and optionally opens them in browser.
Use this when SMTP is blocked or unavailable.

Usage:
    python send_daily_signals_local.py [--open-browser] [--date YYYY-MM-DD]
"""

import argparse
import logging
import webbrowser
import os
from datetime import datetime
import sys

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
    """Main function to generate signals and save as HTML files."""
    parser = argparse.ArgumentParser(description='Generate daily risk signals as HTML files')
    parser.add_argument(
        '--open-browser',
        action='store_true',
        help='Open generated HTML file in default browser'
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

        # Initialize email service (no credentials needed for rendering)
        email_service = EmailService(require_credentials=False)

        # Render HTML
        html_content = email_service.render_daily_signals(signal_data)

        # Prepare output filename
        date_str = signal_data['date']
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"inference/outputs/risk_signals_{date_str}_{timestamp}.html"

        # Save HTML file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(html_content)

        logger.info(f"Risk signals saved to: {output_file}")

        # Open in browser if requested
        if args.open_browser:
            logger.info("Opening in default browser...")
            webbrowser.open(f"file://{os.path.abspath(output_file)}")

        # Print summary
        print("\n" + "="*60)
        print("DAILY RISK SIGNALS SUMMARY")
        print("="*60)
        print(f"Date: {signal_data['date']}")
        print(f"File: {output_file}")
        print(f"Total Traders: {len(signal_data['trader_signals'])}")

        risk_counts = {'high': 0, 'medium': 0, 'low': 0}
        for signal in signal_data['trader_signals']:
            risk_counts[signal['risk_level']] += 1

        print(f"High Risk: {risk_counts['high']}")
        print(f"Medium Risk: {risk_counts['medium']}")
        print(f"Low Risk: {risk_counts['low']}")
        print(f"Critical Alerts: {len(signal_data['alerts'])}")

        if signal_data['alerts']:
            print("\nCRITICAL ALERTS:")
            for alert in signal_data['alerts']:
                print(f"  - Trader {alert['trader_id']}: {alert['message']}")

        print("="*60)
        print("\nTo share this report:")
        print(f"1. Email the file: {output_file}")
        print("2. Upload to shared drive/cloud storage")
        print("3. Copy HTML content to email body")

    except Exception as e:
        logger.error(f"Error generating signals: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
