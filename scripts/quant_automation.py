#!/usr/bin/env python3
"""
Quantitative Risk Tool Automation Script

This script handles the daily and weekly automation tasks for the risk management system.
Designed to run via cron with proper error handling and logging.

Daily Tasks (Monday-Saturday):
- Update database (last 3 days of data)
- Generate and send risk signals

Weekly Tasks (Sunday):
- Update database (last 7 days of data)
- Retrain models with full backtesting
- Generate and send risk signals

Usage:
    python quant_automation.py --mode daily|weekly [--email]
"""

import sys
import subprocess
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
import os

def setup_logging():
    """Set up logging configuration."""
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)

    log_file = log_dir / f"quant_automation_{datetime.now().strftime('%Y%m%d')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def run_command(cmd, description, logger, timeout=1800):
    """
    Run a command with proper error handling and logging.

    Args:
        cmd (list): Command to run
        description (str): Description for logging
        logger: Logger instance
        timeout (int): Timeout in seconds (default 30 minutes)

    Returns:
        bool: True if successful, False otherwise
    """
    logger.info(f"Starting: {description}")
    logger.info(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=Path(__file__).parent.parent
        )

        if result.stdout:
            logger.info(f"Output: {result.stdout}")

        if result.stderr:
            logger.warning(f"Warnings: {result.stderr}")

        if result.returncode != 0:
            logger.error(f"Failed: {description} (return code: {result.returncode})")
            return False

        logger.info(f"✅ Completed: {description}")
        return True

    except subprocess.TimeoutExpired:
        logger.error(f"❌ Timeout: {description} exceeded {timeout} seconds")
        return False
    except Exception as e:
        logger.error(f"❌ Error: {description} failed with exception: {e}")
        return False

def daily_automation(logger, send_email=False):
    """
    Execute daily automation tasks.

    Args:
        logger: Logger instance
        send_email (bool): Whether to send email alerts

    Returns:
        bool: True if all tasks successful, False otherwise
    """
    logger.info("=== DAILY AUTOMATION STARTED ===")
    logger.info("Tasks: Database update (3 days) + Signal generation")

    success = True

    # Step 1: Update database (last 3 days)
    if not run_command(
        [sys.executable, 'scripts/update_database.py', '--days-back', '3'],
        "Daily database update (3 days)",
        logger
    ):
        logger.error("Database update failed - aborting daily automation")
        return False

    # Step 2: Generate and send signals
    signal_cmd = [sys.executable, 'send_daily_signals.py']
    if not send_email:
        signal_cmd.append('--save-only')

    if not run_command(signal_cmd, "Daily signal generation", logger):
        logger.error("Signal generation failed")
        success = False

    logger.info("=== DAILY AUTOMATION COMPLETED ===")
    return success

def weekly_automation(logger, send_email=False):
    """
    Execute weekly automation tasks.

    Args:
        logger: Logger instance
        send_email (bool): Whether to send email alerts

    Returns:
        bool: True if all tasks successful, False otherwise
    """
    logger.info("=== WEEKLY AUTOMATION STARTED ===")
    logger.info("Tasks: Database update (7 days) + Model retraining + Signal generation")

    success = True

    # Step 1: Update database (last 7 days for fresh training data)
    if not run_command(
        [sys.executable, 'scripts/update_database.py', '--days-back', '7'],
        "Weekly database update (7 days)",
        logger
    ):
        logger.error("Database update failed - aborting weekly automation")
        return False

    # Step 2: Retrain models with backtesting
    if not run_command(
        [sys.executable, 'main.py', '--mode', 'backtest'],
        "Weekly model retraining and backtesting",
        logger,
        timeout=3600  # 1 hour timeout for training
    ):
        logger.error("Model retraining failed")
        success = False

    # Step 3: Validate models
    if not run_command(
        [sys.executable, 'main.py', '--mode', 'validate'],
        "Weekly model validation",
        logger,
        timeout=600  # 10 minutes timeout
    ):
        logger.warning("Model validation failed - continuing with signal generation")

    # Step 4: Generate and send signals
    signal_cmd = [sys.executable, 'send_daily_signals.py']
    if not send_email:
        signal_cmd.append('--save-only')

    if not run_command(signal_cmd, "Weekly signal generation", logger):
        logger.error("Signal generation failed")
        success = False

    logger.info("=== WEEKLY AUTOMATION COMPLETED ===")
    return success

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Quantitative Risk Tool Automation'
    )
    parser.add_argument(
        '--mode',
        choices=['daily', 'weekly'],
        required=True,
        help='Automation mode: daily (Mon-Sat) or weekly (Sunday)'
    )
    parser.add_argument(
        '--email',
        action='store_true',
        help='Send email alerts (default: save-only)'
    )
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )

    args = parser.parse_args()

    # Set up logging
    logger = setup_logging()
    logger.setLevel(getattr(logging, args.log_level))

    # Log startup info
    logger.info(f"Quant Automation Started - Mode: {args.mode}")
    logger.info(f"Email notifications: {'Enabled' if args.email else 'Disabled'}")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")

    # Execute automation based on mode
    if args.mode == 'daily':
        success = daily_automation(logger, args.email)
    elif args.mode == 'weekly':
        success = weekly_automation(logger, args.email)
    else:
        logger.error(f"Unknown mode: {args.mode}")
        return 1

    # Log final status
    if success:
        logger.info(f"🎉 {args.mode.upper()} AUTOMATION SUCCESSFUL")
        return 0
    else:
        logger.error(f"❌ {args.mode.upper()} AUTOMATION FAILED")
        return 1

if __name__ == "__main__":
    exit(main())
