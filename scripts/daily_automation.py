#!/usr/bin/env python3
"""
Daily Risk Tool Automation Script

This script orchestrates the complete daily risk management pipeline:
1. Updates database with latest trading data
2. Generates and sends daily risk signals
3. Performs health checks and monitoring
4. Sends notifications on success/failure

Usage:
    python scripts/daily_automation.py [--dry-run] [--skip-db] [--skip-signals]

Options:
    --dry-run: Test run without actually sending emails or updating database
    --skip-db: Skip database update (useful for testing signal generation)
    --skip-signals: Skip signal generation (useful for testing database update)
    --email: Override default email recipients
    --verbose: Enable verbose logging
"""

import sys
import os
import subprocess
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import traceback
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
os.chdir(project_root)

from src.utils import load_config


class DailyAutomation:
    """Main automation orchestrator for daily risk management pipeline."""

    def __init__(self, config_path='configs/main_config.yaml', dry_run=False):
        """Initialize automation with configuration."""
        self.config = load_config(config_path)
        self.dry_run = dry_run
        self.start_time = datetime.now()
        self.results = {
            'db_update': {'success': False, 'duration': 0, 'error': None},
            'signal_generation': {'success': False, 'duration': 0, 'error': None},
            'health_checks': {'success': False, 'duration': 0, 'error': None}
        }

        # Setup logging
        self.setup_logging()

    def setup_logging(self):
        """Configure comprehensive logging."""
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)

        log_filename = f"daily_automation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_path = log_dir / log_filename

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler(sys.stdout)
            ]
        )

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Daily automation started - Log: {log_path}")

    def run_subprocess(self, command, timeout=600):
        """Run subprocess with proper error handling and logging."""
        self.logger.info(f"Executing: {' '.join(command)}")

        if self.dry_run:
            self.logger.info("DRY RUN: Command would be executed")
            return True, "Dry run - command not executed", ""

        try:
            start_time = time.time()
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=project_root
            )

            duration = time.time() - start_time

            # Log output
            if result.stdout:
                self.logger.info(f"STDOUT:\n{result.stdout}")
            if result.stderr:
                self.logger.warning(f"STDERR:\n{result.stderr}")

            success = result.returncode == 0
            if not success:
                self.logger.error(f"Command failed with return code: {result.returncode}")

            return success, result.stdout, result.stderr

        except subprocess.TimeoutExpired:
            self.logger.error(f"Command timed out after {timeout} seconds")
            return False, "", f"Command timed out after {timeout} seconds"
        except Exception as e:
            self.logger.error(f"Error executing command: {str(e)}")
            return False, "", str(e)

    def check_database_health(self):
        """Perform database health checks."""
        self.logger.info("Performing database health checks...")

        try:
            db_path = self.config['paths']['db_path']
            if not os.path.exists(db_path):
                raise Exception(f"Database file not found: {db_path}")

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Check active traders
            cursor.execute("SELECT COUNT(*) FROM accounts WHERE is_active = 1")
            active_traders = cursor.fetchone()[0]

            # Check recent trades
            cursor.execute("""
                SELECT COUNT(*) FROM trades
                WHERE trade_date >= date('now', '-7 days')
            """)
            recent_trades = cursor.fetchone()[0]

            # Check date range
            cursor.execute("SELECT MIN(trade_date), MAX(trade_date) FROM trades")
            date_range = cursor.fetchone()

            conn.close()

            health_info = {
                'active_traders': active_traders,
                'recent_trades': recent_trades,
                'date_range': date_range
            }

            self.logger.info(f"Database health: {health_info}")

            # Validate health
            if active_traders == 0:
                raise Exception("No active traders found in database")
            if recent_trades == 0:
                self.logger.warning("No trades found in last 7 days")

            return True, health_info

        except Exception as e:
            self.logger.error(f"Database health check failed: {str(e)}")
            return False, str(e)

    def update_database(self):
        """Update database with latest trading data."""
        self.logger.info("=" * 60)
        self.logger.info("STEP 1: DATABASE UPDATE")
        self.logger.info("=" * 60)

        start_time = time.time()

        try:
            # Run database update script
            python_executable = sys.executable
            command = [python_executable, "scripts/update_database.py"]

            success, stdout, stderr = self.run_subprocess(command, timeout=1200)  # 20 minutes

            duration = time.time() - start_time
            self.results['db_update']['duration'] = duration

            if success:
                self.logger.info(f"Database update completed successfully in {duration:.1f}s")
                self.results['db_update']['success'] = True
            else:
                error_msg = f"Database update failed: {stderr}"
                self.logger.error(error_msg)
                self.results['db_update']['error'] = error_msg

            return success

        except Exception as e:
            duration = time.time() - start_time
            self.results['db_update']['duration'] = duration
            error_msg = f"Database update exception: {str(e)}"
            self.logger.error(error_msg)
            self.results['db_update']['error'] = error_msg
            return False

    def generate_signals(self, email_recipients=None):
        """Generate and send daily risk signals."""
        self.logger.info("=" * 60)
        self.logger.info("STEP 2: SIGNAL GENERATION")
        self.logger.info("=" * 60)

        start_time = time.time()

        try:
            python_executable = sys.executable
            command = [python_executable, "send_daily_signals.py"]

            # Add email recipients if provided
            if email_recipients:
                for email in email_recipients:
                    command.extend(["--email", email])
            elif self.dry_run:
                command.append("--save-only")

            success, stdout, stderr = self.run_subprocess(command, timeout=600)  # 10 minutes

            duration = time.time() - start_time
            self.results['signal_generation']['duration'] = duration

            if success:
                self.logger.info(f"Signal generation completed successfully in {duration:.1f}s")
                self.results['signal_generation']['success'] = True
            else:
                error_msg = f"Signal generation failed: {stderr}"
                self.logger.error(error_msg)
                self.results['signal_generation']['error'] = error_msg

            return success

        except Exception as e:
            duration = time.time() - start_time
            self.results['signal_generation']['duration'] = duration
            error_msg = f"Signal generation exception: {str(e)}"
            self.logger.error(error_msg)
            self.results['signal_generation']['error'] = error_msg
            return False

    def run_health_checks(self):
        """Run comprehensive health checks."""
        self.logger.info("=" * 60)
        self.logger.info("STEP 3: HEALTH CHECKS")
        self.logger.info("=" * 60)

        start_time = time.time()

        try:
            db_healthy, db_info = self.check_database_health()

            duration = time.time() - start_time
            self.results['health_checks']['duration'] = duration

            if db_healthy:
                self.logger.info(f"Health checks completed successfully in {duration:.1f}s")
                self.results['health_checks']['success'] = True
            else:
                error_msg = f"Health checks failed: {db_info}"
                self.logger.error(error_msg)
                self.results['health_checks']['error'] = error_msg

            return db_healthy

        except Exception as e:
            duration = time.time() - start_time
            self.results['health_checks']['duration'] = duration
            error_msg = f"Health checks exception: {str(e)}"
            self.logger.error(error_msg)
            self.results['health_checks']['error'] = error_msg
            return False

    def send_notification(self, email_recipients, success=True):
        """Send notification email about automation results."""
        if self.dry_run:
            self.logger.info("DRY RUN: Notification email would be sent")
            return

        try:
            # Load email configuration from environment
            smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
            smtp_port = int(os.getenv('SMTP_PORT', '587'))
            email_from = os.getenv('EMAIL_FROM')
            email_password = os.getenv('EMAIL_PASSWORD')

            if not email_from or not email_password:
                self.logger.warning("Email credentials not configured - skipping notification")
                return

            # Prepare email content
            subject = f"Risk Tool Automation {'Success' if success else 'FAILURE'} - {datetime.now().strftime('%Y-%m-%d')}"

            # Create message
            msg = MimeMultipart()
            msg['From'] = email_from
            msg['To'] = ', '.join(email_recipients)
            msg['Subject'] = subject

            # Email body
            total_duration = (datetime.now() - self.start_time).total_seconds()

            body = f"""
Risk Tool Daily Automation Report
=================================

Status: {'SUCCESS' if success else 'FAILURE'}
Total Duration: {total_duration:.1f} seconds
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Step Results:
-------------
1. Database Update: {'✅ SUCCESS' if self.results['db_update']['success'] else '❌ FAILED'} ({self.results['db_update']['duration']:.1f}s)
2. Signal Generation: {'✅ SUCCESS' if self.results['signal_generation']['success'] else '❌ FAILED'} ({self.results['signal_generation']['duration']:.1f}s)
3. Health Checks: {'✅ SUCCESS' if self.results['health_checks']['success'] else '❌ FAILED'} ({self.results['health_checks']['duration']:.1f}s)

"""

            # Add error details if any
            if not success:
                body += "\nError Details:\n--------------\n"
                for step, result in self.results.items():
                    if result['error']:
                        body += f"{step.upper()}: {result['error']}\n\n"

            body += f"\nLogs are available on the server in the logs/ directory.\n"

            msg.attach(MimeText(body, 'plain'))

            # Send email
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(email_from, email_password)
            server.send_message(msg)
            server.quit()

            self.logger.info(f"Notification email sent to: {', '.join(email_recipients)}")

        except Exception as e:
            self.logger.error(f"Failed to send notification email: {str(e)}")

    def run_pipeline(self, skip_db=False, skip_signals=False, email_recipients=None):
        """Run the complete automation pipeline."""
        self.logger.info("Starting daily risk tool automation pipeline")
        self.logger.info(f"Dry run: {self.dry_run}")

        overall_success = True

        try:
            # Step 1: Update database (unless skipped)
            if not skip_db:
                db_success = self.update_database()
                if not db_success:
                    overall_success = False
                    self.logger.error("Database update failed - stopping pipeline")
                    return overall_success
            else:
                self.logger.info("Skipping database update (--skip-db flag)")
                self.results['db_update']['success'] = True  # Mark as success for reporting

            # Step 2: Generate signals (unless skipped)
            if not skip_signals:
                signal_success = self.generate_signals(email_recipients)
                if not signal_success:
                    overall_success = False
                    self.logger.error("Signal generation failed")
            else:
                self.logger.info("Skipping signal generation (--skip-signals flag)")
                self.results['signal_generation']['success'] = True  # Mark as success for reporting

            # Step 3: Health checks
            health_success = self.run_health_checks()
            if not health_success:
                overall_success = False
                self.logger.warning("Health checks failed")

        except Exception as e:
            overall_success = False
            self.logger.error(f"Pipeline exception: {str(e)}")
            self.logger.error(traceback.format_exc())

        # Final reporting
        total_duration = (datetime.now() - self.start_time).total_seconds()

        self.logger.info("=" * 60)
        self.logger.info("AUTOMATION SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Overall Status: {'SUCCESS' if overall_success else 'FAILURE'}")
        self.logger.info(f"Total Duration: {total_duration:.1f} seconds")
        self.logger.info(f"Database Update: {'✅' if self.results['db_update']['success'] else '❌'}")
        self.logger.info(f"Signal Generation: {'✅' if self.results['signal_generation']['success'] else '❌'}")
        self.logger.info(f"Health Checks: {'✅' if self.results['health_checks']['success'] else '❌'}")

        # Send notification if email recipients provided
        if email_recipients:
            self.send_notification(email_recipients, overall_success)

        return overall_success


def main():
    """Main function to run daily automation."""
    parser = argparse.ArgumentParser(description='Daily Risk Tool Automation')
    parser.add_argument('--dry-run', action='store_true', help='Test run without executing changes')
    parser.add_argument('--skip-db', action='store_true', help='Skip database update')
    parser.add_argument('--skip-signals', action='store_true', help='Skip signal generation')
    parser.add_argument('--email', action='append', help='Email for notifications (can use multiple times)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--config', default='configs/main_config.yaml', help='Configuration file path')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize automation
    automation = DailyAutomation(
        config_path=args.config,
        dry_run=args.dry_run
    )

    # Run pipeline
    success = automation.run_pipeline(
        skip_db=args.skip_db,
        skip_signals=args.skip_signals,
        email_recipients=args.email
    )

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
