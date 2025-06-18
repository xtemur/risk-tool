#!/usr/bin/env python
"""
Setup Script - Initialize database and download historical data
Updated for new summaryByDate format
"""

import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime
import os
import requests
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.database_manager import DatabaseManager
from src.data.data_downloader import DataDownloader
from src.data.propreports_parser import PropreReportsParser
from src.data.data_validator import validate_data_fast

# Load environment variables
load_dotenv()


def setup_logging():
    """Configure logging"""
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'logs/setup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )


def authenticate_api():
    """
    Authenticate with PropreReports API to get token

    Returns:
        str: API token for subsequent requests
    """
    logger = logging.getLogger(__name__)

    # Get credentials from environment
    api_url = os.getenv('API_URL', 'https://api.proprereports.com/api.php')
    api_user = os.getenv('API_USER')
    api_pass = os.getenv('API_PASS')

    if not api_user or not api_pass:
        raise ValueError("API_USER and API_PASS must be set in environment variables")

    # Prepare login request
    login_data = {
        'action': 'login',
        'user': api_user,
        'password': api_pass
    }

    logger.info(f"Authenticating with API as user: {api_user}")

    try:
        response = requests.post(
            api_url,
            data=login_data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30
        )

        if response.status_code == 200:
            # Parse response to get token
            response_text = response.text.strip()

            # The API might return the token in different formats
            # It could be just the token, or JSON with a token field
            if response_text.startswith('{'):
                # JSON response
                import json
                data = json.loads(response_text)
                token = data.get('token') or data.get('auth_token') or data.get('access_token')
                if not token:
                    raise ValueError(f"No token found in response: {response_text}")
            else:
                # Plain text token
                token = response_text

            logger.info("Successfully authenticated with API")
            return token

        else:
            raise Exception(f"Authentication failed with status {response.status_code}: {response.text}")

    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise


def setup_database(days_back=1000):
    """
    Setup database and download historical data

    Args:
        days_back (int): Number of days back to download data (default: 1000)

    Returns:
        dict: Download results summary
    """
    # Create necessary directories
    directories = ['data', 'logs', 'config', 'reports']
    for dir_path in directories:
        Path(dir_path).mkdir(exist_ok=True)

    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("Starting Risk Management Tool Setup")
    logger.info(f"Downloading {days_back} days of historical data")
    logger.info("=" * 80)

    try:
        # Authenticate with API first
        logger.info("Authenticating with PropreReports API...")
        token = authenticate_api()

        # Initialize components
        logger.info("Initializing database...")
        db_manager = DatabaseManager()

        logger.info("Initializing parser...")
        parser = PropreReportsParser()

        logger.info("Initializing downloader with API token...")
        downloader = DataDownloader(db_manager, parser, token=token)

        # Download historical data
        logger.info("\nStarting historical data download...")
        logger.info("This may take several minutes depending on the amount of data...")

        # Download specified days of data
        # Use replace_existing=True for recent data downloads to get updated data
        replace_existing = days_back <= 30  # Replace existing data for recent downloads
        results = downloader.download_all_data(
            days_back=days_back,
            data_types=['summary', 'fills'],
            replace_existing=replace_existing
        )

        if replace_existing:
            logger.info(f"Downloading with data replacement enabled (last {days_back} days)")
        else:
            logger.info(f"Downloading with duplicate ignoring (last {days_back} days)")

        # Show summary
        logger.info("\n" + "=" * 80)
        logger.info("DOWNLOAD SUMMARY")
        logger.info("=" * 80)

        success_count = sum(1 for r in results.values() if r.get('success', False))
        total_count = len(results)

        logger.info(f"Total traders processed: {total_count}")
        logger.info(f"Successful downloads: {success_count}")
        logger.info(f"Failed downloads: {total_count - success_count}")

        # Show individual results
        logger.info("\nDetailed Results:")
        for account_id, result in results.items():
            if result.get('success'):
                summary_records = result.get('summary', 0)  # Changed from 'totals'
                fills = result.get('fills', 0)
                logger.info(f"  Account {account_id}: ✓ Success (Summary: {summary_records}, Fills: {fills})")
            else:
                error = result.get('error', 'Unknown error')
                logger.error(f"  Account {account_id}: ✗ Failed - {error}")

        # Database statistics
        logger.info("\n" + "=" * 80)
        logger.info("DATABASE STATISTICS")
        logger.info("=" * 80)

        stats = db_manager.get_database_stats()
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")

        # Show sample data for verification
        logger.info("\n" + "=" * 80)
        logger.info("DATA VERIFICATION")
        logger.info("=" * 80)

        # Get sample account
        accounts = db_manager.get_accounts()
        if not accounts.empty:
            sample_account = accounts.iloc[0]['account_id']
            account_name = accounts.iloc[0]['account_name']

            logger.info(f"\nSample Account: {account_name} ({sample_account})")

            # Show recent daily data
            daily_data = db_manager.get_summary_data(
                account_id=sample_account
            ).tail(5)

            if not daily_data.empty:
                logger.info(f"\nRecent Daily Summary (last 5 days):")
                # Updated column names for new schema
                display_columns = ['date', 'net', 'fills', 'gross', 'end_balance']
                available_columns = [col for col in display_columns if col in daily_data.columns]
                logger.info(daily_data[available_columns].to_string())

        # Run data validation (fast validation during setup)
        logger.info("\n" + "=" * 80)
        logger.info("RUNNING FAST DATA VALIDATION")
        logger.info("=" * 80)

        try:
            validation_result = validate_data_fast()
            validation_stats = validation_result['stats']

            if validation_stats['failed_validation'] == 0:
                logger.info("✓ All data passed fast validation!")
                logger.info("   Run with --validate for full day-by-day validation")
            else:
                logger.warning(f"⚠️  {validation_stats['failed_validation']} accounts failed validation")
                logger.warning(f"   Total discrepancy days: {validation_stats['total_discrepancy_days']}")

        except Exception as e:
            logger.error(f"Data validation failed: {e}")

        # Next steps
        logger.info("\n" + "=" * 80)
        logger.info("SETUP COMPLETE!")

        if success_count < total_count:
            logger.warning("\n⚠️  Some traders failed to download. Check the logs for details.")
            logger.warning("You can retry failed downloads by running this script again.")

        return {
            'success': True,
            'total_traders': total_count,
            'successful_downloads': success_count,
            'failed_downloads': total_count - success_count,
            'results': results
        }

    except Exception as e:
        logger.error(f"Setup failed with error: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e)
        }


def main():
    """Main function with command-line argument parsing"""
    parser = argparse.ArgumentParser(description='Setup database and download historical trading data')
    parser.add_argument(
        '--days-back',
        type=int,
        default=1000,
        help='Number of days back to download data (default: 1000)'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Only run full data validation on existing data (skip download)'
    )
    parser.add_argument(
        '--validate-fast',
        action='store_true',
        help='Only run fast aggregate validation (skip download)'
    )
    parser.add_argument(
        '--account',
        type=str,
        help='Validate specific account ID only (use with --validate)'
    )

    args = parser.parse_args()

    if args.validate or args.validate_fast:
        # Only run validation
        setup_logging()
        logger = logging.getLogger(__name__)

        validation_type = "FAST" if args.validate_fast else "FULL"
        logger.info("=" * 80)
        logger.info(f"RUNNING {validation_type} DATA VALIDATION")
        logger.info("=" * 80)

        try:
            if args.validate_fast:
                validation_result = validate_data_fast()
            else:
                # Import the base validate_data function for account-specific validation
                from src.data.data_validator import validate_data
                validation_result = validate_data(account_id=args.account, fast_only=False)
            validation_stats = validation_result['stats']

            # Show comprehensive validation results
            logger.info(f"Total accounts: {validation_stats['total_accounts']}")
            logger.info(f"Accounts with data: {validation_stats['accounts_with_data']}")

            if validation_stats.get('date_range'):
                date_range = validation_stats['date_range']
                logger.info(f"Data coverage: {date_range['start']} to {date_range['end']} ({date_range['total_days']} days)")

            # Database stats
            if 'database' in validation_stats:
                db_stats = validation_stats['database']
                logger.info(f"Database stats:")
                for key, value in db_stats.items():
                    logger.info(f"  {key}: {value}")

            # Portfolio statistics
            if 'portfolio' in validation_stats:
                ps = validation_stats['portfolio']
                logger.info(f"\nPortfolio statistics:")
                logger.info(f"  Total P&L: ${ps['total_pnl']:,.2f}")
                logger.info(f"  Best account P&L: ${ps['best_account_pnl']:,.2f}")
                logger.info(f"  Worst account P&L: ${ps['worst_account_pnl']:,.2f}")
                logger.info(f"  Best daily P&L: ${ps['best_day_pnl']:,.2f}")
                logger.info(f"  Worst daily P&L: ${ps['worst_day_pnl']:,.2f}")
                logger.info(f"  Total volume: {ps['total_volume']:,.0f} shares")
                logger.info(f"  Total trades: {ps['total_trades']:,}")
                win_rate = ps['total_profitable_days'] / (ps['total_profitable_days'] + ps['total_losing_days']) * 100 if (ps['total_profitable_days'] + ps['total_losing_days']) > 0 else 0
                logger.info(f"  Win rate: {win_rate:.1f}% ({ps['total_profitable_days']} profitable vs {ps['total_losing_days']} losing days)")

            # Account-by-account summary
            if validation_stats['by_account']:
                logger.info("\nAccount summaries:")
                for acc_id, acc_stats in validation_stats['by_account'].items():
                    if acc_stats['has_data']:
                        summary_count = acc_stats['summary_records']
                        fills_count = acc_stats['fills_records']
                        logger.info(f"  {acc_id}: {summary_count} summary, {fills_count} fills")
                        if acc_stats.get('date_range'):
                            dr = acc_stats['date_range']
                            logger.info(f"    Date range: {dr['start']} to {dr['end']} ({dr['days']} days)")

                        # Display trading statistics
                        if acc_stats.get('trading_stats'):
                            ts = acc_stats['trading_stats']
                            logger.info(f"    Total P&L: ${ts['total_pnl']:,.2f}")
                            logger.info(f"    Daily P&L: Min ${ts['min_daily_pnl']:,.2f}, Max ${ts['max_daily_pnl']:,.2f}, Avg ${ts['avg_daily_pnl']:,.2f}")
                            logger.info(f"    Win/Loss: {ts['profitable_days']} profitable, {ts['losing_days']} losing days")
                            logger.info(f"    Trading: {ts['trading_days']} active days, {ts['total_trades']:,} trades, {ts['total_volume']:,.0f} shares")
                    else:
                        logger.warning(f"  {acc_id}: No data")

            # Show any errors or warnings
            if validation_result['errors']:
                logger.error("Validation errors:")
                for error in validation_result['errors']:
                    logger.error(f"  {error}")

            if validation_result['warnings']:
                logger.warning("Validation warnings:")
                for warning in validation_result['warnings']:
                    logger.warning(f"  {warning}")

            # Final result
            if validation_stats['failed_validation'] == 0:
                logger.info("\n✓ All data passed validation!")
                sys.exit(0)
            else:
                logger.error(f"\n✗ {validation_stats['failed_validation']} accounts failed validation")
                sys.exit(1)

        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            sys.exit(1)
    else:
        # Call the setup_database function with the specified days_back parameter
        result = setup_database(days_back=args.days_back)

        if not result['success']:
            sys.exit(1)


if __name__ == "__main__":
    main()
