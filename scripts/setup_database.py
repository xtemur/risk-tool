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
        results = downloader.download_all_data(
            days_back=days_back,
            data_types=['summary', 'fills']  # Changed from 'totals' to 'summary'
        )

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

            # Show account summary
            summary = db_manager.get_account_summary(sample_account)
            logger.info(f"\nSample Account Summary ({sample_account}):")
            for key, value in summary.items():
                if key == 'account_type':
                    logger.info(f"  {key}: {value} Account")
                else:
                    logger.info(f"  {key}: {value}")

            # Show recent daily data - UPDATED METHOD NAME
            daily_data = db_manager.get_account_daily_summary(
                account_id=sample_account
            ).tail(5)

            if not daily_data.empty:
                logger.info(f"\nRecent Daily Summary (last 5 days):")
                # Updated column names for new schema
                display_columns = ['date', 'net', 'fills', 'gross', 'end_balance']
                available_columns = [col for col in display_columns if col in daily_data.columns]
                logger.info(daily_data[available_columns].to_string())

        # Validate downloaded data
        logger.info("\n" + "=" * 80)
        logger.info("DATA VALIDATION")
        logger.info("=" * 80)

        validation_results = downloader.validate_downloads()
        for account_id, validation in validation_results.items():
            trader_name = validation['trader_name']
            account_type = validation.get('account_type', 'Unknown')
            logger.info(f"\n{trader_name} ({account_id}) - {account_type} Account:")
            logger.info(f"  Summary records: {validation['summary_records']}")
            logger.info(f"  Fills records: {validation['fills_records']}")
            if validation['date_range']:
                logger.info(f"  Date range: {validation['date_range'][0]} to {validation['date_range'][1]}")
            if validation['total_pnl'] is not None:
                logger.info(f"  Total P&L: ${validation['total_pnl']:,.2f}")

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

    args = parser.parse_args()

    # Call the setup_database function with the specified days_back parameter
    result = setup_database(days_back=args.days_back)

    if not result['success']:
        sys.exit(1)


if __name__ == "__main__":
    main()
