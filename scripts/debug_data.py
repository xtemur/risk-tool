#!/usr/bin/env python
"""
Debug Script - Diagnose data download and parsing issues
"""

import sys
import logging
from pathlib import Path
import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.database import Database
from src.data_downloader import DataDownloader


def setup_logging():
    """Configure detailed logging for debugging"""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/debug.log')
        ]
    )


def test_single_trader(account_id: str):
    """Test downloading data for a single trader"""
    logger = logging.getLogger(__name__)

    logger.info(f"\nTesting download for account: {account_id}")

    try:
        downloader = DataDownloader()

        # Test with just 7 days of data
        from datetime import date, timedelta
        end_date = date.today()
        start_date = end_date - timedelta(days=30)

        logger.info(f"Downloading from {start_date} to {end_date}")

        # Test totals download
        logger.info("Testing totals download...")
        totals_success = downloader.download_totals(account_id, start_date, end_date)
        logger.info(f"Totals download: {'SUCCESS' if totals_success else 'FAILED'}")

        # Test fills download
        logger.info("Testing fills download...")
        fills_success = downloader.download_fills(account_id, start_date, end_date)
        logger.info(f"Fills download: {'SUCCESS' if fills_success else 'FAILED'}")

        # Check database
        db = Database()
        totals_df, fills_df = db.get_trader_data(account_id)

        logger.info(f"\nDatabase check:")
        logger.info(f"Totals records: {len(totals_df)}")
        logger.info(f"Fills records: {len(fills_df)}")

        if not totals_df.empty:
            logger.info(f"Totals date range: {totals_df['date'].min()} to {totals_df['date'].max()}")
            logger.info(f"Totals columns: {totals_df.columns.tolist()}")

        if not fills_df.empty:
            logger.info(f"Fills date range: {fills_df['datetime'].min()} to {fills_df['datetime'].max()}")
            logger.info(f"Fills columns: {fills_df.columns.tolist()}")
            logger.info(f"Sample fill:")
            logger.info(fills_df.iloc[0].to_dict())

    except Exception as e:
        logger.error(f"Error testing trader {account_id}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())


def test_database_types():
    """Test database type handling"""
    logger = logging.getLogger(__name__)

    logger.info("\nTesting database type handling...")

    # Create test data with various types
    test_fills = pd.DataFrame({
        'datetime': pd.to_datetime(['2024-01-01 10:00:00', '2024-01-01 11:00:00']),
        'symbol': ['AAPL', 'MSFT'],
        'price': [150.5, 300.25],
        'quantity': [100, 200],
        'order_id': ['123', '456'],
        'total_fees': [1.5, 2.0]
    })

    test_totals = pd.DataFrame({
        'date': pd.to_datetime(['2024-01-01', '2024-01-02']).date,
        'orders_count': [10, 15],
        'fills_count': [20, 25],
        'quantity': [1000.0, 1500.0],
        'gross_pnl': [500.0, -200.0],
        'net_pnl': [490.0, -210.0],
        'total_fees': [10.0, 10.0],
        'unrealized_delta': [0.0, 0.0],
        'total_delta': [490.0, -210.0]
    })

    try:
        db = Database()

        # Test saving fills
        logger.info("Testing fills save...")
        db.save_fills(test_fills, 'TEST001')
        logger.info("Fills save: SUCCESS")

        # Test saving totals
        logger.info("Testing totals save...")
        db.save_daily_totals(test_totals, 'TEST001')
        logger.info("Totals save: SUCCESS")

        # Test retrieval
        logger.info("Testing data retrieval...")
        totals_df, fills_df = db.get_trader_data('TEST001')
        logger.info(f"Retrieved {len(totals_df)} totals and {len(fills_df)} fills")

        # Clean up test data
        with db.get_connection() as conn:
            conn.execute("DELETE FROM fills WHERE account_id = 'TEST001'")
            conn.execute("DELETE FROM daily_totals WHERE account_id = 'TEST001'")
            conn.commit()

    except Exception as e:
        logger.error(f"Database type test failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())


def check_all_traders():
    """Check data status for all traders"""
    logger = logging.getLogger(__name__)

    logger.info("\nChecking all traders...")

    db = Database()
    traders_df = db.get_all_traders()

    logger.info(f"Found {len(traders_df)} traders")

    for _, trader in traders_df.iterrows():
        account_id = trader['account_id']
        trader_name = trader['trader_name']
        trading_days = trader['trading_days']
        total_pnl = trader['total_pnl']

        logger.info(f"\n{trader_name} ({account_id}):")
        logger.info(f"  Trading days: {trading_days}")
        logger.info(f"  Total P&L: ${total_pnl:,.2f}")
        logger.info(f"  Date range: {trader['first_trade']} to {trader['last_trade']}")


def main():
    """Main debug function"""
    setup_logging()
    logger = logging.getLogger(__name__)

    import argparse
    parser = argparse.ArgumentParser(description="Debug data issues")
    parser.add_argument('--trader', help='Test specific trader account ID')
    parser.add_argument('--test-db', action='store_true', help='Test database type handling')
    parser.add_argument('--check-all', action='store_true', help='Check all traders')

    args = parser.parse_args()

    logger.info("Starting data debug...")

    if args.trader:
        test_single_trader(args.trader)
    elif args.test_db:
        test_database_types()
    elif args.check_all:
        check_all_traders()
    else:
        # Run all tests
        test_database_types()
        check_all_traders()

        # Test first trader if any exist
        db = Database()
        traders_df = db.get_all_traders()
        if not traders_df.empty:
            first_trader = traders_df.iloc[0]['account_id']
            test_single_trader(first_trader)


if __name__ == "__main__":
    main()
