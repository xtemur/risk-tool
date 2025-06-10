"""
Data Downloader for PropreReports
Simplified API client that downloads and stores trading data
"""

import os
import time
import logging
import requests
import re
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import yaml
from dotenv import load_dotenv
import pandas as pd

from src.data.database_manager import DatabaseManager
from src.data.propreports_parser import PropreReportsParser

load_dotenv()
logger = logging.getLogger(__name__)


class DataDownloader:
    """
    Downloads data from PropreReports API and stores in database
    """

    def __init__(self,
                 db_manager: Optional[DatabaseManager] = None,
                 parser: Optional[PropreReportsParser] = None):
        """
        Initialize downloader

        Args:
            db_manager: Database manager instance
            parser: PropreReports parser instance
        """
        self.token = os.getenv('API_TOKEN')
        self.api_url = os.getenv('API_URL', 'https://api.proprereports.com/api.php')

        if not self.token:
            raise ValueError("API_TOKEN not found in environment")

        self.db = db_manager or DatabaseManager()
        self.parser = parser or PropreReportsParser()

        # Load traders config
        self.traders = self._load_traders()

        # Rate limiting
        self.request_delay = 0.5  # seconds between requests

    def _load_traders(self) -> List[Dict]:
        """Load trader configuration"""
        config_path = Path('config/traders.yaml')
        if not config_path.exists():
            raise FileNotFoundError(f"Traders config not found: {config_path}")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        return config.get('traders', [])

    def _extract_total_pages(self, csv_content: str) -> Optional[int]:
        """Extract total pages from CSV content"""
        # Look for patterns like "Page 1/3" or "Page 1 of 3"
        patterns = [
            r'Page \d+/(\d+)',
            r'Page \d+ of (\d+)',
            r'Page:\s*\d+/(\d+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, csv_content)
            if match:
                return int(match.group(1))

        # If no pagination info found, assume single page
        return 1

    def _has_more_pages(self, csv_content: str, current_page: int) -> Tuple[bool, Optional[int]]:
        """Check if there are more pages to fetch"""
        total_pages = self._extract_total_pages(csv_content)

        if total_pages is None:
            return False, None

        return current_page < total_pages, total_pages

    def download_all_data(self,
                         days_back: int = 365,
                         data_types: List[str] = ['totals', 'fills']) -> Dict[str, Dict]:
        """
        Download all data for all traders

        Args:
            days_back: Number of days to download
            data_types: List of data types to download ('totals', 'fills')

        Returns:
            Dictionary with results for each trader
        """
        end_date = date.today()
        start_date = end_date - timedelta(days=days_back)

        results = {}
        total_traders = len(self.traders)

        for i, trader in enumerate(self.traders, 1):
            account_id = str(trader['account_id'])
            trader_name = trader['name']

            logger.info(f"[{i}/{total_traders}] Processing {trader_name} ({account_id})")

            # Save account info
            self.db.save_account(account_id, trader_name)

            trader_results = {}

            try:
                # Download each data type
                if 'totals' in data_types:
                    totals_count = self._download_totals(account_id, start_date, end_date)
                    trader_results['totals'] = totals_count

                if 'fills' in data_types:
                    fills_count = self._download_fills(account_id, start_date, end_date)
                    trader_results['fills'] = fills_count

                trader_results['success'] = True
                logger.info(f"✓ Successfully downloaded data for {trader_name}")

            except Exception as e:
                logger.error(f"✗ Error downloading data for {trader_name}: {e}")
                trader_results['success'] = False
                trader_results['error'] = str(e)

            results[account_id] = trader_results

        # Summary
        success_count = sum(1 for r in results.values() if r.get('success', False))
        logger.info(f"\nCompleted: {success_count}/{total_traders} traders successful")

        return results

    def _download_totals(self, account_id: str, start_date: date, end_date: date) -> int:
        """Download and save totals by date data"""
        logger.info(f"Downloading totals for {account_id} from {start_date} to {end_date}")

        total_records = 0
        page = 1
        total_pages = None

        while True:
            # API request parameters
            params = {
                'action': 'report',
                'type': 'totalsByDate',
                'token': self.token,
                'accountId': account_id,
                'startDate': start_date.strftime('%Y-%m-%d'),
                'endDate': end_date.strftime('%Y-%m-%d'),
                'page': page
            }

            # Make request
            csv_content = self._make_request(params)
            if not csv_content:
                break

            # Check pagination info
            has_more, extracted_total = self._has_more_pages(csv_content, page)


            if total_pages is None:
                total_pages = extracted_total
                logger.info(f"Total pages to fetch: {total_pages}")

            # Save to temp file for parsing
            temp_file = Path(f"temp_totals_{account_id}_{page}.csv")
            with open(temp_file, 'w') as f:
                f.write(csv_content)

            try:
                # Parse CSV
                df, report_type = self.parser.parse_csv_file(temp_file)

                if not df.empty:
                    # Save to database
                    records = self.db.save_daily_summary(df, account_id)
                    total_records += records
                    logger.info(f"Page {page}/{total_pages or '?'}: Saved {records} records")
                else:
                    logger.warning(f"Page {page} returned empty dataframe")
                    break

            finally:
                # Clean up temp file
                if temp_file.exists():
                    temp_file.unlink()

            # Check if we should continue
            if not has_more:
                logger.info(f"Completed downloading all {page} pages")
                break

            page += 1
            time.sleep(self.request_delay)

        # Record data load
        if total_records > 0:
            self.db.record_data_load(account_id, 'totals', start_date, end_date, total_records)

        logger.info(f"Downloaded {total_records} daily summary records for {account_id}")
        return total_records

    def _download_fills(self, account_id: str, start_date: date, end_date: date) -> int:
        """Download and save fills data"""
        logger.info(f"Downloading fills for {account_id} from {start_date} to {end_date}")

        total_records = 0
        page = 1
        total_pages = None

        while True:
            # API request parameters
            params = {
                'action': 'fills',
                'token': self.token,
                'accountId': account_id,
                'startDate': start_date.strftime('%Y-%m-%d'),
                'endDate': end_date.strftime('%Y-%m-%d'),
                'page': page
            }

            # Make request
            csv_content = self._make_request(params)
            if not csv_content:
                break

            # Check pagination info
            has_more, extracted_total = self._has_more_pages(csv_content, page)


            #remove last line from csv content
            csv_content = '\n'.join(csv_content.split('\n')[:-1])


            if total_pages is None:
                total_pages = extracted_total
                logger.info(f"Total pages to fetch: {total_pages}")

            # Save to temp file for parsing
            temp_file = Path(f"temp_fills_{account_id}_{page}.csv")
            with open(temp_file, 'w') as f:
                f.write(csv_content)

            try:
                # Parse CSV
                df, report_type = self.parser.parse_csv_file(temp_file)

                if not df.empty:
                    # Save to database
                    records = self.db.save_fills(df, account_id)
                    total_records += records
                    logger.info(f"Page {page}/{total_pages or '?'}: Saved {records} records")
                else:
                    logger.warning(f"Page {page} returned empty dataframe")
                    break

            finally:
                # Clean up temp file
                if temp_file.exists():
                    temp_file.unlink()

            # Check if we should continue
            if not has_more:
                logger.info(f"Completed downloading all {page} pages")
                break

            page += 1
            time.sleep(self.request_delay)

        # Record data load
        if total_records > 0:
            self.db.record_data_load(account_id, 'fills', start_date, end_date, total_records)

        logger.info(f"Downloaded {total_records} fill records for {account_id}")
        return total_records

    def _make_request(self, params: Dict) -> Optional[str]:
        """Make API request with error handling"""
        try:
            response = requests.post(
                self.api_url,
                data=params,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=300
            )

            if response.status_code == 200:
                return response.text
            else:
                logger.error(f"API error {response.status_code}: {response.text[:200]}")
                return None

        except Exception as e:
            logger.error(f"Request failed: {e}")
            return None

    def download_recent(self, days_back: int = 7) -> Dict[str, Dict]:
        """Download recent data for daily updates"""
        return self.download_all_data(days_back=days_back)

    def get_download_status(self) -> pd.DataFrame:
        """Get status of data downloads"""
        with self.db.get_connection() as conn:
            query = """
                SELECT
                    dl.account_id,
                    a.account_name,
                    dl.data_type,
                    dl.start_date,
                    dl.end_date,
                    dl.records_loaded,
                    dl.loaded_at,
                    dl.status
                FROM data_loads dl
                JOIN accounts a ON dl.account_id = a.account_id
                ORDER BY dl.loaded_at DESC
                LIMIT 100
            """
            return pd.read_sql_query(query, conn)

    def get_data_summary(self) -> Dict[str, pd.DataFrame]:
        """Get summary of downloaded data"""
        with self.db.get_connection() as conn:
            # Summary by trader
            trader_summary = pd.read_sql_query("""
                SELECT
                    a.account_id,
                    a.account_name,
                    COUNT(DISTINCT ds.date) as trading_days,
                    MIN(ds.date) as first_date,
                    MAX(ds.date) as last_date,
                    SUM(ds.net_pnl) as total_pnl,
                    COUNT(DISTINCT f.fill_id) as total_fills
                FROM accounts a
                LEFT JOIN daily_summary ds ON a.account_id = ds.account_id
                LEFT JOIN fills f ON a.account_id = f.account_id
                GROUP BY a.account_id, a.account_name
                ORDER BY a.account_name
            """, conn)

            # Recent activity
            recent_activity = pd.read_sql_query("""
                SELECT
                    DATE(loaded_at) as load_date,
                    COUNT(DISTINCT account_id) as traders_updated,
                    SUM(records_loaded) as total_records,
                    COUNT(DISTINCT data_type) as data_types
                FROM data_loads
                WHERE loaded_at >= datetime('now', '-7 days')
                GROUP BY DATE(loaded_at)
                ORDER BY load_date DESC
            """, conn)

            return {
                'trader_summary': trader_summary,
                'recent_activity': recent_activity
            }
