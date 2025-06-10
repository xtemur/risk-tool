"""
Data Downloader for PropreReports
Simplified API client that downloads and stores trading data
"""

import os
import time
import logging
import requests
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

                    # Check if there are more pages
                    if 'Page' not in csv_content or f'Page {page}/' not in csv_content:
                        break
                else:
                    break

            finally:
                # Clean up temp file
                if temp_file.exists():
                    temp_file.unlink()

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

                    # Check if there are more pages
                    if 'Page' not in csv_content or f'Page {page}/' not in csv_content:
                        break
                else:
                    break

            finally:
                # Clean up temp file
                if temp_file.exists():
                    temp_file.unlink()

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
