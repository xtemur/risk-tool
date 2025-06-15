"""
Data Downloader for PropreReports
Simplified API client that downloads and stores trading data
Updated to use summaryByDate instead of totalsByDate
Modified to save CSV files for backup/debugging
"""

import os
import time
import logging
import requests
import re
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import yaml
from dotenv import load_dotenv
import pandas as pd
import shutil

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
                 parser: Optional[PropreReportsParser] = None,
                 token: Optional[str] = None,
                 backup_dir: str = "data/csv_backups"):
        """
        Initialize downloader

        Args:
            db_manager: Database manager instance
            parser: PropreReports parser instance
            token: API token for authentication (if not provided, will try to get from env)
            backup_dir: Directory to save CSV backups
        """
        # Use provided token or fall back to environment variable
        self.token = token or os.getenv('API_TOKEN')
        self.api_url = os.getenv('API_URL', 'https://api.proprereports.com/api.php')

        if not self.token:
            raise ValueError("API token not provided and API_TOKEN not found in environment")

        self.db = db_manager or DatabaseManager()
        self.parser = parser or PropreReportsParser()

        # Load traders config
        self.traders = self._load_traders()

        # Create trader lookup dict for easy name access
        self.trader_lookup = {str(t['account_id']): t['name'] for t in self.traders}

        # Rate limiting
        self.request_delay = 0.5  # seconds between requests

        # Backup directory
        self.backup_dir = Path(backup_dir)
        self._ensure_backup_directories()

    def _ensure_backup_directories(self):
        """Create backup directory structure if it doesn't exist"""
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        (self.backup_dir / 'summary').mkdir(exist_ok=True)
        (self.backup_dir / 'fills').mkdir(exist_ok=True)
        logger.info(f"Backup directory created/verified at: {self.backup_dir}")

    def _load_traders(self) -> List[Dict]:
        """Load trader configuration"""
        config_path = Path('config/traders.yaml')
        if not config_path.exists():
            raise FileNotFoundError(f"Traders config not found: {config_path}")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        return config.get('traders', [])

    def _get_backup_filename(self, account_id: str, data_type: str,
                           start_date: date, end_date: date,
                           page: Optional[int] = None) -> Path:
        """
        Generate backup filename with account name and dates

        Args:
            account_id: Account ID
            data_type: 'summary' or 'fills'
            start_date: Start date of data
            end_date: End date of data
            page: Page number for paginated data

        Returns:
            Path to backup file
        """
        # Get trader name, sanitize for filename
        trader_name = self.trader_lookup.get(account_id, f"unknown_{account_id}")
        trader_name = re.sub(r'[^\w\s-]', '', trader_name).strip().replace(' ', '_')

        # Format dates
        date_range = f"{start_date.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}"

        # Create filename
        if page is not None:
            filename = f"{trader_name}_{account_id}_{date_range}_page{page}.csv"
        else:
            filename = f"{trader_name}_{account_id}_{date_range}.csv"

        # Return full path
        return self.backup_dir / data_type / filename

    def _save_backup(self, temp_file: Path, backup_path: Path):
        """Save temporary file as backup"""
        try:
            shutil.copy2(temp_file, backup_path)
            logger.info(f"CSV backup saved: {backup_path.name}")
        except Exception as e:
            logger.error(f"Failed to save backup: {e}")

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
                         days_back: int = 1000,
                         data_types: List[str] = ['summary', 'fills']) -> Dict[str, Dict]:
        """
        Download all data for all traders

        Args:
            days_back: Number of days to download
            data_types: List of data types to download ('summary', 'fills')

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
                if 'summary' in data_types:
                    summary_count = self._download_summary(account_id, start_date, end_date)
                    trader_results['summary'] = summary_count

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

    def _download_summary(self, account_id: str, start_date: date, end_date: date) -> int:
        """Download and save account daily summary data (summaryByDate)"""
        logger.info(f"Downloading summary for {account_id} from {start_date} to {end_date}")

        # API request parameters - summaryByDate has no pagination
        params = {
            'action': 'report',
            'type': 'summaryByDate',  # Changed from totalsByDate
            'token': self.token,
            'accountId': account_id,
            'startDate': start_date.strftime('%Y-%m-%d'),
            'endDate': end_date.strftime('%Y-%m-%d')
        }

        # Make request
        csv_content = self._make_request(params)
        if not csv_content:
            logger.warning(f"No data returned for {account_id}")
            return 0

        # Save to temp file for parsing
        temp_file = Path(f"temp_summary_{account_id}.csv")
        with open(temp_file, 'w') as f:
            f.write(csv_content)

        total_records = 0
        try:
            # Parse CSV
            df, report_type = self.parser.parse_csv_file(temp_file)


            # Save backup copy
            backup_path = self._get_backup_filename(account_id, 'summary', start_date, end_date)
            self._save_backup(temp_file, backup_path)

            if not df.empty:
                # Save to database
                records = self.db.save_account_daily_summary(df, account_id)
                total_records = records
                logger.info(f"Saved {records} summary records")
            else:
                logger.warning(f"Empty dataframe returned for {account_id}")


        finally:
            # Clean up temp file
            if temp_file.exists():
                temp_file.unlink()

        # Record data load
        if total_records > 0:
            self.db.record_data_load(account_id, 'summary', start_date, end_date, total_records)

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

            # Remove last line from csv content if it contains page info
            lines = csv_content.split('\n')
            if lines and 'Page' in lines[-1]:
                csv_content = '\n'.join(lines[:-1])

            if total_pages is None:
                total_pages = extracted_total
                if total_pages and total_pages > 1:
                    logger.info(f"Total pages to fetch: {total_pages}")

            # Save to temp file for parsing
            temp_file = Path(f"temp_fills_{account_id}_{page}.csv")
            with open(temp_file, 'w') as f:
                f.write(csv_content)

            try:
                # Parse CSV
                df, report_type = self.parser.parse_csv_file(temp_file)

                # Save backup copy
                backup_path = self._get_backup_filename(account_id, 'fills', start_date, end_date, page)
                self._save_backup(temp_file, backup_path)

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
        return self.db.get_download_status()

    def get_data_summary(self) -> Dict[str, pd.DataFrame]:
        """Get summary of downloaded data"""
        return self.db.get_data_summary()

    def validate_downloads(self) -> Dict[str, Any]:
        """Validate downloaded data for each trader"""
        validation_results = {}

        for trader in self.traders:
            account_id = str(trader['account_id'])
            trader_name = trader['name']

            # Get summary data
            summary_df = self.db.get_account_daily_summary(account_id=account_id)

            validation = {
                'trader_name': trader_name,
                'has_summary_data': not summary_df.empty,
                'summary_records': len(summary_df) if not summary_df.empty else 0,
                'date_range': None,
                'total_pnl': None,
                'account_type': 'Unknown'
            }

            if not summary_df.empty:
                validation['date_range'] = (
                    summary_df['date'].min().strftime('%Y-%m-%d'),
                    summary_df['date'].max().strftime('%Y-%m-%d')
                )
                validation['total_pnl'] = summary_df['net'].sum()

                # Use the improved account type detection
                validation['account_type'] = self.db.detect_account_type(account_id)

            # Get fills data
            fills_df = self.db.get_fills(account_id=account_id)
            validation['has_fills_data'] = not fills_df.empty
            validation['fills_records'] = len(fills_df) if not fills_df.empty else 0

            validation_results[account_id] = validation

        return validation_results

    def get_backup_info(self) -> Dict[str, Any]:
        """Get information about backed up CSV files"""
        backup_info = {
            'backup_directory': str(self.backup_dir),
            'summary_files': [],
            'fills_files': [],
            'total_size_mb': 0
        }

        # Get summary files
        summary_dir = self.backup_dir / 'summary'
        if summary_dir.exists():
            backup_info['summary_files'] = [f.name for f in summary_dir.glob('*.csv')]

        # Get fills files
        fills_dir = self.backup_dir / 'fills'
        if fills_dir.exists():
            backup_info['fills_files'] = [f.name for f in fills_dir.glob('*.csv')]

        # Calculate total size
        total_size = 0
        for csv_file in self.backup_dir.rglob('*.csv'):
            total_size += csv_file.stat().st_size
        backup_info['total_size_mb'] = round(total_size / (1024 * 1024), 2)

        return backup_info
