import logging
import os
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv

import pandas as pd
import requests
import yaml
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import re
from io import StringIO

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class DownloadConfig:
    """Configuration for data download"""
    api_url: str = "https://neo.propreports.com/api.php"
    headers: Dict[str, str] = None
    max_retries: int = 3
    backoff_factor: float = 0.3
    timeout: int = 1000
    rate_limit_delay: float = 0.1  # Delay between requests

    def __post_init__(self):
        if self.headers is None:
            self.headers = {"Content-Type": "application/x-www-form-urlencoded"}


@dataclass
class TraderAccount:
    """Trader account information"""
    account_id: str
    name: str
    active: bool = True


class APIClient:
    """Handles API communication with retry logic and rate limiting"""

    def __init__(self, config: DownloadConfig):
        self.config = config
        self.session = self._create_session()

    def _create_session(self) -> requests.Session:
        """Create session with retry strategy"""
        session = requests.Session()

        retry_strategy = Retry(
            total=self.config.max_retries,
            backoff_factor=self.config.backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    def make_request(self, form_data: Dict[str, str]) -> Optional[str]:
        """Make API request with error handling and rate limiting"""
        try:
            # Rate limiting
            time.sleep(self.config.rate_limit_delay)

            response = self.session.post(
                self.config.api_url,
                data=form_data,
                headers=self.config.headers,
                timeout=self.config.timeout
            )

            if response.status_code == 200:
                return response.text
            else:
                logger.error(f"HTTP {response.status_code}: {response.text[:200]}")
                return None

        except requests.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            return None


class DataProcessor:
    """Handles CSV parsing and data processing"""

    @staticmethod
    def parse_csv_response(csv_text: str) -> Tuple[List[str], int, int]:
        """Parse CSV response and extract pagination info"""
        import re

        lines = csv_text.strip().splitlines()
        current_page = 1
        total_pages = 1

        # Check for pagination in last line
        if lines and re.match(r"Page\s+\d+/\d+", lines[-1].strip()):
            pagination_line = lines.pop().strip()
            match = re.search(r"Page\s+(\d+)\s*/\s*(\d+)", pagination_line)
            if match:
                current_page = int(match.group(1))
                total_pages = int(match.group(2))

        return lines, current_page, total_pages

    @staticmethod
    def combine_csv_data(header: str, data_lines: List[str]) -> pd.DataFrame:
        """Convert CSV data to DataFrame"""
        if not header or not data_lines:
            return pd.DataFrame()

        # Create CSV string
        csv_content = '\n'.join([header] + data_lines)

        try:
            # Read into DataFrame
            from io import StringIO
            df = pd.read_csv(StringIO(csv_content))

            # Convert Date column if it exists
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df = df.dropna(subset=['Date'])  # Remove invalid dates
                df = df.sort_values('Date')

            return df

        except Exception as e:
            logger.error(f"Error processing CSV data: {str(e)}")
            return pd.DataFrame()


class PropreportsDownloader:
    """Main downloader class with improved efficiency and .env token support"""

    def __init__(self, token: str = None, config_path: str = "config/config.yaml"):
        # Get token from environment if not provided
        self.token = token or os.getenv('API_TOKEN')

        if not self.token:
            raise ValueError("API_TOKEN must be provided either as parameter or in .env file")

        self.config = DownloadConfig()
        self.api_client = APIClient(self.config)
        self.data_processor = DataProcessor()

        # Load trader accounts
        self.traders = self._load_trader_accounts()

        # Setup directories
        self._setup_directories()

        logger.info(f"Initialized downloader with {len(self.traders)} accounts")

    def _load_trader_accounts(self) -> List[TraderAccount]:
        """Load trader accounts from config"""
        try:
            config_path = Path("config/trader_accounts.yaml")
            if not config_path.exists():
                logger.error("config/trader_accounts.yaml not found")
                return []

            with open(config_path, "r") as f:
                config_data = yaml.safe_load(f)

            traders = []
            for trader_data in config_data.get("traders", []):
                traders.append(TraderAccount(
                    account_id=str(trader_data["account_id"]),  # Ensure string
                    name=trader_data.get("name", str(trader_data["account_id"])),
                    active=trader_data.get("active", True)
                ))

            # Filter active traders
            active_traders = [t for t in traders if t.active]
            logger.info(f"Loaded {len(active_traders)} active traders")

            return active_traders

        except Exception as e:
            logger.error(f"Error loading trader accounts: {str(e)}")
            return []

    def _setup_directories(self):
        """Create directory structure"""
        directories = [
            "data/raw",
            "data/processed",
            "data/models",
            "data/predictions",
            "logs"
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    def _fetch_all_pages(self, base_data: Dict[str, str]) -> pd.DataFrame:
        """Fetch all pages for a request and return combined DataFrame"""
        page_num = 0
        header = None
        all_data_lines = []

        logger.debug(f"Fetching data for account {base_data.get('accountId')}")

        while True:
            page_num += 1
            request_data = base_data.copy()
            request_data["page"] = str(page_num)

            csv_text = self.api_client.make_request(request_data)
            if not csv_text:
                logger.warning(f"Failed to fetch page {page_num}")
                break

            lines, current_page, total_pages = self.data_processor.parse_csv_response(csv_text)

            if not lines:
                break

            # Store header from first page
            if header is None:
                header = lines[0]

            # Add data lines (skip header)
            data_lines = lines[1:] if len(lines) > 1 else []
            all_data_lines.extend(data_lines)

            logger.debug(f"Fetched page {current_page}/{total_pages}")

            # Break if we've reached the last page
            if current_page >= total_pages:
                break

        # Convert to DataFrame
        if header and all_data_lines:
            df = self.data_processor.combine_csv_data(header, all_data_lines)
            logger.info(f"Downloaded {len(df)} records")
            return df
        else:
            logger.warning("No data retrieved")
            return pd.DataFrame()

    def _fetch_all_tbd_pages(self, base_data: Dict[str, str]) -> pd.DataFrame:
        """Fetch all pages for totals by date and return combined DataFrame"""
        page_num = 0
        result = pd.DataFrame()

        logger.debug(f"Fetching TBD data for account {base_data.get('accountId')}")

        while True:
            page_num += 1
            request_data = base_data.copy()
            request_data["page"] = str(page_num)

            csv_text = self.api_client.make_request(request_data)
            if not csv_text:
                logger.warning(f"Failed to fetch page {page_num}")
                break

            text, current_page, total_pages = self.data_processor.parse_csv_response(csv_text)

            if not text:
                break

            text_str = "\n".join(text)

            # Split by date blocks
            day_blocks = re.split(r"(?=^\d{1,2}/\d{1,2}/\d{2,4})", text_str, flags=re.MULTILINE)

            for block in day_blocks:
                block_lines = block.strip().splitlines()
                if not block_lines:
                    continue

                # Parse date from block header
                date_str = block_lines[0].strip()
                body_text = "\n".join(block_lines[1:])
                body = re.split(r"(?=^(?:Fee|Daily|Cash))", body_text, flags=re.MULTILINE)[0]

                try:
                    df = pd.read_csv(StringIO(body))
                    if df is not None and not df.empty:
                        df['Date'] = pd.to_datetime(date_str)  # Add date column
                        result = pd.concat([result, df], ignore_index=True)
                except Exception as e:
                    logger.debug(f"Error parsing block for date {date_str}: {e}")
                    continue

            logger.debug(f"Fetched page {current_page}/{total_pages}")

            # Break if we've reached the last page
            if current_page >= total_pages:
                break

        # Return combined DataFrame
        if result is not None and not result.empty:
            logger.info(f"Downloaded {len(result)} TBD records")
            return result
        else:
            logger.warning("No TBD data retrieved")
            return pd.DataFrame()

    def download_totals_by_date(self, start_date: date, end_date: date,
                              account_ids: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """Download totals by date for entire period at once"""
        if account_ids is None:
            account_ids = [trader.account_id for trader in self.traders]

        logger.info(f"Downloading totals from {start_date} to {end_date} for {len(account_ids)} accounts")

        results = {}

        base_request = {
            "action": "report",
            "type": "totalsByDate",
            "token": self.token,
            "startDate": start_date.strftime("%Y-%m-%d"),
            "endDate": end_date.strftime("%Y-%m-%d")
        }

        for account_id in account_ids:
            logger.info(f"Processing totals for account: {account_id}")

            request_data = base_request.copy()
            request_data["accountId"] = account_id

            df = self._fetch_all_tbd_pages(request_data)

            if not df.empty:
                # Add account_id column
                df['account_id'] = account_id

                # Save to file
                output_file = f"data/raw/{account_id}_totals.csv"
                df.to_csv(output_file, index=False)
                logger.info(f"Saved {len(df)} records to {output_file}")

                results[account_id] = df
            else:
                logger.warning(f"No totals data retrieved for {account_id}")

        return results

    def download_fills(self, start_date: date, end_date: date,
                      account_ids: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """Download fills for entire period at once"""
        if account_ids is None:
            account_ids = [trader.account_id for trader in self.traders]

        logger.info(f"Downloading fills from {start_date} to {end_date} for {len(account_ids)} accounts")

        results = {}

        base_request = {
            "action": "fills",
            "token": self.token,
            "startDate": start_date.strftime("%Y-%m-%d"),
            "endDate": end_date.strftime("%Y-%m-%d")
        }

        for account_id in account_ids:
            logger.info(f"Processing fills for account: {account_id}")

            request_data = base_request.copy()
            request_data["accountId"] = account_id

            df = self._fetch_all_pages(request_data)

            if not df.empty:
                # Add account_id column
                df['account_id'] = account_id

                # Save to file
                output_file = f"data/raw/{account_id}_fills.csv"
                df.to_csv(output_file, index=False)
                logger.info(f"Saved {len(df)} records to {output_file}")

                results[account_id] = df
            else:
                logger.warning(f"No fills data retrieved for {account_id}")

        return results

    def get_accounts(self) -> pd.DataFrame:
        """Get list of all accounts"""
        request_data = {
            "action": "accounts",
            "token": self.token
        }

        df = self._fetch_all_pages(request_data)

        if not df.empty:
            logger.info(f"Retrieved {len(df)} accounts")

        return df

    def download_recent_data(self, days_back: int = 30) -> bool:
        """Download recent data for daily updates"""
        end_date = date.today()
        start_date = end_date - timedelta(days=days_back)

        logger.info(f"Downloading recent data: {start_date} to {end_date}")

        try:
            account_ids = [trader.account_id for trader in self.traders]

            # Download both totals and fills
            totals_results = self.download_totals_by_date(start_date, end_date, account_ids)
            fills_results = self.download_fills(start_date, end_date, account_ids)

            # Verify we got data for all accounts
            success = (len(totals_results) == len(account_ids) and
                      len(fills_results) == len(account_ids))

            if success:
                logger.info("Recent data download completed successfully")
            else:
                logger.warning("Some accounts may not have downloaded correctly")

            return success

        except Exception as e:
            logger.error(f"Recent data download failed: {str(e)}")
            return False

    def download_full_history(self, start_date: Optional[date] = None,
                            end_date: Optional[date] = None) -> bool:
        """Download complete historical data"""
        if start_date is None:
            start_date = date(2023, 4, 1)
        if end_date is None:
            end_date = date.today()

        logger.info(f"Downloading full history: {start_date} to {end_date}")

        try:
            account_ids = [trader.account_id for trader in self.traders]

            # Download both totals and fills
            totals_results = self.download_totals_by_date(start_date, end_date, account_ids)
            fills_results = self.download_fills(start_date, end_date, account_ids)

            # Verify data quality
            success = self.verify_data_quality()

            if success:
                logger.info("Full history download completed successfully")
            else:
                logger.warning("Data quality issues detected")

            return success

        except Exception as e:
            logger.error(f"Full history download failed: {str(e)}")
            return False

    def verify_data_quality(self) -> bool:
        """Comprehensive data quality verification"""
        logger.info("Verifying data quality...")

        issues = []

        for trader in self.traders:
            account_id = trader.account_id

            # Check file existence
            totals_file = f"data/raw/{account_id}_totals.csv"
            fills_file = f"data/raw/{account_id}_fills.csv"

            for file_path, file_type in [(totals_file, "totals"), (fills_file, "fills")]:
                if not os.path.exists(file_path):
                    issues.append(f"Missing {file_type} file for {account_id}")
                    continue

                try:
                    df = pd.read_csv(file_path)

                    if df.empty:
                        issues.append(f"Empty {file_type} data for {account_id}")
                        continue

                    # Check required columns based on your data description
                    if file_type == "totals":
                        required_cols = ["Symbol", "Orders", "Fills", "Qty", "Gross", "Net", "Date"]
                    else:  # fills
                        required_cols = ["Date/Time", "Symbol", "Qty", "Price", "Order Id"]

                    missing_cols = [col for col in required_cols if col not in df.columns]
                    if missing_cols:
                        issues.append(f"Missing columns in {account_id} {file_type}: {missing_cols}")

                    # Check data types
                    if 'Date' in df.columns:
                        try:
                            pd.to_datetime(df['Date'])
                        except:
                            issues.append(f"Invalid date format in {account_id} {file_type}")

                    logger.info(f"{account_id} {file_type}: {len(df)} records")

                except Exception as e:
                    issues.append(f"Error reading {file_type} for {account_id}: {str(e)}")

        if issues:
            logger.warning("Data quality issues found:")
            for issue in issues:
                logger.warning(f"  - {issue}")
            return False
        else:
            logger.info("Data quality verification passed!")
            return True


def download_for_risk_tool() -> bool:
    """Function for integration with risk management system"""
    try:
        # Initialize with token from environment
        downloader = PropreportsDownloader()

        # Download recent data
        success = downloader.download_recent_data(days_back=30)

        if success:
            logger.info("Risk tool data update completed successfully")

        return success

    except Exception as e:
        logger.error(f"Risk tool data update failed: {str(e)}")
        return False


def main():
    """Main execution function"""
    try:
        # Initialize downloader (token from .env)
        downloader = PropreportsDownloader()

        if not downloader.traders:
            logger.error("No traders configured. Please set up config/trader_accounts.yaml")
            return

        # Download full history
        start_date = date(2023, 4, 1)
        end_date = date.today()

        success = downloader.download_full_history(start_date, end_date)

        if success:
            logger.info("Download completed successfully!")
            logger.info("You can now proceed with: python main.py --train")
        else:
            logger.error("Download completed with issues. Check logs above.")

    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")


if __name__ == "__main__":
    main()
