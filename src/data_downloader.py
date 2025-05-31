"""
Improved Data Downloader for Risk Management MVP
Downloads data directly into database without intermediate CSV files
"""

import os
import re
import time
import logging
import pandas as pd
from datetime import date, timedelta
from typing import Dict, List, Optional
from io import StringIO
import requests
import yaml
from dotenv import load_dotenv

from src.database import Database

load_dotenv()
logger = logging.getLogger(__name__)


class DataDownloader:
    """Downloads data from PropreReports API directly to database"""

    def __init__(self):
        self.token = os.getenv('API_TOKEN')
        if not self.token:
            raise ValueError("API_TOKEN not found in environment")

        self.api_url = "https://neo2.propreports.com/api.php"
        self.db = Database()
        self.traders = self._load_traders()

    def _load_traders(self) -> List[Dict]:
        """Load trader configuration"""
        with open('config/traders.yaml', 'r') as f:
            config = yaml.safe_load(f)
        return config['traders']

    def _make_request(self, params: Dict) -> Optional[str]:
        """Make API request with error handling"""
        try:
            time.sleep(0.1)  # Rate limiting
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
            logger.error(f"Request failed: {str(e)}")
            return None

    def _parse_totals_by_date(self, csv_text: str) -> pd.DataFrame:
        """Parse totals by date format with multiple days per page"""
        if not csv_text:
            return pd.DataFrame()

        all_data = []

        # Split by date blocks (dates like 9/1/2024)
        date_pattern = r'^\d{1,2}/\d{1,2}/\d{4}$'
        lines = csv_text.strip().split('\n')

        current_date = None
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            # Check if this line is a date
            if re.match(date_pattern, line):
                current_date = pd.to_datetime(line).date()
                i += 1

                # Skip to symbol data
                if i < len(lines) and lines[i].startswith('Symbol,'):
                    i += 1

                    # Read symbol data until we hit summary sections
                    while i < len(lines):
                        line = lines[i].strip()

                        # Stop at summary sections
                        if any(line.startswith(x) for x in ['Fee:', 'Daily Total', 'Equities,', 'Fees,', 'Cash:', 'Page']):
                            break

                        # Skip empty lines
                        if not line:
                            i += 1
                            continue

                        # Parse CSV line
                        try:
                            parts = line.split(',')
                            if len(parts) >= 16 and parts[0] not in ['Symbol', '']:
                                # Extract relevant columns
                                symbol = parts[0]
                                orders = int(parts[1]) if parts[1] else 0
                                fills = int(parts[2]) if parts[2] else 0
                                qty = float(parts[3]) if parts[3] else 0
                                gross = float(parts[4]) if parts[4] else 0
                                net = float(parts[15]) if parts[15] else 0
                                unrealized_delta = float(parts[16]) if len(parts) > 16 and parts[16] else 0
                                total_delta = float(parts[17]) if len(parts) > 17 and parts[17] else 0

                                # Calculate total fees
                                fees = gross - net if gross and net else 0

                                # Only add if there was activity
                                if orders > 0 or fills > 0 or gross != 0:
                                    all_data.append({
                                        'date': current_date,
                                        'symbol': symbol,
                                        'orders_count': orders,
                                        'fills_count': fills,
                                        'quantity': qty,
                                        'gross_pnl': gross,
                                        'net_pnl': net,
                                        'total_fees': fees,
                                        'unrealized_delta': unrealized_delta,
                                        'total_delta': total_delta
                                    })
                        except Exception as e:
                            logger.debug(f"Error parsing line: {line} - {e}")

                        i += 1
            else:
                i += 1

        # Convert to DataFrame and aggregate by date
        if all_data:
            df = pd.DataFrame(all_data)

            # Group by date and sum all metrics
            daily_df = df.groupby('date').agg({
                'orders_count': 'sum',
                'fills_count': 'sum',
                'quantity': 'sum',
                'gross_pnl': 'sum',
                'net_pnl': 'sum',
                'total_fees': 'sum',
                'unrealized_delta': 'sum',
                'total_delta': 'sum'
            }).reset_index()

            return daily_df

        return pd.DataFrame()

    def _parse_fills(self, csv_text: str) -> pd.DataFrame:
        """Parse fills CSV data"""
        if not csv_text:
            return pd.DataFrame()

        try:
            # Remove pagination info
            lines = csv_text.strip().split('\n')
            if lines and re.match(r'Page\s+\d+/\d+', lines[-1]):
                lines = lines[:-1]

            csv_content = '\n'.join(lines)
            df = pd.read_csv(StringIO(csv_content))

            # Log column names for debugging
            logger.debug(f"Fills columns found: {df.columns.tolist()}")

            # Rename columns to match database schema
            column_mapping = {
                'Date/Time': 'datetime',
                'Symbol': 'symbol',
                'Price': 'price',
                'Qty': 'quantity',
                'Order Id': 'order_id'
            }

            # Only rename columns that exist
            columns_to_rename = {k: v for k, v in column_mapping.items() if k in df.columns}
            df = df.rename(columns=columns_to_rename)

            # Calculate total fees
            fee_columns = ['Comm', 'Ecn Fee', 'SEC', 'ORF', 'CAT', 'TAF', 'FTT', 'NSCC', 'Acc', 'Clr', 'Misc']
            existing_fees = [col for col in fee_columns if col in df.columns]
            if existing_fees:
                df['total_fees'] = df[existing_fees].fillna(0).sum(axis=1)
            else:
                df['total_fees'] = 0

            # Keep only required columns
            required_cols = ['datetime', 'symbol', 'price', 'quantity', 'order_id', 'total_fees']
            available_cols = [col for col in required_cols if col in df.columns]

            if 'datetime' not in available_cols:
                logger.error("Missing datetime column in fills data")
                return pd.DataFrame()

            df = df[available_cols]

            # Convert data types
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
                # Remove any rows with invalid dates
                df = df.dropna(subset=['datetime'])

            if 'symbol' in df.columns:
                df['symbol'] = df['symbol'].astype(str)

            if 'order_id' in df.columns:
                df['order_id'] = df['order_id'].astype(str)

            # Convert numeric columns
            for col in ['price', 'quantity', 'total_fees']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

            logger.debug(f"Parsed {len(df)} fills")
            return df

        except Exception as e:
            logger.error(f"Error parsing fills: {str(e)}")
            logger.debug(f"CSV preview: {csv_text[:500]}")
            return pd.DataFrame()

    def download_totals(self, account_id: str, start_date: date, end_date: date) -> bool:
        """Download totals data for an account"""
        logger.info(f"Downloading totals for {account_id} from {start_date} to {end_date}")

        all_data = pd.DataFrame()
        page = 1

        while True:
            params = {
                'action': 'report',
                'type': 'totalsByDate',
                'token': self.token,
                'accountId': account_id,
                'startDate': start_date.strftime('%Y-%m-%d'),
                'endDate': end_date.strftime('%Y-%m-%d'),
                'page': str(page)
            }

            response_text = self._make_request(params)
            if not response_text:
                break

            # Parse the response
            df = self._parse_totals_by_date(response_text)
            if df.empty:
                break

            all_data = pd.concat([all_data, df], ignore_index=True)

            # Check if there are more pages
            if 'Page' not in response_text or f'Page {page}/' not in response_text:
                break

            page += 1

        if not all_data.empty:
            # Save to database
            self.db.save_daily_totals(all_data, account_id)
            logger.info(f"Saved {len(all_data)} daily totals for {account_id}")
            return True

        return False

    def download_fills(self, account_id: str, start_date: date, end_date: date) -> bool:
        """Download fills data for an account"""
        logger.info(f"Downloading fills for {account_id} from {start_date} to {end_date}")

        all_data = pd.DataFrame()
        page = 1

        while True:
            params = {
                'action': 'fills',
                'token': self.token,
                'accountId': account_id,
                'startDate': start_date.strftime('%Y-%m-%d'),
                'endDate': end_date.strftime('%Y-%m-%d'),
                'page': str(page)
            }

            response_text = self._make_request(params)
            if not response_text:
                break

            # Parse the response
            df = self._parse_fills(response_text)
            if df.empty:
                break

            all_data = pd.concat([all_data, df], ignore_index=True)

            # Check if there are more pages
            if 'Page' not in response_text or f'Page {page}/' not in response_text:
                break

            page += 1

        if not all_data.empty:
            # Save to database
            self.db.save_fills(all_data, account_id)
            logger.info(f"Saved {len(all_data)} fills for {account_id}")
            return True

        return False

    def download_all_data(self, days_back: int = 365) -> Dict[str, bool]:
        """Download all data for all traders"""
        end_date = date.today()
        start_date = end_date - timedelta(days=days_back)

        # Save traders to database
        self.db.save_traders(self.traders)

        results = {}
        total = len(self.traders)

        for i, trader in enumerate(self.traders, 1):
            if not trader.get('active', True):
                continue

            account_id = str(trader['account_id'])
            trader_name = trader['name']

            logger.info(f"[{i}/{total}] Processing {trader_name} ({account_id})")

            try:
                # Download totals
                totals_success = self.download_totals(account_id, start_date, end_date)

                # Download fills
                fills_success = self.download_fills(account_id, start_date, end_date)

                results[account_id] = totals_success and fills_success

                if results[account_id]:
                    logger.info(f"✓ Successfully downloaded data for {trader_name}")
                else:
                    logger.warning(f"✗ Failed to download some data for {trader_name}")

            except Exception as e:
                logger.error(f"✗ Error downloading data for {trader_name}: {str(e)}")
                results[account_id] = False

        # Print summary
        success_count = sum(results.values())
        logger.info(f"\nCompleted: {success_count}/{total} traders downloaded successfully")

        # Show database stats
        stats = self.db.get_database_stats()
        logger.info(f"\nDatabase Statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")

        return results

    def download_recent(self, days_back: int = 7) -> Dict[str, bool]:
        """Download recent data for daily updates"""
        return self.download_all_data(days_back=days_back)
