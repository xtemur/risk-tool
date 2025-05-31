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

    # Add this method to src/data_downloader.py to replace the existing _parse_totals_by_date method

    def _parse_totals_by_date(self, csv_text: str) -> pd.DataFrame:
        """Enhanced parser for totals by date format with better P&L extraction"""
        if not csv_text:
            return pd.DataFrame()

        all_data = []
        date_pattern = r'^\d{1,2}/\d{1,2}/\d{4}$'
        lines = csv_text.strip().split('\n')

        current_date = None
        i = 0

        # Debug: Print first few lines to understand format
        logger.debug(f"CSV preview (first 10 lines):")
        for idx, line in enumerate(lines[:10]):
            logger.debug(f"  {idx}: {line}")

        while i < len(lines):
            line = lines[i].strip()

            # Check if this line is a date
            if re.match(date_pattern, line):
                current_date = pd.to_datetime(line).date()
                logger.debug(f"Found date: {current_date}")
                i += 1

                # Skip to symbol data (look for header)
                while i < len(lines) and not lines[i].startswith('Symbol,'):
                    i += 1

                if i < len(lines):
                    header_line = lines[i]
                    logger.debug(f"Header: {header_line}")
                    i += 1

                    # Parse data lines until we hit summary
                    while i < len(lines):
                        line = lines[i].strip()

                        # Stop at summary sections or next date
                        if (any(line.startswith(x) for x in ['Fee:', 'Daily Total', 'Equities,', 'Fees,', 'Cash:', 'Page'])
                            or re.match(date_pattern, line)
                            or not line):
                            break

                        # Parse CSV line
                        try:
                            parts = line.split(',')
                            if len(parts) >= 16 and parts[0] not in ['Symbol', '']:

                                # Debug: Print the parsed parts
                                logger.debug(f"Parsing line: {line}")
                                logger.debug(f"Parts count: {len(parts)}")

                                symbol = parts[0]
                                orders = int(parts[1]) if parts[1] and parts[1] != '' else 0
                                fills = int(parts[2]) if parts[2] and parts[2] != '' else 0
                                qty = float(parts[3]) if parts[3] and parts[3] != '' else 0

                                # Try different column indices for P&L values
                                # The format might be different than expected
                                gross = 0
                                net = 0

                                # Common patterns for P&L columns in PropreReports
                                for idx in range(4, min(len(parts), 20)):
                                    try:
                                        val = float(parts[idx]) if parts[idx] and parts[idx] != '' else 0
                                        # Look for significant non-zero values that could be P&L
                                        if abs(val) > 0.01:  # More than 1 cent
                                            if gross == 0:
                                                gross = val
                                                logger.debug(f"Found potential gross P&L at index {idx}: {val}")
                                            elif net == 0 and abs(val - gross) > 0.01:
                                                net = val
                                                logger.debug(f"Found potential net P&L at index {idx}: {val}")
                                                break
                                    except ValueError:
                                        continue

                                # If we couldn't find distinct gross/net, try standard positions
                                if gross == 0 and net == 0:
                                    try:
                                        # Standard positions based on typical PropreReports format
                                        gross = float(parts[4]) if len(parts) > 4 and parts[4] else 0
                                        net = float(parts[15]) if len(parts) > 15 and parts[15] else 0
                                    except (ValueError, IndexError):
                                        pass

                                # Calculate fees
                                fees = gross - net if gross != 0 and net != 0 else 0

                                # Unrealized and total delta
                                unrealized_delta = 0
                                total_delta = 0
                                try:
                                    if len(parts) > 16:
                                        unrealized_delta = float(parts[16]) if parts[16] else 0
                                    if len(parts) > 17:
                                        total_delta = float(parts[17]) if parts[17] else 0
                                except (ValueError, IndexError):
                                    pass

                                # Only add if there was activity
                                if orders > 0 or fills > 0 or abs(gross) > 0.01 or abs(net) > 0.01:
                                    data_row = {
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
                                    }

                                    logger.debug(f"Adding data row: {data_row}")
                                    all_data.append(data_row)

                        except Exception as e:
                            logger.debug(f"Error parsing line: {line} - {e}")

                        i += 1
            else:
                i += 1

        # Convert to DataFrame and aggregate by date
        if all_data:
            df = pd.DataFrame(all_data)

            # Log some stats before aggregation
            logger.info(f"Before aggregation: {len(df)} rows")
            logger.info(f"Gross P&L range: {df['gross_pnl'].min():.2f} to {df['gross_pnl'].max():.2f}")
            logger.info(f"Net P&L range: {df['net_pnl'].min():.2f} to {df['net_pnl'].max():.2f}")

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

            logger.info(f"After aggregation: {len(daily_df)} rows")
            logger.info(f"Daily Net P&L range: {daily_df['net_pnl'].min():.2f} to {daily_df['net_pnl'].max():.2f}")

            return daily_df

        logger.warning("No data extracted from CSV")
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

    def download_totals(self, account_id: str, start_date, end_date) -> bool:
        """Download totals data for an account - Fixed date handling"""

        # Convert dates to proper format if they're strings
        if isinstance(start_date, str):
            start_date_str = start_date
        else:
            start_date_str = start_date.strftime('%Y-%m-%d')

        if isinstance(end_date, str):
            end_date_str = end_date
        else:
            end_date_str = end_date.strftime('%Y-%m-%d')

        logger.info(f"Downloading totals for {account_id} from {start_date_str} to {end_date_str}")

        all_data = pd.DataFrame()
        page = 1

        while True:
            params = {
                'action': 'report',
                'type': 'totalsByDate',
                'token': self.token,
                'accountId': account_id,
                'startDate': start_date_str,
                'endDate': end_date_str,
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

    def download_fills(self, account_id: str, start_date, end_date) -> bool:
        """Download fills data for an account - Fixed date handling"""

        # Convert dates to proper format if they're strings
        if isinstance(start_date, str):
            start_date_str = start_date
        else:
            start_date_str = start_date.strftime('%Y-%m-%d')

        if isinstance(end_date, str):
            end_date_str = end_date
        else:
            end_date_str = end_date.strftime('%Y-%m-%d')

        logger.info(f"Downloading fills for {account_id} from {start_date_str} to {end_date_str}")

        all_data = pd.DataFrame()
        page = 1

        while True:
            params = {
                'action': 'fills',
                'token': self.token,
                'accountId': account_id,
                'startDate': start_date_str,
                'endDate': end_date_str,
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
