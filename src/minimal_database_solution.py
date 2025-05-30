"""
Minimal Database Solution - Download directly to database, no CSV files
"""

import logging
import os
import sqlite3
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import yaml
from dotenv import load_dotenv
from contextlib import contextmanager

# Import the original downloader for API functionality
from propreports_downloader import PropreportsDownloader

load_dotenv()
logger = logging.getLogger(__name__)


class MinimalDatabaseManager:
    """Minimal database manager - just what we need"""

    def __init__(self, db_path: str = "data/trading_data.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self._init_database()

    @contextmanager
    def get_connection(self):
        """Database connection context manager"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_database(self):
        """Create minimal database schema"""
        with self.get_connection() as conn:
            conn.executescript("""
                -- Traders table
                CREATE TABLE IF NOT EXISTS traders (
                    account_id TEXT PRIMARY KEY,
                    trader_name TEXT NOT NULL UNIQUE,
                    active BOOLEAN DEFAULT 1
                );

                -- Totals table (daily aggregates)
                CREATE TABLE IF NOT EXISTS totals (
                    date DATE NOT NULL,
                    account_id TEXT NOT NULL,
                    symbol TEXT DEFAULT 'TOTAL',
                    orders_count INTEGER DEFAULT 0,
                    fills_count INTEGER DEFAULT 0,
                    qty REAL DEFAULT 0,
                    gross_pnl REAL DEFAULT 0,
                    net_pnl REAL DEFAULT 0,
                    total_fees REAL DEFAULT 0,
                    unrealized_delta REAL DEFAULT 0,
                    total_delta REAL DEFAULT 0,
                    PRIMARY KEY (date, account_id, symbol),
                    FOREIGN KEY (account_id) REFERENCES traders(account_id)
                );

                -- Fills table (transactions)
                CREATE TABLE IF NOT EXISTS fills (
                    datetime TIMESTAMP NOT NULL,
                    account_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    price REAL NOT NULL,
                    qty REAL NOT NULL,
                    order_id TEXT,
                    total_fees REAL DEFAULT 0,
                    FOREIGN KEY (account_id) REFERENCES traders(account_id)
                );

                -- Indexes for performance
                CREATE INDEX IF NOT EXISTS idx_totals_date ON totals(date);
                CREATE INDEX IF NOT EXISTS idx_totals_account ON totals(account_id);
                CREATE INDEX IF NOT EXISTS idx_fills_datetime ON fills(datetime);
                CREATE INDEX IF NOT EXISTS idx_fills_account ON fills(account_id);
            """)
            conn.commit()

    def save_totals(self, df: pd.DataFrame, account_id: str):
        """Save totals data to database"""
        if df.empty:
            return

        # Create a copy to avoid modifying original
        df = df.copy()

        # Add account_id
        df['account_id'] = account_id

        # Rename columns to match database schema
        column_mapping = {
            'Orders': 'orders_count',
            'Fills': 'fills_count',
            'Qty': 'qty',
            'Gross': 'gross_pnl',
            'Net': 'net_pnl',
            'Unrealized δ': 'unrealized_delta',
            'Total δ': 'total_delta',
            'Date': 'date'
        }

        # Rename columns that exist
        df = df.rename(columns=column_mapping)

        # Calculate total_fees if we have individual fee columns
        fee_columns = ['Comm', 'Ecn Fee', 'SEC', 'ORF', 'CAT', 'TAF', 'FTT', 'NSCC', 'Acc', 'Clr', 'Misc']
        existing_fee_cols = [col for col in fee_columns if col in df.columns]
        if existing_fee_cols:
            df['total_fees'] = df[existing_fee_cols].fillna(0).sum(axis=1)
            # Drop individual fee columns
            df = df.drop(columns=existing_fee_cols)
        elif 'total_fees' not in df.columns:
            df['total_fees'] = 0

        # Ensure required columns exist with defaults
        required_cols = {
            'orders_count': 0,
            'fills_count': 0,
            'qty': 0.0,
            'gross_pnl': 0.0,
            'net_pnl': 0.0,
            'total_fees': 0.0,
            'unrealized_delta': 0.0,
            'total_delta': 0.0
        }

        for col, default in required_cols.items():
            if col not in df.columns:
                df[col] = default

        # Ensure numeric types
        int_cols = ['orders_count', 'fills_count']
        for col in int_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

        float_cols = ['qty', 'gross_pnl', 'net_pnl', 'total_fees', 'unrealized_delta', 'total_delta']
        for col in float_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

        # Ensure date column
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']).dt.date
        else:
            df['date'] = pd.to_datetime(df['Date']).dt.date

        # Add symbol column if missing (totals by date are aggregated across all symbols)
        if 'symbol' not in df.columns and 'Symbol' not in df.columns:
            df['symbol'] = 'TOTAL'
        elif 'Symbol' in df.columns and 'symbol' not in df.columns:
            df['symbol'] = df['Symbol'].fillna('TOTAL')
        else:
            df['symbol'] = df['symbol'].fillna('TOTAL')

        # Select only columns that exist in database
        db_columns = ['date', 'account_id', 'symbol', 'orders_count', 'fills_count', 'qty',
                     'gross_pnl', 'net_pnl', 'total_fees', 'unrealized_delta', 'total_delta']

        # Keep only columns that exist
        available_columns = [col for col in db_columns if col in df.columns]
        df = df[available_columns]

        with self.get_connection() as conn:
            # Save to database (replace existing)
            df.to_sql('totals', conn, if_exists='append', index=False)

    def save_fills(self, df: pd.DataFrame, account_id: str):
        """Save fills data to database"""
        if df.empty:
            return

        # Create a copy to avoid modifying original
        df = df.copy()

        # Add account_id
        df['account_id'] = account_id

        # Rename columns to match database schema
        column_mapping = {
            'Date/Time': 'datetime',
            'Symbol': 'symbol',
            'Price': 'price',
            'Qty': 'qty',
            'Order Id': 'order_id'
        }

        # Rename columns that exist
        df = df.rename(columns=column_mapping)

        # Calculate total_fees if we have individual fee columns
        fee_columns = ['Comm', 'Ecn Fee', 'SEC', 'ORF', 'CAT', 'TAF', 'FTT', 'NSCC', 'Acc', 'Clr', 'Misc']
        existing_fee_cols = [col for col in fee_columns if col in df.columns]
        if existing_fee_cols:
            df['total_fees'] = df[existing_fee_cols].fillna(0).sum(axis=1)
            # Drop individual fee columns
            df = df.drop(columns=existing_fee_cols)
        elif 'total_fees' not in df.columns:
            df['total_fees'] = 0

        # Drop unnecessary columns
        cols_to_drop = ['Account', 'B/S', 'Route', 'Liq', 'Fill Id', 'Currency',
                       'ISIN', 'CUSIP', 'Status', 'PropReports Id']
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

        # Ensure numeric types
        numeric_cols = ['price', 'qty', 'total_fees']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Ensure string types
        if 'order_id' in df.columns:
            df['order_id'] = df['order_id'].astype(str)
        if 'symbol' in df.columns:
            df['symbol'] = df['symbol'].astype(str)

        # Ensure datetime
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])

        # Select only columns that exist in database
        db_columns = ['datetime', 'account_id', 'symbol', 'price', 'qty', 'order_id', 'total_fees']

        # Keep only columns that exist
        available_columns = [col for col in db_columns if col in df.columns]
        df = df[available_columns]

        with self.get_connection() as conn:
            # Save to database
            df.to_sql('fills', conn, if_exists='append', index=False)

    def clear_trader_data(self, account_id: str):
        """Clear existing data for a trader before fresh download"""
        with self.get_connection() as conn:
            conn.execute("DELETE FROM totals WHERE account_id = ?", (account_id,))
            conn.execute("DELETE FROM fills WHERE account_id = ?", (account_id,))
            conn.commit()

    def get_trader_data(self, identifier: Union[str, int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get trader data by account_id or trader_name"""
        with self.get_connection() as conn:
            # Try to find trader
            trader = conn.execute("""
                SELECT account_id FROM traders
                WHERE account_id = ? OR trader_name = ?
            """, (str(identifier), str(identifier))).fetchone()

            if not trader:
                return pd.DataFrame(), pd.DataFrame()

            account_id = trader['account_id']

            # Get data
            totals = pd.read_sql_query(
                "SELECT * FROM totals WHERE account_id = ? ORDER BY date",
                conn, params=[account_id]
            )

            fills = pd.read_sql_query(
                "SELECT * FROM fills WHERE account_id = ? ORDER BY datetime",
                conn, params=[account_id]
            )

            return fills, totals


class MinimalDatabaseDownloader(PropreportsDownloader):
    """Minimal downloader that saves directly to database"""

    def __init__(self, token: str = None, reset_db: bool = False):
        super().__init__(token)

        # Reset database if requested or if schema is old
        if reset_db or self._needs_schema_update():
            self._reset_database()

        self.db = MinimalDatabaseManager()
        self._setup_traders()

    def _needs_schema_update(self) -> bool:
        """Check if database needs schema update"""
        db_path = Path("data/trading_data.db")
        if not db_path.exists():
            return False

        try:
            with sqlite3.connect(db_path) as conn:
                # Check if symbol column exists in totals
                cursor = conn.execute("PRAGMA table_info(totals)")
                columns = [row[1] for row in cursor.fetchall()]
                return 'symbol' not in columns
        except:
            return True

    def _reset_database(self):
        """Reset database with new schema"""
        db_path = Path("data/trading_data.db")
        if db_path.exists():
            logger.info("Resetting database with updated schema...")
            db_path.unlink()  # Delete old database

    def _setup_traders(self):
        """Add traders to database"""
        with self.db.get_connection() as conn:
            for trader in self.traders:
                conn.execute("""
                    INSERT OR REPLACE INTO traders (account_id, trader_name, active)
                    VALUES (?, ?, ?)
                """, (trader.account_id, trader.name, trader.active))
            conn.commit()

    def download_all_data(self, days_back: int = 365) -> Dict[str, bool]:
        """Download all data for all traders"""
        end_date = date.today()
        start_date = end_date - timedelta(days=days_back)

        results = {}
        total = len(self.traders)

        logger.info(f"Downloading data from {start_date} to {end_date}")

        for i, trader in enumerate(self.traders, 1):
            account_id = trader.account_id
            logger.info(f"[{i}/{total}] Downloading {trader.name}...")

            try:
                # Clear existing data
                self.db.clear_trader_data(account_id)

                # Download totals
                totals_df = self._fetch_all_tbd_pages({
                    "action": "report",
                    "type": "totalsByDate",
                    "token": self.token,
                    "accountId": account_id,
                    "startDate": start_date.strftime("%Y-%m-%d"),
                    "endDate": end_date.strftime("%Y-%m-%d")
                })

                if not totals_df.empty:
                    self.db.save_totals(totals_df, account_id)

                # Download fills
                fills_df = self._fetch_all_pages({
                    "action": "fills",
                    "token": self.token,
                    "accountId": account_id,
                    "startDate": start_date.strftime("%Y-%m-%d"),
                    "endDate": end_date.strftime("%Y-%m-%d")
                })

                if not fills_df.empty:
                    self.db.save_fills(fills_df, account_id)

                results[account_id] = True
                logger.info(f"  ✓ Success: {len(totals_df)} days, {len(fills_df)} fills")

            except Exception as e:
                results[account_id] = False
                logger.error(f"  ✗ Failed: {str(e)}")

        # Summary
        success_count = sum(results.values())
        logger.info(f"\nCompleted: {success_count}/{total} traders downloaded successfully")

        # Show database status
        self._show_status()

        return results

    def _show_status(self):
        """Show database status after download"""
        with self.db.get_connection() as conn:
            # Get summary stats
            total_traders = conn.execute("SELECT COUNT(*) FROM traders").fetchone()[0]
            total_days = conn.execute("SELECT COUNT(*) FROM totals").fetchone()[0]
            total_fills = conn.execute("SELECT COUNT(*) FROM fills").fetchone()[0]

            # Get date range
            date_range = conn.execute("""
                SELECT MIN(date), MAX(date) FROM totals
            """).fetchone()

            # Get total P&L
            total_pnl = conn.execute("""
                SELECT SUM(net_pnl) FROM totals
            """).fetchone()[0] or 0

            print("\n" + "="*50)
            print("DATABASE STATUS")
            print("="*50)
            print(f"Traders: {total_traders}")
            print(f"Total Records: {total_days:,} daily totals, {total_fills:,} fills")
            print(f"Date Range: {date_range[0]} to {date_range[1]}")
            print(f"Total P&L: ${total_pnl:,.2f}")
            print(f"Database: data/trading_data.db")
            print("="*50)

    def download_recent_updates(self, days_back: int = 7) -> Dict[str, bool]:
        """Download only recent data"""
        return self.download_all_data(days_back)


class MinimalDataLoader:
    """Minimal data loader that works with database"""

    def __init__(self):
        self.db = MinimalDatabaseManager()

    def load_trader_data(self, identifier: Union[str, int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load trader data from database"""
        return self.db.get_trader_data(identifier)

    def load_all_traders_data(self) -> Dict[str, Dict]:
        """Load all traders data"""
        all_data = {}

        with self.db.get_connection() as conn:
            traders = pd.read_sql_query("SELECT * FROM traders WHERE active = 1", conn)

            for _, trader in traders.iterrows():
                fills, totals = self.load_trader_data(trader['account_id'])

                if not totals.empty:
                    all_data[trader['account_id']] = {
                        'fills': fills,
                        'totals': totals,
                        'name': trader['trader_name']
                    }

        return all_data


# Simple functions for daily use
def download_everything(days_back: int = 365):
    """Download all data from scratch"""
    downloader = MinimalDatabaseDownloader()
    return downloader.download_all_data(days_back)


def update_recent_data(days_back: int = 7):
    """Update with recent data only"""
    downloader = MinimalDatabaseDownloader()
    return downloader.download_recent_updates(days_back)


def get_trader_data(trader_identifier: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Get trader data by name or ID"""
    loader = MinimalDataLoader()
    return loader.load_trader_data(trader_identifier)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Minimal Database Solution")
    parser.add_argument("--download", action="store_true", help="Download all data")
    parser.add_argument("--update", action="store_true", help="Update recent data (7 days)")
    parser.add_argument("--days", type=int, default=365, help="Days to download")
    parser.add_argument("--trader", help="Get data for specific trader")
    parser.add_argument("--reset", action="store_true", help="Reset database before download")

    args = parser.parse_args()

    if args.download:
        print(f"Downloading {args.days} days of data...")
        if args.reset:
            print("Resetting database first...")
        downloader = MinimalDatabaseDownloader(reset_db=args.reset)
        results = downloader.download_all_data(args.days)
        print("Download complete!")

    elif args.update:
        print("Updating recent data...")
        downloader = MinimalDatabaseDownloader()
        results = downloader.download_recent_updates()
        print("Update complete!")

    elif args.trader:
        fills, totals = get_trader_data(args.trader)
        print(f"Trader: {args.trader}")
        print(f"Totals: {len(totals)} days")
        print(f"Fills: {len(fills)} transactions")
        if not totals.empty:
            print(f"Date range: {totals['date'].min()} to {totals['date'].max()}")
            print(f"Total P&L: ${totals['net_pnl'].sum():,.2f}")

    else:
        parser.print_help()
