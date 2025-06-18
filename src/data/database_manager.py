"""
Simple Database Manager for Risk Tool
Essential operations only: insert data and get data
No complex SQLite queries - keep it simple and clean
"""

import sqlite3
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
from pathlib import Path
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Simple database manager with only essential operations:
    - Insert data (with duplicate handling)
    - Get data
    - Basic stats
    """

    def __init__(self, db_path: str = "data/trading_data.db"):
        """Initialize database manager"""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)

        # Create tables if they don't exist
        self._create_tables()

        # Run database migrations for existing databases
        self._migrate_database()

    def _create_tables(self):
        """Create database tables"""
        with sqlite3.connect(self.db_path) as conn:
            # Accounts table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS accounts (
                    account_id TEXT PRIMARY KEY,
                    account_name TEXT,
                    account_type TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Daily summary table - essential fields only
            # UNIQUE constraint on (account_id, date) prevents duplicate daily summaries
            conn.execute("""
                CREATE TABLE IF NOT EXISTS account_daily_summary (
                    account_id TEXT,
                    date DATE,
                    type TEXT,
                    orders INTEGER,
                    fills INTEGER,
                    qty REAL,
                    gross REAL,
                    net REAL,
                    trade_fees REAL,
                    total_delta REAL,
                    end_balance REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (account_id, date),
                    UNIQUE(account_id, date)
                )
            """)

            # Fills table - essential fields only
            # Need to add order_id and fill_id columns for proper uniqueness
            conn.execute("""
                CREATE TABLE IF NOT EXISTS fills (
                    account_id TEXT,
                    datetime TIMESTAMP,
                    symbol TEXT,
                    side TEXT,
                    qty REAL,
                    price REAL,
                    total_fee REAL,
                    order_id TEXT,
                    fill_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(account_id, fill_id)
                )
            """)

            # Create indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_summary_account_date ON account_daily_summary (account_id, date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_fills_account_datetime ON fills (account_id, datetime)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_fills_symbol ON fills (symbol)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_fills_order_id ON fills (order_id)")

    def _migrate_database(self):
        """Handle database migrations for existing databases"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Check if fills table has the new columns
            cursor.execute("PRAGMA table_info(fills)")
            columns = [column[1] for column in cursor.fetchall()]

            # Add order_id column if it doesn't exist
            if 'order_id' not in columns:
                logger.info("Adding order_id column to fills table")
                cursor.execute("ALTER TABLE fills ADD COLUMN order_id TEXT")

            # Add fill_id column if it doesn't exist
            if 'fill_id' not in columns:
                logger.info("Adding fill_id column to fills table")
                cursor.execute("ALTER TABLE fills ADD COLUMN fill_id TEXT")

            # Add unique constraint if it doesn't exist
            try:
                cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_fills_unique ON fills (account_id, fill_id)")
                logger.info("Created unique index on fills table")
            except Exception as e:
                logger.debug(f"Unique index might already exist: {e}")

    def insert_summary_data(self, df: pd.DataFrame, account_id: str, replace_existing: bool = False) -> int:
        """
        Insert daily summary data, with option to replace existing data

        Args:
            df: DataFrame with summary data
            account_id: Account ID
            replace_existing: If True, replace existing data; if False, ignore duplicates

        Returns:
            Number of records inserted/updated
        """
        if df.empty:
            return 0

        # Prepare data with only essential fields
        data = []
        for _, row in df.iterrows():
            # Convert date to string if it's a pandas Timestamp
            date_value = row.get('Date')
            if pd.notna(date_value):
                if hasattr(date_value, 'strftime'):
                    date_str = date_value.strftime('%Y-%m-%d')
                else:
                    date_str = str(date_value)
            else:
                date_str = None

            record = (
                str(account_id),  # Ensure string
                date_str,
                str(row.get('Type', 'Eq')),
                int(row.get('Orders', 0)) if pd.notna(row.get('Orders')) else 0,
                int(row.get('Fills', 0)) if pd.notna(row.get('Fills')) else 0,
                float(row.get('Qty', 0)) if pd.notna(row.get('Qty')) else 0.0,
                float(row.get('Gross', 0)) if pd.notna(row.get('Gross')) else 0.0,
                float(row.get('Net', 0)) if pd.notna(row.get('Net')) else 0.0,
                float(row.get('Trade Fees', 0)) if pd.notna(row.get('Trade Fees')) else 0.0,
                float(row.get('Total Δ', 0)) if pd.notna(row.get('Total Δ')) else 0.0,
                float(row.get('End Balance', 0)) if pd.notna(row.get('End Balance')) else 0.0
            )
            data.append(record)

        # Insert with ignore duplicates
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Store account info
            cursor.execute("""
                INSERT OR REPLACE INTO accounts (account_id, account_name, account_type)
                VALUES (?, ?, ?)
            """, (str(account_id), f"Account_{account_id}", str(data[0][2] if data else 'Eq')))

            # Insert summary data with appropriate strategy
            inserted = 0
            for record in data:
                try:
                    if replace_existing:
                        # Use INSERT OR REPLACE to update existing records
                        cursor.execute("""
                            INSERT OR REPLACE INTO account_daily_summary
                            (account_id, date, type, orders, fills, qty, gross, net, trade_fees, total_delta, end_balance)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, record)
                        inserted += 1
                    else:
                        # Use INSERT OR IGNORE to skip duplicates
                        cursor.execute("""
                            INSERT OR IGNORE INTO account_daily_summary
                            (account_id, date, type, orders, fills, qty, gross, net, trade_fees, total_delta, end_balance)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, record)
                        if cursor.rowcount > 0:
                            inserted += 1
                except Exception as e:
                    logger.warning(f"Failed to insert summary record: {e}")

        logger.info(f"Inserted {inserted} new summary records for {account_id}")
        return inserted

    def insert_fills_data(self, df: pd.DataFrame, account_id: str, replace_existing: bool = False) -> int:
        """
        Insert fills data with option to replace existing data

        Args:
            df: DataFrame with fills data
            account_id: Account ID
            replace_existing: If True, delete existing data for the date range first

        Returns:
            Number of records inserted
        """
        if df.empty:
            return 0

        # Calculate total fees (sum of all fee columns)
        fee_columns = ['Comm', 'Ecn Fee', 'SEC', 'ORF', 'CAT', 'TAF', 'FTT', 'NSCC', 'Acc', 'Clr', 'Misc']
        df['total_fee'] = 0
        for col in fee_columns:
            if col in df.columns:
                df['total_fee'] += df[col].fillna(0)

        # Prepare data with only essential fields
        data = []
        for _, row in df.iterrows():
            # Convert datetime to string if it's a pandas Timestamp
            datetime_value = row.get('Date/Time')
            if pd.notna(datetime_value):
                if hasattr(datetime_value, 'strftime'):
                    datetime_str = datetime_value.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    datetime_str = str(datetime_value)
            else:
                datetime_str = None

            record = (
                str(account_id),  # Ensure string
                datetime_str,
                str(row.get('Symbol', '')).strip().upper(),
                str(row.get('B/S', '')).strip().upper(),
                float(row.get('Qty', 0)) if pd.notna(row.get('Qty')) else 0.0,
                float(row.get('Price', 0)) if pd.notna(row.get('Price')) else 0.0,
                float(row.get('total_fee', 0)) if pd.notna(row.get('total_fee')) else 0.0,
                str(row.get('Order Id', '')) if pd.notna(row.get('Order Id')) else None,
                str(row.get('Fill Id', '')) if pd.notna(row.get('Fill Id')) else None
            )
            data.append(record)

        # Insert data
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # If replacing existing data, delete fills for the date range first
            if replace_existing and data:
                # Get date range from the data
                dates = [record[1] for record in data if record[1]]  # datetime column
                if dates:
                    # Extract just the date part for deletion
                    date_strs = list(set(dt.split(' ')[0] if ' ' in dt else dt for dt in dates if dt))
                    if date_strs:
                        placeholders = ','.join(['?' for _ in date_strs])
                        cursor.execute(f"""
                            DELETE FROM fills
                            WHERE account_id = ? AND date(datetime) IN ({placeholders})
                        """, [str(account_id)] + date_strs)
                        deleted_count = cursor.rowcount
                        if deleted_count > 0:
                            logger.info(f"Deleted {deleted_count} existing fill records for replacement")

            inserted = 0
            for record in data:
                try:
                    if replace_existing:
                        # Use INSERT OR REPLACE to update existing records
                        cursor.execute("""
                            INSERT OR REPLACE INTO fills
                            (account_id, datetime, symbol, side, qty, price, total_fee, order_id, fill_id)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, record)
                    else:
                        # Use INSERT OR IGNORE to skip duplicates
                        cursor.execute("""
                            INSERT OR IGNORE INTO fills
                            (account_id, datetime, symbol, side, qty, price, total_fee, order_id, fill_id)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, record)
                    if cursor.rowcount > 0:
                        inserted += 1
                except Exception as e:
                    logger.warning(f"Failed to insert fill record: {e}")

        logger.info(f"Inserted {inserted} fill records for {account_id}")
        return inserted

    def get_accounts(self) -> pd.DataFrame:
        """Get all accounts"""
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query("SELECT * FROM accounts", conn)

    def get_summary_data(self, account_id: Optional[str] = None,
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get daily summary data

        Args:
            account_id: Filter by account ID
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with summary data
        """
        query = "SELECT * FROM account_daily_summary WHERE 1=1"
        params = []

        if account_id:
            query += " AND account_id = ?"
            params.append(account_id)

        if start_date:
            query += " AND date >= ?"
            params.append(start_date)

        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        query += " ORDER BY account_id, date"

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)
            if not df.empty and 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            return df

    def get_fills_data(self, account_id: Optional[str] = None,
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None,
                      symbol: Optional[str] = None) -> pd.DataFrame:
        """
        Get fills data

        Args:
            account_id: Filter by account ID
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            symbol: Filter by symbol

        Returns:
            DataFrame with fills data
        """
        query = "SELECT * FROM fills WHERE 1=1"
        params = []

        if account_id:
            query += " AND account_id = ?"
            params.append(account_id)

        if start_date:
            query += " AND date(datetime) >= ?"
            params.append(start_date)

        if end_date:
            query += " AND date(datetime) <= ?"
            params.append(end_date)

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol.upper())

        query += " ORDER BY account_id, datetime"

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)
            if not df.empty and 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
            return df

    def get_database_stats(self) -> Dict[str, Any]:
        """Get basic database statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Count records in each table
            cursor.execute("SELECT COUNT(*) FROM accounts")
            accounts_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM account_daily_summary")
            summary_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM fills")
            fills_count = cursor.fetchone()[0]

            # Date ranges
            cursor.execute("SELECT MIN(date), MAX(date) FROM account_daily_summary")
            summary_dates = cursor.fetchone()

            cursor.execute("SELECT MIN(date(datetime)), MAX(date(datetime)) FROM fills")
            fills_dates = cursor.fetchone()

            stats = {
                'Total Accounts': accounts_count,
                'Daily Summary Records': summary_count,
                'Fills Records': fills_count,
                'Summary Date Range': f"{summary_dates[0]} to {summary_dates[1]}" if summary_dates[0] else "No data",
                'Fills Date Range': f"{fills_dates[0]} to {fills_dates[1]}" if fills_dates[0] else "No data"
            }

        return stats

    def delete_data_for_period(self, account_id: str, start_date: str, end_date: str):
        """
        Delete data for a specific account and date range
        Used before reinserting updated data

        Args:
            account_id: Account ID
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Delete summary data
            cursor.execute("""
                DELETE FROM account_daily_summary
                WHERE account_id = ? AND date >= ? AND date <= ?
            """, (account_id, start_date, end_date))
            summary_deleted = cursor.rowcount

            # Delete fills data
            cursor.execute("""
                DELETE FROM fills
                WHERE account_id = ? AND date(datetime) >= ? AND date(datetime) <= ?
            """, (account_id, start_date, end_date))
            fills_deleted = cursor.rowcount

            logger.info(f"Deleted {summary_deleted} summary and {fills_deleted} fills records for {account_id} ({start_date} to {end_date})")
