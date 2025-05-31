"""
Simplified Database Handler for Risk Management MVP
Handles all database operations with clean interface
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


class Database:
    """Single database handler for all operations"""

    def __init__(self, db_path: str = "data/trading.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self._init_database()

    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_database(self):
        """Initialize database with simplified schema"""
        with self.get_connection() as conn:
            conn.executescript("""
                -- Traders table
                CREATE TABLE IF NOT EXISTS traders (
                    account_id TEXT PRIMARY KEY,
                    trader_name TEXT NOT NULL,
                    active BOOLEAN DEFAULT 1
                );

                -- Daily totals table (aggregated by date)
                CREATE TABLE IF NOT EXISTS daily_totals (
                    date DATE NOT NULL,
                    account_id TEXT NOT NULL,
                    orders_count INTEGER DEFAULT 0,
                    fills_count INTEGER DEFAULT 0,
                    quantity REAL DEFAULT 0,
                    gross_pnl REAL DEFAULT 0,
                    net_pnl REAL DEFAULT 0,
                    total_fees REAL DEFAULT 0,
                    unrealized_delta REAL DEFAULT 0,
                    total_delta REAL DEFAULT 0,
                    PRIMARY KEY (date, account_id),
                    FOREIGN KEY (account_id) REFERENCES traders(account_id)
                );

                -- Fills table (transaction details)
                CREATE TABLE IF NOT EXISTS fills (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    datetime TIMESTAMP NOT NULL,
                    account_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    price REAL NOT NULL,
                    quantity REAL NOT NULL,
                    order_id TEXT,
                    total_fees REAL DEFAULT 0,
                    FOREIGN KEY (account_id) REFERENCES traders(account_id)
                );

                -- Model predictions table
                CREATE TABLE IF NOT EXISTS predictions (
                    date DATE NOT NULL,
                    account_id TEXT NOT NULL,
                    predicted_pnl REAL,
                    risk_score REAL,
                    confidence REAL,
                    PRIMARY KEY (date, account_id)
                );

                -- Indexes for performance
                CREATE INDEX IF NOT EXISTS idx_totals_date ON daily_totals(date);
                CREATE INDEX IF NOT EXISTS idx_totals_account ON daily_totals(account_id);
                CREATE INDEX IF NOT EXISTS idx_fills_datetime ON fills(datetime);
                CREATE INDEX IF NOT EXISTS idx_fills_account ON fills(account_id);
            """)
            conn.commit()

    def save_traders(self, traders: List[Dict]):
        """Save trader information"""
        with self.get_connection() as conn:
            for trader in traders:
                conn.execute("""
                    INSERT OR REPLACE INTO traders (account_id, trader_name, active)
                    VALUES (?, ?, ?)
                """, (trader['account_id'], trader['name'], trader.get('active', True)))
            conn.commit()

    def save_daily_totals(self, df: pd.DataFrame, account_id: str):
        """Save daily totals data"""
        if df.empty:
            return

        df = df.copy()
        df['account_id'] = account_id

        # Ensure date column
        if 'date' not in df.columns:
            df['date'] = pd.to_datetime(df.index).date
        else:
            df['date'] = pd.to_datetime(df['date']).dt.date

        # Convert date to string format for SQLite
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

        # Ensure numeric columns
        numeric_cols = ['orders_count', 'fills_count', 'quantity', 'gross_pnl',
                       'net_pnl', 'total_fees', 'unrealized_delta', 'total_delta']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        with self.get_connection() as conn:
            # Clear existing data for date range
            if len(df) > 0:
                min_date = df['date'].min()
                max_date = df['date'].max()
                conn.execute("""
                    DELETE FROM daily_totals
                    WHERE account_id = ? AND date BETWEEN ? AND ?
                """, (account_id, min_date, max_date))

            # Insert new data
            try:
                df.to_sql('daily_totals', conn, if_exists='append', index=False)
                conn.commit()
                logger.info(f"Saved {len(df)} daily totals for {account_id}")
            except Exception as e:
                logger.error(f"Error saving totals: {str(e)}")
                logger.debug(f"DataFrame columns: {df.columns.tolist()}")
                logger.debug(f"DataFrame dtypes: {df.dtypes.to_dict()}")
                raise

    def save_fills(self, df: pd.DataFrame, account_id: str):
        """Save fills data"""
        if df.empty:
            return

        df = df.copy()
        df['account_id'] = account_id

        # Ensure datetime column is properly formatted
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            # Convert to string format for SQLite
            df['datetime'] = df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')

        # Convert numeric columns
        numeric_cols = ['price', 'quantity', 'total_fees']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        with self.get_connection() as conn:
            # Clear existing data for date range
            if 'datetime' in df.columns and len(df) > 0:
                # Convert to date strings for SQL
                min_date = pd.to_datetime(df['datetime']).min().strftime('%Y-%m-%d')
                max_date = pd.to_datetime(df['datetime']).max().strftime('%Y-%m-%d')

                conn.execute("""
                    DELETE FROM fills
                    WHERE account_id = ? AND date(datetime) BETWEEN ? AND ?
                """, (account_id, min_date, max_date))

            # Insert new data
            try:
                df.to_sql('fills', conn, if_exists='append', index=False)
                conn.commit()
                logger.info(f"Saved {len(df)} fills for {account_id}")
            except Exception as e:
                logger.error(f"Error saving fills: {str(e)}")
                logger.debug(f"DataFrame columns: {df.columns.tolist()}")
                logger.debug(f"DataFrame dtypes: {df.dtypes.to_dict()}")
                raise

    def get_trader_data(self, account_id: str,
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get trader data (totals and fills)"""
        with self.get_connection() as conn:
            # Build date filter
            date_filter = ""
            params = [account_id]

            if start_date:
                date_filter += " AND date >= ?"
                params.append(start_date)
            if end_date:
                date_filter += " AND date <= ?"
                params.append(end_date)

            # Get daily totals
            totals_query = f"""
                SELECT * FROM daily_totals
                WHERE account_id = ? {date_filter}
                ORDER BY date
            """
            totals_df = pd.read_sql_query(totals_query, conn, params=params)

            # Get fills
            fills_params = [account_id]
            fills_date_filter = date_filter.replace('date', 'date(datetime)')
            if start_date:
                fills_params.append(start_date)
            if end_date:
                fills_params.append(end_date)

            fills_query = f"""
                SELECT * FROM fills
                WHERE account_id = ? {fills_date_filter}
                ORDER BY datetime
            """
            fills_df = pd.read_sql_query(fills_query, conn, params=fills_params)

            # Convert date columns
            if not totals_df.empty:
                totals_df['date'] = pd.to_datetime(totals_df['date'])
            if not fills_df.empty:
                fills_df['datetime'] = pd.to_datetime(fills_df['datetime'])

            return totals_df, fills_df

    def get_all_traders(self) -> pd.DataFrame:
        """Get all active traders"""
        with self.get_connection() as conn:
            return pd.read_sql_query("""
                SELECT t.*,
                       COUNT(DISTINCT dt.date) as trading_days,
                       MIN(dt.date) as first_trade,
                       MAX(dt.date) as last_trade,
                       SUM(dt.net_pnl) as total_pnl
                FROM traders t
                LEFT JOIN daily_totals dt ON t.account_id = dt.account_id
                WHERE t.active = 1
                GROUP BY t.account_id
            """, conn)

    def save_predictions(self, predictions: List[Dict]):
        """Save model predictions"""
        df = pd.DataFrame(predictions)
        df['date'] = pd.Timestamp.now().date()

        with self.get_connection() as conn:
            df.to_sql('predictions', conn, if_exists='append', index=False)

    def get_latest_predictions(self) -> pd.DataFrame:
        """Get latest predictions"""
        with self.get_connection() as conn:
            return pd.read_sql_query("""
                SELECT p.*, t.trader_name
                FROM predictions p
                JOIN traders t ON p.account_id = t.account_id
                WHERE p.date = (SELECT MAX(date) FROM predictions)
                ORDER BY p.risk_score DESC
            """, conn)

    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        with self.get_connection() as conn:
            stats = {}

            # Count records
            for table in ['traders', 'daily_totals', 'fills', 'predictions']:
                count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                stats[f'{table}_count'] = count

            # Date ranges
            date_range = conn.execute("""
                SELECT MIN(date), MAX(date) FROM daily_totals
            """).fetchone()
            stats['date_range'] = f"{date_range[0]} to {date_range[1]}" if date_range[0] else "No data"

            # Database size
            stats['db_size_mb'] = self.db_path.stat().st_size / (1024 * 1024) if self.db_path.exists() else 0

            return stats
