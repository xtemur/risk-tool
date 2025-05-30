"""
Database Manager for Risk Management System
Handles data storage, updates, and retrieval with SQLite
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple, Union
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages SQLite database for trading data"""

    def __init__(self, db_path: str = "data/trading_data.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)

        # Initialize database
        self._init_database()

    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
        finally:
            conn.close()

    def _init_database(self):
        """Initialize database schema"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Traders table (maps account_id to trader_name)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS traders (
                    account_id TEXT PRIMARY KEY,
                    trader_name TEXT NOT NULL UNIQUE,
                    strategy TEXT,
                    active BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create index on trader_name for fast lookups
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_trader_name
                ON traders(trader_name)
            """)

            # Fills table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS fills (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    datetime TIMESTAMP NOT NULL,
                    account_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    price REAL NOT NULL,
                    qty REAL NOT NULL,
                    trade_side TEXT,
                    order_id TEXT,
                    total_fees REAL DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (account_id) REFERENCES traders(account_id),
                    UNIQUE(datetime, account_id, order_id, symbol, price, qty)
                )
            """)

            # Indexes for fills
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_fills_account_date
                ON fills(account_id, datetime)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_fills_date
                ON fills(datetime)
            """)

            # Totals table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS totals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE NOT NULL,
                    account_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    orders_count INTEGER DEFAULT 0,
                    fills_count INTEGER DEFAULT 0,
                    qty REAL DEFAULT 0,
                    gross_pnl REAL DEFAULT 0,
                    net_pnl REAL DEFAULT 0,
                    total_fees REAL DEFAULT 0,
                    unrealized_delta REAL DEFAULT 0,
                    total_delta REAL DEFAULT 0,
                    unrealized_pnl REAL DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (account_id) REFERENCES traders(account_id),
                    UNIQUE(date, account_id, symbol)
                )
            """)

            # Indexes for totals
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_totals_account_date
                ON totals(account_id, date)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_totals_date
                ON totals(date)
            """)

            # Data update log table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS update_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    table_name TEXT NOT NULL,
                    account_id TEXT,
                    start_date DATE,
                    end_date DATE,
                    records_inserted INTEGER DEFAULT 0,
                    records_updated INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.commit()

        logger.info(f"Database initialized at {self.db_path}")

    def upsert_traders(self, traders_df: pd.DataFrame) -> Dict[str, int]:
        """Insert or update trader information"""
        stats = {"inserted": 0, "updated": 0}

        with self.get_connection() as conn:
            cursor = conn.cursor()

            for _, trader in traders_df.iterrows():
                cursor.execute("""
                    INSERT INTO traders (account_id, trader_name, strategy, active)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(account_id) DO UPDATE SET
                        trader_name = excluded.trader_name,
                        strategy = excluded.strategy,
                        active = excluded.active,
                        updated_at = CURRENT_TIMESTAMP
                """, (
                    str(trader['account_id']),
                    trader.get('trader_name', trader['account_id']),
                    trader.get('strategy', 'Unknown'),
                    trader.get('active', True)
                ))

                if cursor.rowcount > 0:
                    stats["inserted"] += 1
                else:
                    stats["updated"] += 1

            conn.commit()

        logger.info(f"Traders updated: {stats}")
        return stats

    def upsert_fills(self, fills_df: pd.DataFrame, account_id: str) -> Dict[str, int]:
        """Insert or update fills data"""
        stats = {"inserted": 0, "skipped": 0, "errors": 0}

        if fills_df.empty:
            return stats

        # Ensure datetime column
        fills_df['datetime'] = pd.to_datetime(fills_df['datetime'])

        with self.get_connection() as conn:
            cursor = conn.cursor()

            for _, fill in fills_df.iterrows():
                try:
                    # Clean and convert data types
                    price = float(pd.to_numeric(fill.get('price', 0), errors='coerce') or 0)
                    qty = float(pd.to_numeric(fill.get('qty', 0), errors='coerce') or 0)
                    total_fees = float(pd.to_numeric(fill.get('total_fees', 0), errors='coerce') or 0)

                    cursor.execute("""
                        INSERT OR IGNORE INTO fills
                        (datetime, account_id, symbol, price, qty, trade_side,
                         order_id, total_fees)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        fill['datetime'].isoformat(),
                        account_id,
                        str(fill.get('symbol', '')),
                        price,
                        qty,
                        str(fill.get('trade_side', '')),
                        str(fill.get('order_id', '')),
                        total_fees
                    ))

                    if cursor.rowcount > 0:
                        stats["inserted"] += 1
                    else:
                        stats["skipped"] += 1

                except Exception as e:
                    logger.error(f"Error inserting fill: {e}")
                    logger.debug(f"Problem record: datetime={fill.get('datetime')}, symbol={fill.get('symbol')}")
                    stats["errors"] += 1

            conn.commit()

            # Log the update
            if stats["inserted"] > 0:
                cursor.execute("""
                    INSERT INTO update_log
                    (table_name, account_id, start_date, end_date, records_inserted)
                    VALUES ('fills', ?, ?, ?, ?)
                """, (
                    account_id,
                    fills_df['datetime'].min().date().isoformat(),
                    fills_df['datetime'].max().date().isoformat(),
                    stats["inserted"]
                ))
                conn.commit()

        logger.info(f"Fills updated for {account_id}: {stats}")
        return stats

    def upsert_totals(self, totals_df: pd.DataFrame, account_id: str) -> Dict[str, int]:
        """Insert or update totals data"""
        stats = {"inserted": 0, "updated": 0, "errors": 0}

        if totals_df.empty:
            return stats

        # Ensure date column
        totals_df['date'] = pd.to_datetime(totals_df['date'])

        with self.get_connection() as conn:
            cursor = conn.cursor()

            for _, total in totals_df.iterrows():
                try:
                    # Clean and convert data types
                    orders_count = int(pd.to_numeric(total.get('orders_count', 0), errors='coerce') or 0)
                    fills_count = int(pd.to_numeric(total.get('fills_count', 0), errors='coerce') or 0)
                    qty = float(pd.to_numeric(total.get('qty', 0), errors='coerce') or 0)
                    gross_pnl = float(pd.to_numeric(total.get('gross_pnl', 0), errors='coerce') or 0)
                    net_pnl = float(pd.to_numeric(total.get('net_pnl', 0), errors='coerce') or 0)
                    total_fees = float(pd.to_numeric(total.get('total_fees', 0), errors='coerce') or 0)
                    unrealized_delta = float(pd.to_numeric(total.get('unrealized_delta', 0), errors='coerce') or 0)
                    total_delta = float(pd.to_numeric(total.get('total_delta', 0), errors='coerce') or 0)
                    unrealized_pnl = float(pd.to_numeric(total.get('unrealized_pnl', 0), errors='coerce') or 0)

                    cursor.execute("""
                        INSERT INTO totals
                        (date, account_id, symbol, orders_count, fills_count, qty,
                        gross_pnl, net_pnl, total_fees, unrealized_delta,
                        total_delta, unrealized_pnl)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(date, account_id, symbol) DO UPDATE SET
                        orders_count    = excluded.orders_count,
                        fills_count     = excluded.fills_count,
                        qty             = excluded.qty,
                        gross_pnl       = excluded.gross_pnl,
                        net_pnl         = excluded.net_pnl,
                        total_fees      = excluded.total_fees,
                        unrealized_delta= excluded.unrealized_delta,
                        total_delta     = excluded.total_delta,
                        unrealized_pnl  = excluded.unrealized_pnl,
                        updated_at      = CURRENT_TIMESTAMP
                    """, (
                        total['date'].date().isoformat(),
                        account_id,
                        symbol,
                        orders_count,
                        fills_count,
                        qty,
                        gross_pnl,
                        net_pnl,
                        total_fees,
                        unrealized_delta,
                        total_delta,
                        unrealized_pnl
                    ))

                    if cursor.lastrowid:
                        stats["inserted"] += 1
                    else:
                        stats["updated"] += 1

                except Exception as e:
                    stats["errors"] += 1
                    logger.warning(f"Error inserting totals record: {e}")
                    logger.debug(f"Problem record: date={total.get('date')}, symbol={total.get('symbol')}")

            conn.commit()

            # Log the update
            if stats["inserted"] > 0 or stats["updated"] > 0:
                cursor.execute("""
                    INSERT INTO update_log
                    (table_name, account_id, start_date, end_date,
                     records_inserted, records_updated)
                    VALUES ('totals', ?, ?, ?, ?, ?)
                """, (
                    account_id,
                    totals_df['date'].min().date().isoformat(),
                    totals_df['date'].max().date().isoformat(),
                    stats["inserted"],
                    stats["updated"]
                ))
                conn.commit()

        logger.info(f"Totals updated for {account_id}: {stats}")
        return stats

    def get_trader_data(self, identifier: Union[str, int],
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get trader data by account_id or trader_name"""

        # First, resolve the identifier to account_id
        account_id = self._resolve_trader_identifier(identifier)
        if not account_id:
            logger.warning(f"Trader not found: {identifier}")
            return pd.DataFrame(), pd.DataFrame()

        # Build date filter
        date_filter = ""
        params = [account_id]

        if start_date:
            date_filter += " AND date >= ?"
            params.append(start_date)
        if end_date:
            date_filter += " AND date <= ?"
            params.append(end_date)

        # Get fills
        fills_query = f"""
            SELECT f.*, t.trader_name
            FROM fills f
            JOIN traders t ON f.account_id = t.account_id
            WHERE f.account_id = ?
            {date_filter.replace('date', 'datetime')}
            ORDER BY datetime
        """

        # Get totals
        totals_query = f"""
            SELECT tot.*, t.trader_name
            FROM totals tot
            JOIN traders t ON tot.account_id = t.account_id
            WHERE tot.account_id = ?
            {date_filter}
            ORDER BY date
        """

        with self.get_connection() as conn:
            fills_df = pd.read_sql_query(fills_query, conn, params=params)
            totals_df = pd.read_sql_query(totals_query, conn, params=params)

        # Convert datetime columns
        if not fills_df.empty:
            fills_df['datetime'] = pd.to_datetime(fills_df['datetime'])
        if not totals_df.empty:
            totals_df['date'] = pd.to_datetime(totals_df['date'])

        return fills_df, totals_df

    def _resolve_trader_identifier(self, identifier: Union[str, int]) -> Optional[str]:
        """Resolve trader_name or account_id to account_id"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Try as account_id first
            cursor.execute(
                "SELECT account_id FROM traders WHERE account_id = ?",
                (str(identifier),)
            )
            result = cursor.fetchone()

            if result:
                return result[0]

            # Try as trader_name
            cursor.execute(
                "SELECT account_id FROM traders WHERE trader_name = ?",
                (str(identifier),)
            )
            result = cursor.fetchone()

            return result[0] if result else None

    def get_all_traders(self) -> pd.DataFrame:
        """Get all trader information"""
        query = """
            SELECT t.*,
                   COUNT(DISTINCT tot.date) as trading_days,
                   MIN(tot.date) as first_trade_date,
                   MAX(tot.date) as last_trade_date,
                   SUM(tot.net_pnl) as total_pnl
            FROM traders t
            LEFT JOIN totals tot ON t.account_id = tot.account_id
            WHERE t.active = 1
            GROUP BY t.account_id
            ORDER BY t.trader_name
        """

        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn)

    def get_latest_data_dates(self) -> pd.DataFrame:
        """Get the latest data date for each trader"""
        query = """
            SELECT
                t.account_id,
                t.trader_name,
                MAX(f.datetime) as latest_fill,
                MAX(tot.date) as latest_total
            FROM traders t
            LEFT JOIN fills f ON t.account_id = f.account_id
            LEFT JOIN totals tot ON t.account_id = tot.account_id
            GROUP BY t.account_id
            ORDER BY t.trader_name
        """

        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn)

    def get_update_history(self, days: int = 7) -> pd.DataFrame:
        """Get recent update history"""
        query = """
            SELECT * FROM update_log
            WHERE created_at >= datetime('now', '-' || ? || ' days')
            ORDER BY created_at DESC
        """

        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=[days])

    def vacuum_database(self):
        """Optimize database size and performance"""
        with self.get_connection() as conn:
            conn.execute("VACUUM")
        logger.info("Database vacuumed")

    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        stats = {}

        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Table sizes
            tables = ['traders', 'fills', 'totals']
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                stats[f"{table}_count"] = cursor.fetchone()[0]

            # Database file size
            stats['db_size_mb'] = self.db_path.stat().st_size / (1024 * 1024)

            # Date ranges
            cursor.execute("SELECT MIN(date), MAX(date) FROM totals")
            min_date, max_date = cursor.fetchone()
            stats['date_range'] = f"{min_date} to {max_date}"

        return stats
