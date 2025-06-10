"""
Database Manager for Risk Tool
Handles all database operations with proper schema for PropreReports data
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from contextlib import contextmanager
import logging
from datetime import datetime, date

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Centralized database manager for trading data
    """

    def __init__(self, db_path: str = "data/trading_risk.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True, parents=True)
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
        """Initialize database with proper schema"""
        with self.get_connection() as conn:
            # Enable foreign keys
            conn.execute("PRAGMA foreign_keys = ON")

            # Create tables
            conn.executescript("""
                -- Accounts table
                CREATE TABLE IF NOT EXISTS accounts (
                    account_id TEXT PRIMARY KEY,
                    account_name TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                -- Daily summary table (from totals by date reports)
                CREATE TABLE IF NOT EXISTS daily_summary (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    account_id TEXT NOT NULL,
                    date DATE NOT NULL,
                    symbol TEXT NOT NULL,
                    orders INTEGER DEFAULT 0,
                    fills INTEGER DEFAULT 0,
                    shares REAL DEFAULT 0,
                    gross_pl REAL DEFAULT 0,
                    net_pl REAL DEFAULT 0,
                    unrealized REAL DEFAULT 0,
                    total REAL DEFAULT 0,
                    volume REAL DEFAULT 0,
                    high_price REAL,
                    low_price REAL,
                    open_shares REAL DEFAULT 0,
                    closed_pl REAL DEFAULT 0,
                    trades INTEGER DEFAULT 0,
                    UNIQUE(account_id, date, symbol),
                    FOREIGN KEY (account_id) REFERENCES accounts(account_id)
                );

                -- Fills table (individual trades)
                CREATE TABLE IF NOT EXISTS fills (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    account_id TEXT NOT NULL,
                    datetime TIMESTAMP NOT NULL,
                    side TEXT CHECK(side IN ('B', 'S', 'T')),
                    quantity INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    price REAL NOT NULL,
                    route TEXT,
                    liquidity TEXT,
                    commission REAL DEFAULT 0,
                    ecn_fee REAL DEFAULT 0,
                    sec_fee REAL DEFAULT 0,
                    orf_fee REAL DEFAULT 0,
                    cat_fee REAL DEFAULT 0,
                    taf_fee REAL DEFAULT 0,
                    ftt_fee REAL DEFAULT 0,
                    nscc_fee REAL DEFAULT 0,
                    acc_fee REAL DEFAULT 0,
                    clr_fee REAL DEFAULT 0,
                    misc_fee REAL DEFAULT 0,
                    total_fees REAL DEFAULT 0,
                    order_id TEXT,
                    fill_id TEXT,
                    currency TEXT DEFAULT 'USD',
                    status TEXT,
                    propreports_id INTEGER,
                    FOREIGN KEY (account_id) REFERENCES accounts(account_id)
                );

                -- Indexes for performance
                CREATE INDEX IF NOT EXISTS idx_daily_date ON daily_summary(date);
                CREATE INDEX IF NOT EXISTS idx_daily_account ON daily_summary(account_id);
                CREATE INDEX IF NOT EXISTS idx_daily_symbol ON daily_summary(symbol);
                CREATE INDEX IF NOT EXISTS idx_daily_account_date ON daily_summary(account_id, date);
                CREATE INDEX IF NOT EXISTS idx_fills_datetime ON fills(datetime);
                CREATE INDEX IF NOT EXISTS idx_fills_account ON fills(account_id);
                CREATE INDEX IF NOT EXISTS idx_fills_symbol ON fills(symbol);
                CREATE INDEX IF NOT EXISTS idx_fills_account_datetime ON fills(account_id, datetime);
                CREATE INDEX IF NOT EXISTS idx_fills_fill_id ON fills(fill_id);

                -- Metadata table for tracking data loads
                CREATE TABLE IF NOT EXISTS data_loads (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    account_id TEXT NOT NULL,
                    data_type TEXT NOT NULL,
                    start_date DATE NOT NULL,
                    end_date DATE NOT NULL,
                    records_loaded INTEGER,
                    loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'completed'
                );
            """)
            conn.commit()

    def save_account(self, account_id: str, account_name: str) -> None:
        """Save or update account information"""
        with self.get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO accounts (account_id, account_name, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            """, (account_id, account_name))
            conn.commit()

    def save_daily_summary(self, df: pd.DataFrame, account_id: str,
                          handle_duplicates: str = 'replace') -> int:
        """
        Save daily summary data (totals by date)

        Args:
            df: DataFrame with daily totals
            account_id: Account identifier
            handle_duplicates: How to handle duplicates - 'replace', 'ignore', or 'error'

        Returns:
            Number of records saved
        """
        if df.empty:
            return 0

        # Prepare data
        df = df.copy()
        df['account_id'] = account_id

        # Ensure proper column names
        column_mapping = {
            'Date': 'date',
            'Symbol': 'symbol',
            'Orders': 'orders',
            'Fills': 'fills',
            'Shares': 'shares',
            'Gross P&L': 'gross_pl',
            'Net P&L': 'net_pl',
            'Unrealized': 'unrealized',
            'Total': 'total',
            'Volume': 'volume',
            'High': 'high_price',
            'Low': 'low_price',
            'Open Shares': 'open_shares',
            'Closed P&L': 'closed_pl',
            'Trades': 'trades'
        }

        df = df.rename(columns=column_mapping)

        # Select only columns that exist
        available_cols = [col for col in column_mapping.values() if col in df.columns]
        available_cols = ['account_id'] + available_cols
        df = df[available_cols]

        # Convert date if needed - IMPORTANT: Convert to string for SQLite
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            # Convert to string format that SQLite understands
            df['date'] = df['date'].dt.strftime('%Y-%m-%d')

        records_saved = 0

        with self.get_connection() as conn:
            if handle_duplicates == 'replace':
                # Delete existing records for these dates
                if 'date' in df.columns and 'symbol' in df.columns:
                    for _, row in df.iterrows():
                        conn.execute("""
                            DELETE FROM daily_summary
                            WHERE account_id = ? AND date = ? AND symbol = ?
                        """, (account_id, str(row['date']), str(row['symbol'])))

                # Insert new records
                df.to_sql('daily_summary', conn, if_exists='append', index=False, method='multi')
                records_saved = len(df)

            elif handle_duplicates == 'ignore':
                # Insert only new records
                for _, row in df.iterrows():
                    try:
                        row_dict = row.to_dict()
                        columns = ', '.join(row_dict.keys())
                        placeholders = ', '.join(['?' for _ in row_dict])

                        conn.execute(f"""
                            INSERT OR IGNORE INTO daily_summary ({columns})
                            VALUES ({placeholders})
                        """, list(row_dict.values()))

                        if conn.total_changes > 0:
                            records_saved += 1
                    except Exception as e:
                        logger.debug(f"Error inserting row: {e}")

            elif handle_duplicates == 'error':
                # Raise error on duplicates
                df.to_sql('daily_summary', conn, if_exists='append', index=False, method='multi')
                records_saved = len(df)

            conn.commit()

        logger.info(f"Saved {records_saved} daily summary records for account {account_id}")
        return records_saved

    def save_fills(self, df: pd.DataFrame, account_id: str,
                   handle_duplicates: str = 'check') -> int:
        """
        Save fills (individual trades) data

        Args:
            df: DataFrame with fills data
            account_id: Account identifier
            handle_duplicates: How to handle duplicates - 'check', 'force', or 'error'
                - 'check': Skip duplicate fill_ids (recommended)
                - 'force': Insert all records (may create duplicates)
                - 'error': Raise error on duplicates

        Returns:
            Number of records saved
        """
        if df.empty:
            return 0

        # Prepare data
        df = df.copy()
        df['account_id'] = account_id

        # Column mapping
        column_mapping = {
            'Date/Time': 'datetime',
            'B/S': 'side',
            'Qty': 'quantity',
            'Symbol': 'symbol',
            'Price': 'price',
            'Route': 'route',
            'Liq': 'liquidity',
            'Comm': 'commission',
            'Ecn Fee': 'ecn_fee',
            'SEC': 'sec_fee',
            'ORF': 'orf_fee',
            'CAT': 'cat_fee',
            'TAF': 'taf_fee',
            'FTT': 'ftt_fee',
            'NSCC': 'nscc_fee',
            'Acc': 'acc_fee',
            'Clr': 'clr_fee',
            'Misc': 'misc_fee',
            'Order Id': 'order_id',
            'Fill Id': 'fill_id',
            'Currency': 'currency',
            'Status': 'status',
            'PropReports Id': 'propreports_id'
        }

        df = df.rename(columns=column_mapping)

        # Calculate total fees
        fee_columns = ['commission', 'ecn_fee', 'sec_fee', 'orf_fee', 'cat_fee',
                      'taf_fee', 'ftt_fee', 'nscc_fee', 'acc_fee', 'clr_fee', 'misc_fee']
        existing_fees = [col for col in fee_columns if col in df.columns]
        df['total_fees'] = df[existing_fees].fillna(0).sum(axis=1)

        # Convert datetime - IMPORTANT: Convert to string for SQLite
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            # Convert to string format that SQLite understands
            df['datetime'] = df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')

        # Select columns to save
        db_columns = ['account_id', 'datetime', 'side', 'quantity', 'symbol', 'price',
                      'route', 'liquidity', 'commission', 'ecn_fee', 'sec_fee', 'orf_fee',
                      'cat_fee', 'taf_fee', 'ftt_fee', 'nscc_fee', 'acc_fee', 'clr_fee',
                      'misc_fee', 'total_fees', 'order_id', 'fill_id', 'currency',
                      'status', 'propreports_id']

        available_cols = [col for col in db_columns if col in df.columns]
        df = df[available_cols]

        records_saved = 0

        with self.get_connection() as conn:
            if handle_duplicates == 'check':
                # Check for existing fill_ids
                if 'fill_id' in df.columns:
                    existing_fills = set()
                    for fill_id in df['fill_id'].dropna().unique():
                        result = conn.execute(
                            "SELECT 1 FROM fills WHERE fill_id = ? AND account_id = ?",
                            (str(fill_id), account_id)
                        ).fetchone()
                        if result:
                            existing_fills.add(fill_id)

                    # Filter out existing fills
                    if existing_fills:
                        logger.info(f"Found {len(existing_fills)} duplicate fill_ids, skipping...")
                        df = df[~df['fill_id'].isin(existing_fills)]

                # # Also check by datetime/symbol/quantity to catch duplicates without fill_id
                # if 'datetime' in df.columns and 'symbol' in df.columns:
                #     duplicates_to_remove = []
                #     for idx, row in df.iterrows():
                #         result = conn.execute("""
                #             SELECT 1 FROM fills
                #             WHERE account_id = ?
                #             AND datetime = ?
                #             AND symbol = ?
                #             AND quantity = ?
                #             AND price = ?
                #             AND side = ?
                #         """, (account_id, str(row['datetime']), str(row['symbol']),
                #               float(row['quantity']), float(row['price']), str(row.get('side', ''))))

                #         if result.fetchone():
                #             duplicates_to_remove.append(idx)

                #     if duplicates_to_remove:
                #         logger.info(f"Found {len(duplicates_to_remove)} potential duplicates by transaction details")
                #         df = df.drop(duplicates_to_remove)

            if not df.empty:
                df.to_sql('fills', conn, if_exists='append', index=False, method='multi')
                records_saved = len(df)

            conn.commit()

        logger.info(f"Saved {records_saved} fill records for account {account_id}")
        return records_saved

    def get_daily_summary(self,
                         account_id: Optional[str] = None,
                         start_date: Optional[date] = None,
                         end_date: Optional[date] = None,
                         symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """Get daily summary data with filters - ALWAYS sorted by date"""

        query = "SELECT * FROM daily_summary WHERE 1=1"
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

        if symbols:
            placeholders = ','.join(['?' for _ in symbols])
            query += f" AND symbol IN ({placeholders})"
            params.extend(symbols)

        # ALWAYS sort by date and symbol for consistent ordering
        query += " ORDER BY account_id, date, symbol"

        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)

        # Convert date column
        if not df.empty and 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            # Set date as index for time series operations
            df = df.sort_values(['account_id', 'date', 'symbol'])

        return df

    def get_fills(self,
                  account_id: Optional[str] = None,
                  start_date: Optional[datetime] = None,
                  end_date: Optional[datetime] = None,
                  symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """Get fills data with filters - ALWAYS sorted by datetime"""

        query = "SELECT * FROM fills WHERE 1=1"
        params = []

        if account_id:
            query += " AND account_id = ?"
            params.append(account_id)

        if start_date:
            query += " AND datetime >= ?"
            params.append(start_date)

        if end_date:
            query += " AND datetime <= ?"
            params.append(end_date)

        if symbols:
            placeholders = ','.join(['?' for _ in symbols])
            query += f" AND symbol IN ({placeholders})"
            params.extend(symbols)

        # ALWAYS sort by datetime for consistent ordering
        query += " ORDER BY account_id, datetime, symbol"

        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)

        # Convert datetime column
        if not df.empty and 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            # Ensure proper ordering
            df = df.sort_values(['account_id', 'datetime'])

        return df

    def get_accounts(self) -> pd.DataFrame:
        """Get all accounts"""
        with self.get_connection() as conn:
            return pd.read_sql_query("SELECT * FROM accounts ORDER BY account_id", conn)

    def get_account_summary(self, account_id: str) -> Dict[str, Any]:
        """Get summary statistics for an account"""
        with self.get_connection() as conn:
            # Get date range
            date_range = conn.execute("""
                SELECT MIN(date) as first_date, MAX(date) as last_date,
                       COUNT(DISTINCT date) as trading_days
                FROM daily_summary
                WHERE account_id = ?
            """, (account_id,)).fetchone()

            # Get P&L summary
            pl_summary = conn.execute("""
                SELECT SUM(net_pl) as total_pl,
                       AVG(net_pl) as avg_daily_pl,
                       MAX(net_pl) as best_day,
                       MIN(net_pl) as worst_day
                FROM (
                    SELECT date, SUM(net_pl) as net_pl
                    FROM daily_summary
                    WHERE account_id = ?
                    GROUP BY date
                )
            """, (account_id,)).fetchone()

            # Get trade statistics
            trade_stats = conn.execute("""
                SELECT COUNT(*) as total_fills,
                       COUNT(DISTINCT symbol) as unique_symbols,
                       SUM(total_fees) as total_fees
                FROM fills
                WHERE account_id = ?
            """, (account_id,)).fetchone()

        return {
            'account_id': account_id,
            'first_date': date_range['first_date'],
            'last_date': date_range['last_date'],
            'trading_days': date_range['trading_days'],
            'total_pl': pl_summary['total_pl'] or 0,
            'avg_daily_pl': pl_summary['avg_daily_pl'] or 0,
            'best_day': pl_summary['best_day'] or 0,
            'worst_day': pl_summary['worst_day'] or 0,
            'total_fills': trade_stats['total_fills'] or 0,
            'unique_symbols': trade_stats['unique_symbols'] or 0,
            'total_fees': trade_stats['total_fees'] or 0
        }

    def record_data_load(self, account_id: str, data_type: str,
                        start_date: date, end_date: date, records: int) -> None:
        """Record data load for tracking"""
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO data_loads (account_id, data_type, start_date, end_date, records_loaded)
                VALUES (?, ?, ?, ?, ?)
            """, (account_id, data_type, start_date, end_date, records))
            conn.commit()

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        with self.get_connection() as conn:
            stats = {}

            # Table sizes
            for table in ['accounts', 'daily_summary', 'fills', 'data_loads']:
                count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                stats[f'{table}_count'] = count

            # Date ranges
            date_range = conn.execute("""
                SELECT MIN(date) as min_date, MAX(date) as max_date
                FROM daily_summary
            """).fetchone()

            stats['date_range'] = f"{date_range['min_date']} to {date_range['max_date']}" if date_range['min_date'] else "No data"

            # Database size
            stats['db_size_mb'] = self.db_path.stat().st_size / (1024 * 1024) if self.db_path.exists() else 0

        return stats

    def remove_duplicates(self) -> Dict[str, int]:
        """
        Remove duplicate records from database

        Returns:
            Dictionary with number of duplicates removed per table
        """
        removed = {}

        with self.get_connection() as conn:
            # Remove duplicate daily summaries (keep the most recent insert)
            result = conn.execute("""
                DELETE FROM daily_summary
                WHERE id NOT IN (
                    SELECT MAX(id)
                    FROM daily_summary
                    GROUP BY account_id, date, symbol
                )
            """)
            removed['daily_summary'] = result.rowcount

            # Remove duplicate fills
            # First by fill_id (if available)
            result = conn.execute("""
                DELETE FROM fills
                WHERE id NOT IN (
                    SELECT MIN(id)
                    FROM fills
                    WHERE fill_id IS NOT NULL
                    GROUP BY account_id, fill_id
                )
                AND fill_id IS NOT NULL
            """)
            removed['fills_by_fill_id'] = result.rowcount

            # Then by transaction details for fills without fill_id
            result = conn.execute("""
                DELETE FROM fills
                WHERE id NOT IN (
                    SELECT MIN(id)
                    FROM fills
                    GROUP BY account_id, datetime, symbol, quantity, price, side
                )
            """)
            removed['fills_by_details'] = result.rowcount

            conn.commit()

        logger.info(f"Removed duplicates: {removed}")
        return removed

    def check_duplicates(self) -> Dict[str, pd.DataFrame]:
        """
        Check for duplicate records in database

        Returns:
            Dictionary with DataFrames showing duplicates per table
        """
        duplicates = {}

        with self.get_connection() as conn:
            # Check daily_summary duplicates
            daily_dupes = pd.read_sql_query("""
                SELECT account_id, date, symbol, COUNT(*) as duplicate_count
                FROM daily_summary
                GROUP BY account_id, date, symbol
                HAVING COUNT(*) > 1
                ORDER BY duplicate_count DESC
            """, conn)

            if not daily_dupes.empty:
                duplicates['daily_summary'] = daily_dupes

            # Check fills duplicates by fill_id
            fill_dupes_by_id = pd.read_sql_query("""
                SELECT account_id, fill_id, COUNT(*) as duplicate_count
                FROM fills
                WHERE fill_id IS NOT NULL
                GROUP BY account_id, fill_id
                HAVING COUNT(*) > 1
                ORDER BY duplicate_count DESC
            """, conn)

            if not fill_dupes_by_id.empty:
                duplicates['fills_by_fill_id'] = fill_dupes_by_id

            # Check fills duplicates by transaction details
            fill_dupes_by_details = pd.read_sql_query("""
                SELECT account_id, datetime, symbol, quantity, price, side,
                       COUNT(*) as duplicate_count
                FROM fills
                GROUP BY account_id, datetime, symbol, quantity, price, side
                HAVING COUNT(*) > 1
                ORDER BY duplicate_count DESC
                LIMIT 20
            """, conn)

            if not fill_dupes_by_details.empty:
                duplicates['fills_by_details'] = fill_dupes_by_details

        return duplicates

    def check_time_order_issues(self) -> Dict[str, pd.DataFrame]:
        """
        Check for time order issues in the database

        Returns:
            Dictionary with DataFrames showing time order problems
        """
        issues = {}

        with self.get_connection() as conn:
            # Check if daily_summary has proper date ordering
            # This query finds cases where a later id has an earlier date
            daily_issues = pd.read_sql_query("""
                SELECT
                    t1.id as id1,
                    t1.account_id,
                    t1.date as date1,
                    t1.symbol,
                    t2.id as id2,
                    t2.date as date2
                FROM daily_summary t1
                JOIN daily_summary t2
                    ON t1.account_id = t2.account_id
                    AND t1.symbol = t2.symbol
                    AND t1.id > t2.id
                    AND t1.date < t2.date
                LIMIT 100
            """, conn)

            if not daily_issues.empty:
                issues['daily_summary_order'] = daily_issues

            # Check fills for time order issues
            fills_issues = pd.read_sql_query("""
                SELECT
                    t1.id as id1,
                    t1.account_id,
                    t1.datetime as datetime1,
                    t1.symbol,
                    t2.id as id2,
                    t2.datetime as datetime2
                FROM fills t1
                JOIN fills t2
                    ON t1.account_id = t2.account_id
                    AND t1.id > t2.id
                    AND t1.datetime < t2.datetime
                LIMIT 100
            """, conn)

            if not fills_issues.empty:
                issues['fills_order'] = fills_issues

        return issues

    def create_time_ordered_views(self):
        """
        Create views that always return data in proper time order
        """
        with self.get_connection() as conn:
            # Create ordered view for daily summaries
            conn.execute("""
                CREATE VIEW IF NOT EXISTS daily_summary_ordered AS
                SELECT * FROM daily_summary
                ORDER BY account_id, date, symbol
            """)

            # Create ordered view for fills
            conn.execute("""
                CREATE VIEW IF NOT EXISTS fills_ordered AS
                SELECT * FROM fills
                ORDER BY account_id, datetime
            """)

            # Create a view for aggregated daily P&L
            conn.execute("""
                CREATE VIEW IF NOT EXISTS daily_pnl AS
                SELECT
                    account_id,
                    date,
                    SUM(net_pl) as total_net_pl,
                    SUM(gross_pl) as total_gross_pl,
                    SUM(trades) as total_trades,
                    COUNT(DISTINCT symbol) as symbols_traded
                FROM daily_summary
                GROUP BY account_id, date
                ORDER BY account_id, date
            """)

            conn.commit()
            logger.info("Created time-ordered views")

    def get_trader_time_series(self, account_id: str,
                               start_date: Optional[date] = None,
                               end_date: Optional[date] = None) -> pd.DataFrame:
        """
        Get properly ordered time series data for a trader
        Ensures data is ready for time series analysis
        """
        # Convert dates to strings if they're date objects
        if isinstance(start_date, date):
            start_date = start_date.strftime('%Y-%m-%d')
        if isinstance(end_date, date):
            end_date = end_date.strftime('%Y-%m-%d')

        # Get daily data
        daily_data = self.get_daily_summary(
            account_id=account_id,
            start_date=start_date,
            end_date=end_date
        )

        if daily_data.empty:
            return pd.DataFrame()

        # Aggregate by date
        time_series = daily_data.groupby('date').agg({
            'net_pl': 'sum',
            'gross_pl': 'sum',
            'trades': 'sum',
            'orders': 'sum',
            'fills': 'sum',
            'volume': 'sum',
            'symbol': 'count'  # Number of symbols traded
        }).rename(columns={'symbol': 'symbols_traded'})

        # Ensure continuous time series (fill missing trading days)
        if len(time_series) > 1:
            # Create date range for all trading days
            date_range = pd.date_range(
                start=time_series.index.min(),
                end=time_series.index.max(),
                freq='B'  # Business days
            )

            # Reindex to include all trading days
            time_series = time_series.reindex(date_range)

            # Fill missing values appropriately
            fill_values = {
                'net_pl': 0,
                'gross_pl': 0,
                'trades': 0,
                'orders': 0,
                'fills': 0,
                'volume': 0,
                'symbols_traded': 0
            }
            time_series = time_series.fillna(fill_values)

            # Add cumulative columns
            time_series['cumulative_pl'] = time_series['net_pl'].cumsum()
            time_series['trading_days'] = (time_series['trades'] > 0).cumsum()

        return time_series

    def validate_time_consistency(self, account_id: str) -> Dict[str, Any]:
        """
        Validate time consistency for an account

        Returns:
            Dictionary with validation results
        """
        validation = {
            'is_valid': True,
            'issues': [],
            'stats': {}
        }

        # Get time series
        ts = self.get_trader_time_series(account_id)

        if ts.empty:
            validation['is_valid'] = False
            validation['issues'].append("No data found for account")
            return validation

        # Check for gaps
        expected_days = pd.bdate_range(ts.index.min(), ts.index.max())
        missing_days = expected_days.difference(ts.index)

        if len(missing_days) > 0:
            validation['stats']['missing_trading_days'] = len(missing_days)
            if len(missing_days) > len(expected_days) * 0.1:  # More than 10% missing
                validation['issues'].append(f"Missing {len(missing_days)} trading days")

        # Check for duplicate dates in raw data
        daily_data = self.get_daily_summary(account_id=account_id)
        date_counts = daily_data.groupby(['date', 'symbol']).size()
        duplicates = date_counts[date_counts > 1]

        if not duplicates.empty:
            validation['is_valid'] = False
            validation['issues'].append(f"Found {len(duplicates)} duplicate date/symbol combinations")
            validation['stats']['duplicates'] = duplicates.to_dict()

        # Check time order consistency
        if not daily_data['date'].is_monotonic_increasing:
            validation['issues'].append("Data is not properly time-ordered")

        validation['stats']['date_range'] = (ts.index.min(), ts.index.max())
        validation['stats']['total_days'] = len(ts)
        validation['stats']['trading_days'] = (ts['trades'] > 0).sum()

        return validation
