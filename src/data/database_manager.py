"""
Database Manager for Risk Tool
Handles all database operations with proper schema for PropreReports data
Updated to support new summaryByDate format
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

        # Optimize for performance with large datasets
        conn.execute("PRAGMA cache_size = -64000")  # 64MB cache
        conn.execute("PRAGMA temp_store = MEMORY")
        conn.execute("PRAGMA journal_mode = WAL")    # Write-Ahead Logging
        conn.execute("PRAGMA synchronous = NORMAL")  # Faster writes

        try:
            yield conn
        finally:
            conn.close()

    def _init_database(self):
        """Initialize database with proper schema"""
        with self.get_connection() as conn:
            # Enable foreign keys and optimize settings
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA busy_timeout = 5000")  # 5 second timeout for locks

            # Create tables
            conn.executescript("""
                -- Accounts table
                CREATE TABLE IF NOT EXISTS accounts (
                    account_id TEXT PRIMARY KEY,
                    account_name TEXT,
                    account_type TEXT DEFAULT NULL,  -- Cached account type
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                -- Account daily summary table (from summaryByDate reports)
                -- Includes all possible columns for both equities and options accounts
                CREATE TABLE IF NOT EXISTS account_daily_summary (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    account_id TEXT NOT NULL,
                    date DATE NOT NULL,
                    type TEXT DEFAULT 'Eq',
                    orders INTEGER DEFAULT 0,
                    fills INTEGER DEFAULT 0,
                    qty INTEGER DEFAULT 0,
                    gross REAL DEFAULT 0,
                    comm REAL DEFAULT 0,
                    ecn_fee REAL DEFAULT 0,
                    sec REAL DEFAULT 0,
                    orf REAL DEFAULT 0,
                    cat REAL DEFAULT 0,
                    taf REAL DEFAULT 0,
                    ftt REAL DEFAULT 0,
                    nscc REAL DEFAULT 0,
                    acc REAL DEFAULT 0,
                    clr REAL DEFAULT 0,
                    misc REAL DEFAULT 0,
                    trade_fees REAL DEFAULT 0,
                    net REAL DEFAULT 0,
                    fee_software_md REAL DEFAULT NULL,      -- Equities only
                    fee_vat REAL DEFAULT NULL,              -- Equities only
                    fee_daily_interest REAL DEFAULT NULL,   -- Options only
                    adj_fees REAL DEFAULT 0,
                    adj_net REAL DEFAULT 0,
                    unrealized_delta REAL DEFAULT 0,
                    total_delta REAL DEFAULT 0,
                    transfer_deposit REAL DEFAULT 0,
                    transfers REAL DEFAULT 0,
                    cash REAL DEFAULT 0,
                    unrealized REAL DEFAULT 0,
                    end_balance REAL DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(account_id, date),
                    FOREIGN KEY (account_id) REFERENCES accounts(account_id)
                );

                -- Fills table (individual trades)
                CREATE TABLE IF NOT EXISTS fills (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    account_id TEXT NOT NULL,
                    datetime TIMESTAMP NOT NULL,
                    side TEXT CHECK(side IN ('B', 'S', 'T')),
                    quantity INTEGER NOT NULL,
                    symbol TEXT DEFAULT 'MISSING',
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
                CREATE INDEX IF NOT EXISTS idx_summary_date ON account_daily_summary(date);
                CREATE INDEX IF NOT EXISTS idx_summary_account ON account_daily_summary(account_id);
                CREATE INDEX IF NOT EXISTS idx_summary_account_date ON account_daily_summary(account_id, date);
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

    def save_account(self, account_id: str, account_name: str, account_type: Optional[str] = None) -> None:
        """Save or update account information"""
        with self.get_connection() as conn:
            if account_type:
                conn.execute("""
                    INSERT OR REPLACE INTO accounts (account_id, account_name, account_type, updated_at)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                """, (account_id, account_name, account_type))
            else:
                conn.execute("""
                    INSERT OR REPLACE INTO accounts (account_id, account_name, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                """, (account_id, account_name))
            conn.commit()

    def update_account_type(self, account_id: str, force_refresh: bool = False) -> str:
        """
        Update account type in database

        Args:
            account_id: Account ID
            force_refresh: Force detection even if already cached

        Returns:
            Detected account type
        """
        with self.get_connection() as conn:
            # Check if we already have a cached type
            if not force_refresh:
                cached = conn.execute(
                    "SELECT account_type FROM accounts WHERE account_id = ?",
                    (account_id,)
                ).fetchone()

                if cached and cached['account_type']:
                    return cached['account_type']

            # Detect account type
            account_type = self.detect_account_type(account_id)

            # Update cache
            conn.execute("""
                UPDATE accounts
                SET account_type = ?, updated_at = CURRENT_TIMESTAMP
                WHERE account_id = ?
            """, (account_type, account_id))
            conn.commit()

            return account_type

    def save_account_daily_summary(self, df: pd.DataFrame, account_id: str,
                                 handle_duplicates: str = 'replace') -> int:
        """
        Save account daily summary data (summaryByDate)

        Args:
            df: DataFrame with daily summary
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

        # Ensure proper column names - handle both equities and options columns
        column_mapping = {
            'Date': 'date',
            'Type': 'type',
            'Orders': 'orders',
            'Fills': 'fills',
            'Qty': 'qty',
            'Gross': 'gross',
            'Comm': 'comm',
            'Ecn Fee': 'ecn_fee',
            'SEC': 'sec',
            'ORF': 'orf',
            'CAT': 'cat',
            'TAF': 'taf',
            'FTT': 'ftt',
            'NSCC': 'nscc',
            'Acc': 'acc',
            'Clr': 'clr',
            'Misc': 'misc',
            'Trade Fees': 'trade_fees',
            'Net': 'net',
            'Fee: Software & MD': 'fee_software_md',      # Equities
            'Fee: VAT': 'fee_vat',                        # Equities
            'Fee: Daily Interest': 'fee_daily_interest',  # Options
            'Adj Fees': 'adj_fees',
            'Adj Net': 'adj_net',
            'Unrealized Δ': 'unrealized_delta',
            'Total Δ': 'total_delta',
            'Transfer: Deposit': 'transfer_deposit',
            'Transfers': 'transfers',
            'Cash': 'cash',
            'Unrealized': 'unrealized',
            'End Balance': 'end_balance'
        }

        df = df.rename(columns=column_mapping)

        # Select only columns that exist in the dataframe
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
                if 'date' in df.columns:
                    dates = df['date'].unique()
                    # Process in batches to avoid too many SQL variables
                    batch_size = 500
                    for i in range(0, len(dates), batch_size):
                        batch_dates = dates[i:i + batch_size]
                        placeholders = ','.join(['?' for _ in batch_dates])
                        conn.execute(
                            f"DELETE FROM account_daily_summary WHERE account_id = ? AND date IN ({placeholders})",
                            [account_id] + list(batch_dates)
                        )

                # Insert new records in batches
                num_columns = len(df.columns)
                batch_size = min(100, 900 // num_columns)  # Conservative batch size

                for i in range(0, len(df), batch_size):
                    batch_df = df.iloc[i:i + batch_size]
                    batch_df.to_sql('account_daily_summary', conn, if_exists='append', index=False, method='multi')
                    records_saved += len(batch_df)

            elif handle_duplicates == 'ignore':
                # Insert only new records
                for _, row in df.iterrows():
                    try:
                        row_dict = row.to_dict()
                        columns = ', '.join(row_dict.keys())
                        placeholders = ', '.join(['?' for _ in row_dict])

                        conn.execute(f"""
                            INSERT OR IGNORE INTO account_daily_summary ({columns})
                            VALUES ({placeholders})
                        """, list(row_dict.values()))

                        if conn.total_changes > 0:
                            records_saved += 1
                    except Exception as e:
                        logger.debug(f"Error inserting row: {e}")

            elif handle_duplicates == 'error':
                # Raise error on duplicates - use batching
                num_columns = len(df.columns)
                batch_size = min(100, 900 // num_columns)

                for i in range(0, len(df), batch_size):
                    batch_df = df.iloc[i:i + batch_size]
                    batch_df.to_sql('account_daily_summary', conn, if_exists='append', index=False, method='multi')
                    records_saved += len(batch_df)

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

        # Convert datetime
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
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

                    # Process fill_ids in batches to avoid too many SQL variables
                    unique_fill_ids = df['fill_id'].dropna().unique()
                    batch_size = 500

                    for i in range(0, len(unique_fill_ids), batch_size):
                        batch_ids = unique_fill_ids[i:i + batch_size]
                        placeholders = ','.join(['?' for _ in batch_ids])
                        query = f"""
                            SELECT fill_id FROM fills
                            WHERE fill_id IN ({placeholders})
                            AND account_id = ?
                        """
                        params = list(batch_ids) + [account_id]
                        results = conn.execute(query, params).fetchall()
                        existing_fills.update(row[0] for row in results)

                    if existing_fills:
                        logger.info(f"Found {len(existing_fills)} duplicate fill_ids, skipping...")
                        df = df[~df['fill_id'].isin(existing_fills)]

            if not df.empty:
                # Batch inserts to avoid "too many SQL variables" error
                # SQLite limit is typically 999 variables, so calculate batch size based on columns
                num_columns = len(available_cols)
                batch_size = min(500, 900 // num_columns)  # Conservative batch size

                for i in range(0, len(df), batch_size):
                    batch_df = df.iloc[i:i + batch_size]
                    batch_df.to_sql('fills', conn, if_exists='append', index=False, method='multi')
                    records_saved += len(batch_df)

                    if i + batch_size < len(df):
                        logger.debug(f"Saved batch {i//batch_size + 1}, {records_saved}/{len(df)} fills")

            conn.commit()

        logger.info(f"Saved {records_saved} fill records for account {account_id}")
        return records_saved

    def get_account_daily_summary(self,
                                account_id: Optional[str] = None,
                                start_date: Optional[date] = None,
                                end_date: Optional[date] = None) -> pd.DataFrame:
        """Get account daily summary data with filters - ALWAYS sorted by date"""

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

        # ALWAYS sort by date for consistent ordering
        query += " ORDER BY account_id, date"

        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)

        # Convert date column
        if not df.empty and 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values(['account_id', 'date'])

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
            df = df.sort_values(['account_id', 'datetime'])

        return df

    def get_accounts(self) -> pd.DataFrame:
        """Get all accounts"""
        with self.get_connection() as conn:
            df = pd.read_sql_query("""
                SELECT
                    account_id,
                    account_name,
                    account_type,
                    created_at,
                    updated_at
                FROM accounts
                ORDER BY account_id
            """, conn)

            # Update account types if missing
            for idx, row in df.iterrows():
                if pd.isna(row['account_type']) or row['account_type'] is None:
                    account_type = self.update_account_type(row['account_id'])
                    df.at[idx, 'account_type'] = account_type

            return df

    def get_account_summary(self, account_id: str) -> Dict[str, Any]:
        """Get summary statistics for an account"""
        with self.get_connection() as conn:
            # Get date range
            date_range = conn.execute("""
                SELECT MIN(date) as first_date, MAX(date) as last_date,
                       COUNT(DISTINCT date) as trading_days
                FROM account_daily_summary
                WHERE account_id = ?
            """, (account_id,)).fetchone()

            # Get P&L summary
            pl_summary = conn.execute("""
                SELECT SUM(net) as total_pl,
                       AVG(net) as avg_daily_pl,
                       MAX(net) as best_day,
                       MIN(net) as worst_day
                FROM account_daily_summary
                WHERE account_id = ?
            """, (account_id,)).fetchone()

            # Get trade statistics
            trade_stats = conn.execute("""
                SELECT COUNT(*) as total_fills,
                       COUNT(DISTINCT symbol) as unique_symbols,
                       SUM(total_fees) as total_fees
                FROM fills
                WHERE account_id = ?
            """, (account_id,)).fetchone()

            # Detect account type using improved method
            account_type = self.detect_account_type(account_id)

        return {
            'account_id': account_id,
            'account_type': account_type,
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

    def detect_account_type(self, account_id: str) -> str:
        """
        Detect account type using multiple indicators

        Priority:
        1. Type column in daily summary ('Op' = Options, 'Eq' = Equities)
        2. Symbol patterns in fills (options have specific formats)
        3. Fee columns as fallback
        """
        with self.get_connection() as conn:
            # First, check the Type column - most reliable
            type_check = conn.execute("""
                SELECT type, COUNT(*) as count
                FROM account_daily_summary
                WHERE account_id = ?
                GROUP BY type
                ORDER BY count DESC
                LIMIT 1
            """, (account_id,)).fetchone()

            if type_check and type_check['type']:
                account_type = type_check['type'].strip().upper()
                if account_type in ['OP', 'OPTIONS']:
                    return 'Options'
                elif account_type in ['EQ', 'EQUITIES']:
                    return 'Equities'

            # Second, check symbols in fills for options patterns
            # Options symbols typically have formats like: SPY210319C00380000, AAPL 210319C00125000
            symbol_check = conn.execute("""
                SELECT symbol, COUNT(*) as trade_count
                FROM fills
                WHERE account_id = ?
                    AND symbol IS NOT NULL
                GROUP BY symbol
                ORDER BY trade_count DESC
                LIMIT 10
            """, (account_id,)).fetchall()

            if symbol_check:
                options_patterns = 0
                equity_patterns = 0

                for row in symbol_check:
                    symbol = str(row['symbol']).strip()
                    # Check for options patterns:
                    # - Contains numbers in middle/end (expiration date)
                    # - Contains 'C' or 'P' (Call/Put)
                    # - Length > 10 (options symbols are longer)
                    if (len(symbol) > 10 and
                        any(c in symbol for c in ['C', 'P']) and
                        any(char.isdigit() for char in symbol[3:])):  # digits after first few chars
                        options_patterns += row['trade_count']
                    else:
                        equity_patterns += row['trade_count']

                # If more than 70% of trades look like options
                if options_patterns > 0 and options_patterns / (options_patterns + equity_patterns) > 0.7:
                    return 'Options'
                elif equity_patterns > options_patterns:
                    return 'Equities'

            # Third, check fee columns as last resort
            fee_check = conn.execute("""
                SELECT
                    CASE
                        WHEN SUM(fee_daily_interest) > 0 THEN 'Options'
                        WHEN SUM(fee_software_md) > 0 OR SUM(fee_vat) > 0 THEN 'Equities'
                        ELSE 'Unknown'
                    END as account_type
                FROM account_daily_summary
                WHERE account_id = ?
            """, (account_id,)).fetchone()

            if fee_check and fee_check['account_type'] != 'Unknown':
                return fee_check['account_type']

            return 'Unknown'

    def parse_options_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Parse options symbol to extract components

        Common formats:
        - OCC format: AAPL210319C00125000 (AAPL, 2021-03-19, Call, $125.00)
        - Some brokers: AAPL 210319C00125000 or AAPL_210319C125

        Returns dict with: underlying, expiration, type, strike
        """
        import re

        symbol = symbol.strip().upper()

        # Try different patterns
        patterns = [
            # OCC format: AAPL210319C00125000
            r'^([A-Z]+)(\d{6})([CP])(\d+)$',
            # With space: AAPL 210319C00125000
            r'^([A-Z]+)\s+(\d{6})([CP])(\d+)$',
            # Underscore format: AAPL_210319C125
            r'^([A-Z]+)_(\d{6})([CP])(\d+)$',
        ]

        for pattern in patterns:
            match = re.match(pattern, symbol)
            if match:
                underlying = match.group(1)
                date_str = match.group(2)
                option_type = 'Call' if match.group(3) == 'C' else 'Put'
                strike_str = match.group(4)

                # Parse date (YYMMDD format)
                try:
                    year = 2000 + int(date_str[:2])
                    month = int(date_str[2:4])
                    day = int(date_str[4:6])
                    expiration = f"{year}-{month:02d}-{day:02d}"
                except:
                    expiration = date_str

                # Parse strike price
                try:
                    # OCC format has 8 digits with 3 decimal places
                    if len(strike_str) == 8:
                        strike = float(strike_str) / 1000
                    else:
                        strike = float(strike_str)
                except:
                    strike = strike_str

                return {
                    'underlying': underlying,
                    'expiration': expiration,
                    'type': option_type,
                    'strike': strike,
                    'original': symbol
                }

        return None

    def get_symbol_statistics(self, account_id: str) -> Dict[str, Any]:
        """
        Get symbol trading statistics for an account
        Useful for understanding trading patterns and account type
        """
        with self.get_connection() as conn:
            # Get top traded symbols
            top_symbols = pd.read_sql_query("""
                SELECT
                    symbol,
                    COUNT(*) as trade_count,
                    SUM(quantity) as total_volume,
                    AVG(price) as avg_price,
                    MIN(datetime) as first_trade,
                    MAX(datetime) as last_trade
                FROM fills
                WHERE account_id = ?
                GROUP BY symbol
                ORDER BY trade_count DESC
                LIMIT 20
            """, conn, params=[account_id])

            # Analyze symbol patterns
            options_symbols = []
            equity_symbols = []

            for _, row in top_symbols.iterrows():
                symbol = str(row['symbol'])
                options_data = self.parse_options_symbol(symbol)

                if options_data:
                    options_symbols.append({
                        'symbol': symbol,
                        'underlying': options_data['underlying'],
                        'type': options_data['type'],
                        'strike': options_data['strike'],
                        'expiration': options_data['expiration'],
                        'trades': row['trade_count']
                    })
                else:
                    equity_symbols.append({
                        'symbol': symbol,
                        'trades': row['trade_count']
                    })

            # Get trading style metrics
            style_metrics = conn.execute("""
                SELECT
                    AVG(quantity) as avg_trade_size,
                    COUNT(DISTINCT DATE(datetime)) as trading_days,
                    COUNT(DISTINCT symbol) as unique_symbols,
                    COUNT(*) as total_trades,
                    COUNT(CASE WHEN side = 'B' THEN 1 END) as buy_trades,
                    COUNT(CASE WHEN side = 'S' THEN 1 END) as sell_trades
                FROM fills
                WHERE account_id = ?
            """, (account_id,)).fetchone()

            # Get underlying distribution for options
            underlying_dist = {}
            if options_symbols:
                for opt in options_symbols:
                    underlying = opt['underlying']
                    if underlying not in underlying_dist:
                        underlying_dist[underlying] = {'calls': 0, 'puts': 0, 'total': 0}
                    underlying_dist[underlying]['total'] += opt['trades']
                    if opt['type'] == 'Call':
                        underlying_dist[underlying]['calls'] += opt['trades']
                    else:
                        underlying_dist[underlying]['puts'] += opt['trades']

            return {
                'top_symbols': top_symbols.to_dict('records') if not top_symbols.empty else [],
                'symbol_analysis': {
                    'options_symbols': len(options_symbols),
                    'equity_symbols': len(equity_symbols),
                    'likely_type': 'Options' if len(options_symbols) > len(equity_symbols) else 'Equities',
                    'options_details': options_symbols[:10],  # Top 10 options
                    'equity_details': equity_symbols[:10],   # Top 10 equities
                    'underlying_distribution': underlying_dist
                },
                'trading_metrics': dict(style_metrics) if style_metrics else {}
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
            for table in ['accounts', 'account_daily_summary', 'fills', 'data_loads']:
                count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                stats[f'{table}_count'] = count

            # Date ranges
            date_range = conn.execute("""
                SELECT MIN(date) as min_date, MAX(date) as max_date
                FROM account_daily_summary
            """).fetchone()

            stats['date_range'] = f"{date_range['min_date']} to {date_range['max_date']}" if date_range['min_date'] else "No data"

            # Database size
            stats['db_size_mb'] = self.db_path.stat().st_size / (1024 * 1024) if self.db_path.exists() else 0

        return stats

    def get_download_status(self) -> pd.DataFrame:
        """Get status of data downloads"""
        with self.get_connection() as conn:
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

    def get_data_summary(self) -> Dict[str, pd.DataFrame]:
        """Get summary of downloaded data"""
        with self.get_connection() as conn:
            # Summary by trader
            trader_summary = pd.read_sql_query("""
                SELECT
                    a.account_id,
                    a.account_name,
                    COALESCE(a.account_type, 'Unknown') as account_type,
                    COUNT(DISTINCT ads.date) as trading_days,
                    MIN(ads.date) as first_date,
                    MAX(ads.date) as last_date,
                    SUM(ads.net) as total_pnl,
                    COUNT(DISTINCT f.fill_id) as total_fills
                FROM accounts a
                LEFT JOIN account_daily_summary ads ON a.account_id = ads.account_id
                LEFT JOIN fills f ON a.account_id = f.account_id
                GROUP BY a.account_id, a.account_name, a.account_type
                ORDER BY a.account_name
            """, conn)

            # Update account types if missing
            for idx, row in trader_summary.iterrows():
                if row['account_type'] == 'Unknown':
                    account_type = self.update_account_type(row['account_id'])
                    trader_summary.at[idx, 'account_type'] = account_type

            # Recent activity
            recent_activity = pd.read_sql_query("""
                SELECT
                    DATE(loaded_at) as load_date,
                    COUNT(DISTINCT account_id) as traders_updated,
                    SUM(records_loaded) as total_records,
                    COUNT(DISTINCT data_type) as data_types
                FROM data_loads
                WHERE loaded_at >= datetime('now', '-7 days')
                GROUP BY DATE(loaded_at)
                ORDER BY load_date DESC
            """, conn)

            return {
                'trader_summary': trader_summary,
                'recent_activity': recent_activity
            }

    def get_trader_time_series(self, account_id: str,
                               start_date: Optional[date] = None,
                               end_date: Optional[date] = None) -> pd.DataFrame:
        """
        Get properly ordered time series data for a trader
        """
        # Convert dates to strings if they're date objects
        if isinstance(start_date, date):
            start_date = start_date.strftime('%Y-%m-%d')
        if isinstance(end_date, date):
            end_date = end_date.strftime('%Y-%m-%d')

        # Get daily data
        daily_data = self.get_account_daily_summary(
            account_id=account_id,
            start_date=start_date,
            end_date=end_date
        )

        if daily_data.empty:
            return pd.DataFrame()

        # Set date as index
        daily_data = daily_data.set_index('date')

        # Add derived columns
        daily_data['cumulative_pl'] = daily_data['net'].cumsum()
        daily_data['trading_days'] = (daily_data['fills'] > 0).cumsum()

        return daily_data

    def remove_duplicates(self) -> Dict[str, int]:
        """Remove duplicate records from database"""
        removed = {}

        with self.get_connection() as conn:
            # Remove duplicate daily summaries
            result = conn.execute("""
                DELETE FROM account_daily_summary
                WHERE id NOT IN (
                    SELECT MAX(id)
                    FROM account_daily_summary
                    GROUP BY account_id, date
                )
            """)
            removed['account_daily_summary'] = result.rowcount

            # Remove duplicate fills
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

            conn.commit()

        logger.info(f"Removed duplicates: {removed}")
        return removed

    def create_time_ordered_views(self):
        """Create views that always return data in proper time order"""
        with self.get_connection() as conn:
            # Create ordered view for account daily summaries
            conn.execute("""
                CREATE VIEW IF NOT EXISTS account_summary_ordered AS
                SELECT * FROM account_daily_summary
                ORDER BY account_id, date
            """)

            # Create ordered view for fills
            conn.execute("""
                CREATE VIEW IF NOT EXISTS fills_ordered AS
                SELECT * FROM fills
                ORDER BY account_id, datetime
            """)

            conn.commit()
            logger.info("Created time-ordered views")
