"""
Database Manager for Risk Tool
Handles core database operations with schema creation, data insertion, and simple queries
"""

import sqlite3
import pandas as pd
from pathlib import Path
from typing import Optional, List
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Simple database manager for trading data"""

    def __init__(self, db_path: str = "data/trading_risk.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True, parents=True)
        self._create_schemas()

    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _create_schemas(self):
        """Create database schemas"""
        with self.get_connection() as conn:
            conn.executescript("""
                -- Accounts table
                CREATE TABLE IF NOT EXISTS accounts (
                    account_id TEXT PRIMARY KEY,
                    account_name TEXT,
                    account_type TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                -- Account daily summary table
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
                    fee_software_md REAL DEFAULT NULL,
                    fee_vat REAL DEFAULT NULL,
                    fee_daily_interest REAL DEFAULT NULL,
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

                -- Fills table
                CREATE TABLE IF NOT EXISTS fills (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    account_id TEXT NOT NULL,
                    datetime TIMESTAMP NOT NULL,
                    date DATE NOT NULL,
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
                    UNIQUE(account_id, date, fill_id, order_id),
                    FOREIGN KEY (account_id) REFERENCES accounts(account_id)
                );

                -- Indexes
                CREATE INDEX IF NOT EXISTS idx_summary_account_date ON account_daily_summary(account_id, date);
                CREATE INDEX IF NOT EXISTS idx_fills_account_datetime ON fills(account_id, datetime);
                CREATE INDEX IF NOT EXISTS idx_fills_symbol ON fills(symbol);
            """)
            conn.commit()

    def insert_account(self, account_id: str, account_name: str, account_type: str = None):
        """Insert or update account"""
        with self.get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO accounts (account_id, account_name, account_type)
                VALUES (?, ?, ?)
            """, (account_id, account_name, account_type))
            conn.commit()

    def insert_summary_data(self, df: pd.DataFrame, account_id: str):
        """Insert account daily summary data"""
        if df.empty:
            return 0

        df = df.copy()
        df['account_id'] = account_id

        # Column mapping for PropreReports data
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
            'Fee: Software & MD': 'fee_software_md',
            'Fee: VAT': 'fee_vat',
            'Fee: Daily Interest': 'fee_daily_interest',
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

        # Convert date column
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

        with self.get_connection() as conn:
            # Delete existing records for these dates first
            if 'date' in df.columns:
                dates = df['date'].unique()
                placeholders = ','.join(['?' for _ in dates])
                conn.execute(
                    f"DELETE FROM account_daily_summary WHERE account_id = ? AND date IN ({placeholders})",
                    [account_id] + list(dates)
                )

            # Insert new records
            df.to_sql('account_daily_summary', conn, if_exists='append', index=False)
            conn.commit()

        return len(df)

    def insert_fills_data(self, df: pd.DataFrame, account_id: str):
        """Insert fills data"""
        if df.empty:
            return 0

        df = df.copy()
        df['account_id'] = account_id

        # Column mapping for PropreReports fills data
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

        # Convert datetime and extract date
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df['date'] = df['datetime'].dt.date
            df['datetime'] = df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')

        with self.get_connection() as conn:
            try:
                df.to_sql('fills', conn, if_exists='append', index=False)
                conn.commit()
                return len(df)
            except sqlite3.IntegrityError:
                # Handle duplicates by inserting one by one
                records_saved = 0
                for _, row in df.iterrows():
                    try:
                        row.to_frame().T.to_sql('fills', conn, if_exists='append', index=False)
                        records_saved += 1
                    except sqlite3.IntegrityError:
                        pass  # Skip duplicates
                conn.commit()
                return records_saved

    def get_accounts(self) -> pd.DataFrame:
        """Get all accounts"""
        with self.get_connection() as conn:
            return pd.read_sql_query("""
                SELECT account_id, account_name, account_type, created_at
                FROM accounts
                ORDER BY account_id
            """, conn)

    def get_summary_data(self, account_id: Optional[str] = None,
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None) -> pd.DataFrame:
        """Get account daily summary data"""
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

        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)

        if not df.empty and 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])

        return df

    def get_fills_data(self, account_id: Optional[str] = None,
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None,
                      symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """Get fills data"""
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

        query += " ORDER BY account_id, datetime"

        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)

        if not df.empty and 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])

        return df
