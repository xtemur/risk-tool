import sqlite3
import pandas as pd
from datetime import datetime
from typing import Tuple, Optional

class DataLoader:
    def __init__(self, db_path: str = 'data/risk_tool.db'):
        self.db_path = db_path

    def load_trades_data(self, test_cutoff_date: str = '2025-04-01') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load trades data and split into train/test based on cutoff date.
        Focus only on realized PNL from closed trades.
        """
        conn = sqlite3.connect(self.db_path)

        query = """
        SELECT
            account_id,
            trade_date,
            net as realized_pnl,
            CASE WHEN net > 0 THEN 1 ELSE 0 END as is_winner
        FROM trades
        WHERE trade_date IS NOT NULL
        ORDER BY account_id, trade_date
        """

        df = pd.read_sql_query(query, conn)
        conn.close()

        # Convert trade_date to datetime
        df['trade_date'] = pd.to_datetime(df['trade_date'])

        # Split train/test based on date
        train_df = df[df['trade_date'] < test_cutoff_date].copy()
        test_df = df[df['trade_date'] >= test_cutoff_date].copy()

        print(f"Train data: {len(train_df)} trades from {train_df['trade_date'].min()} to {train_df['trade_date'].max()}")
        print(f"Test data: {len(test_df)} trades from {test_df['trade_date'].min()} to {test_df['trade_date'].max()}")
        print(f"Unique traders in train: {train_df['account_id'].nunique()}")
        print(f"Unique traders in test: {test_df['account_id'].nunique()}")

        return train_df, test_df

    def get_account_info(self) -> pd.DataFrame:
        """Load account information for reference."""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM accounts", conn)
        conn.close()
        return df
