"""Repository for trader data access."""

import sqlite3
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import logging

from .base import Repository
from ..models.domain import Trader, TradingMetrics

logger = logging.getLogger(__name__)


class TraderRepository(Repository):
    """Repository for managing trader data access."""

    def __init__(self, db_path: str):
        """Initialize repository with database path."""
        self.db_path = db_path

    def find_by_id(self, trader_id: int) -> Optional[Trader]:
        """Find trader by ID."""
        query = """
        SELECT account_id, account_name, created_at, updated_at
        FROM accounts
        WHERE account_id = ?
        """

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, (trader_id,))
            row = cursor.fetchone()

            if row:
                return Trader(
                    id=row[0],
                    name=row[1] if row[1] else f"Trader {row[0]}",
                    created_at=row[2],
                    updated_at=row[3]
                )
        return None

    def find_all(self) -> List[Trader]:
        """Get all traders."""
        query = """
        SELECT account_id, account_name, created_at, updated_at
        FROM accounts
        ORDER BY account_id
        """

        traders = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query)

            for row in cursor.fetchall():
                traders.append(Trader(
                    id=row[0],
                    name=row[1] if row[1] else f"Trader {row[0]}",
                    created_at=row[2],
                    updated_at=row[3]
                ))

        return traders

    def get_trader_fills(self, trader_id: int,
                        start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Get trader fills within date range."""
        query = """
        SELECT * FROM fills
        WHERE account = ?
        """
        params = [trader_id]

        if start_date:
            query += " AND date >= ?"
            params.append(start_date.strftime('%Y-%m-%d'))

        if end_date:
            query += " AND date <= ?"
            params.append(end_date.strftime('%Y-%m-%d'))

        query += " ORDER BY date"

        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=params)

    def get_trader_metrics(self, trader_id: int,
                          lookback_days: int = 30) -> Optional[TradingMetrics]:
        """Calculate trading metrics for a trader."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        fills = self.get_trader_fills(trader_id, start_date, end_date)

        if fills.empty:
            return None

        # Calculate metrics
        total_trades = len(fills)
        winning_trades = len(fills[fills['pnl'] > 0])
        losing_trades = len(fills[fills['pnl'] < 0])

        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        wl_ratio = (winning_trades / losing_trades) if losing_trades > 0 else float('inf')

        # Calculate Sharpe ratio
        if 'pnl' in fills.columns and len(fills) > 1:
            daily_returns = fills.groupby('date')['pnl'].sum()
            if len(daily_returns) > 1 and daily_returns.std() > 0:
                sharpe = (daily_returns.mean() / daily_returns.std()) * (252 ** 0.5)
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0

        return TradingMetrics(
            bat_30d=win_rate,
            wl_ratio=wl_ratio,
            sharpe=sharpe,
            total_trades=total_trades,
            total_pnl=fills['pnl'].sum() if 'pnl' in fills.columns else 0.0
        )

    def get_active_traders(self, min_trades: int = 10,
                          days: int = 30) -> List[int]:
        """Get list of active trader IDs."""
        query = """
        SELECT account, COUNT(*) as trade_count
        FROM fills
        WHERE date >= date('now', '-' || ? || ' days')
        GROUP BY account
        HAVING trade_count >= ?
        ORDER BY account
        """

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, (days, min_trades))
            return [row[0] for row in cursor.fetchall()]

    def save(self, trader: Trader) -> Trader:
        """Save or update trader."""
        query = """
        INSERT OR REPLACE INTO accounts (account_id, account_name, created_at, updated_at)
        VALUES (?, ?, ?, ?)
        """

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, (
                trader.id,
                trader.name,
                trader.created_at,
                datetime.now().isoformat()
            ))
            conn.commit()

        return trader

    def delete(self, trader_id: int) -> bool:
        """Delete trader (not recommended for audit trail)."""
        logger.warning(f"Attempting to delete trader {trader_id}")
        return False  # Prevent deletion for data integrity

    def exists(self, trader_id: int) -> bool:
        """Check if trader exists."""
        query = "SELECT 1 FROM accounts WHERE account_id = ? LIMIT 1"

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, (trader_id,))
            return cursor.fetchone() is not None
