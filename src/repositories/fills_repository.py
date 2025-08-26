"""Repository for fills/trades data access."""

import sqlite3
import pandas as pd
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging

from .base import Repository
from ..exceptions import DatabaseConnectionError

logger = logging.getLogger(__name__)


class FillsRepository(Repository):
    """Repository for managing fills/trades data."""

    def __init__(self, db_path: str):
        """Initialize repository with database path."""
        self.db_path = db_path

    def find_by_id(self, fill_id: int) -> Optional[Dict[str, Any]]:
        """Find a specific fill by ID."""
        query = "SELECT * FROM fills WHERE id = ?"

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(query, (fill_id,))
                row = cursor.fetchone()

                if row:
                    return dict(row)
                return None

        except sqlite3.Error as e:
            raise DatabaseConnectionError(self.db_path, e)

    def find_all(self) -> List[Dict[str, Any]]:
        """Get all fills (limited to recent for performance)."""
        # Limit to last 30 days by default
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        query = "SELECT * FROM fills WHERE date >= ? ORDER BY date DESC LIMIT 10000"

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(query, (start_date,))

                return [dict(row) for row in cursor.fetchall()]

        except sqlite3.Error as e:
            raise DatabaseConnectionError(self.db_path, e)

    def get_fills_by_trader(self,
                           trader_id: int,
                           start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None,
                           limit: Optional[int] = None) -> pd.DataFrame:
        """
        Get fills for a specific trader.

        Args:
            trader_id: The trader's account ID
            start_date: Optional start date filter
            end_date: Optional end date filter
            limit: Maximum number of records

        Returns:
            DataFrame with fills data
        """
        query = "SELECT * FROM fills WHERE account = ?"
        params = [trader_id]

        if start_date:
            query += " AND date >= ?"
            params.append(start_date.strftime('%Y-%m-%d'))

        if end_date:
            query += " AND date <= ?"
            params.append(end_date.strftime('%Y-%m-%d'))

        query += " ORDER BY date DESC"

        if limit:
            query += f" LIMIT {limit}"

        try:
            with sqlite3.connect(self.db_path) as conn:
                return pd.read_sql_query(query, conn, params=params)

        except sqlite3.Error as e:
            raise DatabaseConnectionError(self.db_path, e)

    def get_daily_summary(self,
                         trader_id: Optional[int] = None,
                         date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Get daily summary statistics.

        Args:
            trader_id: Optional trader filter
            date: Optional date filter (defaults to today)

        Returns:
            DataFrame with daily summary
        """
        if date is None:
            date = datetime.now()

        date_str = date.strftime('%Y-%m-%d')

        if trader_id:
            query = """
            SELECT
                account,
                date,
                COUNT(*) as num_trades,
                SUM(pnl) as total_pnl,
                AVG(pnl) as avg_pnl,
                MAX(pnl) as max_pnl,
                MIN(pnl) as min_pnl,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades
            FROM fills
            WHERE account = ? AND date = ?
            GROUP BY account, date
            """
            params = (trader_id, date_str)
        else:
            query = """
            SELECT
                account,
                date,
                COUNT(*) as num_trades,
                SUM(pnl) as total_pnl,
                AVG(pnl) as avg_pnl,
                MAX(pnl) as max_pnl,
                MIN(pnl) as min_pnl,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades
            FROM fills
            WHERE date = ?
            GROUP BY account, date
            """
            params = (date_str,)

        try:
            with sqlite3.connect(self.db_path) as conn:
                return pd.read_sql_query(query, conn, params=params)

        except sqlite3.Error as e:
            raise DatabaseConnectionError(self.db_path, e)

    def get_performance_metrics(self,
                              trader_id: int,
                              lookback_days: int = 30) -> Dict[str, float]:
        """
        Calculate performance metrics for a trader.

        Args:
            trader_id: The trader's account ID
            lookback_days: Number of days to look back

        Returns:
            Dictionary with performance metrics
        """
        start_date = (datetime.now() - timedelta(days=lookback_days))
        fills = self.get_fills_by_trader(trader_id, start_date)

        if fills.empty:
            return {
                'total_trades': 0,
                'total_pnl': 0.0,
                'avg_pnl': 0.0,
                'win_rate': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0
            }

        # Calculate metrics
        total_trades = len(fills)
        total_pnl = fills['pnl'].sum()
        avg_pnl = fills['pnl'].mean()

        winning_trades = len(fills[fills['pnl'] > 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        # Sharpe ratio (simplified)
        if 'pnl' in fills.columns and len(fills) > 1:
            daily_returns = fills.groupby('date')['pnl'].sum()
            if len(daily_returns) > 1 and daily_returns.std() > 0:
                sharpe = (daily_returns.mean() / daily_returns.std()) * (252 ** 0.5)
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0

        # Max drawdown
        cumulative_pnl = fills['pnl'].cumsum()
        running_max = cumulative_pnl.expanding().max()
        drawdown = cumulative_pnl - running_max
        max_drawdown = drawdown.min() if not drawdown.empty else 0.0

        return {
            'total_trades': total_trades,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown
        }

    def save(self, fill_data: Dict[str, Any]) -> Dict[str, Any]:
        """Save a new fill record."""
        required_fields = ['account', 'date', 'symbol', 'qty', 'price', 'pnl']

        # Validate required fields
        for field in required_fields:
            if field not in fill_data:
                raise ValueError(f"Missing required field: {field}")

        query = """
        INSERT INTO fills (account, date, symbol, qty, price, pnl, side, order_type)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, (
                    fill_data['account'],
                    fill_data['date'],
                    fill_data['symbol'],
                    fill_data['qty'],
                    fill_data['price'],
                    fill_data['pnl'],
                    fill_data.get('side', 'BUY'),
                    fill_data.get('order_type', 'MARKET')
                ))
                conn.commit()

                fill_data['id'] = cursor.lastrowid
                logger.info(f"Saved new fill with ID {fill_data['id']}")

                return fill_data

        except sqlite3.Error as e:
            logger.error(f"Failed to save fill: {e}")
            raise DatabaseConnectionError(self.db_path, e)

    def delete(self, fill_id: int) -> bool:
        """Delete a fill (not recommended for audit trail)."""
        logger.warning(f"Attempting to delete fill {fill_id} - not recommended")
        return False  # Prevent deletion for data integrity

    def exists(self, fill_id: int) -> bool:
        """Check if a fill exists."""
        query = "SELECT 1 FROM fills WHERE id = ? LIMIT 1"

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, (fill_id,))
                return cursor.fetchone() is not None

        except sqlite3.Error as e:
            raise DatabaseConnectionError(self.db_path, e)

    def get_symbols_traded(self, trader_id: int) -> List[str]:
        """Get list of symbols traded by a trader."""
        query = """
        SELECT DISTINCT symbol
        FROM fills
        WHERE account = ?
        ORDER BY symbol
        """

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, (trader_id,))
                return [row[0] for row in cursor.fetchall()]

        except sqlite3.Error as e:
            raise DatabaseConnectionError(self.db_path, e)
