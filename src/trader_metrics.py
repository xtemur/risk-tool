"""
Professional Trader Metrics Module

This module provides comprehensive trader performance metrics using pandas for
efficient data processing and analysis.

Features:
- BAT (Batting Average): Percentage of winning trades
- W/L (Win/Loss Ratio): Average net gain in winning trades / average net loss in losing trades
- Sharpe Ratio: Risk-adjusted return measure with proper bounds
- Traditional metrics: PnL statistics, volatility measures
- Heatmap support for visual representation
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from functools import lru_cache

logger = logging.getLogger(__name__)


@dataclass
class TraderMetrics:
    """Data class for trader metrics."""
    trader_id: int

    # Trading activity
    trading_days: int
    total_trades: int
    last_trade_date: str

    # Performance metrics - 30 day
    sharpe_30d: float
    bat_30d: float
    wl_ratio_30d: float
    avg_daily_pnl: float
    total_pnl: float

    # Performance metrics - all time
    sharpe_all_time: float
    bat_all_time: float
    wl_ratio_all_time: float
    all_time_avg_daily_pnl: float

    # Risk metrics
    highest_pnl: float
    lowest_pnl: float
    all_time_highest_pnl: float
    all_time_lowest_pnl: float

    # Trade statistics
    avg_winning_trade: float
    avg_losing_trade: float
    winning_trades_30d: int
    total_trades_30d: int
    winning_trades_all_time: int
    total_trades_all_time: int

    # Additional metrics
    last_trading_day_pnl: float
    avg_win_30d: float
    avg_loss_30d: float
    avg_win_all_time: float
    avg_loss_all_time: float


class TraderMetricsProvider:
    """
    Professional trader metrics provider using pandas for efficient data processing.

    This class provides comprehensive trader performance analytics including:
    - Batting averages and win/loss ratios
    - Risk-adjusted returns (Sharpe ratios)
    - Traditional performance metrics
    - Heatmap support for visualization
    """

    def __init__(self, config: Dict):
        """Initialize with configuration."""
        self.config = config
        self.db_path = config['paths']['db_path']
        self._trader_names_cache = None

    @lru_cache(maxsize=1)
    def get_trader_names(self) -> Dict[int, str]:
        """Get trader account names from database with caching."""
        if self._trader_names_cache is not None:
            return self._trader_names_cache

        try:
            query = "SELECT account_id, account_name FROM accounts WHERE is_active = 1"
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(query, conn)

            trader_names = dict(zip(df['account_id'], df['account_name']))
            self._trader_names_cache = trader_names
            logger.info(f"Retrieved names for {len(trader_names)} traders")
            return trader_names

        except Exception as e:
            logger.warning(f"Could not load trader names: {e}")
            return {}

    def _load_trades_data(self, lookback_days: Optional[int] = None) -> pd.DataFrame:
        """Load and prepare trades data efficiently."""
        base_query = """
        SELECT
            account_id,
            trade_date,
            net,
            gross,
            (comm + ecn_fee + sec + orf + cat + taf + ftt + nscc + acc + clr + misc) as fees,
            qty as volume
        FROM trades
        """

        params = []
        if lookback_days:
            cutoff_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
            base_query += " WHERE trade_date >= ?"
            params.append(cutoff_date)

        base_query += " ORDER BY account_id, trade_date"

        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(base_query, conn, params=params)

            # Convert date column
            df['trade_date'] = pd.to_datetime(df['trade_date'])

            # Ensure numeric columns
            numeric_cols = ['net', 'gross', 'fees', 'volume']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            logger.info(f"Loaded {len(df)} trades for analysis")
            return df

        except Exception as e:
            logger.error(f"Error loading trades data: {e}")
            return pd.DataFrame()

    def _calculate_daily_aggregates(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate daily aggregates efficiently using pandas."""
        if trades_df.empty:
            return pd.DataFrame()

        # Group by trader and date to get daily metrics
        daily_agg = trades_df.groupby(['account_id', 'trade_date']).agg({
            'net': ['sum', 'count'],
            'gross': 'sum',
            'fees': 'sum',
            'volume': 'sum'
        }).reset_index()

        # Flatten column names
        daily_agg.columns = [
            'account_id', 'trade_date', 'daily_pnl', 'trade_count',
            'daily_gross', 'daily_fees', 'daily_volume'
        ]

        return daily_agg

    def _calculate_batting_average(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate batting average efficiently using pandas."""
        if trades_df.empty:
            return pd.DataFrame()

        # Calculate winning trades and total trades per trader
        bat_stats = trades_df.groupby('account_id').agg({
            'net': [
                lambda x: (x > 0).sum(),  # winning trades
                'count'                    # total trades
            ]
        }).reset_index()

        # Flatten columns
        bat_stats.columns = ['account_id', 'winning_trades', 'total_trades']

        # Calculate batting average
        bat_stats['bat'] = np.where(
            bat_stats['total_trades'] > 0,
            (bat_stats['winning_trades'] / bat_stats['total_trades']) * 100,
            0
        )

        return bat_stats[['account_id', 'bat', 'winning_trades', 'total_trades']]

    def _calculate_win_loss_ratio(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate win/loss ratio efficiently using pandas."""
        if trades_df.empty:
            return pd.DataFrame()

        # Separate winning and losing trades
        winning_trades = trades_df[trades_df['net'] > 0]
        losing_trades = trades_df[trades_df['net'] < 0]

        # Calculate averages
        avg_wins = winning_trades.groupby('account_id')['net'].mean()
        avg_losses = losing_trades.groupby('account_id')['net'].apply(lambda x: x.abs().mean())

        # Combine and calculate ratio
        wl_data = pd.DataFrame({
            'avg_win': avg_wins,
            'avg_loss': avg_losses
        }).fillna(0).reset_index()

        wl_data['wl_ratio'] = np.where(
            wl_data['avg_loss'] > 0,
            wl_data['avg_win'] / wl_data['avg_loss'],
            0
        )

        return wl_data[['account_id', 'wl_ratio', 'avg_win', 'avg_loss']]

    def _calculate_sharpe_ratio(self, daily_df: pd.DataFrame,
                               min_days: int = 10,
                               max_sharpe: float = 5.0) -> pd.DataFrame:
        """Calculate Sharpe ratio with proper bounds using pandas."""
        if daily_df.empty:
            return pd.DataFrame()

        def sharpe_calc(group):
            if len(group) < min_days:
                return 0.0

            daily_returns = group['daily_pnl']
            mean_return = daily_returns.mean()
            std_return = daily_returns.std(ddof=1)

            if std_return == 0 or pd.isna(std_return):
                return 0.0

            # Annualized Sharpe ratio
            sharpe = (mean_return / std_return) * np.sqrt(252)

            # Apply bounds
            return np.clip(sharpe, -max_sharpe, max_sharpe)

        sharpe_df = daily_df.groupby('account_id', group_keys=False).apply(sharpe_calc).reset_index()
        sharpe_df.columns = ['account_id', 'sharpe_ratio']

        return sharpe_df

    def _calculate_pnl_statistics(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate PnL statistics efficiently."""
        if daily_df.empty:
            return pd.DataFrame()

        pnl_stats = daily_df.groupby('account_id').agg({
            'daily_pnl': ['sum', 'mean', 'std', 'min', 'max', 'count'],
            'trade_date': 'max'
        }).reset_index()

        # Flatten columns
        pnl_stats.columns = [
            'account_id', 'total_pnl', 'avg_daily_pnl', 'pnl_std',
            'lowest_pnl', 'highest_pnl', 'trading_days', 'last_trade_date'
        ]

        return pnl_stats

    def _get_last_trading_day_pnl(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        """Get last trading day PnL for each trader."""
        if daily_df.empty:
            return pd.DataFrame()

        # Get the last trading day for each trader
        last_day_pnl = daily_df.loc[
            daily_df.groupby('account_id')['trade_date'].idxmax()
        ][['account_id', 'daily_pnl']]

        last_day_pnl.columns = ['account_id', 'last_trading_day_pnl']
        return last_day_pnl

    def get_comprehensive_trader_metrics(self, lookback_days: int = 30) -> Dict[int, Dict]:
        """
        Get comprehensive trader metrics using efficient pandas operations.

        Args:
            lookback_days: Number of days to look back for 30-day metrics

        Returns:
            Dictionary mapping trader_id to comprehensive metrics
        """
        try:
            # Load all data once for all-time metrics
            all_trades = self._load_trades_data()
            recent_trades = self._load_trades_data(lookback_days)

            if all_trades.empty:
                logger.warning("No trades data available")
                return {}

            # Calculate daily aggregates
            all_daily = self._calculate_daily_aggregates(all_trades)
            recent_daily = self._calculate_daily_aggregates(recent_trades)

            # Calculate all metrics in parallel
            metrics_components = {}

            # 30-day metrics
            if not recent_trades.empty and not recent_daily.empty:
                metrics_components['bat_30d'] = self._calculate_batting_average(recent_trades)
                metrics_components['wl_30d'] = self._calculate_win_loss_ratio(recent_trades)
                metrics_components['sharpe_30d'] = self._calculate_sharpe_ratio(recent_daily, min_days=10, max_sharpe=5.0)
                metrics_components['pnl_30d'] = self._calculate_pnl_statistics(recent_daily)
                metrics_components['last_day_pnl'] = self._get_last_trading_day_pnl(recent_daily)

            # All-time metrics
            metrics_components['bat_all'] = self._calculate_batting_average(all_trades)
            metrics_components['wl_all'] = self._calculate_win_loss_ratio(all_trades)
            metrics_components['sharpe_all'] = self._calculate_sharpe_ratio(all_daily, min_days=30, max_sharpe=3.0)
            metrics_components['pnl_all'] = self._calculate_pnl_statistics(all_daily)

            # Combine all metrics
            return self._combine_metrics_components(metrics_components)

        except Exception as e:
            logger.error(f"Error calculating comprehensive metrics: {e}")
            return {}

    def _combine_metrics_components(self, components: Dict[str, pd.DataFrame]) -> Dict[int, Dict]:
        """Combine all metric components into final structure."""
        final_metrics = {}

        # Get all unique trader IDs
        all_trader_ids = set()
        for df in components.values():
            if not df.empty and 'account_id' in df.columns:
                all_trader_ids.update(df['account_id'].unique())

        # Build metrics for each trader
        for trader_id in all_trader_ids:
            trader_metrics = {'account_id': trader_id}

            # Helper function to safely get value from DataFrame
            def get_value(df, trader_id, column, default=0):
                if df.empty:
                    return default
                row = df[df['account_id'] == trader_id]
                if row.empty:
                    return default
                return row[column].iloc[0]

            # 30-day metrics
            if 'bat_30d' in components:
                trader_metrics.update({
                    'bat_30d': get_value(components['bat_30d'], trader_id, 'bat', 0),
                    'winning_trades_30d': get_value(components['bat_30d'], trader_id, 'winning_trades', 0),
                    'total_trades_30d': get_value(components['bat_30d'], trader_id, 'total_trades', 0)
                })

            if 'wl_30d' in components:
                trader_metrics.update({
                    'wl_ratio_30d': get_value(components['wl_30d'], trader_id, 'wl_ratio', 0),
                    'avg_win_30d': get_value(components['wl_30d'], trader_id, 'avg_win', 0),
                    'avg_loss_30d': get_value(components['wl_30d'], trader_id, 'avg_loss', 0)
                })

            if 'sharpe_30d' in components:
                trader_metrics['sharpe_30d'] = get_value(components['sharpe_30d'], trader_id, 'sharpe_ratio', 0)

            if 'pnl_30d' in components:
                trader_metrics.update({
                    'total_pnl': get_value(components['pnl_30d'], trader_id, 'total_pnl', 0),
                    'avg_daily_pnl': get_value(components['pnl_30d'], trader_id, 'avg_daily_pnl', 0),
                    'highest_pnl': get_value(components['pnl_30d'], trader_id, 'highest_pnl', 0),
                    'lowest_pnl': get_value(components['pnl_30d'], trader_id, 'lowest_pnl', 0),
                    'trading_days': get_value(components['pnl_30d'], trader_id, 'trading_days', 0),
                    'last_trade_date': get_value(components['pnl_30d'], trader_id, 'last_trade_date', '')
                })

            if 'last_day_pnl' in components:
                trader_metrics['last_trading_day_pnl'] = get_value(components['last_day_pnl'], trader_id, 'last_trading_day_pnl', 0)

            # All-time metrics
            trader_metrics.update({
                'bat_all_time': get_value(components['bat_all'], trader_id, 'bat', 0),
                'winning_trades_all_time': get_value(components['bat_all'], trader_id, 'winning_trades', 0),
                'total_trades_all_time': get_value(components['bat_all'], trader_id, 'total_trades', 0),
                'wl_ratio_all_time': get_value(components['wl_all'], trader_id, 'wl_ratio', 0),
                'avg_win_all_time': get_value(components['wl_all'], trader_id, 'avg_win', 0),
                'avg_loss_all_time': get_value(components['wl_all'], trader_id, 'avg_loss', 0),
                'all_time_sharpe': get_value(components['sharpe_all'], trader_id, 'sharpe_ratio', 0),
                'all_time_avg_daily_pnl': get_value(components['pnl_all'], trader_id, 'avg_daily_pnl', 0),
                'all_time_highest_pnl': get_value(components['pnl_all'], trader_id, 'highest_pnl', 0),
                'all_time_lowest_pnl': get_value(components['pnl_all'], trader_id, 'lowest_pnl', 0)
            })

            # Legacy field mappings for backward compatibility
            trader_metrics.update({
                'avg_winning_trade': trader_metrics.get('avg_win_30d', 0),
                'avg_losing_trade': trader_metrics.get('avg_loss_30d', 0),
                'all_time_avg_winning_trade': trader_metrics.get('avg_win_all_time', 0),
                'all_time_avg_losing_trade': trader_metrics.get('avg_loss_all_time', 0)
            })

            final_metrics[trader_id] = trader_metrics

        logger.info(f"Calculated comprehensive metrics for {len(final_metrics)} traders")
        return final_metrics

    def get_trader_metrics_for_email(self, lookback_days: int = 30) -> Dict[int, Dict]:
        """
        Get trader metrics formatted specifically for email display.

        Args:
            lookback_days: Number of days for recent metrics

        Returns:
            Dictionary with trader metrics formatted for email
        """
        return self.get_comprehensive_trader_metrics(lookback_days)

    def get_metrics_summary(self, lookback_days: int = 30) -> pd.DataFrame:
        """
        Get a summary DataFrame of all trader metrics.

        Args:
            lookback_days: Number of days for recent metrics

        Returns:
            DataFrame with summary metrics
        """
        metrics = self.get_comprehensive_trader_metrics(lookback_days)

        if not metrics:
            return pd.DataFrame()

        # Convert to DataFrame for easier analysis
        df = pd.DataFrame.from_dict(metrics, orient='index')

        # Add trader names if available
        trader_names = self.get_trader_names()
        if trader_names:
            df['trader_name'] = df['account_id'].map(trader_names)

        return df

    def calculate_batting_average(self, account_id: int, lookback_days: int = 30) -> Dict[str, float]:
        """
        Calculate BAT (Batting Average) for a specific trader.

        Args:
            account_id: Trader account ID
            lookback_days: Number of days to look back

        Returns:
            Dictionary with batting average metrics
        """
        all_metrics = self.get_comprehensive_trader_metrics(lookback_days)

        if account_id not in all_metrics:
            return {
                'bat_30d': 0,
                'bat_all_time': 0,
                'winning_trades_30d': 0,
                'total_trades_30d': 0,
                'winning_trades_all_time': 0,
                'total_trades_all_time': 0
            }

        trader_metrics = all_metrics[account_id]
        return {
            'bat_30d': trader_metrics.get('bat_30d', 0),
            'bat_all_time': trader_metrics.get('bat_all_time', 0),
            'winning_trades_30d': trader_metrics.get('winning_trades_30d', 0),
            'total_trades_30d': trader_metrics.get('total_trades_30d', 0),
            'winning_trades_all_time': trader_metrics.get('winning_trades_all_time', 0),
            'total_trades_all_time': trader_metrics.get('total_trades_all_time', 0)
        }

    def calculate_win_loss_ratio(self, account_id: int, lookback_days: int = 30) -> Dict[str, float]:
        """
        Calculate W/L (Win/Loss Ratio) for a specific trader.

        Args:
            account_id: Trader account ID
            lookback_days: Number of days to look back

        Returns:
            Dictionary with win/loss ratio metrics
        """
        all_metrics = self.get_comprehensive_trader_metrics(lookback_days)

        if account_id not in all_metrics:
            return {
                'wl_ratio_30d': 0,
                'wl_ratio_all_time': 0,
                'avg_win_30d': 0,
                'avg_loss_30d': 0,
                'avg_win_all_time': 0,
                'avg_loss_all_time': 0
            }

        trader_metrics = all_metrics[account_id]
        return {
            'wl_ratio_30d': trader_metrics.get('wl_ratio_30d', 0),
            'wl_ratio_all_time': trader_metrics.get('wl_ratio_all_time', 0),
            'avg_win_30d': trader_metrics.get('avg_win_30d', 0),
            'avg_loss_30d': trader_metrics.get('avg_loss_30d', 0),
            'avg_win_all_time': trader_metrics.get('avg_win_all_time', 0),
            'avg_loss_all_time': trader_metrics.get('avg_loss_all_time', 0)
        }
