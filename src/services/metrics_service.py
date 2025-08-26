"""Service for calculating and managing metrics."""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from ..repositories.fills_repository import FillsRepository
from ..repositories.trader_repository import TraderRepository
from ..models.domain import TradingMetrics
from ..config.config_manager import ConfigManager

logger = logging.getLogger(__name__)


class MetricsService:
    """Service for calculating trading metrics."""

    def __init__(self,
                 fills_repo: FillsRepository,
                 trader_repo: TraderRepository,
                 config: ConfigManager):
        """Initialize metrics service."""
        self.fills_repo = fills_repo
        self.trader_repo = trader_repo
        self.config = config

    def calculate_trader_metrics(self,
                                trader_id: int,
                                lookback_days: int = 30) -> TradingMetrics:
        """
        Calculate comprehensive metrics for a trader.

        Args:
            trader_id: The trader ID
            lookback_days: Days to look back

        Returns:
            TradingMetrics object
        """
        # Get fills data
        start_date = datetime.now() - timedelta(days=lookback_days)
        fills = self.fills_repo.get_fills_by_trader(trader_id, start_date)

        if fills.empty:
            return TradingMetrics(
                bat_30d=0.0,
                wl_ratio=0.0,
                sharpe=0.0,
                total_trades=0,
                total_pnl=0.0
            )

        # Calculate metrics
        metrics = self._calculate_base_metrics(fills)
        metrics['max_drawdown'] = self._calculate_max_drawdown(fills)
        metrics['avg_trade_size'] = self._calculate_avg_trade_size(fills)
        metrics['daily_volume'] = self._calculate_daily_volume(fills)

        return TradingMetrics(**metrics)

    def calculate_sharpe_ratio(self,
                              returns: pd.Series,
                              risk_free_rate: float = 0.0) -> float:
        """
        Calculate Sharpe ratio.

        Args:
            returns: Series of returns
            risk_free_rate: Risk-free rate (annualized)

        Returns:
            Sharpe ratio
        """
        if returns.empty or returns.std() == 0:
            return 0.0

        # Daily Sharpe, annualized
        excess_returns = returns - risk_free_rate / 252
        sharpe = excess_returns.mean() / excess_returns.std()

        # Annualize
        return sharpe * np.sqrt(252)

    def calculate_sortino_ratio(self,
                               returns: pd.Series,
                               target_return: float = 0.0) -> float:
        """
        Calculate Sortino ratio.

        Args:
            returns: Series of returns
            target_return: Target return

        Returns:
            Sortino ratio
        """
        if returns.empty:
            return 0.0

        excess_returns = returns - target_return
        downside_returns = excess_returns[excess_returns < 0]

        if downside_returns.empty:
            return float('inf')

        downside_std = downside_returns.std()

        if downside_std == 0:
            return float('inf') if excess_returns.mean() > 0 else 0.0

        return excess_returns.mean() / downside_std * np.sqrt(252)

    def calculate_calmar_ratio(self,
                              returns: pd.Series,
                              max_drawdown: float) -> float:
        """
        Calculate Calmar ratio.

        Args:
            returns: Series of returns
            max_drawdown: Maximum drawdown

        Returns:
            Calmar ratio
        """
        if max_drawdown == 0:
            return float('inf') if returns.mean() > 0 else 0.0

        annual_return = returns.mean() * 252
        return annual_return / abs(max_drawdown)

    def aggregate_metrics(self, trader_ids: List[int]) -> Dict[str, Any]:
        """
        Calculate aggregate metrics for multiple traders.

        Args:
            trader_ids: List of trader IDs

        Returns:
            Dictionary with aggregate metrics
        """
        all_metrics = []

        for trader_id in trader_ids:
            metrics = self.calculate_trader_metrics(trader_id)
            all_metrics.append({
                'trader_id': trader_id,
                'metrics': metrics
            })

        if not all_metrics:
            return {}

        # Calculate aggregates
        total_pnl = sum(m['metrics'].total_pnl for m in all_metrics)
        total_trades = sum(m['metrics'].total_trades for m in all_metrics)
        avg_sharpe = np.mean([m['metrics'].sharpe for m in all_metrics])
        avg_win_rate = np.mean([m['metrics'].bat_30d for m in all_metrics])

        return {
            'total_pnl': total_pnl,
            'total_trades': total_trades,
            'avg_sharpe': avg_sharpe,
            'avg_win_rate': avg_win_rate,
            'num_traders': len(trader_ids),
            'individual_metrics': all_metrics,
            'calculated_at': datetime.now().isoformat()
        }

    def _calculate_base_metrics(self, fills: pd.DataFrame) -> Dict[str, float]:
        """Calculate base trading metrics."""
        total_trades = len(fills)

        if 'pnl' in fills.columns:
            winning_trades = len(fills[fills['pnl'] > 0])
            losing_trades = len(fills[fills['pnl'] < 0])
            total_pnl = fills['pnl'].sum()

            # Win rate (BAT)
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

            # Win/Loss ratio
            if losing_trades > 0:
                avg_win = fills[fills['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
                avg_loss = abs(fills[fills['pnl'] < 0]['pnl'].mean()) if losing_trades > 0 else 1
                wl_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')
            else:
                wl_ratio = float('inf') if winning_trades > 0 else 0

            # Sharpe ratio
            if 'date' in fills.columns:
                daily_returns = fills.groupby('date')['pnl'].sum()
                sharpe = self.calculate_sharpe_ratio(daily_returns)
            else:
                sharpe = 0.0
        else:
            win_rate = 0
            wl_ratio = 0
            total_pnl = 0
            sharpe = 0

        return {
            'bat_30d': win_rate,
            'wl_ratio': wl_ratio,
            'sharpe': sharpe,
            'total_trades': total_trades,
            'total_pnl': total_pnl
        }

    def _calculate_max_drawdown(self, fills: pd.DataFrame) -> float:
        """Calculate maximum drawdown."""
        if 'pnl' not in fills.columns or fills.empty:
            return 0.0

        cumulative_pnl = fills['pnl'].cumsum()
        running_max = cumulative_pnl.expanding().max()
        drawdown = cumulative_pnl - running_max

        return abs(drawdown.min()) if not drawdown.empty else 0.0

    def _calculate_avg_trade_size(self, fills: pd.DataFrame) -> float:
        """Calculate average trade size."""
        if 'qty' in fills.columns and 'price' in fills.columns:
            trade_sizes = fills['qty'].abs() * fills['price']
            return trade_sizes.mean() if not trade_sizes.empty else 0.0
        return 0.0

    def _calculate_daily_volume(self, fills: pd.DataFrame) -> float:
        """Calculate average daily volume."""
        if 'qty' in fills.columns and 'price' in fills.columns and 'date' in fills.columns:
            fills['volume'] = fills['qty'].abs() * fills['price']
            daily_volume = fills.groupby('date')['volume'].sum()
            return daily_volume.mean() if not daily_volume.empty else 0.0
        return 0.0
