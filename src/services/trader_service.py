"""Service for trader management."""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from ..repositories.trader_repository import TraderRepository
from ..repositories.fills_repository import FillsRepository
from ..models.domain import Trader, TradingMetrics, TraderProfile
from ..exceptions import TraderNotFoundError
from ..config.config_manager import ConfigManager

logger = logging.getLogger(__name__)


class TraderService:
    """Service for managing traders and their data."""

    def __init__(self,
                 trader_repo: TraderRepository,
                 fills_repo: FillsRepository,
                 config: ConfigManager):
        """Initialize trader service."""
        self.trader_repo = trader_repo
        self.fills_repo = fills_repo
        self.config = config

    def get_trader(self, trader_id: int) -> Trader:
        """
        Get trader by ID.

        Args:
            trader_id: The trader ID

        Returns:
            Trader object

        Raises:
            TraderNotFoundError: If trader not found
        """
        trader = self.trader_repo.find_by_id(trader_id)
        if not trader:
            raise TraderNotFoundError(trader_id)
        return trader

    def get_all_traders(self) -> List[Trader]:
        """Get all traders."""
        return self.trader_repo.find_all()

    def get_active_traders(self) -> List[Trader]:
        """Get all active traders."""
        active_ids = self.trader_repo.get_active_traders(
            min_trades=self.config.risk.min_trades,
            days=self.config.risk.lookback_days
        )

        traders = []
        for trader_id in active_ids:
            trader = self.trader_repo.find_by_id(trader_id)
            if trader:
                traders.append(trader)

        return traders

    def get_trader_profile(self, trader_id: int) -> TraderProfile:
        """
        Get complete trader profile with metrics.

        Args:
            trader_id: The trader ID

        Returns:
            TraderProfile object
        """
        trader = self.get_trader(trader_id)

        # Get current metrics
        current_metrics = self.trader_repo.get_trader_metrics(
            trader_id,
            self.config.risk.lookback_days
        )

        # Get historical metrics (last 3 periods)
        historical_metrics = []
        for i in range(1, 4):
            start_days = self.config.risk.lookback_days * (i + 1)
            end_days = self.config.risk.lookback_days * i

            metrics = self.trader_repo.get_trader_metrics(
                trader_id,
                start_days
            )
            if metrics:
                historical_metrics.append(metrics)

        return TraderProfile(
            trader=trader,
            current_metrics=current_metrics or TradingMetrics(0, 0, 0),
            historical_metrics=historical_metrics,
            last_update=datetime.now()
        )

    def update_trader_status(self, trader_id: int, status: str) -> Trader:
        """
        Update trader status.

        Args:
            trader_id: The trader ID
            status: New status

        Returns:
            Updated trader object
        """
        trader = self.get_trader(trader_id)
        trader.status = status
        return self.trader_repo.save(trader)

    def get_trader_performance(self,
                              trader_id: int,
                              days: int = 30) -> Dict[str, Any]:
        """
        Get trader performance summary.

        Args:
            trader_id: The trader ID
            days: Number of days to analyze

        Returns:
            Performance dictionary
        """
        metrics = self.fills_repo.get_performance_metrics(trader_id, days)

        # Add trader info
        trader = self.get_trader(trader_id)
        metrics['trader_id'] = trader.id
        metrics['trader_name'] = trader.display_name

        # Add risk assessment
        trading_metrics = self.trader_repo.get_trader_metrics(trader_id, days)
        if trading_metrics:
            metrics['risk_score'] = trading_metrics.get_risk_score()
            metrics['is_high_risk'] = trading_metrics.is_high_risk()

        return metrics

    def compare_traders(self, trader_ids: List[int]) -> Dict[str, Any]:
        """
        Compare multiple traders.

        Args:
            trader_ids: List of trader IDs to compare

        Returns:
            Comparison dictionary
        """
        comparisons = []

        for trader_id in trader_ids:
            try:
                performance = self.get_trader_performance(trader_id)
                comparisons.append(performance)
            except TraderNotFoundError:
                logger.warning(f"Trader {trader_id} not found for comparison")
                continue

        # Sort by total PnL
        comparisons.sort(key=lambda x: x.get('total_pnl', 0), reverse=True)

        return {
            'traders': comparisons,
            'best_performer': comparisons[0] if comparisons else None,
            'worst_performer': comparisons[-1] if comparisons else None,
            'comparison_date': datetime.now().isoformat()
        }

    def get_symbols_by_trader(self, trader_id: int) -> List[str]:
        """Get list of symbols traded by a trader."""
        return self.fills_repo.get_symbols_traded(trader_id)
