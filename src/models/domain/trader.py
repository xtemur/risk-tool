"""Domain model for Trader and related entities."""

from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime
from enum import Enum


class TraderStatus(Enum):
    """Trader account status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    UNDER_REVIEW = "under_review"


@dataclass
class Trader:
    """Domain model for a trader."""
    id: int
    name: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    status: TraderStatus = TraderStatus.ACTIVE
    risk_level: Optional[str] = None

    @property
    def display_name(self) -> str:
        """Get display name for trader."""
        return self.name if self.name else f"Trader {self.id}"

    def is_active(self) -> bool:
        """Check if trader is active."""
        return self.status == TraderStatus.ACTIVE


@dataclass
class TradingMetrics:
    """Trading performance metrics."""
    bat_30d: float  # Win rate over 30 days
    wl_ratio: float  # Win/Loss ratio
    sharpe: float  # Sharpe ratio
    total_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: Optional[float] = None
    avg_trade_size: Optional[float] = None
    daily_volume: Optional[float] = None

    def is_high_risk(self) -> bool:
        """Determine if metrics indicate high risk."""
        return (
            self.sharpe < 0.5 or
            self.bat_30d < 40 or
            self.wl_ratio < 1.0
        )

    def is_low_performance(self) -> bool:
        """Check if trader has low performance."""
        return self.total_pnl < 0 and self.sharpe < 0

    def get_risk_score(self) -> float:
        """Calculate composite risk score (0-100, higher is riskier)."""
        score = 0.0

        # Sharpe ratio component (0-40 points)
        if self.sharpe < 0:
            score += 40
        elif self.sharpe < 0.5:
            score += 30
        elif self.sharpe < 1.0:
            score += 20
        elif self.sharpe < 1.5:
            score += 10

        # Win rate component (0-30 points)
        if self.bat_30d < 30:
            score += 30
        elif self.bat_30d < 40:
            score += 20
        elif self.bat_30d < 50:
            score += 10

        # Win/Loss ratio component (0-30 points)
        if self.wl_ratio < 0.5:
            score += 30
        elif self.wl_ratio < 1.0:
            score += 20
        elif self.wl_ratio < 1.5:
            score += 10

        return min(score, 100.0)


@dataclass
class TraderProfile:
    """Complete trader profile with metrics and history."""
    trader: Trader
    current_metrics: TradingMetrics
    historical_metrics: List[TradingMetrics] = field(default_factory=list)
    last_update: datetime = field(default_factory=datetime.now)

    def get_trend(self) -> str:
        """Analyze performance trend."""
        if len(self.historical_metrics) < 2:
            return "insufficient_data"

        recent = self.historical_metrics[-1]
        previous = self.historical_metrics[-2]

        if recent.sharpe > previous.sharpe * 1.1:
            return "improving"
        elif recent.sharpe < previous.sharpe * 0.9:
            return "deteriorating"
        else:
            return "stable"
