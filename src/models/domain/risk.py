"""Domain models for risk assessment."""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum


class RiskLevel(Enum):
    """Risk levels for traders."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @classmethod
    def from_score(cls, score: float) -> 'RiskLevel':
        """Determine risk level from score (0-100)."""
        if score < 25:
            return cls.LOW
        elif score < 50:
            return cls.MEDIUM
        elif score < 75:
            return cls.HIGH
        else:
            return cls.CRITICAL

    def get_color(self) -> str:
        """Get color code for risk level."""
        colors = {
            self.LOW: "green",
            self.MEDIUM: "yellow",
            self.HIGH: "orange",
            self.CRITICAL: "red"
        }
        return colors.get(self, "gray")


@dataclass
class RiskAssessment:
    """Risk assessment for a trader."""
    trader_id: int
    risk_level: RiskLevel
    risk_score: float  # 0-100
    var_95: float  # Value at Risk at 95% confidence
    expected_shortfall: float
    max_probable_loss: float
    assessment_date: datetime
    confidence: float  # Model confidence 0-1
    factors: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize factors if not provided."""
        if self.factors is None:
            self.factors = {}

    def is_actionable(self) -> bool:
        """Check if risk requires action."""
        return self.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]

    def get_recommendation(self) -> str:
        """Get risk management recommendation."""
        if self.risk_level == RiskLevel.CRITICAL:
            return "Immediate intervention required - suspend trading"
        elif self.risk_level == RiskLevel.HIGH:
            return "Close monitoring required - consider position limits"
        elif self.risk_level == RiskLevel.MEDIUM:
            return "Regular monitoring - review weekly"
        else:
            return "Standard monitoring sufficient"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'trader_id': self.trader_id,
            'risk_level': self.risk_level.value,
            'risk_score': self.risk_score,
            'var_95': self.var_95,
            'expected_shortfall': self.expected_shortfall,
            'max_probable_loss': self.max_probable_loss,
            'assessment_date': self.assessment_date.isoformat(),
            'confidence': self.confidence,
            'factors': self.factors,
            'recommendation': self.get_recommendation()
        }


@dataclass
class RiskAlert:
    """Risk alert for notification."""
    trader_id: int
    alert_type: str
    severity: RiskLevel
    message: str
    timestamp: datetime
    metrics: Dict[str, float]
    resolved: bool = False

    def format_message(self) -> str:
        """Format alert message for notification."""
        return (f"[{self.severity.value.upper()}] Trader {self.trader_id}: "
                f"{self.message} ({self.timestamp.strftime('%Y-%m-%d %H:%M')})")

    def should_escalate(self) -> bool:
        """Determine if alert should be escalated."""
        return self.severity in [RiskLevel.HIGH, RiskLevel.CRITICAL] and not self.resolved
