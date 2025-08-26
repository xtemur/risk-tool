"""Domain models for predictions."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class PredictionStatus(Enum):
    """Status of a prediction."""
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass
class Prediction:
    """Individual prediction for a trader."""
    trader_id: int
    prediction_date: datetime
    predicted_value: float
    confidence: float  # 0-1
    model_version: Optional[str] = None
    features_used: Optional[List[str]] = None
    status: PredictionStatus = PredictionStatus.PENDING

    def is_valid(self) -> bool:
        """Check if prediction is valid."""
        return (
            self.status == PredictionStatus.COMPLETED and
            0 <= self.confidence <= 1
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'trader_id': self.trader_id,
            'prediction_date': self.prediction_date.isoformat(),
            'predicted_value': self.predicted_value,
            'confidence': self.confidence,
            'model_version': self.model_version,
            'features_used': self.features_used,
            'status': self.status.value
        }


@dataclass
class PredictionResult:
    """Aggregated prediction results."""
    trader_id: int
    prediction_date: datetime
    var_prediction: float  # Value at Risk
    loss_probability: float  # Probability of loss
    expected_return: float
    confidence_interval: tuple  # (lower, upper)
    risk_metrics: Dict[str, float] = field(default_factory=dict)
    model_metadata: Dict[str, Any] = field(default_factory=dict)

    def get_risk_signal(self) -> str:
        """Get risk signal based on predictions."""
        if self.loss_probability > 0.7:
            return "HIGH_RISK"
        elif self.loss_probability > 0.5:
            return "MEDIUM_RISK"
        elif self.loss_probability > 0.3:
            return "LOW_RISK"
        else:
            return "MINIMAL_RISK"

    def should_alert(self, threshold: float = 0.6) -> bool:
        """Determine if prediction warrants an alert."""
        return self.loss_probability > threshold

    def format_summary(self) -> str:
        """Format prediction summary."""
        return (
            f"Trader {self.trader_id} - "
            f"VaR: ${self.var_prediction:,.2f}, "
            f"Loss Prob: {self.loss_probability:.1%}, "
            f"Expected Return: ${self.expected_return:,.2f}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'trader_id': self.trader_id,
            'prediction_date': self.prediction_date.isoformat(),
            'var_prediction': self.var_prediction,
            'loss_probability': self.loss_probability,
            'expected_return': self.expected_return,
            'confidence_interval': self.confidence_interval,
            'risk_metrics': self.risk_metrics,
            'model_metadata': self.model_metadata,
            'risk_signal': self.get_risk_signal()
        }


@dataclass
class ModelPrediction:
    """Model-specific prediction with metadata."""
    model_id: str
    model_type: str
    trader_id: int
    prediction: float
    feature_importance: Dict[str, float]
    training_date: datetime
    validation_score: float

    def is_stale(self, days: int = 7) -> bool:
        """Check if model prediction is stale."""
        age = (datetime.now() - self.training_date).days
        return age > days

    def get_top_features(self, n: int = 5) -> List[tuple]:
        """Get top N important features."""
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_features[:n]
