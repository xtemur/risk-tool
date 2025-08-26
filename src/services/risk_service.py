"""Risk assessment service layer."""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from ..repositories.trader_repository import TraderRepository
from ..repositories.model_repository import ModelRepository
from ..models.domain import (
    Trader, TradingMetrics, RiskAssessment,
    RiskLevel, RiskAlert
)
from ..exceptions import (
    TraderNotFoundError, ModelNotFoundError,
    InsufficientDataError, RiskAssessmentError
)
from ..config.config_manager import ConfigManager
from ..constants import MIN_TRADES_FOR_ANALYSIS

logger = logging.getLogger(__name__)


class RiskService:
    """Service for risk assessment and management."""

    def __init__(self,
                 trader_repo: TraderRepository,
                 model_repo: ModelRepository,
                 config: ConfigManager):
        """Initialize risk service with dependencies."""
        self.trader_repo = trader_repo
        self.model_repo = model_repo
        self.config = config

    def assess_trader_risk(self, trader_id: int) -> RiskAssessment:
        """
        Assess risk for a specific trader.

        Args:
            trader_id: The trader to assess

        Returns:
            RiskAssessment object

        Raises:
            TraderNotFoundError: If trader doesn't exist
            ModelNotFoundError: If model not found
            InsufficientDataError: If not enough data
            RiskAssessmentError: If assessment fails
        """
        # Verify trader exists
        trader = self.trader_repo.find_by_id(trader_id)
        if not trader:
            raise TraderNotFoundError(trader_id)

        # Get trader metrics
        metrics = self.trader_repo.get_trader_metrics(
            trader_id,
            self.config.risk.lookback_days
        )

        if not metrics:
            raise InsufficientDataError(
                trader_id,
                MIN_TRADES_FOR_ANALYSIS,
                0
            )

        if metrics.total_trades < self.config.risk.min_trades:
            raise InsufficientDataError(
                trader_id,
                self.config.risk.min_trades,
                metrics.total_trades
            )

        try:
            # Load model and generate predictions
            model_data = self.model_repo.load_model(trader_id)
            if not model_data:
                raise ModelNotFoundError(trader_id)

            # Calculate risk metrics
            risk_score = metrics.get_risk_score()
            risk_level = RiskLevel.from_score(risk_score)

            # Calculate VaR and ES using model predictions
            var_95, es = self._calculate_var_es(trader_id, model_data, metrics)
            max_probable_loss = self._calculate_max_loss(trader_id, var_95, es)

            # Create assessment
            assessment = RiskAssessment(
                trader_id=trader_id,
                risk_level=risk_level,
                risk_score=risk_score,
                var_95=var_95,
                expected_shortfall=es,
                max_probable_loss=max_probable_loss,
                assessment_date=datetime.now(),
                confidence=self._calculate_confidence(model_data, metrics),
                factors={
                    'sharpe': metrics.sharpe,
                    'win_rate': metrics.bat_30d,
                    'wl_ratio': metrics.wl_ratio,
                    'total_trades': metrics.total_trades,
                    'total_pnl': metrics.total_pnl
                }
            )

            logger.info(f"Risk assessment completed for trader {trader_id}: {risk_level.value}")
            return assessment

        except Exception as e:
            logger.error(f"Risk assessment failed for trader {trader_id}: {str(e)}")
            raise RiskAssessmentError(trader_id, str(e))

    def assess_all_traders(self) -> List[RiskAssessment]:
        """Assess risk for all active traders."""
        assessments = []
        active_traders = self.trader_repo.get_active_traders(
            min_trades=self.config.risk.min_trades,
            days=self.config.risk.lookback_days
        )

        for trader_id in active_traders:
            try:
                assessment = self.assess_trader_risk(trader_id)
                assessments.append(assessment)
            except (InsufficientDataError, ModelNotFoundError) as e:
                logger.warning(f"Skipping trader {trader_id}: {str(e)}")
            except Exception as e:
                logger.error(f"Failed to assess trader {trader_id}: {str(e)}")

        return assessments

    def get_high_risk_traders(self) -> List[RiskAssessment]:
        """Get all traders with high or critical risk."""
        assessments = self.assess_all_traders()
        return [
            a for a in assessments
            if a.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
        ]

    def generate_risk_alerts(self,
                           assessments: List[RiskAssessment]) -> List[RiskAlert]:
        """Generate alerts for risk assessments."""
        alerts = []

        for assessment in assessments:
            if assessment.is_actionable():
                alert = RiskAlert(
                    trader_id=assessment.trader_id,
                    alert_type='risk_threshold_exceeded',
                    severity=assessment.risk_level,
                    message=assessment.get_recommendation(),
                    timestamp=assessment.assessment_date,
                    metrics={
                        'risk_score': assessment.risk_score,
                        'var_95': assessment.var_95,
                        'expected_shortfall': assessment.expected_shortfall
                    }
                )
                alerts.append(alert)

        return alerts

    def _calculate_var_es(self, trader_id: int,
                         model_data: dict,
                         metrics: TradingMetrics) -> tuple:
        """Calculate Value at Risk and Expected Shortfall."""
        # This would use the actual model predictions
        # Simplified calculation for now
        if 'model' in model_data and 'predictions' in model_data:
            predictions = model_data['predictions']
            var_95 = np.percentile(predictions, 5)
            es = predictions[predictions <= var_95].mean()
        else:
            # Fallback to simple calculation
            var_95 = metrics.total_pnl * 0.05
            es = var_95 * 1.2

        return abs(var_95), abs(es)

    def _calculate_max_loss(self, trader_id: int,
                           var_95: float,
                           es: float) -> float:
        """Calculate maximum probable loss."""
        # Weighted average of VaR and ES
        return var_95 * 0.7 + es * 0.3

    def _calculate_confidence(self, model_data: dict,
                            metrics: TradingMetrics) -> float:
        """Calculate model confidence score."""
        confidence = 1.0

        # Reduce confidence for low trade counts
        if metrics.total_trades < 100:
            confidence *= 0.8
        elif metrics.total_trades < 50:
            confidence *= 0.6

        # Check model performance metrics if available
        if 'validation_score' in model_data:
            confidence *= model_data['validation_score']

        return min(confidence, 1.0)
