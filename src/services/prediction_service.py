"""Service for generating predictions."""

import logging
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any
from datetime import datetime

from ..repositories.model_repository import ModelRepository
from ..repositories.trader_repository import TraderRepository
from ..models.domain import PredictionResult, Prediction, PredictionStatus
from ..exceptions import ModelNotFoundError, PredictionError
from ..config.config_manager import ConfigManager

logger = logging.getLogger(__name__)


class PredictionService:
    """Service for generating and managing predictions."""

    def __init__(self,
                 model_repo: ModelRepository,
                 trader_repo: TraderRepository,
                 config: ConfigManager):
        """Initialize prediction service."""
        self.model_repo = model_repo
        self.trader_repo = trader_repo
        self.config = config

    def generate_prediction(self, trader_id: int) -> PredictionResult:
        """
        Generate prediction for a trader.

        Args:
            trader_id: The trader ID

        Returns:
            PredictionResult object

        Raises:
            ModelNotFoundError: If model not found
            PredictionError: If prediction fails
        """
        try:
            # Load model
            model_data = self.model_repo.load_model(trader_id)
            if not model_data:
                raise ModelNotFoundError(trader_id)

            # Get recent data for prediction
            recent_fills = self.trader_repo.get_trader_fills(
                trader_id,
                start_date=datetime.now() - pd.Timedelta(days=30)
            )

            if recent_fills.empty:
                raise PredictionError(trader_id, "No recent data available")

            # Generate features (simplified)
            features = self._generate_features(recent_fills)

            # Make prediction
            model = model_data.get('model')
            if model is None:
                raise PredictionError(trader_id, "Model object not found")

            prediction = model.predict(features)[0]
            prediction_proba = model.predict_proba(features)[0] if hasattr(model, 'predict_proba') else [0.5, 0.5]

            # Calculate risk metrics
            var_95 = self._calculate_var(recent_fills, confidence=0.95)
            loss_prob = prediction_proba[1] if len(prediction_proba) > 1 else 0.5
            expected_return = self._calculate_expected_return(recent_fills)
            confidence_interval = self._calculate_confidence_interval(recent_fills)

            return PredictionResult(
                trader_id=trader_id,
                prediction_date=datetime.now(),
                var_prediction=var_95,
                loss_probability=loss_prob,
                expected_return=expected_return,
                confidence_interval=confidence_interval,
                risk_metrics={
                    'model_confidence': model_data.get('validation_score', 0.0),
                    'data_points': len(recent_fills),
                    'prediction_raw': float(prediction)
                },
                model_metadata={
                    'model_version': model_data.get('version', 'unknown'),
                    'trained_date': str(model_data.get('trained_date', 'unknown')),
                    'features_used': model_data.get('feature_names', [])
                }
            )

        except Exception as e:
            logger.error(f"Prediction failed for trader {trader_id}: {str(e)}")
            raise PredictionError(trader_id, str(e))

    def generate_batch_predictions(self, trader_ids: List[int]) -> List[PredictionResult]:
        """
        Generate predictions for multiple traders.

        Args:
            trader_ids: List of trader IDs

        Returns:
            List of PredictionResult objects
        """
        predictions = []

        for trader_id in trader_ids:
            try:
                prediction = self.generate_prediction(trader_id)
                predictions.append(prediction)
            except (ModelNotFoundError, PredictionError) as e:
                logger.warning(f"Skipping trader {trader_id}: {str(e)}")
                continue

        return predictions

    def get_high_risk_predictions(self,
                                threshold: float = 0.6) -> List[PredictionResult]:
        """
        Get predictions with high risk.

        Args:
            threshold: Loss probability threshold

        Returns:
            List of high-risk predictions
        """
        active_traders = self.trader_repo.get_active_traders(
            min_trades=self.config.risk.min_trades,
            days=self.config.risk.lookback_days
        )

        predictions = self.generate_batch_predictions(active_traders)

        return [p for p in predictions if p.loss_probability > threshold]

    def _generate_features(self, fills_df: pd.DataFrame) -> np.ndarray:
        """Generate features from fills data."""
        # Simplified feature generation
        features = []

        # Basic statistics
        features.append(fills_df['pnl'].mean() if 'pnl' in fills_df else 0)
        features.append(fills_df['pnl'].std() if 'pnl' in fills_df else 0)
        features.append(len(fills_df))

        # Win rate
        if 'pnl' in fills_df:
            wins = len(fills_df[fills_df['pnl'] > 0])
            features.append(wins / len(fills_df) if len(fills_df) > 0 else 0)
        else:
            features.append(0)

        # Recent performance
        if 'pnl' in fills_df and len(fills_df) > 5:
            recent_pnl = fills_df.tail(5)['pnl'].mean()
            features.append(recent_pnl)
        else:
            features.append(0)

        return np.array(features).reshape(1, -1)

    def _calculate_var(self, fills_df: pd.DataFrame,
                      confidence: float = 0.95) -> float:
        """Calculate Value at Risk."""
        if 'pnl' not in fills_df or fills_df.empty:
            return 0.0

        # Simple historical VaR
        pnl_values = fills_df['pnl'].values
        var_percentile = (1 - confidence) * 100
        var_value = np.percentile(pnl_values, var_percentile)

        return abs(var_value)

    def _calculate_expected_return(self, fills_df: pd.DataFrame) -> float:
        """Calculate expected return."""
        if 'pnl' not in fills_df or fills_df.empty:
            return 0.0

        return fills_df['pnl'].mean()

    def _calculate_confidence_interval(self, fills_df: pd.DataFrame,
                                      confidence: float = 0.95) -> tuple:
        """Calculate confidence interval for returns."""
        if 'pnl' not in fills_df or fills_df.empty:
            return (0.0, 0.0)

        pnl_values = fills_df['pnl'].values
        mean = np.mean(pnl_values)
        std = np.std(pnl_values)

        # Simple confidence interval
        z_score = 1.96  # 95% confidence
        margin = z_score * std / np.sqrt(len(pnl_values))

        return (mean - margin, mean + margin)
