"""
Risk Predictor Module

This module provides a clean interface for risk prediction based on the causal impact evaluation approach.
It handles model loading, feature preparation, and prediction generation for trader-specific risk assessment.
"""

import pandas as pd
import numpy as np
import pickle
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class RiskPredictor:
    """
    Risk predictor using position sizing optimization approach.
    Provides position sizing predictions and loss probabilities for trader risk assessment.
    """

    def __init__(self, config: Dict):
        """
        Initialize risk predictor with configuration.

        Args:
            config: Configuration dictionary containing paths and settings
        """
        self.config = config
        self.models_path = Path(config['paths']['model_dir'])
        self.thresholds_path = Path(config['paths'].get('optimal_thresholds_dir', 'configs/optimal_thresholds')) / 'optimal_thresholds.json'

        # Storage for loaded models and thresholds
        self.trader_models = {}
        self.optimal_thresholds = {}

        # Load optimal thresholds
        self._load_optimal_thresholds()

    def _load_optimal_thresholds(self):
        """Load optimal thresholds for each trader from configuration."""
        try:
            with open(self.thresholds_path, 'r') as f:
                threshold_data = json.load(f)

            for trader_threshold in threshold_data['thresholds']:
                trader_id = int(trader_threshold['trader_id'])
                self.optimal_thresholds[trader_id] = {
                    'position_threshold': trader_threshold.get('position_threshold', 0.5),
                    'loss_prob_threshold': trader_threshold['loss_prob_threshold']
                }

            logger.info(f"Loaded optimal thresholds for {len(self.optimal_thresholds)} traders")

        except Exception as e:
            logger.error(f"Error loading optimal thresholds: {e}")
            # Set default thresholds if loading fails
            for trader_id in self.config.get('active_traders', []):
                self.optimal_thresholds[trader_id] = {
                    'position_threshold': 0.5,
                    'loss_prob_threshold': 0.15
                }

    def load_trader_model(self, trader_id: int) -> bool:
        """
        Load model for a specific trader.

        Args:
            trader_id: Trader account ID

        Returns:
            True if model loaded successfully, False otherwise
        """
        model_path = self.models_path / f'{trader_id}_tuned_validated.pkl'

        try:
            with open(model_path, 'rb') as f:
                self.trader_models[trader_id] = pickle.load(f)
            logger.info(f"Loaded model for trader {trader_id}")
            return True

        except Exception as e:
            logger.error(f"Error loading model for trader {trader_id}: {e}")
            return False

    def load_all_trader_models(self):
        """Load models for all active traders."""
        logger.info("Loading trader-specific models...")

        for trader_id in self.config.get('active_traders', []):
            self.load_trader_model(trader_id)

        logger.info(f"Successfully loaded {len(self.trader_models)} trader models")

    def prepare_features(self, data_series: pd.Series, trader_id: int) -> Optional[pd.DataFrame]:
        """
        Prepare features for prediction based on model requirements.

        Args:
            data_series: Feature data for the trader
            trader_id: Trader account ID

        Returns:
            Prepared feature DataFrame or None if preparation fails
        """
        if trader_id not in self.trader_models:
            logger.warning(f"No model found for trader {trader_id}")
            return None

        model_data = self.trader_models[trader_id]

        try:
            # Get feature names from model
            feature_names = model_data.get('feature_names', [])

            # Convert series to dataframe
            data_df = pd.DataFrame([data_series])

            # Prepare features for prediction
            if feature_names:
                missing_features = [f for f in feature_names if f not in data_df.columns]
                if missing_features:
                    logger.warning(f"Missing features for trader {trader_id}: {missing_features}")
                    available_features = [f for f in feature_names if f in data_df.columns]
                    if not available_features:
                        logger.error(f"No features available for trader {trader_id}")
                        return None
                    X_test = data_df[available_features]
                else:
                    X_test = data_df[feature_names]
            else:
                # Fallback: use all available features except metadata columns
                feature_cols = [col for col in data_df.columns if col not in [
                    'trader_id', 'date', 'target_pnl', 'target_large_loss', 'trade_date', 'daily_pnl'
                ]]
                X_test = data_df[feature_cols]

            return X_test

        except Exception as e:
            logger.error(f"Error preparing features for trader {trader_id}: {e}")
            return None

    def generate_prediction(self, trader_id: int, data_series: pd.Series) -> Optional[Dict[str, float]]:
        """
        Generate position sizing and loss probability prediction for a trader.

        Args:
            trader_id: Trader account ID
            data_series: Feature data for the trader

        Returns:
            Dictionary with prediction results or None if prediction fails
        """
        if trader_id not in self.trader_models:
            if not self.load_trader_model(trader_id):
                return None

        model_data = self.trader_models[trader_id]

        try:
            # Prepare features
            X_test = self.prepare_features(data_series, trader_id)
            if X_test is None:
                return None

            # Get models
            var_model = model_data.get('var_model')
            classification_model = model_data.get('classification_model')

            if var_model is None or classification_model is None:
                logger.error(f"Missing var_model or classification_model for trader {trader_id}")
                return None

            # Check if this is a position sizing model
            using_position_sizing = model_data.get('using_position_sizing', False) or model_data.get('model_type', 'legacy') == 'position_sizing'

            # Generate predictions
            raw_position_prediction = var_model.predict(X_test)[0]
            loss_probability = classification_model.predict_proba(X_test)[0, 1]  # Probability of loss

            if using_position_sizing:
                # For position sizing models, ensure output is in valid range (0.0 to 1.5)
                position_prediction = max(0.0, min(1.5, raw_position_prediction))
                logger.debug(f"Position sizing prediction for trader {trader_id}: {position_prediction:.3f}")
            else:
                # Legacy VAR model - convert to position sizing approximation
                # Negative VAR values suggest reducing position
                if raw_position_prediction <= -5000:
                    position_prediction = 0.3  # Reduce position significantly
                elif raw_position_prediction <= -2000:
                    position_prediction = 0.6  # Reduce position moderately
                elif raw_position_prediction <= 0:
                    position_prediction = 0.8  # Slightly reduce position
                else:
                    position_prediction = 1.0  # Normal position
                logger.debug(f"Legacy VAR converted to position sizing for trader {trader_id}: VAR={raw_position_prediction:.0f} -> Position={position_prediction:.3f}")

            # Calculate model confidence
            model_confidence = max(loss_probability, 1 - loss_probability)

            return {
                'predicted_position_size': position_prediction,
                'loss_probability': loss_probability,
                'model_confidence': model_confidence,
                'feature_count': len(X_test.columns)
            }

        except Exception as e:
            logger.error(f"Error generating prediction for trader {trader_id}: {e}")
            return None

    def generate_predictions_batch(self, trader_data_dict: Dict[int, pd.Series]) -> Dict[int, Dict[str, float]]:
        """
        Generate predictions for multiple traders in batch.

        Args:
            trader_data_dict: Dictionary mapping trader_id to feature data

        Returns:
            Dictionary mapping trader_id to prediction results
        """
        predictions = {}

        for trader_id, data_series in trader_data_dict.items():
            prediction = self.generate_prediction(trader_id, data_series)
            if prediction is not None:
                predictions[trader_id] = prediction
                logger.info(f"Generated prediction for trader {trader_id}: Position={prediction['predicted_position_size']:.2%}, P(Loss)={prediction['loss_probability']:.3f}")

        logger.info(f"Generated predictions for {len(predictions)} traders")
        return predictions

    def classify_position_level(self, trader_id: int, position_prediction: float, loss_probability: float) -> str:
        """
        Classify position level using position sizing logic.

        Args:
            trader_id: Trader account ID
            position_prediction: Position sizing prediction value
            loss_probability: Loss probability prediction

        Returns:
            Position level ('reduce', 'conservative', 'normal', 'aggressive')
        """
        # Get trader-specific thresholds
        trader_thresholds = self.optimal_thresholds.get(trader_id, {
            'position_threshold': 0.5,
            'loss_prob_threshold': 0.15
        })

        position_threshold = trader_thresholds['position_threshold']
        loss_prob_threshold = trader_thresholds['loss_prob_threshold']

        # Classify based on position size and loss probability
        if position_prediction <= 0.5 or loss_probability >= loss_prob_threshold:
            return 'reduce'  # High risk - reduce position
        elif position_prediction <= 0.8:
            return 'conservative'
        elif position_prediction >= 1.2:
            return 'aggressive'
        else:
            return 'normal'


    def generate_intervention_recommendation(self, trader_id: int, position_prediction: float, loss_probability: float) -> Dict[str, Any]:
        """
        Generate position sizing recommendation based on thresholds.

        Args:
            trader_id: Trader account ID
            position_prediction: Position sizing prediction value
            loss_probability: Loss probability prediction

        Returns:
            Dictionary with intervention details
        """
        # Get trader-specific thresholds
        trader_thresholds = self.optimal_thresholds.get(trader_id, {})
        var_threshold = trader_thresholds.get('var_threshold', -5000)
        loss_prob_threshold = trader_thresholds.get('loss_prob_threshold', 0.15)

        # Check if intervention is recommended
        should_intervene = (
            (var_prediction <= var_threshold) or
            (loss_probability >= loss_prob_threshold)
        )

        # Determine severity if intervention is needed
        severity = 'standard'
        if should_intervene:
            var_ratio = var_prediction / var_threshold if var_threshold != 0 else 1
            prob_ratio = loss_probability / loss_prob_threshold if loss_prob_threshold != 0 else 1

            if var_ratio <= 0.5 or prob_ratio >= 2.0:
                severity = 'critical'

        return {
            'should_intervene': should_intervene,
            'severity': severity,
            'var_threshold': var_threshold,
            'loss_prob_threshold': loss_prob_threshold,
            'var_ratio': var_prediction / var_threshold if var_threshold != 0 else 1,
            'prob_ratio': loss_probability / loss_prob_threshold if loss_prob_threshold != 0 else 1,
            'recommendation': self._format_recommendation(should_intervene, severity, var_prediction, loss_probability, var_threshold, loss_prob_threshold)
        }

    def _format_recommendation(self, should_intervene: bool, severity: str, var_pred: float, loss_prob: float, var_thresh: float, loss_thresh: float) -> str:
        """Format intervention recommendation message."""
        if not should_intervene:
            return "Normal trading conditions. No intervention required."

        if severity == 'critical':
            return f"CRITICAL INTERVENTION: Model strongly recommends reducing position. VaR: ${var_pred:,.0f} (threshold: ${var_thresh:,.0f}), Loss Prob: {loss_prob:.1%} (threshold: {loss_thresh:.1%})"
        else:
            return f"INTERVENTION RECOMMENDED: Consider reducing position size by 50% based on model prediction."

    def get_model_info(self, trader_id: int) -> Optional[Dict[str, Any]]:
        """
        Get information about a trader's model.

        Args:
            trader_id: Trader account ID

        Returns:
            Model information dictionary or None
        """
        if trader_id not in self.trader_models:
            return None

        model_data = self.trader_models[trader_id]

        return {
            'feature_names': model_data.get('feature_names', []),
            'feature_count': len(model_data.get('feature_names', [])),
            'test_metrics': model_data.get('test_metrics', {}),
            'has_var_model': 'var_model' in model_data,
            'has_classification_model': 'classification_model' in model_data
        }

    def validate_model_predictions(self, trader_id: int, test_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate model predictions against test data (for debugging/testing).

        Args:
            trader_id: Trader account ID
            test_data: Test dataset for validation

        Returns:
            Validation results
        """
        if trader_id not in self.trader_models:
            if not self.load_trader_model(trader_id):
                return {'error': 'Model not available'}

        model_data = self.trader_models[trader_id]

        try:
            # Get feature names from model
            feature_names = model_data.get('feature_names', [])

            # Prepare test features
            X_test = test_data[feature_names]

            # Get models
            var_model = model_data.get('var_model')
            classification_model = model_data.get('classification_model')

            # Generate predictions
            var_predictions = var_model.predict(X_test)
            loss_probabilities = classification_model.predict_proba(X_test)[:, 1]

            # Create validation results
            validation_results = {
                'trader_id': trader_id,
                'num_predictions': len(var_predictions),
                'var_predictions_range': {
                    'min': float(np.min(var_predictions)),
                    'max': float(np.max(var_predictions)),
                    'mean': float(np.mean(var_predictions))
                },
                'loss_probabilities_range': {
                    'min': float(np.min(loss_probabilities)),
                    'max': float(np.max(loss_probabilities)),
                    'mean': float(np.mean(loss_probabilities))
                },
                'feature_count': len(feature_names),
                'model_performance': model_data.get('test_metrics', {})
            }

            return validation_results

        except Exception as e:
            logger.error(f"Error validating model predictions for trader {trader_id}: {e}")
            return {'error': str(e)}
