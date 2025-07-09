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
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class RiskPredictor:
    """
    Risk predictor using causal impact evaluation approach.
    Provides VaR predictions and loss probabilities for trader risk assessment.
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
                    'var_threshold': trader_threshold['var_threshold'],
                    'loss_prob_threshold': trader_threshold['loss_prob_threshold']
                }

            logger.info(f"Loaded optimal thresholds for {len(self.optimal_thresholds)} traders")

        except Exception as e:
            logger.error(f"Error loading optimal thresholds: {e}")
            # Set default thresholds if loading fails
            for trader_id in self.config.get('active_traders', []):
                self.optimal_thresholds[trader_id] = {
                    'var_threshold': -5000,
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
        Generate VaR and loss probability prediction for a trader.

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

            # Generate predictions using causal impact approach
            var_prediction = var_model.predict(X_test)[0]
            loss_probability = classification_model.predict_proba(X_test)[0, 1]  # Probability of loss

            # Calculate model confidence
            model_confidence = max(loss_probability, 1 - loss_probability)

            return {
                'var_prediction': var_prediction,
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
                logger.info(f"Generated prediction for trader {trader_id}: VaR=${prediction['var_prediction']:.2f}, P(Loss)={prediction['loss_probability']:.3f}")

        logger.info(f"Generated predictions for {len(predictions)} traders")
        return predictions

    def classify_risk_level(self, trader_id: int, var_prediction: float, loss_probability: float) -> str:
        """
        Classify risk level using causal impact intervention logic.

        Args:
            trader_id: Trader account ID
            var_prediction: VaR prediction value
            loss_probability: Loss probability prediction

        Returns:
            Risk level ('high' or 'low')
        """
        # Get trader-specific thresholds
        trader_thresholds = self.optimal_thresholds.get(trader_id, {
            'var_threshold': -5000,
            'loss_prob_threshold': 0.15
        })

        var_threshold = trader_thresholds['var_threshold']
        loss_prob_threshold = trader_thresholds['loss_prob_threshold']

        # Apply intervention logic from causal impact evaluation
        # High risk if model suggests intervention (don't trade)
        should_intervene = (
            (var_prediction <= var_threshold) or
            (loss_probability >= loss_prob_threshold)
        )

        return 'high' if should_intervene else 'low'

    def generate_intervention_recommendation(self, trader_id: int, var_prediction: float, loss_probability: float) -> Dict[str, Any]:
        """
        Generate intervention recommendation based on thresholds.

        Args:
            trader_id: Trader account ID
            var_prediction: VaR prediction value
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
