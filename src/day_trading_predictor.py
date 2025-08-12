"""
Day Trading Predictor Module

This module provides a clean interface for day trading predictions using the enhanced
models focused on position sizing, volatility regimes, and risk management signals.
"""

import pandas as pd
import numpy as np
import pickle
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class DayTradingPredictor:
    """
    Day trading predictor using enhanced models for position sizing and risk management.
    Provides actionable signals for day traders including position sizing recommendations,
    volatility regime predictions, and risk management alerts.
    """

    def __init__(self, config: Dict, model_suffix: str = "_day_trading"):
        """
        Initialize day trading predictor with configuration.

        Args:
            config: Configuration dictionary containing paths and settings
            model_suffix: Model file suffix (default: "_day_trading")
        """
        self.config = config
        self.models_path = Path(config['paths']['model_dir'])
        self.model_suffix = model_suffix

        # Storage for loaded models
        self.trader_models = {}

        # Define signal mappings
        self.signal_mappings = {
            'position_sizing': {
                0: 'normal_size',
                1: 'reduce_size'
            },
            'favorable_conditions': {
                0: 'normal_conditions',
                1: 'favorable_conditions'
            },
            'stop_trading': {
                0: 'continue_trading',
                1: 'stop_trading'
            },
            'volatility_regime': {
                0: 'low_volatility',
                1: 'normal_volatility',
                2: 'high_volatility'
            },
            'underperformance': {
                0: 'normal_performance',
                1: 'likely_underperform'
            }
        }

    def load_trader_model(self, trader_id: int) -> bool:
        """
        Load day trading models for a specific trader.

        Args:
            trader_id: Trader account ID

        Returns:
            True if models loaded successfully, False otherwise
        """
        model_path = self.models_path / f'{trader_id}{self.model_suffix}.pkl'

        try:
            with open(model_path, 'rb') as f:
                self.trader_models[trader_id] = pickle.load(f)
            logger.info(f"Loaded day trading models for trader {trader_id}")
            return True

        except Exception as e:
            logger.error(f"Error loading day trading models for trader {trader_id}: {e}")
            return False

    def load_all_trader_models(self):
        """Load day trading models for all active traders."""
        logger.info("Loading day trading models...")

        for trader_id in self.config.get('active_traders', []):
            self.load_trader_model(trader_id)

        logger.info(f"Successfully loaded day trading models for {len(self.trader_models)} traders")

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
            logger.warning(f"No day trading models found for trader {trader_id}")
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
                    'trader_id', 'date', 'target_pnl', 'target_large_loss', 'trade_date', 'daily_pnl',
                    'target_reduce_size', 'target_increase_size', 'target_stop_trading',
                    'target_vol_regime', 'target_underperform', 'target_outperform'
                ]]
                X_test = data_df[feature_cols]

            return X_test

        except Exception as e:
            logger.error(f"Error preparing features for trader {trader_id}: {e}")
            return None

    def generate_prediction(self, trader_id: int, data_series: pd.Series) -> Optional[Dict[str, Any]]:
        """
        Generate comprehensive day trading predictions for a trader.

        Args:
            trader_id: Trader account ID
            data_series: Feature data for the trader

        Returns:
            Dictionary with day trading predictions or None if prediction fails
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
            models = model_data.get('models', {})
            if not models:
                logger.error(f"No trained models found for trader {trader_id}")
                return None

            predictions = {}
            probabilities = {}

            # Generate predictions for each target
            for target_name, model_info in models.items():
                model = model_info.get('model')
                target_config = model_info.get('target_config', {})

                if model is None:
                    continue

                try:
                    if target_config.get('model_type') == 'classification':
                        # Binary classification
                        pred_proba = model.predict_proba(X_test)[0]
                        prediction = model.predict(X_test)[0]

                        # Store probability of positive class
                        probabilities[target_name] = pred_proba[1] if len(pred_proba) > 1 else pred_proba[0]
                        predictions[target_name] = int(prediction)

                    elif target_config.get('model_type') == 'multiclass':
                        # Multiclass (volatility regime)
                        pred_proba = model.predict_proba(X_test)[0]
                        prediction = model.predict(X_test)[0]

                        probabilities[target_name] = pred_proba.tolist()
                        predictions[target_name] = int(prediction)

                except Exception as e:
                    logger.warning(f"Error predicting {target_name} for trader {trader_id}: {e}")
                    continue

            # Convert predictions to actionable signals
            signals = self._convert_to_signals(predictions, probabilities)

            # Calculate overall risk score and position multiplier
            risk_score, position_multiplier = self._calculate_risk_metrics(
                predictions, probabilities, data_series
            )

            return {
                'trader_id': trader_id,
                'predictions': predictions,
                'probabilities': probabilities,
                'signals': signals,
                'risk_score': risk_score,
                'position_multiplier': position_multiplier,
                'feature_count': len(X_test.columns),
                'model_count': len(models)
            }

        except Exception as e:
            logger.error(f"Error generating day trading prediction for trader {trader_id}: {e}")
            return None

    def _convert_to_signals(self, predictions: Dict, probabilities: Dict) -> Dict[str, Any]:
        """Convert raw predictions to actionable signals"""
        signals = {}

        for target_name, prediction in predictions.items():
            if target_name in self.signal_mappings:
                signal_mapping = self.signal_mappings[target_name]
                signals[target_name] = {
                    'signal': signal_mapping.get(prediction, 'unknown'),
                    'confidence': probabilities.get(target_name, 0.5),
                    'raw_prediction': prediction
                }

        return signals

    def _calculate_risk_metrics(self, predictions: Dict, probabilities: Dict,
                              data_series: pd.Series) -> tuple[float, float]:
        """Calculate overall risk score and position sizing multiplier"""

        # Base risk score calculation
        risk_factors = []

        # Factor 1: Position sizing signal
        if 'position_sizing' in probabilities:
            reduce_prob = probabilities['position_sizing']
            risk_factors.append(reduce_prob)

        # Factor 2: Stop trading signal
        if 'stop_trading' in probabilities:
            stop_prob = probabilities['stop_trading']
            risk_factors.append(stop_prob * 2)  # Double weight for stop signal

        # Factor 3: Volatility regime
        if 'volatility_regime' in predictions:
            vol_regime = predictions['volatility_regime']
            if vol_regime == 2:  # High volatility
                risk_factors.append(0.8)
            elif vol_regime == 0:  # Low volatility
                risk_factors.append(-0.3)  # Negative = less risky

        # Factor 4: Underperformance likelihood
        if 'underperformance' in probabilities:
            underperform_prob = probabilities['underperformance']
            risk_factors.append(underperform_prob)

        # Calculate weighted risk score (0-1 scale)
        if risk_factors:
            risk_score = np.clip(np.mean(risk_factors), 0, 1)
        else:
            risk_score = 0.5  # Neutral if no factors

        # Calculate position multiplier based on signals
        base_multiplier = 1.0

        # Adjust based on primary signals
        if predictions.get('stop_trading') == 1:
            position_multiplier = 0.0  # Stop trading
        elif predictions.get('position_sizing') == 1:
            position_multiplier = 0.5  # Reduce size
        elif predictions.get('favorable_conditions') == 1:
            position_multiplier = 1.5  # Increase size
        else:
            position_multiplier = base_multiplier

        # Adjust for volatility regime
        vol_regime = predictions.get('volatility_regime', 1)
        if vol_regime == 2:  # High volatility
            position_multiplier *= 0.7
        elif vol_regime == 0:  # Low volatility
            position_multiplier *= 1.2

        # Use current position_size_multiplier from features if available
        if hasattr(data_series, 'position_size_multiplier'):
            feature_multiplier = data_series.position_size_multiplier
            if not pd.isna(feature_multiplier) and feature_multiplier > 0:
                position_multiplier = (position_multiplier + feature_multiplier) / 2

        # Clamp position multiplier to reasonable range
        position_multiplier = np.clip(position_multiplier, 0.0, 2.0)

        return float(risk_score), float(position_multiplier)

    def generate_predictions_batch(self, trader_data_dict: Dict[int, pd.Series]) -> Dict[int, Dict]:
        """
        Generate day trading predictions for multiple traders in batch.

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

                # Log key signals
                signals = prediction.get('signals', {})
                position_mult = prediction.get('position_multiplier', 1.0)
                risk_score = prediction.get('risk_score', 0.5)

                logger.info(f"Day Trading Signals for trader {trader_id}: "
                          f"Position={position_mult:.2f}x, Risk={risk_score:.3f}")

                # Log critical alerts
                if signals.get('stop_trading', {}).get('signal') == 'stop_trading':
                    logger.warning(f"STOP TRADING ALERT for trader {trader_id}")

        logger.info(f"Generated day trading predictions for {len(predictions)} traders")

        return predictions

    def get_trading_recommendations(self, trader_id: int, data_series: pd.Series) -> Dict[str, str]:
        """
        Get human-readable trading recommendations for a trader.

        Args:
            trader_id: Trader account ID
            data_series: Feature data for the trader

        Returns:
            Dictionary with trading recommendations
        """
        prediction = self.generate_prediction(trader_id, data_series)
        if prediction is None:
            return {"error": "Unable to generate predictions"}

        signals = prediction.get('signals', {})
        position_multiplier = prediction.get('position_multiplier', 1.0)
        risk_score = prediction.get('risk_score', 0.5)

        recommendations = {
            'position_sizing': f"Use {position_multiplier:.1f}x normal position size",
            'overall_risk': f"Risk level: {risk_score:.1%}",
        }

        # Add specific recommendations based on signals
        if signals.get('stop_trading', {}).get('signal') == 'stop_trading':
            recommendations['action'] = "🛑 STOP TRADING - High risk conditions detected"
        elif signals.get('position_sizing', {}).get('signal') == 'reduce_size':
            recommendations['action'] = "⚠️  REDUCE POSITION SIZE - Unfavorable conditions"
        elif signals.get('favorable_conditions', {}).get('signal') == 'favorable_conditions':
            recommendations['action'] = "✅ FAVORABLE CONDITIONS - Consider larger positions"
        else:
            recommendations['action'] = "➡️  NORMAL TRADING - Standard position sizing"

        # Add volatility info
        vol_signal = signals.get('volatility_regime', {}).get('signal', 'normal_volatility')
        recommendations['volatility'] = f"Market volatility: {vol_signal.replace('_', ' ')}"

        return recommendations
