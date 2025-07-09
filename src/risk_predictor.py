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

    def calculate_linear_risk_score(self, var_prediction: float, loss_probability: float,
                                  alpha: float, beta: float, bias: float,
                                  var_range: Tuple[float, float]) -> float:
        """
        Calculate risk score using linear regression formula.

        Formula: risk_score = alpha * normalized_VaR + beta * normalized_LossProb + bias

        Args:
            var_prediction: VaR prediction value (typically negative)
            loss_probability: Loss probability prediction (0-1 scale)
            alpha: Linear coefficient for VaR component
            beta: Linear coefficient for loss probability component
            bias: Bias term
            var_range: Tuple of (min_var, max_var) for normalization

        Returns:
            Linear risk score (unbounded)
        """
        var_min, var_max = var_range

        # Normalize VaR to [0, 1] (higher = higher risk)
        if var_max != var_min:
            # Invert VaR normalization (lower VaR = higher risk)
            normalized_var = (var_max - var_prediction) / (var_max - var_min)
            normalized_var = max(0, min(1, normalized_var))  # Clamp to 0-1
        else:
            normalized_var = 0.5  # Neutral if no variance

        # Loss probability is already 0-1 scale (higher = higher risk)
        normalized_loss_prob = max(0, min(1, loss_probability))

        # Calculate linear risk score
        risk_score = alpha * normalized_var + beta * normalized_loss_prob + bias

        return risk_score

    def classify_risk_level_weighted(self, var_prediction: float, loss_probability: float,
                                   alpha: float = 0.6, beta: float = 0.4,
                                   thresholds: Dict[str, float] = None,
                                   var_range: Tuple[float, float] = None) -> str:
        """
        Classify risk level using weighted risk score with 4 levels.

        Args:
            var_prediction: VaR prediction value
            loss_probability: Loss probability prediction
            alpha: Weight for VaR component
            beta: Weight for loss probability component
            thresholds: Risk level thresholds
            var_range: Tuple of (min_var, max_var) for normalization

        Returns:
            Risk level ('High Risk', 'Medium Risk', 'Low Risk', or 'Neutral')
        """
        if thresholds is None:
            thresholds = {
                'high_threshold': 0.7,
                'medium_threshold': 0.5,
                'low_threshold': 0.3
            }

        # Calculate weighted risk score
        risk_score = self.calculate_weighted_risk_score(
            var_prediction, loss_probability, alpha, beta, var_range, 'sigmoid'
        )

        # Classify based on thresholds
        if risk_score >= thresholds['high_threshold']:
            return 'High Risk'
        elif risk_score >= thresholds['medium_threshold']:
            return 'Medium Risk'
        elif risk_score >= thresholds['low_threshold']:
            return 'Low Risk'
        else:
            return 'Neutral'

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

    def train_linear_regression_coefficients(self, trader_id: int = None) -> Dict[str, Any]:
        """
        Train alpha, beta, and bias coefficients using simple linear regression.

        Formula: risk_score = alpha * normalized_VaR + beta * normalized_LossProb + bias

        METHODOLOGY:
        1. Load TRAINING data to fit linear regression
        2. Evaluate performance on separate TEST data
        3. Clear train/test separation - no data leakage

        Args:
            trader_id: Specific trader ID to train coefficients for (None for all traders)

        Returns:
            Dictionary with trained coefficients and evaluation results
        """
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score
        from sklearn.preprocessing import StandardScaler

        if trader_id is not None:
            trader_ids = [trader_id]
        else:
            trader_ids = self.config.get('active_traders', [])

        # Lists to store training and test data separately
        train_features = []
        train_labels = []
        test_features = []
        test_labels = []

        # Load training and test data for all traders
        for tid in trader_ids:
            if not self.load_trader_model(tid):
                logger.warning(f"Could not load model for trader {tid}")
                continue

            # Load TRAINING data
            train_data_path = Path(self.config['paths']['processed_features']) / str(tid) / "train_data.parquet"
            test_data_path = Path(self.config['paths']['processed_features']) / str(tid) / "test_data.parquet"

            if not train_data_path.exists() or not test_data_path.exists():
                logger.warning(f"Missing train/test data for trader {tid}")
                continue

            try:
                # Load data
                train_data = pd.read_parquet(train_data_path)
                test_data = pd.read_parquet(test_data_path)

                # Verify no overlap between train and test dates
                train_dates = set(train_data['date'])
                test_dates = set(test_data['date'])
                overlap = train_dates.intersection(test_dates)
                if overlap:
                    logger.error(f"Data leakage detected for trader {tid}: {len(overlap)} overlapping dates")
                    continue

                logger.info(f"Trader {tid}: Train dates {train_data['date'].min()} to {train_data['date'].max()}")
                logger.info(f"Trader {tid}: Test dates {test_data['date'].min()} to {test_data['date'].max()}")

                # Get model and generate predictions
                model_data = self.trader_models[tid]
                feature_names = model_data.get('feature_names', [])
                var_model = model_data.get('var_model')
                classification_model = model_data.get('classification_model')

                # Generate predictions for TRAINING data
                X_train = train_data[feature_names]
                train_var_pred = var_model.predict(X_train)
                train_loss_prob = classification_model.predict_proba(X_train)[:, 1]

                # Get training labels (risk outcome)
                if 'target_large_loss' in train_data.columns:
                    train_risk_labels = train_data['target_large_loss'].values
                else:
                    train_risk_labels = (train_data['daily_pnl'] < 0).astype(int)

                # Generate predictions for TEST data
                X_test = test_data[feature_names]
                test_var_pred = var_model.predict(X_test)
                test_loss_prob = classification_model.predict_proba(X_test)[:, 1]

                # Get test labels
                if 'target_large_loss' in test_data.columns:
                    test_risk_labels = test_data['target_large_loss'].values
                else:
                    test_risk_labels = (test_data['daily_pnl'] < 0).astype(int)

                # Store training data
                train_features.extend(list(zip(train_var_pred, train_loss_prob)))
                train_labels.extend(train_risk_labels)

                # Store test data
                test_features.extend(list(zip(test_var_pred, test_loss_prob)))
                test_labels.extend(test_risk_labels)

                logger.info(f"Loaded {len(train_var_pred)} training and {len(test_var_pred)} test samples for trader {tid}")

            except Exception as e:
                logger.error(f"Error loading data for trader {tid}: {e}")
                continue

        if not train_features or not test_features:
            logger.error("No training or test data available")
            return {'error': 'No data available'}

        # Convert to numpy arrays
        X_train = np.array(train_features)
        y_train = np.array(train_labels)
        X_test = np.array(test_features)
        y_test = np.array(test_labels)

        logger.info(f"Training linear regression on {len(X_train)} training samples")
        logger.info(f"Will evaluate on {len(X_test)} test samples")

        # Normalize VaR feature (column 0)
        var_min = X_train[:, 0].min()
        var_max = X_train[:, 0].max()

        # Normalize VaR to [0, 1] (higher = higher risk)
        X_train_norm = X_train.copy()
        X_test_norm = X_test.copy()

        if var_max != var_min:
            # Invert VaR normalization (lower VaR = higher risk)
            X_train_norm[:, 0] = (var_max - X_train[:, 0]) / (var_max - var_min)
            X_test_norm[:, 0] = (var_max - X_test[:, 0]) / (var_max - var_min)
        else:
            X_train_norm[:, 0] = 0.5
            X_test_norm[:, 0] = 0.5

        # Loss probability (column 1) is already normalized [0, 1]

        # Fit linear regression
        model = LinearRegression()
        model.fit(X_train_norm, y_train)

        # Get coefficients
        alpha = model.coef_[0]  # VaR coefficient
        beta = model.coef_[1]   # Loss probability coefficient
        bias = model.intercept_  # Bias term

        # Make predictions
        train_pred = model.predict(X_train_norm)
        test_pred = model.predict(X_test_norm)

        # Calculate metrics
        train_mse = mean_squared_error(y_train, train_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)

        # Calculate AUC
        try:
            train_auc = roc_auc_score(y_train, train_pred)
            test_auc = roc_auc_score(y_test, test_pred)
        except Exception:
            train_auc = 0.5
            test_auc = 0.5

        # Calculate risk thresholds based on training predictions
        train_pred_sorted = sorted(train_pred)
        thresholds = {
            'high_threshold': np.percentile(train_pred_sorted, 75),
            'medium_threshold': np.percentile(train_pred_sorted, 50),
            'low_threshold': np.percentile(train_pred_sorted, 25)
        }

        # Classification accuracy on test data
        high_risk_test = (test_pred >= thresholds['high_threshold'])
        low_risk_test = (test_pred < thresholds['low_threshold'])

        if high_risk_test.sum() > 0:
            high_risk_accuracy = y_test[high_risk_test].mean()
        else:
            high_risk_accuracy = 0.0

        if low_risk_test.sum() > 0:
            low_risk_accuracy = 1 - y_test[low_risk_test].mean()
        else:
            low_risk_accuracy = 0.0

        results = {
            'alpha': alpha,
            'beta': beta,
            'bias': bias,
            'var_range': (var_min, var_max),
            'thresholds': thresholds,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'traders_used': len(trader_ids),
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_auc': train_auc,
            'test_auc': test_auc,
            'test_high_risk_accuracy': high_risk_accuracy,
            'test_low_risk_accuracy': low_risk_accuracy
        }

        logger.info(f"Linear regression training completed:")
        logger.info(f"  Alpha (VaR): {alpha:.3f}")
        logger.info(f"  Beta (LossProb): {beta:.3f}")
        logger.info(f"  Bias: {bias:.3f}")
        logger.info(f"  Training R²: {train_r2:.3f}")
        logger.info(f"  Test R²: {test_r2:.3f}")
        logger.info(f"  Training AUC: {train_auc:.3f}")
        logger.info(f"  Test AUC: {test_auc:.3f}")

        return results

    def generate_risk_signal_weighted(self, trader_id: int, var_prediction: float,
                                    loss_probability: float, alpha: float = 0.6,
                                    beta: float = 0.4, thresholds: Dict[str, float] = None,
                                    var_range: Tuple[float, float] = None) -> Dict[str, Any]:
        """
        Generate risk signal using weighted formula with 4-level classification.

        Args:
            trader_id: Trader account ID
            var_prediction: VaR prediction value
            loss_probability: Loss probability prediction
            alpha: Weight for VaR component
            beta: Weight for loss probability component
            thresholds: Risk level thresholds
            var_range: Tuple of (min_var, max_var) for normalization

        Returns:
            Dictionary with risk signal information
        """
        # Calculate weighted risk score
        risk_score = self.calculate_weighted_risk_score(
            var_prediction, loss_probability, alpha, beta, var_range, 'sigmoid'
        )

        # Classify risk level
        risk_level = self.classify_risk_level_weighted(
            var_prediction, loss_probability, alpha, beta, thresholds, var_range
        )

        # Generate intervention recommendation
        intervention = self.generate_intervention_recommendation_weighted(
            trader_id, risk_score, risk_level, thresholds
        )

        return {
            'trader_id': trader_id,
            'var_prediction': var_prediction,
            'loss_probability': loss_probability,
            'risk_score': risk_score,
            'risk_level': risk_level,
            'alpha': alpha,
            'beta': beta,
            'var_range': var_range,
            'thresholds': thresholds,
            'intervention': intervention
        }

    def generate_intervention_recommendation_weighted(self, trader_id: int, risk_score: float,
                                                   risk_level: str, thresholds: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Generate intervention recommendation based on weighted risk score.

        Args:
            trader_id: Trader account ID
            risk_score: Weighted risk score (0-1)
            risk_level: Risk level classification
            thresholds: Risk level thresholds

        Returns:
            Dictionary with intervention details
        """
        if thresholds is None:
            thresholds = {
                'high_threshold': 0.7,
                'medium_threshold': 0.5,
                'low_threshold': 0.3
            }

        # Determine intervention level based on risk score
        if risk_score >= thresholds['high_threshold']:
            should_intervene = True
            intervention_level = 'high'
            reduction_factor = 0.7  # 70% reduction
            message = f"HIGH RISK: Strong intervention recommended. Risk score: {risk_score:.3f}. Consider reducing position by {reduction_factor*100:.0f}%."
        elif risk_score >= thresholds['medium_threshold']:
            should_intervene = True
            intervention_level = 'medium'
            reduction_factor = 0.5  # 50% reduction
            message = f"MEDIUM RISK: Moderate intervention recommended. Risk score: {risk_score:.3f}. Consider reducing position by {reduction_factor*100:.0f}%."
        elif risk_score >= thresholds['low_threshold']:
            should_intervene = True
            intervention_level = 'low'
            reduction_factor = 0.3  # 30% reduction
            message = f"LOW RISK: Light intervention recommended. Risk score: {risk_score:.3f}. Consider reducing position by {reduction_factor*100:.0f}%."
        else:
            should_intervene = False
            intervention_level = 'none'
            reduction_factor = 0.0  # No reduction
            message = f"NEUTRAL: No intervention required. Risk score: {risk_score:.3f}. Normal trading conditions."

        return {
            'should_intervene': should_intervene,
            'intervention_level': intervention_level,
            'risk_score': risk_score,
            'risk_level': risk_level,
            'reduction_factor': reduction_factor,
            'thresholds': thresholds,
            'message': message
        }
