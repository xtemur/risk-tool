"""
Base Model Class
Abstract base class for all trading models
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import joblib
from pathlib import Path
from datetime import datetime
import json

from src.core.constants import TradingConstants as TC

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    Abstract base class for trading models
    Ensures consistent interface and best practices
    """

    def __init__(self,
                 model_name: str,
                 model_type: str = 'regression',
                 random_state: int = 42):
        """
        Initialize base model

        Args:
            model_name: Name of the model
            model_type: Type of model ('regression', 'classification', 'ranking')
            random_state: Random seed for reproducibility
        """
        self.model_name = model_name
        self.model_type = model_type
        self.random_state = random_state

        # Model components
        self.model_ = None
        self.feature_names_ = None
        self.feature_importance_ = None
        self.is_fitted_ = False

        # Metadata
        self.metadata_ = {
            'model_name': model_name,
            'model_type': model_type,
            'created_at': datetime.now().isoformat(),
            'version': '1.0',
            'training_info': {}
        }

        # Performance tracking
        self.training_history_ = []
        self.validation_scores_ = {}

    @abstractmethod
    def _create_model(self, **kwargs) -> Any:
        """Create the underlying model instance"""
        pass

    @abstractmethod
    def _get_feature_importance(self) -> Dict[str, float]:
        """Extract feature importance from fitted model"""
        pass

    def fit(self,
            X: pd.DataFrame,
            y: pd.Series,
            sample_weight: Optional[pd.Series] = None,
            validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
            **kwargs) -> 'BaseModel':
        """
        Fit the model

        Args:
            X: Training features
            y: Training target
            sample_weight: Optional sample weights
            validation_data: Optional (X_val, y_val) tuple
            **kwargs: Additional arguments for specific model

        Returns:
            Self
        """
        # Validate inputs
        X, y = self._validate_input(X, y)

        # Store feature names
        self.feature_names_ = X.columns.tolist()

        # Record training start
        train_start = datetime.now()

        # Create model if not exists
        if self.model_ is None:
            self.model_ = self._create_model(**kwargs)

        # Prepare training arguments
        fit_args = {'X': X, 'y': y}

        if sample_weight is not None:
            fit_args['sample_weight'] = sample_weight

        # Add validation data if provided
        if validation_data is not None:
            X_val, y_val = self._validate_input(validation_data[0], validation_data[1])
            fit_args['eval_set'] = [(X_val, y_val)]

            # For LightGBM-style models
            if hasattr(self.model_, 'set_params'):
                fit_args['eval_names'] = ['valid']
                fit_args['eval_metric'] = kwargs.get('eval_metric', 'rmse')
                fit_args['callbacks'] = kwargs.get('callbacks', None)

        # Fit model
        logger.info(f"Training {self.model_name} on {len(X)} samples with {len(self.feature_names_)} features")

        try:
            self.model_.fit(**fit_args)
            self.is_fitted_ = True

            # Extract feature importance
            self.feature_importance_ = self._get_feature_importance()

            # Record training info
            train_time = (datetime.now() - train_start).total_seconds()
            self.metadata_['training_info'] = {
                'n_samples': len(X),
                'n_features': len(self.feature_names_),
                'training_time': train_time,
                'completed_at': datetime.now().isoformat()
            }

            # Evaluate on training data
            train_score = self.score(X, y)
            self.training_history_.append({
                'timestamp': datetime.now(),
                'train_score': train_score,
                'val_score': self.score(X_val, y_val) if validation_data else None
            })

            logger.info(f"Training completed in {train_time:.2f} seconds. Train score: {train_score:.4f}")

        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions

        Args:
            X: Features to predict on

        Returns:
            Predictions array
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")

        # Validate input
        X = self._validate_features(X)

        # Make predictions
        try:
            predictions = self.model_.predict(X)

            # Post-process if needed
            predictions = self._post_process_predictions(predictions)

            return predictions

        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities (for classification)

        Args:
            X: Features to predict on

        Returns:
            Probability array
        """
        if self.model_type != 'classification':
            raise ValueError("predict_proba only available for classification models")

        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")

        X = self._validate_features(X)

        if hasattr(self.model_, 'predict_proba'):
            return self.model_.predict_proba(X)
        else:
            raise NotImplementedError("Model does not support probability prediction")

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Score the model

        Args:
            X: Features
            y: True labels

        Returns:
            Score (R² for regression, accuracy for classification)
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before scoring")

        X, y = self._validate_input(X, y)

        if hasattr(self.model_, 'score'):
            return self.model_.score(X, y)
        else:
            # Manual scoring
            predictions = self.predict(X)

            if self.model_type == 'regression':
                # R² score
                ss_res = np.sum((y - predictions) ** 2)
                ss_tot = np.sum((y - y.mean()) ** 2)
                return 1 - (ss_res / (ss_tot + TC.MIN_VARIANCE))
            else:
                # Accuracy
                return np.mean(predictions == y)

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters"""
        if hasattr(self.model_, 'get_params'):
            return self.model_.get_params()
        return {}

    def set_params(self, **params) -> 'BaseModel':
        """Set model parameters"""
        if hasattr(self.model_, 'set_params'):
            self.model_.set_params(**params)
        return self

    def save(self, path: Union[str, Path]) -> None:
        """
        Save model to disk

        Args:
            path: Path to save model
        """
        path = Path(path)
        path.parent.mkdir(exist_ok=True, parents=True)

        # Save model
        model_data = {
            'model': self.model_,
            'feature_names': self.feature_names_,
            'feature_importance': self.feature_importance_,
            'metadata': self.metadata_,
            'training_history': self.training_history_,
            'validation_scores': self.validation_scores_,
            'model_class': self.__class__.__name__
        }

        joblib.dump(model_data, path)

        # Save metadata separately for easy access
        metadata_path = path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            # Convert non-serializable objects
            safe_metadata = self.metadata_.copy()
            safe_metadata['feature_importance'] = (
                dict(sorted(self.feature_importance_.items(),
                           key=lambda x: x[1],
                           reverse=True)[:20])  # Top 20 features
                if self.feature_importance_ else {}
            )
            json.dump(safe_metadata, f, indent=2, default=str)

        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'BaseModel':
        """
        Load model from disk

        Args:
            path: Path to load model from

        Returns:
            Loaded model instance
        """
        path = Path(path)

        # Load model data
        model_data = joblib.load(path)

        # Create instance
        instance = cls(model_name=model_data['metadata']['model_name'])

        # Restore state
        instance.model_ = model_data['model']
        instance.feature_names_ = model_data['feature_names']
        instance.feature_importance_ = model_data['feature_importance']
        instance.metadata_ = model_data['metadata']
        instance.training_history_ = model_data['training_history']
        instance.validation_scores_ = model_data['validation_scores']
        instance.is_fitted_ = True

        logger.info(f"Model loaded from {path}")

        return instance

    def _validate_input(self,
                       X: pd.DataFrame,
                       y: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Validate input data"""

        # Ensure DataFrame
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        # Remove non-numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_cols]

        # Handle missing values
        if X.isnull().any().any():
            logger.warning(f"Found {X.isnull().sum().sum()} missing values, filling with 0")
            X = X.fillna(0)

        # Handle infinite values
        if np.isinf(X.values).any():
            logger.warning("Found infinite values, clipping")
            X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Validate target if provided
        if y is not None:
            if not isinstance(y, pd.Series):
                y = pd.Series(y, index=X.index)

            # Align indices
            X, y = X.align(y, join='inner', axis=0)

            # Check for NaN in target
            if y.isnull().any():
                logger.warning(f"Found {y.isnull().sum()} NaN values in target, dropping")
                mask = ~y.isnull()
                X = X[mask]
                y = y[mask]

        return X, y

    def _validate_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Validate features match training features"""

        if self.feature_names_ is None:
            raise ValueError("Model has no stored feature names")

        # Get common features
        common_features = [f for f in self.feature_names_ if f in X.columns]
        missing_features = [f for f in self.feature_names_ if f not in X.columns]

        if missing_features:
            logger.warning(f"Missing {len(missing_features)} features: {missing_features[:5]}...")

            # Create missing features with zeros
            for feature in missing_features:
                X[feature] = 0

        # Ensure correct order
        X = X[self.feature_names_]

        # Validate
        X, _ = self._validate_input(X)

        return X

    def _post_process_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """Post-process predictions (can be overridden)"""
        return predictions

    def get_feature_importance(self, top_k: Optional[int] = None) -> pd.DataFrame:
        """
        Get feature importance as DataFrame

        Args:
            top_k: Number of top features to return

        Returns:
            DataFrame with feature names and importance scores
        """
        if self.feature_importance_ is None:
            return pd.DataFrame()

        importance_df = pd.DataFrame({
            'feature': list(self.feature_importance_.keys()),
            'importance': list(self.feature_importance_.values())
        }).sort_values('importance', ascending=False)

        if top_k:
            importance_df = importance_df.head(top_k)

        return importance_df

    def explain_prediction(self,
                          X: pd.DataFrame,
                          index: int = 0) -> Dict[str, float]:
        """
        Explain a single prediction (basic version)

        Args:
            X: Features
            index: Which sample to explain

        Returns:
            Dictionary of feature contributions
        """
        # This is a simple version - can be enhanced with SHAP/LIME
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before explaining")

        sample = X.iloc[index:index+1]
        prediction = self.predict(sample)[0]

        # Simple explanation based on feature importance and values
        explanations = {}

        if self.feature_importance_:
            for feature in self.feature_names_:
                if feature in self.feature_importance_:
                    value = sample[feature].iloc[0]
                    importance = self.feature_importance_[feature]

                    # Simple contribution estimate
                    contribution = value * importance
                    explanations[feature] = contribution

        return explanations
