"""
Model Trainer

Flexible model training framework supporting multiple algorithms
with hyperparameter optimization and preprocessing.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from pathlib import Path
import pickle
import joblib

# Scikit-learn imports
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

# Handle XGBoost import gracefully
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBRegressor = None
    XGBOOST_AVAILABLE = False

from .config import ModelConfig
from .time_series_validator import TimeSeriesValidator

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Flexible model trainer supporting multiple algorithms
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize the trainer

        Args:
            config: Model configuration object
        """
        self.config = config or ModelConfig()
        self.validator = TimeSeriesValidator(config)
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.training_history = {}

    def _get_model_class(self, model_type: str):
        """
        Get model class by name

        Args:
            model_type: Model type string

        Returns:
            Model class
        """
        model_map = {
            'LinearRegression': LinearRegression,
            'Ridge': Ridge,
            'Lasso': Lasso,
            'ElasticNet': ElasticNet,
            'XGBRegressor': XGBRegressor if XGBOOST_AVAILABLE else None
        }

        model_class = model_map.get(model_type)
        if model_class is None:
            if model_type == 'XGBRegressor' and not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost is not installed. Install with: pip install xgboost")
            else:
                raise ValueError(f"Unknown model type: {model_type}")

        return model_class

    def _get_scaler(self, scaler_type: str):
        """
        Get scaler by name

        Args:
            scaler_type: Scaler type string

        Returns:
            Scaler instance
        """
        scaler_map = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler(),
            'none': None
        }

        return scaler_map.get(scaler_type, StandardScaler())

    def _get_feature_selector(self, method: str, n_features: int):
        """
        Get feature selector

        Args:
            method: Selection method
            n_features: Number of features to select

        Returns:
            Feature selector instance
        """
        if method == 'f_regression':
            return SelectKBest(score_func=f_regression, k=n_features)
        elif method == 'mutual_info':
            return SelectKBest(score_func=mutual_info_regression, k=n_features)
        else:
            raise ValueError(f"Unknown feature selection method: {method}")

    def preprocess_data(self, X: pd.DataFrame, y: pd.Series,
                       fit_preprocessors: bool = True,
                       preprocessor_key: str = 'default') -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess features and target

        Args:
            X: Feature matrix
            y: Target vector
            fit_preprocessors: Whether to fit new preprocessors
            preprocessor_key: Key to store/retrieve preprocessors

        Returns:
            Tuple of (processed_X, processed_y)
        """
        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
            X = X.values
        else:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]

        if isinstance(y, pd.Series):
            y = y.values

        # Handle missing values
        if self.config.HANDLE_MISSING == 'median':
            if fit_preprocessors:
                self._median_values = np.nanmedian(X, axis=0)
                # Store for later use
                setattr(self, f'_median_values_{preprocessor_key}', self._median_values)
            else:
                # Use stored values
                self._median_values = getattr(self, f'_median_values_{preprocessor_key}', np.nanmedian(X, axis=0))

            # Fill missing values
            mask = np.isnan(X)
            if np.any(mask):
                for col_idx in range(X.shape[1]):
                    col_mask = mask[:, col_idx]
                    if np.any(col_mask):
                        X[col_mask, col_idx] = self._median_values[col_idx]

        # Handle outliers (cap at z-score threshold)
        if self.config.HANDLE_OUTLIERS:
            if fit_preprocessors:
                outlier_bounds = {}
                for i in range(X.shape[1]):
                    mean_val = np.mean(X[:, i])
                    std_val = np.std(X[:, i])
                    outlier_bounds[i] = (
                        mean_val - self.config.OUTLIER_THRESHOLD * std_val,
                        mean_val + self.config.OUTLIER_THRESHOLD * std_val
                    )
                setattr(self, f'_outlier_bounds_{preprocessor_key}', outlier_bounds)
            else:
                outlier_bounds = getattr(self, f'_outlier_bounds_{preprocessor_key}', {})

            for i, (lower, upper) in outlier_bounds.items():
                if i < X.shape[1]:  # Safety check
                    X[:, i] = np.clip(X[:, i], lower, upper)

        # Feature selection
        if self.config.FEATURE_SELECTION and fit_preprocessors:
            n_features = min(self.config.MAX_FEATURES, X.shape[1])
            selector = self._get_feature_selector(
                self.config.FEATURE_SELECTION_METHOD,
                n_features
            )
            X = selector.fit_transform(X, y)
            self.feature_selectors[preprocessor_key] = selector

        elif self.config.FEATURE_SELECTION and preprocessor_key in self.feature_selectors:
            X = self.feature_selectors[preprocessor_key].transform(X)

        # Feature scaling
        if self.config.FEATURE_SCALING != 'none':
            if fit_preprocessors:
                scaler = self._get_scaler(self.config.FEATURE_SCALING)
                X = scaler.fit_transform(X)
                self.scalers[preprocessor_key] = scaler
            elif preprocessor_key in self.scalers:
                X = self.scalers[preprocessor_key].transform(X)
            else:
                logger.warning(f"No scaler found for key {preprocessor_key}")

        return X, y

    def train_model(self, X_train: Union[pd.DataFrame, np.ndarray],
                   y_train: Union[pd.Series, np.ndarray],
                   model_type: str = 'ridge',
                   model_key: str = 'default',
                   optimize_hyperparams: bool = True) -> Dict[str, Any]:
        """
        Train a single model

        Args:
            X_train: Training features
            y_train: Training targets
            model_type: Type of model to train
            model_key: Key to store the model
            optimize_hyperparams: Whether to optimize hyperparameters

        Returns:
            Training results dictionary
        """
        logger.info(f"Training {model_type} model with key '{model_key}'")

        # Preprocess data
        X_processed, y_processed = self.preprocess_data(
            X_train, y_train,
            fit_preprocessors=True,
            preprocessor_key=model_key
        )

        # Get model configuration
        if model_type not in self.config.MODEL_TYPES:
            raise ValueError(f"Model type {model_type} not in configuration")

        model_config = self.config.MODEL_TYPES[model_type]
        model_class = self._get_model_class(model_config['class'])
        base_params = model_config['params']

        # Hyperparameter optimization
        if optimize_hyperparams and model_type in self.config.HYPERPARAMETER_GRIDS:
            logger.info(f"Optimizing hyperparameters for {model_type}")

            # Create base model
            model = model_class(**base_params)

            # Get hyperparameter grid
            param_grid = self.config.HYPERPARAMETER_GRIDS[model_type]

            # Create temporal cross-validation splits
            cv_splits = self.validator.get_temporal_cv_splits(
                pd.DataFrame({'date': range(len(X_processed))}),
                n_splits=self.config.CV_FOLDS
            )

            if not cv_splits:
                logger.warning("No CV splits available, using default parameters")
                model = model_class(**base_params)
            else:
                # Grid search with temporal CV
                grid_search = GridSearchCV(
                    model,
                    param_grid,
                    cv=cv_splits,
                    scoring=self.config.CV_SCORING,
                    n_jobs=-1
                )
                grid_search.fit(X_processed, y_processed)
                model = grid_search.best_estimator_

                logger.info(f"Best hyperparameters: {grid_search.best_params_}")
                logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        else:
            # Use default parameters
            model = model_class(**base_params)

        # Train final model
        model.fit(X_processed, y_processed)

        # Store model
        self.models[model_key] = model

        # Calculate training metrics
        train_pred = model.predict(X_processed)
        train_metrics = {
            'mae': mean_absolute_error(y_processed, train_pred),
            'rmse': np.sqrt(mean_squared_error(y_processed, train_pred)),
            'r2': r2_score(y_processed, train_pred)
        }

        # Store training history
        self.training_history[model_key] = {
            'model_type': model_type,
            'train_samples': len(X_processed),
            'train_metrics': train_metrics,
            'hyperparams_optimized': optimize_hyperparams,
            'feature_dim': X_processed.shape[1]
        }

        logger.info(f"Training completed. MAE: {train_metrics['mae']:.4f}, R²: {train_metrics['r2']:.4f}")

        return {
            'model': model,
            'metrics': train_metrics,
            'model_key': model_key
        }

    def predict(self, X_test: Union[pd.DataFrame, np.ndarray],
               model_key: str = 'default') -> np.ndarray:
        """
        Make predictions with trained model

        Args:
            X_test: Test features
            model_key: Key of the model to use

        Returns:
            Predictions array
        """
        if model_key not in self.models:
            raise ValueError(f"No model found with key '{model_key}'")

        # Preprocess test data (don't fit new preprocessors)
        X_processed, _ = self.preprocess_data(
            X_test,
            np.zeros(len(X_test)),  # Dummy target
            fit_preprocessors=False,
            preprocessor_key=model_key
        )

        # Make predictions
        model = self.models[model_key]
        predictions = model.predict(X_processed)

        return predictions

    def train_trader_models(self, trader_data: Dict[str, pd.DataFrame],
                           model_type: str = 'ridge') -> Dict[str, Dict[str, Any]]:
        """
        Train individual models for each trader

        Args:
            trader_data: Dictionary mapping account_id to dataframe
            model_type: Type of model to train

        Returns:
            Dictionary mapping account_id to training results
        """
        results = {}

        for account_id, data in trader_data.items():
            logger.info(f"Training model for trader {account_id}")

            # Prepare features and target
            feature_cols = [col for col in data.columns
                           if col not in ['account_id', 'date', 'target_next_pnl']]

            X = data[feature_cols]
            y = data['target_next_pnl']

            # Train model for this trader
            try:
                result = self.train_model(
                    X, y,
                    model_type=model_type,
                    model_key=account_id,
                    optimize_hyperparams=True
                )
                results[account_id] = result

            except Exception as e:
                logger.error(f"Failed to train model for trader {account_id}: {e}")
                results[account_id] = {'error': str(e)}

        return results

    def save_models(self, save_path: str = None) -> str:
        """
        Save all trained models

        Args:
            save_path: Directory to save models

        Returns:
            Path where models were saved
        """
        if save_path is None:
            save_path = self.config.MODEL_SAVE_PATH

        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save models
        models_file = save_path / "models.pkl"
        with open(models_file, 'wb') as f:
            pickle.dump(self.models, f)

        # Save preprocessors
        scalers_file = save_path / "scalers.pkl"
        with open(scalers_file, 'wb') as f:
            pickle.dump(self.scalers, f)

        selectors_file = save_path / "feature_selectors.pkl"
        with open(selectors_file, 'wb') as f:
            pickle.dump(self.feature_selectors, f)

        # Save training history
        history_file = save_path / "training_history.pkl"
        with open(history_file, 'wb') as f:
            pickle.dump(self.training_history, f)

        logger.info(f"Models saved to {save_path}")
        return str(save_path)

    def load_models(self, load_path: str) -> None:
        """
        Load saved models

        Args:
            load_path: Directory to load models from
        """
        load_path = Path(load_path)

        # Load models
        models_file = load_path / "models.pkl"
        if models_file.exists():
            with open(models_file, 'rb') as f:
                self.models = pickle.load(f)

        # Load preprocessors
        scalers_file = load_path / "scalers.pkl"
        if scalers_file.exists():
            with open(scalers_file, 'rb') as f:
                self.scalers = pickle.load(f)

        selectors_file = load_path / "feature_selectors.pkl"
        if selectors_file.exists():
            with open(selectors_file, 'rb') as f:
                self.feature_selectors = pickle.load(f)

        # Load training history
        history_file = load_path / "training_history.pkl"
        if history_file.exists():
            with open(history_file, 'rb') as f:
                self.training_history = pickle.load(f)

        logger.info(f"Models loaded from {load_path}")

    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get summary of all trained models

        Returns:
            Summary dictionary
        """
        summary = {
            'num_models': len(self.models),
            'model_keys': list(self.models.keys()),
            'training_history': self.training_history
        }

        if self.training_history:
            # Aggregate statistics
            all_metrics = [hist['train_metrics'] for hist in self.training_history.values()
                          if 'train_metrics' in hist]

            if all_metrics:
                summary['aggregate_metrics'] = {
                    'avg_mae': np.mean([m['mae'] for m in all_metrics]),
                    'avg_rmse': np.mean([m['rmse'] for m in all_metrics]),
                    'avg_r2': np.mean([m['r2'] for m in all_metrics])
                }

        return summary
