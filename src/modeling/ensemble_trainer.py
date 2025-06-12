"""
Ensemble Model Trainer

Advanced ensemble methods for trading PnL prediction combining multiple
gradient boosting algorithms with sophisticated weighting strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import BaseEstimator, RegressorMixin
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# Gradient boosting imports
try:
    import xgboost as xgb
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import catboost as cb
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

from .config import ModelConfig
from .time_series_validator import TimeSeriesValidator

logger = logging.getLogger(__name__)


class EnsembleTrainer:
    """
    Advanced ensemble trainer combining multiple gradient boosting algorithms
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize the ensemble trainer

        Args:
            config: Model configuration object
        """
        self.config = config or ModelConfig()
        self.validator = TimeSeriesValidator(config)

        # Check available algorithms
        self.available_algorithms = []
        if XGBOOST_AVAILABLE:
            self.available_algorithms.append('xgboost')
        if LIGHTGBM_AVAILABLE:
            self.available_algorithms.append('lightgbm')
        if CATBOOST_AVAILABLE:
            self.available_algorithms.append('catboost')

        if not self.available_algorithms:
            raise ImportError("No gradient boosting libraries available. Install xgboost, lightgbm, or catboost")

        logger.info(f"Available algorithms: {self.available_algorithms}")

        # Model storage
        self.base_models = {}
        self.ensemble_weights = {}
        self.best_ensemble = None
        self.training_history = {}

    def get_algorithm_param_space(self, algorithm: str, trial) -> Dict[str, Any]:
        """
        Get algorithm-specific parameter space for Optuna optimization

        Args:
            algorithm: Algorithm name ('xgboost', 'lightgbm', 'catboost')
            trial: Optuna trial object

        Returns:
            Parameter dictionary for the algorithm
        """
        if algorithm == 'xgboost':
            return {
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                'gamma': trial.suggest_float('gamma', 0, 0.5, step=0.1),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0, step=0.1),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0, step=0.1),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10, log=True),
            }
        elif algorithm == 'lightgbm':
            return {
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0, step=0.1),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0, step=0.1),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 10, 300),
            }
        elif algorithm == 'catboost':
            return {
                'depth': trial.suggest_int('depth', 3, 8),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10, log=True),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'iterations': trial.suggest_int('iterations', 100, 1000, step=100),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0, step=0.1),
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0, step=0.1),
            }
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def create_model(self, algorithm: str, params: Dict[str, Any]) -> BaseEstimator:
        """
        Create model instance with given parameters

        Args:
            algorithm: Algorithm name
            params: Model parameters

        Returns:
            Model instance
        """
        if algorithm == 'xgboost':
            base_params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'mae',
                'random_state': 42,
                'n_jobs': -1,
                'tree_method': 'hist',
                'verbosity': 0,
            }
            final_params = {**base_params, **params}
            return XGBRegressor(**final_params)

        elif algorithm == 'lightgbm':
            base_params = {
                'objective': 'regression',
                'metric': 'mae',
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1,
                'force_col_wise': True,
            }
            final_params = {**base_params, **params}
            return LGBMRegressor(**final_params)

        elif algorithm == 'catboost':
            base_params = {
                'loss_function': 'MAE',
                'random_seed': 42,
                'thread_count': -1,
                'verbose': False,
                'allow_writing_files': False,
            }
            final_params = {**base_params, **params}
            return CatBoostRegressor(**final_params)

        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def optimize_single_algorithm(self, algorithm: str, X_train: np.ndarray, y_train: np.ndarray,
                                 cv_splits: List[Tuple[np.ndarray, np.ndarray]],
                                 n_trials: int = 50) -> Tuple[Dict[str, Any], float]:
        """
        Optimize hyperparameters for a single algorithm

        Args:
            algorithm: Algorithm name
            X_train: Training features
            y_train: Training targets
            cv_splits: Cross-validation splits
            n_trials: Number of optimization trials

        Returns:
            Tuple of (best_params, best_score)
        """
        logger.info(f"Optimizing {algorithm} hyperparameters...")

        def objective(trial):
            try:
                # Get algorithm-specific parameters
                params = self.get_algorithm_param_space(algorithm, trial)

                # Cross-validation scores
                cv_scores = []

                for train_idx, val_idx in cv_splits:
                    X_tr, X_val = X_train[train_idx], X_train[val_idx]
                    y_tr, y_val = y_train[train_idx], y_train[val_idx]

                    # Skip if too few samples
                    if len(X_tr) < 10 or len(X_val) < 5:
                        continue

                    # Create and train model
                    model = self.create_model(algorithm, params)

                    # Fit model with early stopping for applicable algorithms
                    if algorithm in ['xgboost', 'lightgbm']:
                        model.fit(
                            X_tr, y_tr,
                            eval_set=[(X_val, y_val)],
                            early_stopping_rounds=20,
                            verbose=False
                        )
                    else:  # CatBoost
                        model.fit(
                            X_tr, y_tr,
                            eval_set=[(X_val, y_val)],
                            early_stopping_rounds=20,
                            verbose=False
                        )

                    # Evaluate
                    val_pred = model.predict(X_val)
                    val_mae = mean_absolute_error(y_val, val_pred)

                    if not np.isfinite(val_mae) or val_mae > 1e6:
                        continue

                    cv_scores.append(val_mae)

                if not cv_scores:
                    return np.inf

                mean_score = np.mean(cv_scores)

                # Report for pruning
                trial.report(mean_score, step=0)

                return mean_score

            except Exception as e:
                logger.warning(f"Trial failed for {algorithm}: {e}")
                return np.inf

        # Create study
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )

        # Optimize
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        if study.best_params:
            logger.info(f"Best {algorithm} params: {study.best_params}")
            logger.info(f"Best {algorithm} score: {study.best_value:.4f}")
            return study.best_params, study.best_value
        else:
            logger.warning(f"No valid parameters found for {algorithm}")
            return {}, np.inf

    def train_base_models(self, X_train: Union[pd.DataFrame, np.ndarray],
                         y_train: Union[pd.Series, np.ndarray],
                         X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
                         y_val: Optional[Union[pd.Series, np.ndarray]] = None,
                         optimization_trials: int = 50,
                         cv_folds: int = 5) -> Dict[str, Any]:
        """
        Train all available base models with hyperparameter optimization

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Optional validation features
            y_val: Optional validation targets
            optimization_trials: Number of trials per algorithm
            cv_folds: Number of cross-validation folds

        Returns:
            Training results dictionary
        """
        logger.info("Training ensemble base models...")

        # Convert to numpy arrays
        if isinstance(X_train, pd.DataFrame):
            feature_names = X_train.columns.tolist()
            X_train = X_train.values
        else:
            feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]

        if isinstance(y_train, pd.Series):
            y_train = y_train.values

        # Handle validation split
        if X_val is None or y_val is None:
            split_idx = int(0.8 * len(X_train))
            X_val = X_train[split_idx:]
            y_val = y_train[split_idx:]
            X_train = X_train[:split_idx]
            y_train = y_train[:split_idx]
        else:
            if isinstance(X_val, pd.DataFrame):
                X_val = X_val.values
            if isinstance(y_val, pd.Series):
                y_val = y_val.values

        # Create temporal cross-validation splits
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        cv_splits = list(tscv.split(X_train))

        # Train each algorithm
        algorithm_results = {}

        for algorithm in self.available_algorithms:
            try:
                # Optimize hyperparameters
                best_params, best_score = self.optimize_single_algorithm(
                    algorithm, X_train, y_train, cv_splits, optimization_trials
                )

                if best_score == np.inf:
                    logger.warning(f"Skipping {algorithm} due to optimization failure")
                    continue

                # Train final model with best parameters
                logger.info(f"Training final {algorithm} model...")
                final_model = self.create_model(algorithm, best_params)

                # Train with early stopping
                if algorithm in ['xgboost', 'lightgbm']:
                    final_model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        early_stopping_rounds=50,
                        verbose=False
                    )
                else:  # CatBoost
                    final_model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        early_stopping_rounds=50,
                        verbose=False
                    )

                # Evaluate
                train_pred = final_model.predict(X_train)
                val_pred = final_model.predict(X_val)

                metrics = {
                    'train_mae': mean_absolute_error(y_train, train_pred),
                    'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
                    'train_r2': r2_score(y_train, train_pred),
                    'val_mae': mean_absolute_error(y_val, val_pred),
                    'val_rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
                    'val_r2': r2_score(y_val, val_pred),
                }

                # Store results
                algorithm_results[algorithm] = {
                    'model': final_model,
                    'best_params': best_params,
                    'metrics': metrics,
                    'cv_score': best_score
                }

                self.base_models[algorithm] = final_model

                logger.info(f"{algorithm} results - Val MAE: {metrics['val_mae']:.4f}, "
                           f"Val R²: {metrics['val_r2']:.4f}")

            except Exception as e:
                logger.error(f"Failed to train {algorithm}: {e}")
                continue

        if not algorithm_results:
            raise RuntimeError("No algorithms trained successfully")

        return algorithm_results

    def optimize_ensemble_weights(self, X_val: np.ndarray, y_val: np.ndarray,
                                 n_trials: int = 100) -> Dict[str, float]:
        """
        Optimize ensemble weights using Optuna

        Args:
            X_val: Validation features
            y_val: Validation targets
            n_trials: Number of optimization trials

        Returns:
            Optimal ensemble weights
        """
        if len(self.base_models) < 2:
            logger.warning("Need at least 2 models for ensemble. Using single model.")
            return {list(self.base_models.keys())[0]: 1.0}

        logger.info("Optimizing ensemble weights...")

        # Generate predictions from all base models
        base_predictions = {}
        for name, model in self.base_models.items():
            try:
                pred = model.predict(X_val)
                if np.all(np.isfinite(pred)):
                    base_predictions[name] = pred
                else:
                    logger.warning(f"Invalid predictions from {name}, skipping")
            except Exception as e:
                logger.warning(f"Prediction failed for {name}: {e}")

        if len(base_predictions) < 2:
            logger.warning("Insufficient valid predictions for ensemble")
            return {list(base_predictions.keys())[0]: 1.0}

        algorithm_names = list(base_predictions.keys())

        def objective(trial):
            # Generate weights (they will be normalized)
            raw_weights = []
            for i, name in enumerate(algorithm_names):
                if i == len(algorithm_names) - 1:
                    # Last weight is determined by others to sum to 1
                    break
                weight = trial.suggest_float(f'weight_{name}', 0.0, 1.0)
                raw_weights.append(weight)

            # Normalize weights to sum to 1
            if len(raw_weights) == len(algorithm_names) - 1:
                last_weight = max(0, 1.0 - sum(raw_weights))
                raw_weights.append(last_weight)

            # Renormalize to ensure sum = 1
            total_weight = sum(raw_weights)
            if total_weight > 0:
                weights = [w / total_weight for w in raw_weights]
            else:
                weights = [1.0 / len(algorithm_names)] * len(algorithm_names)

            # Create ensemble prediction
            ensemble_pred = np.zeros_like(y_val, dtype=float)
            for weight, name in zip(weights, algorithm_names):
                ensemble_pred += weight * base_predictions[name]

            # Calculate error
            mae = mean_absolute_error(y_val, ensemble_pred)
            return mae

        # Optimize weights
        study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        # Extract best weights
        best_trial = study.best_trial
        best_weights = {}

        raw_weights = []
        for i, name in enumerate(algorithm_names):
            if i == len(algorithm_names) - 1:
                break
            raw_weights.append(best_trial.params.get(f'weight_{name}', 0.0))

        # Add last weight and normalize
        if len(raw_weights) == len(algorithm_names) - 1:
            last_weight = max(0, 1.0 - sum(raw_weights))
            raw_weights.append(last_weight)

        total_weight = sum(raw_weights)
        if total_weight > 0:
            normalized_weights = [w / total_weight for w in raw_weights]
        else:
            normalized_weights = [1.0 / len(algorithm_names)] * len(algorithm_names)

        for name, weight in zip(algorithm_names, normalized_weights):
            best_weights[name] = weight

        logger.info(f"Optimal ensemble weights: {best_weights}")
        logger.info(f"Ensemble validation MAE: {study.best_value:.4f}")

        self.ensemble_weights = best_weights
        return best_weights

    def predict_ensemble(self, X: np.ndarray, weights: Optional[Dict[str, float]] = None) -> np.ndarray:
        """
        Make ensemble predictions

        Args:
            X: Features for prediction
            weights: Optional custom weights (uses optimized weights if None)

        Returns:
            Ensemble predictions
        """
        if not self.base_models:
            raise RuntimeError("No base models trained")

        weights = weights or self.ensemble_weights
        if not weights:
            # Equal weights
            weights = {name: 1.0 / len(self.base_models) for name in self.base_models.keys()}

        # Generate predictions
        ensemble_pred = np.zeros(X.shape[0], dtype=float)
        total_weight = 0

        for name, model in self.base_models.items():
            if name in weights:
                weight = weights[name]
                pred = model.predict(X)
                ensemble_pred += weight * pred
                total_weight += weight

        # Normalize if weights don't sum to 1
        if total_weight > 0 and abs(total_weight - 1.0) > 1e-6:
            ensemble_pred /= total_weight

        return ensemble_pred

    def get_ensemble_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive ensemble summary

        Returns:
            Ensemble summary dictionary
        """
        if not self.base_models:
            return {"error": "No models trained"}

        summary = {
            'available_algorithms': self.available_algorithms,
            'trained_models': list(self.base_models.keys()),
            'ensemble_weights': self.ensemble_weights,
            'model_count': len(self.base_models)
        }

        return summary
