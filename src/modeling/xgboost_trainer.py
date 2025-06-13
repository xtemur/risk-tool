"""
Comprehensive XGBoost Model Trainer

Advanced XGBoost implementation with:
- Extensive hyperparameter optimization
- Early stopping and regularization
- Training/validation monitoring
- Feature importance analysis
- Cross-validation with temporal splits
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# XGBoost imports
try:
    import xgboost as xgb
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from .config import ModelConfig
from .time_series_validator import TimeSeriesValidator

logger = logging.getLogger(__name__)


class AdvancedXGBoostTrainer:
    """
    Comprehensive XGBoost trainer with advanced features for preventing overfitting
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize the trainer

        Args:
            config: Model configuration object
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not available. Install with: pip install xgboost")

        self.config = config or ModelConfig()
        self.validator = TimeSeriesValidator(config)

        # Training history
        self.training_history = {}
        self.validation_curves = {}
        self.feature_importance = {}

        # Best model storage
        self.best_model = None
        self.best_params = None
        self.best_score = None

    def get_comprehensive_param_space(self) -> Dict[str, Any]:
        """
        Get comprehensive hyperparameter space for optimization

        Returns:
            Parameter space dictionary
        """
        return {
            # Tree structure parameters
            'max_depth': [3, 4, 5, 6, 7, 8],
            'min_child_weight': [1, 3, 5, 7],
            'gamma': [0, 0.1, 0.2, 0.3, 0.4, 0.5],

            # Sampling parameters
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bylevel': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bynode': [0.6, 0.7, 0.8, 0.9, 1.0],

            # Learning parameters
            'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.15, 0.2],
            'n_estimators': [100, 200, 300, 500, 800, 1000],

            # Regularization parameters
            'reg_alpha': [0, 0.001, 0.01, 0.1, 1, 10],
            'reg_lambda': [0, 0.001, 0.01, 0.1, 1, 10],

            # Additional parameters
            'max_delta_step': [0, 1, 5, 10],
            'scale_pos_weight': [1],  # For balanced dataset
        }

    def get_optuna_param_space(self, trial) -> Dict[str, Any]:
        """
        Get Optuna parameter space for Bayesian optimization

        Args:
            trial: Optuna trial object

        Returns:
            Parameter dictionary for the trial
        """
        return {
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'gamma': trial.suggest_float('gamma', 0, 0.5, step=0.1),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0, step=0.1),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0, step=0.1),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0, step=0.1),
            'colsample_bynode': trial.suggest_float('colsample_bynode', 0.6, 1.0, step=0.1),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10, log=True),
            'max_delta_step': trial.suggest_int('max_delta_step', 0, 10),
        }

    def create_xgb_model(self, params: Dict[str, Any], early_stopping_rounds: int = 50) -> XGBRegressor:
        """
        Create XGBoost model with given parameters

        Args:
            params: Model parameters
            early_stopping_rounds: Early stopping rounds

        Returns:
            XGBRegressor instance
        """
        # Base parameters for regression
        base_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'mae',  # Set evaluation metric
            'random_state': 42,
            'n_jobs': -1,
            'tree_method': 'hist',  # Faster training
            'verbosity': 0,  # Reduce output
            'early_stopping_rounds': early_stopping_rounds,  # Add early stopping
        }

        # Merge with provided parameters
        final_params = {**base_params, **params}

        return XGBRegressor(**final_params)

    def train_with_early_stopping(self, X_train: np.ndarray, y_train: np.ndarray,
                                 X_val: np.ndarray, y_val: np.ndarray,
                                 params: Dict[str, Any],
                                 early_stopping_rounds: int = 50) -> Tuple[XGBRegressor, Dict[str, List]]:
        """
        Train XGBoost model with early stopping and monitoring

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            params: Model parameters
            early_stopping_rounds: Early stopping patience

        Returns:
            Tuple of (trained_model, training_history)
        """
        # Ensure data is clean
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_val = np.nan_to_num(X_val, nan=0.0)
        y_train = np.nan_to_num(y_train, nan=0.0)
        y_val = np.nan_to_num(y_val, nan=0.0)

        # Create model
        model = self.create_xgb_model(params, early_stopping_rounds)

        # Training with evaluation
        eval_set = [(X_train, y_train), (X_val, y_val)]
        eval_names = ['train', 'validation']

        # Fit with early stopping
        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )

        # Extract training history
        evals_result = model.evals_result_
        training_history = {
            'train_mae': evals_result.get('validation_0', {}).get('mae', []),
            'train_rmse': evals_result.get('validation_0', {}).get('rmse', []),
            'validation_mae': evals_result.get('validation_1', {}).get('mae', []),
            'validation_rmse': evals_result.get('validation_1', {}).get('rmse', []),
            'best_iteration': getattr(model, 'best_iteration', len(evals_result.get('validation_1', {}).get('mae', []))),
            'best_score': getattr(model, 'best_score', np.inf)
        }

        # If we only have MAE, use that
        if not training_history['train_rmse']:
            training_history['train_rmse'] = training_history['train_mae']
        if not training_history['validation_rmse']:
            training_history['validation_rmse'] = training_history['validation_mae']

        return model, training_history

    def evaluate_overfitting(self, training_history: Dict[str, List]) -> Dict[str, float]:
        """
        Evaluate overfitting based on training curves

        Args:
            training_history: Training history from XGBoost

        Returns:
            Overfitting metrics
        """
        train_mae = training_history['train_mae']
        val_mae = training_history['validation_mae']
        train_rmse = training_history['train_rmse']
        val_rmse = training_history['validation_rmse']

        # Calculate overfitting metrics
        best_iter = training_history['best_iteration']

        # MAE ratios
        final_train_mae = train_mae[best_iter]
        final_val_mae = val_mae[best_iter]
        mae_ratio = final_val_mae / final_train_mae if final_train_mae > 0 else np.inf

        # RMSE ratios
        final_train_rmse = train_rmse[best_iter]
        final_val_rmse = val_rmse[best_iter]
        rmse_ratio = final_val_rmse / final_train_rmse if final_train_rmse > 0 else np.inf

        # Performance degradation from minimum
        min_val_mae = min(val_mae)
        min_val_rmse = min(val_rmse)
        mae_degradation = (final_val_mae - min_val_mae) / min_val_mae if min_val_mae > 0 else 0
        rmse_degradation = (final_val_rmse - min_val_rmse) / min_val_rmse if min_val_rmse > 0 else 0

        return {
            'train_val_mae_ratio': mae_ratio,
            'train_val_rmse_ratio': rmse_ratio,
            'mae_degradation': mae_degradation,
            'rmse_degradation': rmse_degradation,
            'best_iteration': best_iter,
            'total_iterations': len(train_mae),
            'early_stopped': best_iter < len(train_mae) - 1
        }

    def optuna_objective(self, trial, X_train: np.ndarray, y_train: np.ndarray,
                        cv_splits: List[Tuple[np.ndarray, np.ndarray]]) -> float:
        """
        Optuna objective function for hyperparameter optimization

        Args:
            trial: Optuna trial
            X_train: Training features
            y_train: Training targets
            cv_splits: Cross-validation splits

        Returns:
            Objective score (lower is better)
        """
        try:
            # Get trial parameters
            params = self.get_optuna_param_space(trial)

            # Cross-validation scores
            cv_scores = []
            overfitting_scores = []

            for train_idx, val_idx in cv_splits:
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]

                # Skip if too few samples
                if len(X_tr) < 10 or len(X_val) < 5:
                    continue

                # Handle missing values
                X_tr = np.nan_to_num(X_tr, nan=0.0)
                X_val = np.nan_to_num(X_val, nan=0.0)
                y_tr = np.nan_to_num(y_tr, nan=0.0)
                y_val = np.nan_to_num(y_val, nan=0.0)

                # Train with early stopping
                model, history = self.train_with_early_stopping(
                    X_tr, y_tr, X_val, y_val, params,
                    early_stopping_rounds=20  # Reduced for faster trials
                )

                # Evaluate performance
                val_pred = model.predict(X_val)
                val_mae = mean_absolute_error(y_val, val_pred)

                # Check for valid predictions
                if not np.isfinite(val_mae) or val_mae > 1e6:
                    continue

                cv_scores.append(val_mae)

                # Evaluate overfitting
                overfitting = self.evaluate_overfitting(history)
                overfitting_penalty = max(overfitting['train_val_mae_ratio'] - 1.2, 0) * 0.1
                overfitting_scores.append(overfitting_penalty)

            # Check if we have valid scores
            if not cv_scores:
                return np.inf

            # Combine validation score with overfitting penalty
            mean_cv_score = np.mean(cv_scores)
            mean_overfitting_penalty = np.mean(overfitting_scores) if overfitting_scores else 0

            # Final objective (minimize)
            objective = mean_cv_score + mean_overfitting_penalty

            # Check for valid objective
            if not np.isfinite(objective):
                return np.inf

            # Report intermediate values for pruning
            trial.report(objective, step=0)

            return objective

        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            return np.inf

    def train_comprehensive_model(self, X_train: Union[pd.DataFrame, np.ndarray],
                                 y_train: Union[pd.Series, np.ndarray],
                                 X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
                                 y_val: Optional[Union[pd.Series, np.ndarray]] = None,
                                 optimization_method: str = 'optuna',
                                 n_trials: int = 100,
                                 cv_folds: int = 3) -> Dict[str, Any]:
        """
        Train comprehensive XGBoost model with advanced optimization

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Optional validation features
            y_val: Optional validation targets
            optimization_method: 'optuna', 'grid', or 'random'
            n_trials: Number of optimization trials
            cv_folds: Number of cross-validation folds

        Returns:
            Training results dictionary
        """
        logger.info(f"Starting comprehensive XGBoost training with {optimization_method} optimization")

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
            # Create validation split
            split_idx = int(0.8 * len(X_train))
            X_val = X_train[split_idx:]
            y_val = y_train[split_idx:]
            X_train = X_train[:split_idx]
            y_train = y_train[:split_idx]
            logger.info(f"Created validation split: {len(X_train)} train, {len(X_val)} validation")
        else:
            if isinstance(X_val, pd.DataFrame):
                X_val = X_val.values
            if isinstance(y_val, pd.Series):
                y_val = y_val.values

        # Create temporal cross-validation splits
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        cv_splits = list(tscv.split(X_train))

        # Hyperparameter optimization
        if optimization_method == 'optuna':
            # Optuna optimization
            study = optuna.create_study(
                direction='minimize',
                sampler=TPESampler(seed=42),
                pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
            )

            objective_func = lambda trial: self.optuna_objective(trial, X_train, y_train, cv_splits)
            study.optimize(objective_func, n_trials=n_trials, show_progress_bar=True)

            best_params = study.best_params
            logger.info(f"Best parameters found: {best_params}")

        elif optimization_method == 'grid':
            # Grid search optimization
            param_space = self.get_comprehensive_param_space()

            # Reduce grid size for tractability
            reduced_space = {
                'max_depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1, 0.15],
                'n_estimators': [200, 500],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
                'reg_alpha': [0, 0.1, 1],
                'reg_lambda': [0, 0.1, 1]
            }

            best_score = np.inf
            best_params = None

            for params in ParameterGrid(reduced_space):
                cv_scores = []
                for train_idx, val_idx in cv_splits:
                    X_tr, X_val_cv = X_train[train_idx], X_train[val_idx]
                    y_tr, y_val_cv = y_train[train_idx], y_train[val_idx]

                    model, _ = self.train_with_early_stopping(
                        X_tr, y_tr, X_val_cv, y_val_cv, params
                    )

                    val_pred = model.predict(X_val_cv)
                    cv_scores.append(mean_absolute_error(y_val_cv, val_pred))

                mean_score = np.mean(cv_scores)
                if mean_score < best_score:
                    best_score = mean_score
                    best_params = params

            logger.info(f"Best parameters found: {best_params}")

        else:  # Default parameters
            best_params = {
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 500,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1,
            }

        # Train final model with best parameters
        logger.info("Training final model with best parameters")
        final_model, training_history = self.train_with_early_stopping(
            X_train, y_train, X_val, y_val, best_params,
            early_stopping_rounds=50
        )

        # Evaluate overfitting
        overfitting_metrics = self.evaluate_overfitting(training_history)

        # Calculate final predictions and metrics
        train_pred = final_model.predict(X_train)
        val_pred = final_model.predict(X_val)

        final_metrics = {
            'train_mae': mean_absolute_error(y_train, train_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'train_r2': r2_score(y_train, train_pred),
            'val_mae': mean_absolute_error(y_val, val_pred),
            'val_rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
            'val_r2': r2_score(y_val, val_pred),
        }

        # Feature importance
        feature_importance = dict(zip(feature_names, final_model.feature_importances_))
        sorted_importance = dict(sorted(feature_importance.items(),
                                      key=lambda x: x[1], reverse=True))

        # Store results
        self.best_model = final_model
        self.best_params = best_params
        self.best_score = final_metrics['val_mae']
        self.training_history = training_history
        self.feature_importance = sorted_importance

        logger.info(f"Training completed. Val MAE: {final_metrics['val_mae']:.4f}, "
                   f"Val R²: {final_metrics['val_r2']:.4f}")
        logger.info(f"Overfitting check - Train/Val MAE ratio: {overfitting_metrics['train_val_mae_ratio']:.3f}")

        return {
            'model': final_model,
            'best_params': best_params,
            'metrics': final_metrics,
            'overfitting_metrics': overfitting_metrics,
            'training_history': training_history,
            'feature_importance': sorted_importance,
            'cv_folds': cv_folds,
            'optimization_method': optimization_method
        }

    def plot_training_curves(self, save_path: Optional[str] = None) -> None:
        """
        Plot training and validation curves

        Args:
            save_path: Optional path to save the plot
        """
        if not self.training_history:
            logger.warning("No training history available for plotting")
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # MAE curves
        iterations = range(len(self.training_history['train_mae']))
        ax1.plot(iterations, self.training_history['train_mae'], label='Training MAE', color='blue')
        ax1.plot(iterations, self.training_history['validation_mae'], label='Validation MAE', color='red')
        ax1.axvline(x=self.training_history['best_iteration'], color='green',
                   linestyle='--', label='Best Iteration')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('MAE')
        ax1.set_title('Mean Absolute Error')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # RMSE curves
        ax2.plot(iterations, self.training_history['train_rmse'], label='Training RMSE', color='blue')
        ax2.plot(iterations, self.training_history['validation_rmse'], label='Validation RMSE', color='red')
        ax2.axvline(x=self.training_history['best_iteration'], color='green',
                   linestyle='--', label='Best Iteration')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('RMSE')
        ax2.set_title('Root Mean Square Error')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Feature importance (top 15)
        if self.feature_importance:
            top_features = dict(list(self.feature_importance.items())[:15])
            features = list(top_features.keys())
            importances = list(top_features.values())

            ax3.barh(range(len(features)), importances)
            ax3.set_yticks(range(len(features)))
            ax3.set_yticklabels(features)
            ax3.set_xlabel('Importance')
            ax3.set_title('Top 15 Feature Importance')
            ax3.grid(True, alpha=0.3)

        # Overfitting analysis
        train_val_ratio = np.array(self.training_history['validation_mae']) / np.array(self.training_history['train_mae'])
        ax4.plot(iterations, train_val_ratio, color='purple', label='Val/Train MAE Ratio')
        ax4.axhline(y=1.0, color='black', linestyle='-', alpha=0.5, label='Perfect Fit')
        ax4.axhline(y=1.2, color='red', linestyle='--', alpha=0.7, label='Overfitting Threshold')
        ax4.axvline(x=self.training_history['best_iteration'], color='green',
                   linestyle='--', label='Best Iteration')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Ratio')
        ax4.set_title('Overfitting Monitor (Val/Train Ratio)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training curves saved to {save_path}")

        plt.show()

    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive model summary

        Returns:
            Model summary dictionary
        """
        if self.best_model is None:
            return {"error": "No model trained yet"}

        overfitting_metrics = self.evaluate_overfitting(self.training_history)

        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'overfitting_metrics': overfitting_metrics,
            'feature_importance_top10': dict(list(self.feature_importance.items())[:10]),
            'training_iterations': self.training_history['best_iteration'],
            'early_stopping_triggered': overfitting_metrics['early_stopped']
        }
