import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class RiskModel:
    """
    XGBoost-based risk model for predicting trader performance.
    Uses MAE objective for robustness to outliers.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = None
        self.feature_importance = None
        self.best_params = None

    def get_default_params(self) -> Dict:
        """Get default XGBoost parameters optimized for financial time series."""
        return {
            'objective': 'reg:absoluteerror',  # MAE objective for robustness
            'eval_metric': 'mae',
            'random_state': self.random_state,
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,  # L1 regularization
            'reg_lambda': 1.0,  # L2 regularization
            'verbosity': 0
        }

    def tune_hyperparameters(self,
                           X_train: pd.DataFrame,
                           y_train: pd.Series,
                           cv_folds: int = 3,
                           n_jobs: int = -1) -> Dict:
        """
        Tune hyperparameters using time series cross-validation.

        Args:
            X_train: Training features
            y_train: Training targets
            cv_folds: Number of CV folds
            n_jobs: Number of parallel jobs

        Returns:
            Best parameters dictionary
        """
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.05, 0.1, 0.2],
            'reg_alpha': [0, 0.1, 1.0],
            'reg_lambda': [0.1, 1.0, 10.0]
        }

        base_params = self.get_default_params()

        # Use time series split for CV (preserves temporal order)
        from sklearn.model_selection import TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=cv_folds)

        xgb_model = xgb.XGBRegressor(**base_params)

        grid_search = GridSearchCV(
            estimator=xgb_model,
            param_grid=param_grid,
            cv=tscv,
            scoring='neg_mean_absolute_error',
            n_jobs=n_jobs,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        self.best_params = {**base_params, **grid_search.best_params_}

        print(f"Best CV Score: {-grid_search.best_score_:.4f}")
        print(f"Best Parameters: {grid_search.best_params_}")

        return self.best_params

    def train(self,
              X_train: pd.DataFrame,
              y_train: pd.Series,
              X_val: pd.DataFrame = None,
              y_val: pd.Series = None,
              params: Dict = None) -> None:
        """
        Train the XGBoost model.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            params: Model parameters (if None, use default)
        """
        if params is None:
            params = self.best_params if self.best_params else self.get_default_params()

        self.model = xgb.XGBRegressor(**params)

        # Train model with early stopping if validation set provided
        if X_val is not None and y_val is not None:
            # Add early stopping for training with validation set
            training_params = params.copy()
            training_params['early_stopping_rounds'] = 10
            self.model = xgb.XGBRegressor(**training_params)

            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            # Train without early stopping if no validation set
            self.model.fit(X_train, y_train)

        # Store feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        return self.model.predict(X)

    def evaluate(self,
                X_test: pd.DataFrame,
                y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            X_test: Test features
            y_test: Test targets

        Returns:
            Evaluation metrics dictionary
        """
        y_pred = self.predict(X_test)

        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': self.model.score(X_test, y_test)
        }

        return metrics

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get top N most important features."""
        if self.feature_importance is None:
            raise ValueError("Model not trained yet. Call train() first.")

        return self.feature_importance.head(top_n)

    def generate_risk_signals(self,
                            predictions: np.ndarray,
                            percentiles: Tuple[float, float] = (10, 75)) -> np.ndarray:
        """
        Convert predictions to risk signals.

        Args:
            predictions: Model predictions
            percentiles: (low_risk_threshold, high_risk_threshold) percentiles

        Returns:
            Risk signals: 0 = High Risk, 1 = Neutral, 2 = Low Risk/High Conviction
        """
        low_threshold = np.percentile(predictions, percentiles[0])
        high_threshold = np.percentile(predictions, percentiles[1])

        signals = np.ones(len(predictions))  # Default to neutral
        signals[predictions < low_threshold] = 0  # High risk
        signals[predictions > high_threshold] = 2  # Low risk/high conviction

        return signals.astype(int)
