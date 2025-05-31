"""
Simplified Model Trainer for Risk Management MVP
Personal LightGBM models only - no global model, no ARIMA
"""

import logging
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train personal LightGBM models for each trader"""

    def __init__(self, models_path: str = "data/models"):
        self.models_path = Path(models_path)
        self.models_path.mkdir(exist_ok=True)

        # LightGBM parameters optimized for trading P&L prediction
        self.base_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 15,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'min_data_in_leaf': 20,
            'lambda_l1': 1.0,
            'lambda_l2': 1.0,
            'random_state': 42,
            'verbose': -1
        }

        self.min_training_days = 30  # Minimum days required to train a model

    def train_personal_model(self, features_df: pd.DataFrame,
                           account_id: str,
                           feature_columns: list) -> Optional[Dict]:
        """Train a personal model for a specific trader"""

        if len(features_df) < self.min_training_days:
            logger.warning(f"Insufficient data for {account_id}: {len(features_df)} days")
            return None

        logger.info(f"Training model for {account_id} with {len(features_df)} days of data")

        # Prepare data
        X = features_df[feature_columns].values
        y = features_df['target'].values

        # Time-based train/validation split (80/20)
        split_idx = int(len(features_df) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Train model with early stopping
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # Dynamic parameters based on data size
        params = self.base_params.copy()
        params['n_estimators'] = min(1000, max(100, len(X_train) // 10))

        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(50),
                lgb.log_evaluation(0)
            ]
        )

        # Evaluate on validation set
        val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        val_mae = mean_absolute_error(y_val, val_pred)

        # Calculate feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)

        # Calculate risk threshold (for binary classification of risk)
        # Use validation predictions to find optimal threshold
        risk_scores = model.predict(X_val, num_iteration=model.best_iteration)
        threshold = self._find_optimal_threshold(y_val, risk_scores)

        # Save model
        model_path = self.models_path / f"model_{account_id}.pkl"
        joblib.dump({
            'model': model,
            'feature_columns': feature_columns,
            'threshold': threshold,
            'validation_metrics': {
                'rmse': val_rmse,
                'mae': val_mae,
                'n_train': len(X_train),
                'n_val': len(X_val)
            },
            'feature_importance': feature_importance,
            'training_date': pd.Timestamp.now()
        }, model_path)

        logger.info(f"Model saved for {account_id}: RMSE={val_rmse:.2f}, MAE={val_mae:.2f}")

        return {
            'account_id': account_id,
            'rmse': val_rmse,
            'mae': val_mae,
            'threshold': threshold,
            'n_samples': len(features_df),
            'top_features': feature_importance.head(5)['feature'].tolist()
        }

    def _find_optimal_threshold(self, y_true: np.ndarray, predictions: np.ndarray) -> float:
        """Find optimal threshold for risk classification"""
        # Simple approach: find threshold that maximizes profit
        thresholds = np.percentile(predictions, np.arange(10, 90, 10))
        best_threshold = 0
        best_profit = -np.inf

        for threshold in thresholds:
            # Trade only when prediction is above threshold
            trade_signals = predictions > threshold
            profit = np.sum(y_true[trade_signals])

            if profit > best_profit:
                best_profit = profit
                best_threshold = threshold

        return best_threshold

    def train_all_models(self, all_features: Dict[str, pd.DataFrame],
                        feature_columns: list) -> Dict[str, Dict]:
        """Train models for all traders"""
        results = {}

        for account_id, features_df in all_features.items():
            result = self.train_personal_model(features_df, account_id, feature_columns)
            if result:
                results[account_id] = result

        logger.info(f"Trained {len(results)} models successfully")

        # Save training summary
        summary_df = pd.DataFrame(results).T
        summary_df.to_csv(self.models_path / 'training_summary.csv')

        return results

    def load_model(self, account_id: str) -> Optional[Dict]:
        """Load a trained model"""
        model_path = self.models_path / f"model_{account_id}.pkl"

        if not model_path.exists():
            logger.warning(f"No model found for {account_id}")
            return None

        try:
            return joblib.load(model_path)
        except Exception as e:
            logger.error(f"Error loading model for {account_id}: {str(e)}")
            return None

    def get_all_models(self) -> Dict[str, Dict]:
        """Load all available models"""
        models = {}

        for model_file in self.models_path.glob("model_*.pkl"):
            account_id = model_file.stem.replace("model_", "")
            model_data = self.load_model(account_id)
            if model_data:
                models[account_id] = model_data

        return models
