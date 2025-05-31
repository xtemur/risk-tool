"""
Enhanced Model Trainer for Risk Management MVP
Includes hyperparameter tuning on validation set and proper time series validation
"""

import logging
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, Tuple, Optional, List

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Enhanced trainer with hyperparameter tuning for personal LightGBM models"""

    def __init__(self, models_path: str = "data/models"):
        self.models_path = Path(models_path)
        self.models_path.mkdir(exist_ok=True)

        # Hyperparameter search space
        self.param_grid = {
            'num_leaves': [10, 15, 20, 31],
            'learning_rate': [0.03, 0.05, 0.08, 0.1],
            'feature_fraction': [0.6, 0.8, 0.9],
            'min_data_in_leaf': [15, 20, 30],
            'lambda_l1': [0.5, 1.0, 2.0],
            'lambda_l2': [0.5, 1.0, 2.0]
        }

        # Base parameters
        self.base_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'random_state': 42,
            'verbose': -1,
            'force_col_wise': True
        }

        self.min_training_days = 30
        self.max_trials = 20  # Limit hyperparameter search for MVP

    def create_time_splits(self, features_df: pd.DataFrame,
                          val_size: float = 0.15,
                          test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create proper time series splits to avoid data leakage

        Returns:
            train_df: First 65% for training
            val_df: Next 15% for validation/hyperparameter tuning
            test_df: Last 20% for final evaluation
        """
        n = len(features_df)

        train_end = int(n * (1 - val_size - test_size))
        val_end = int(n * (1 - test_size))

        train_df = features_df.iloc[:train_end].copy()
        val_df = features_df.iloc[train_end:val_end].copy()
        test_df = features_df.iloc[val_end:].copy()

        logger.info(f"Time splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

        return train_df, val_df, test_df

    def hyperparameter_search(self, X_train: np.ndarray, y_train: np.ndarray,
                             X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """
        Grid search with early stopping on validation set
        """
        best_score = float('inf')
        best_params = None
        best_model = None

        # Create parameter combinations (limited for MVP)
        param_combinations = list(ParameterGrid(self.param_grid))

        # Limit search space for faster training
        if len(param_combinations) > self.max_trials:
            import random
            random.seed(42)
            param_combinations = random.sample(param_combinations, self.max_trials)

        logger.info(f"Testing {len(param_combinations)} parameter combinations")

        for i, params in enumerate(param_combinations):
            try:
                # Combine with base parameters
                full_params = {**self.base_params, **params}

                # Dynamic n_estimators based on data size
                full_params['n_estimators'] = min(1000, max(100, len(X_train) // 5))

                # Create datasets
                train_data = lgb.Dataset(X_train, label=y_train)
                val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

                # Train with early stopping
                model = lgb.train(
                    full_params,
                    train_data,
                    valid_sets=[val_data],
                    callbacks=[
                        lgb.early_stopping(50),
                        lgb.log_evaluation(0)
                    ]
                )

                # Evaluate on validation set
                val_pred = model.predict(X_val, num_iteration=model.best_iteration)
                val_score = mean_squared_error(y_val, val_pred)

                if val_score < best_score:
                    best_score = val_score
                    best_params = full_params
                    best_model = model

                logger.debug(f"Trial {i+1}/{len(param_combinations)}: RMSE={np.sqrt(val_score):.3f}")

            except Exception as e:
                logger.warning(f"Trial {i+1} failed: {str(e)}")
                continue

        if best_model is None:
            raise ValueError("All hyperparameter trials failed")

        logger.info(f"Best validation RMSE: {np.sqrt(best_score):.3f}")

        return {
            'model': best_model,
            'best_params': best_params,
            'best_score': best_score,
            'trials_completed': len(param_combinations)
        }

    def train_personal_model(self, features_df: pd.DataFrame,
                           account_id: str,
                           feature_columns: List[str]) -> Optional[Dict]:
        """Train a personal model with hyperparameter tuning"""

        if len(features_df) < self.min_training_days:
            logger.warning(f"Insufficient data for {account_id}: {len(features_df)} days")
            return None

        logger.info(f"Training model for {account_id} with {len(features_df)} days of data")

        # Create time series splits
        train_df, val_df, test_df = self.create_time_splits(features_df)

        if len(val_df) < 5:  # Need minimum validation set
            logger.warning(f"Validation set too small for {account_id}: {len(val_df)} days")
            return None

        # Prepare data
        X_train = train_df[feature_columns].values
        y_train = train_df['target'].values
        X_val = val_df[feature_columns].values
        y_val = val_df['target'].values

        # Hyperparameter search
        try:
            search_result = self.hyperparameter_search(X_train, y_train, X_val, y_val)
            model = search_result['model']
            best_params = search_result['best_params']

        except Exception as e:
            logger.error(f"Hyperparameter search failed for {account_id}: {str(e)}")
            return None

        # Final evaluation metrics
        val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        val_mae = mean_absolute_error(y_val, val_pred)

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)

        # Calculate optimal threshold for trading signals
        threshold = self._find_optimal_threshold(y_val, val_pred)

        # Test set evaluation (final performance)
        test_metrics = {}
        if len(test_df) > 0:
            X_test = test_df[feature_columns].values
            y_test = test_df['target'].values
            test_pred = model.predict(X_test, num_iteration=model.best_iteration)

            test_metrics = {
                'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
                'test_mae': mean_absolute_error(y_test, test_pred),
                'directional_accuracy': np.mean((y_test > 0) == (test_pred > 0))
            }

        # Save model with all metadata
        model_data = {
            'model': model,
            'feature_columns': feature_columns,
            'threshold': threshold,
            'best_params': best_params,
            'validation_metrics': {
                'rmse': val_rmse,
                'mae': val_mae,
                'n_train': len(X_train),
                'n_val': len(X_val)
            },
            'test_metrics': test_metrics,
            'feature_importance': feature_importance,
            'training_date': pd.Timestamp.now(),
            'data_splits': {
                'train_dates': (train_df['date'].min(), train_df['date'].max()),
                'val_dates': (val_df['date'].min(), val_df['date'].max()),
                'test_dates': (test_df['date'].min(), test_df['date'].max()) if len(test_df) > 0 else None
            }
        }

        model_path = self.models_path / f"model_{account_id}.pkl"
        joblib.dump(model_data, model_path)

        logger.info(f"Model saved for {account_id}: Val RMSE={val_rmse:.2f}, "
                   f"Test RMSE={test_metrics.get('test_rmse', 'N/A')}")

        return {
            'account_id': account_id,
            'rmse': val_rmse,
            'mae': val_mae,
            'test_rmse': test_metrics.get('test_rmse', None),
            'directional_accuracy': test_metrics.get('directional_accuracy', None),
            'threshold': threshold,
            'n_samples': len(features_df),
            'top_features': feature_importance.head(5)['feature'].tolist(),
            'best_params': best_params
        }

    def _find_optimal_threshold(self, y_true: np.ndarray, predictions: np.ndarray) -> float:
        """Find optimal threshold for risk classification using validation data"""

        # Try different percentiles as thresholds
        thresholds = np.percentile(predictions, np.arange(10, 91, 10))
        best_threshold = 0
        best_profit = -np.inf

        for threshold in thresholds:
            # Simulate trading strategy: trade only when prediction > threshold
            trade_signals = predictions > threshold

            if np.sum(trade_signals) > 0:
                profit = np.sum(y_true[trade_signals])

                # Penalize strategies that trade too infrequently
                trade_frequency = np.mean(trade_signals)
                if trade_frequency < 0.1:  # Less than 10% trading days
                    profit *= trade_frequency / 0.1  # Penalty

                if profit > best_profit:
                    best_profit = profit
                    best_threshold = threshold

        return best_threshold

    def retrain_from_scratch(self, account_id: str, end_date: str = None) -> bool:
        """
        Retrain model from scratch with all available data up to end_date
        Used for daily prediction pipeline
        """
        from database import Database
        from feature_engineer import FeatureEngineer

        db = Database()
        feature_engineer = FeatureEngineer()

        # Get all data up to end_date
        totals_df, fills_df = db.get_trader_data(account_id, end_date=end_date)

        if totals_df.empty:
            logger.warning(f"No data found for {account_id}")
            return False

        # Create features
        features_df = feature_engineer.create_features(totals_df, fills_df)

        if len(features_df) < self.min_training_days:
            logger.warning(f"Insufficient data for retraining {account_id}: {len(features_df)} days")
            return False

        # Train model (this will use time series splits internally)
        result = self.train_personal_model(features_df, account_id, feature_engineer.get_feature_columns())

        return result is not None

    def train_all_models(self, all_features: Dict[str, pd.DataFrame],
                        feature_columns: List[str]) -> Dict[str, Dict]:
        """Train models for all traders with hyperparameter tuning"""
        results = {}

        for account_id, features_df in all_features.items():
            logger.info(f"Training model for {account_id}...")
            result = self.train_personal_model(features_df, account_id, feature_columns)
            if result:
                results[account_id] = result

        logger.info(f"Trained {len(results)} models successfully")

        # Save training summary
        if results:
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

    def get_model_summary(self) -> pd.DataFrame:
        """Get summary of all trained models"""
        models = self.get_all_models()

        if not models:
            return pd.DataFrame()

        summary_data = []
        for account_id, model_data in models.items():
            val_metrics = model_data.get('validation_metrics', {})
            test_metrics = model_data.get('test_metrics', {})

            summary_data.append({
                'account_id': account_id,
                'training_date': model_data.get('training_date'),
                'val_rmse': val_metrics.get('rmse'),
                'val_mae': val_metrics.get('mae'),
                'test_rmse': test_metrics.get('test_rmse'),
                'directional_accuracy': test_metrics.get('directional_accuracy'),
                'threshold': model_data.get('threshold'),
                'n_features': len(model_data.get('feature_columns', [])),
                'top_feature': model_data.get('feature_importance', pd.DataFrame()).iloc[0]['feature'] if not model_data.get('feature_importance', pd.DataFrame()).empty else None
            })

        return pd.DataFrame(summary_data)
