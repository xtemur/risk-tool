import pandas as pd
import numpy as np
import lightgbm as lgb
from typing import Dict, Tuple, Optional, List
from pathlib import Path
import logging
import yaml
import pickle
import json
from collections import defaultdict
import warnings
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, mean_absolute_error
import optuna
from datetime import datetime

warnings.filterwarnings('ignore')

# Removed data processing imports - using preprocessed parquet files
# from src.data_processing import create_trader_day_panel
# from src.feature_engineering import build_features

logger = logging.getLogger(__name__)


class TraderSpecificTrainer:
    """
    Trainer that prepares, tunes, validates and trains models for each trader
    using exactly 80% of their data for training/validation.
    """

    def __init__(self, data_dir: str = 'data/processed/trader_splits', sequence_length: int = 7):
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.models_dir = Path('models/trader_specific_80pct')
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.raw_feature_cols = [
            'daily_pnl', 'daily_gross', 'daily_fees', 'daily_volume',
            'n_trades', 'gross_profit', 'gross_loss'
        ]

        # Base hyperparameter search space
        self.param_search_space = {
            'num_leaves': [15, 31, 63],
            'learning_rate': [0.01, 0.05, 0.1],
            'feature_fraction': [0.6, 0.8, 1.0],
            'bagging_fraction': [0.6, 0.8, 1.0],
            'min_child_samples': [10, 20, 50],
            'reg_alpha': [0.0, 0.1, 0.5],
            'reg_lambda': [0.0, 0.1, 0.5]
        }

        self.base_params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42,
            'force_row_wise': True
        }

        logger.info(f"TraderSpecificTrainer initialized: data_dir={data_dir}, sequence_length={sequence_length}")

    def load_trader_data(self, trader_id: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """Load preprocessed training and test data for a trader"""
        trader_dir = self.data_dir / str(trader_id)

        if not trader_dir.exists():
            logger.warning(f"No data directory found for trader {trader_id}")
            return None, None, None

        # Load training and test data
        train_path = trader_dir / 'train_data.parquet'
        test_path = trader_dir / 'test_data.parquet'
        metadata_path = trader_dir / 'metadata.json'

        if not all(p.exists() for p in [train_path, test_path, metadata_path]):
            logger.warning(f"Missing data files for trader {trader_id}")
            return None, None, None

        train_data = pd.read_parquet(train_path)
        test_data = pd.read_parquet(test_path)

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Convert date columns to datetime
        train_data['date'] = pd.to_datetime(train_data['date'])
        test_data['date'] = pd.to_datetime(test_data['date'])

        logger.info(f"Loaded trader {trader_id}: Train={len(train_data)} days, Test={len(test_data)} days")

        return train_data, test_data, metadata

    def prepare_trader_features(self, train_data: pd.DataFrame, trader_id: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
        """Prepare features from preprocessed training data"""
        trader_data = train_data.sort_values('date').copy()

        # Get all feature columns (exclude metadata and targets)
        exclude_cols = ['trader_id', 'date', 'target_pnl', 'target_large_loss']
        feature_cols = [col for col in trader_data.columns if col not in exclude_cols]

        # Remove features that are all null
        valid_feature_cols = []
        for col in feature_cols:
            if not trader_data[col].isnull().all():
                valid_feature_cols.append(col)

        logger.info(f"Trader {trader_id}: Using {len(valid_feature_cols)} features")

        # Prepare data with sequence_length lookback
        features_list = []
        targets_cls = []
        targets_pnl = []
        feature_dates = []

        for i in range(self.sequence_length, len(trader_data)):
            # Get current row features
            current_features = trader_data.iloc[i][valid_feature_cols]
            target_cls = trader_data.iloc[i]['target_large_loss']
            target_pnl = trader_data.iloc[i]['target_pnl'] if 'target_pnl' in trader_data.columns else trader_data.iloc[i]['daily_pnl']
            current_date = trader_data.iloc[i]['date']

            # Check for valid data
            features_valid = not pd.isna(current_features.values).any()
            targets_valid = not pd.isna(target_cls) and not pd.isna(target_pnl)

            if features_valid and targets_valid:
                features_list.append(current_features.values)
                targets_cls.append(target_cls)
                targets_pnl.append(target_pnl)
                feature_dates.append(current_date)

        if len(features_list) == 0:
            return None, None, None, None

        # Convert to arrays/DataFrame
        features_df = pd.DataFrame(features_list, columns=valid_feature_cols)
        features_df['date'] = feature_dates
        targets_cls = np.array(targets_cls, dtype=np.float32)
        targets_pnl = np.array(targets_pnl, dtype=np.float32)

        # Clean data
        features_df[valid_feature_cols] = features_df[valid_feature_cols].replace([np.inf, -np.inf], np.nan)
        features_df[valid_feature_cols] = features_df[valid_feature_cols].fillna(
            features_df[valid_feature_cols].median()
        )
        features_df[valid_feature_cols] = features_df[valid_feature_cols].clip(-1e6, 1e6)

        logger.info(f"Trader {trader_id}: Prepared {len(features_df)} samples with {len(valid_feature_cols)} features")

        return features_df, targets_cls, targets_pnl, valid_feature_cols

    def hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: np.ndarray,
                            X_val: pd.DataFrame, y_val: np.ndarray,
                            trader_id: str, n_trials: int = 50) -> Dict:
        """Perform hyperparameter tuning using Optuna"""

        def objective(trial):
            params = self.base_params.copy()
            params.update({
                'num_leaves': trial.suggest_categorical('num_leaves', [15, 31, 63]),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0)
            })

            # Ensure we have both classes in training data
            if len(np.unique(y_train)) < 2:
                return 0.5  # Return baseline AUC

            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )

            if len(np.unique(y_val)) < 2:
                return 0.5

            y_pred = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_pred)
            return auc

        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_params = self.base_params.copy()
        best_params.update(study.best_params)

        logger.info(f"Trader {trader_id}: Best AUC={study.best_value:.4f} with params: {study.best_params}")

        return best_params

    def validate_model(self, model: lgb.LGBMClassifier, X_val: pd.DataFrame,
                      y_val: np.ndarray, trader_id: str) -> Dict:
        """Validate model performance on validation set"""
        if len(np.unique(y_val)) < 2:
            logger.warning(f"Trader {trader_id}: Validation set has only one class")
            return {}

        y_pred_proba = model.predict_proba(X_val)[:, 1]
        y_pred = model.predict(X_val)

        metrics = {
            'auc': roc_auc_score(y_val, y_pred_proba),
            'ap': average_precision_score(y_val, y_pred_proba),
            'pos_rate': np.mean(y_val),
            'pred_pos_rate': np.mean(y_pred),
            'validation_samples': len(y_val)
        }

        logger.info(f"Trader {trader_id} validation metrics: "
                   f"AUC={metrics['auc']:.4f}, AP={metrics['ap']:.4f}")

        return metrics

    def split_train_validation(self, features_df: pd.DataFrame, targets_cls: np.ndarray, targets_pnl: np.ndarray, validation_split: float = 0.8) -> Tuple:
        """Split training data into train/validation sets"""
        split_idx = int(len(features_df) * validation_split)

        feature_cols = [col for col in features_df.columns if col != 'date']

        # Training set
        X_train = features_df.iloc[:split_idx][feature_cols]
        y_train_cls = targets_cls[:split_idx]
        y_train_pnl = targets_pnl[:split_idx]

        # Validation set
        X_val = features_df.iloc[split_idx:][feature_cols]
        y_val_cls = targets_cls[split_idx:]
        y_val_pnl = targets_pnl[split_idx:]

        return X_train, y_train_cls, y_train_pnl, X_val, y_val_cls, y_val_pnl

    def hyperparameter_tuning_var(self, X_train: pd.DataFrame, y_train: np.ndarray,
                                X_val: pd.DataFrame, y_val: np.ndarray,
                                trader_id: str, n_trials: int = 30) -> Dict:
        """Perform hyperparameter tuning for VaR model using Optuna"""

        # VaR model base parameters from config
        var_base_params = {
            'objective': 'quantile',
            'alpha': 0.05,  # 5% VaR
            'boosting_type': 'gbdt',
            'verbose': -1,
            'random_state': 42,
            'force_row_wise': True
        }

        def objective(trial):
            params = var_base_params.copy()
            params.update({
                'num_leaves': trial.suggest_categorical('num_leaves', [15, 31, 63]),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0)
            })

            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )

            y_pred = model.predict(X_val)
            # Use MAE as optimization metric for VaR
            mae = mean_absolute_error(y_val, y_pred)
            return mae

        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_params = var_base_params.copy()
        best_params.update(study.best_params)

        logger.info(f"Trader {trader_id} VaR model: Best MAE={study.best_value:.4f} with params: {study.best_params}")

        return best_params

    def validate_var_model(self, model: lgb.LGBMRegressor, X_val: pd.DataFrame,
                         y_val: np.ndarray, trader_id: str) -> Dict:
        """Validate VaR model performance on validation set"""
        y_pred = model.predict(X_val)

        # VaR-specific metrics
        mae = mean_absolute_error(y_val, y_pred)

        # Calculate actual vs predicted VaR coverage
        actual_var_5pct = np.percentile(y_val, 5)
        predicted_var_5pct = np.percentile(y_pred, 5)

        # Count violations (actual losses worse than predicted VaR)
        violations = np.sum(y_val < y_pred)
        violation_rate = violations / len(y_val)
        expected_violation_rate = 0.05  # 5% VaR

        metrics = {
            'mae': mae,
            'actual_var_5pct': actual_var_5pct,
            'predicted_var_5pct': predicted_var_5pct,
            'violation_rate': violation_rate,
            'expected_violation_rate': expected_violation_rate,
            'violation_difference': abs(violation_rate - expected_violation_rate),
            'validation_samples': len(y_val),
            'pnl_mean': np.mean(y_val),
            'pnl_std': np.std(y_val)
        }

        logger.info(f"Trader {trader_id} VaR validation metrics: "
                   f"MAE={metrics['mae']:.4f}, Violation Rate={metrics['violation_rate']:.4f}")

        return metrics

    def train_trader_model(self, trader_id: str) -> Optional[Dict]:
        """Train, tune, validate and save both classification and VaR models for a specific trader using preprocessed data"""

        # Load preprocessed data
        train_data, test_data, metadata = self.load_trader_data(trader_id)
        if train_data is None:
            return None

        # Prepare features
        features_df, targets_cls, targets_pnl, feature_names = self.prepare_trader_features(train_data, trader_id)
        if features_df is None:
            logger.warning(f"No valid features for trader {trader_id}")
            return None

        # Split training data into train/validation (80/20)
        X_train, y_train_cls, y_train_pnl, X_val, y_val_cls, y_val_pnl = self.split_train_validation(
            features_df, targets_cls, targets_pnl
        )

        logger.info(f"Trader {trader_id}: Train samples={len(X_train)}, Val samples={len(X_val)}")

        # Check if we have both classes in training data
        if len(np.unique(y_train_cls)) < 2:
            logger.warning(f"Trader {trader_id}: Training data has only one class")
            return None

        # Train Classification Model
        logger.info(f"Training classification model for trader {trader_id}")
        best_params_cls = self.hyperparameter_tuning(X_train, y_train_cls, X_val, y_val_cls, trader_id)

        # Train final classification model on full training data
        feature_cols = [col for col in features_df.columns if col != 'date']
        X_full_train = features_df[feature_cols]
        y_full_train_cls = targets_cls

        final_cls_model = lgb.LGBMClassifier(**best_params_cls)
        final_cls_model.fit(X_full_train, y_full_train_cls)

        cls_validation_metrics = self.validate_model(final_cls_model, X_val, y_val_cls, trader_id)

        # Train VaR Model
        logger.info(f"Training VaR model for trader {trader_id}")
        best_params_var = self.hyperparameter_tuning_var(X_train, y_train_pnl, X_val, y_val_pnl, trader_id)

        y_full_train_pnl = targets_pnl
        final_var_model = lgb.LGBMRegressor(**best_params_var)
        final_var_model.fit(X_full_train, y_full_train_pnl)

        var_validation_metrics = self.validate_var_model(final_var_model, X_val, y_val_pnl, trader_id)

        # Save both models and metadata
        model_data = {
            'classification_model': final_cls_model,
            'var_model': final_var_model,
            'feature_names': feature_cols,
            'sequence_length': self.sequence_length,
            'metadata': metadata,
            'best_params_cls': best_params_cls,
            'best_params_var': best_params_var,
            'cls_validation_metrics': cls_validation_metrics,
            'var_validation_metrics': var_validation_metrics,
            'training_samples': len(X_full_train),
            'trader_id': trader_id,
            'trained_date': datetime.now().isoformat()
        }

        model_path = self.models_dir / f"{trader_id}_tuned_validated.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"Saved classification and VaR models for trader {trader_id} to {model_path}")

        return model_data

    def evaluate_on_test_data(self, trader_id: str) -> Optional[Dict]:
        """Evaluate trained models on unseen test data"""
        # Load trained model
        model_data = self.load_trader_model(trader_id)
        if model_data is None:
            logger.warning(f"No trained model found for trader {trader_id}")
            return None

        # Load test data
        train_data, test_data, metadata = self.load_trader_data(trader_id)
        if test_data is None:
            logger.warning(f"No test data found for trader {trader_id}")
            return None

        # Prepare test features (same way as training features)
        test_features_df, test_targets_cls, test_targets_pnl, test_feature_names = self.prepare_trader_features(test_data, trader_id)
        if test_features_df is None:
            logger.warning(f"Could not prepare test features for trader {trader_id}")
            return None

        # Get models and feature names
        cls_model = model_data['classification_model']
        var_model = model_data['var_model']
        feature_names = model_data['feature_names']

        # Ensure feature alignment
        test_feature_cols = [col for col in test_features_df.columns if col != 'date']
        if set(test_feature_cols) != set(feature_names):
            logger.warning(f"Feature mismatch for trader {trader_id}. Expected: {len(feature_names)}, Got: {len(test_feature_cols)}")
            # Align features
            common_features = [f for f in feature_names if f in test_feature_cols]
            X_test = test_features_df[common_features]
            logger.info(f"Using {len(common_features)} common features for evaluation")
        else:
            X_test = test_features_df[feature_names]

        # Classification predictions
        cls_test_metrics = {}
        if len(np.unique(test_targets_cls)) >= 2:
            y_pred_proba_cls = cls_model.predict_proba(X_test)[:, 1]
            y_pred_cls = cls_model.predict(X_test)

            cls_test_metrics = {
                'auc': roc_auc_score(test_targets_cls, y_pred_proba_cls),
                'ap': average_precision_score(test_targets_cls, y_pred_proba_cls),
                'pos_rate': np.mean(test_targets_cls),
                'pred_pos_rate': np.mean(y_pred_cls),
                'test_samples': len(test_targets_cls)
            }
        else:
            logger.warning(f"Test set for trader {trader_id} has only one class for classification")
            cls_test_metrics = {'test_samples': len(test_targets_cls), 'note': 'only_one_class'}

        # VaR predictions
        y_pred_var = var_model.predict(X_test)

        var_test_metrics = {
            'mae': mean_absolute_error(test_targets_pnl, y_pred_var),
            'actual_var_5pct': np.percentile(test_targets_pnl, 5),
            'predicted_var_5pct': np.percentile(y_pred_var, 5),
            'test_samples': len(test_targets_pnl),
            'pnl_mean': np.mean(test_targets_pnl),
            'pnl_std': np.std(test_targets_pnl)
        }

        # Calculate VaR violations
        violations = np.sum(test_targets_pnl < y_pred_var)
        var_test_metrics['violation_rate'] = violations / len(test_targets_pnl)
        var_test_metrics['expected_violation_rate'] = 0.05
        var_test_metrics['violation_difference'] = abs(var_test_metrics['violation_rate'] - 0.05)

        test_results = {
            'trader_id': trader_id,
            'cls_test_metrics': cls_test_metrics,
            'var_test_metrics': var_test_metrics,
            'test_date_range': {
                'start': test_features_df['date'].min().isoformat(),
                'end': test_features_df['date'].max().isoformat()
            },
            'evaluation_date': datetime.now().isoformat()
        }

        logger.info(f"Trader {trader_id} test evaluation:")
        if 'auc' in cls_test_metrics:
            logger.info(f"  Classification - AUC: {cls_test_metrics['auc']:.4f}, AP: {cls_test_metrics['ap']:.4f}")
        logger.info(f"  VaR - MAE: {var_test_metrics['mae']:.2f}, Violation Rate: {var_test_metrics['violation_rate']:.4f}")

        return test_results

    def evaluate_all_traders_on_test(self, trader_ids: Optional[List] = None) -> Dict:
        """Evaluate all trained models on test data"""
        if trader_ids is None:
            trader_ids = self.get_available_traders()

        test_results = {}
        successful_evals = 0

        logger.info(f"Evaluating {len(trader_ids)} traders on test data")

        for trader_id in trader_ids:
            logger.info(f"\n=== Evaluating trader {trader_id} on test data ===")

            try:
                result = self.evaluate_on_test_data(trader_id)
                if result is not None:
                    test_results[trader_id] = result
                    successful_evals += 1
                else:
                    logger.warning(f"Failed to evaluate trader {trader_id}")
            except Exception as e:
                logger.error(f"Error evaluating trader {trader_id}: {str(e)}")
                continue

        logger.info(f"\nTest evaluation completed: {successful_evals}/{len(trader_ids)} traders successful")

        # Save test results
        test_summary = {
            'total_traders': len(trader_ids),
            'successful_evaluations': successful_evals,
            'evaluated_traders': list(test_results.keys()),
            'evaluation_date': datetime.now().isoformat()
        }

        results_path = self.models_dir / 'test_evaluation_results.pkl'
        with open(results_path, 'wb') as f:
            pickle.dump({'summary': test_summary, 'results': test_results}, f)

        logger.info(f"Test results saved to {results_path}")

        return test_results

    def load_trader_model(self, trader_id: str) -> Optional[Dict]:
        """Load a saved trader model"""
        model_path = self.models_dir / f"{trader_id}_tuned_validated.pkl"
        if not model_path.exists():
            logger.warning(f"No saved model found for trader {trader_id}")
            return None

        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        return model_data

    def get_available_traders(self) -> List[str]:
        """Get list of available trader IDs from data directory"""
        trader_ids = []
        for trader_dir in self.data_dir.iterdir():
            if trader_dir.is_dir() and trader_dir.name.isdigit():
                # Check if required files exist
                required_files = ['train_data.parquet', 'test_data.parquet', 'metadata.json']
                if all((trader_dir / f).exists() for f in required_files):
                    trader_ids.append(trader_dir.name)
        return sorted(trader_ids)

    def train_all_traders(self, trader_ids: Optional[List] = None) -> Dict:
        """Train models for all specified traders using preprocessed data"""
        if trader_ids is None:
            trader_ids = self.get_available_traders()

        results = {}
        successful_trains = 0

        logger.info(f"Found {len(trader_ids)} traders to train: {trader_ids}")

        for trader_id in trader_ids:
            logger.info(f"\n=== Training model for trader {trader_id} ===")

            try:
                model_data = self.train_trader_model(trader_id)
                if model_data is not None:
                    results[trader_id] = model_data
                    successful_trains += 1
                    logger.info(f"Successfully trained model for trader {trader_id}")
                else:
                    logger.warning(f"Failed to train model for trader {trader_id}")
            except Exception as e:
                logger.error(f"Error training model for trader {trader_id}: {str(e)}")
                continue

        logger.info(f"\nTraining completed: {successful_trains}/{len(trader_ids)} traders successful")

        # Save summary
        summary = {
            'total_traders': len(trader_ids),
            'successful_trains': successful_trains,
            'trained_traders': list(results.keys()),
            'training_date': datetime.now().isoformat(),
            'model_types': ['classification', 'var']
        }

        summary_path = self.models_dir / 'training_summary.pkl'
        with open(summary_path, 'wb') as f:
            pickle.dump(summary, f)

        return results


def run_trader_specific_training(data_dir: str = 'data/processed/trader_splits', sequence_length: int = 7, trader_ids: Optional[List] = None):
    """Main function to run trader-specific training with both classification and VaR models using preprocessed data"""

    logging.basicConfig(level=logging.INFO)
    logger.info(f"Starting trader-specific training (Classification + VaR) with sequence_length={sequence_length}")
    logger.info(f"Using preprocessed data from: {data_dir}")

    # Initialize trainer with data directory
    trainer = TraderSpecificTrainer(data_dir=data_dir, sequence_length=sequence_length)

    # Train all models (both classification and VaR)
    results = trainer.train_all_traders(trader_ids)

    logger.info(f"Training completed for {len(results)} traders (Classification + VaR models)")

    return results


def run_classification_only_training(data_dir: str = 'data/processed/trader_splits', sequence_length: int = 7, trader_ids: Optional[List] = None):
    """Run training with classification models only (backward compatibility)"""

    # For backward compatibility, create a version that only trains classification models
    # This can be done by modifying the trainer to skip VaR training
    logger.warning("Classification-only training mode - consider using full training with both models")

    return run_trader_specific_training(data_dir, sequence_length, trader_ids)


def evaluate_models_on_test_data(data_dir: str = 'data/processed/trader_splits', trader_ids: Optional[List] = None):
    """Evaluate trained models on test data"""

    logging.basicConfig(level=logging.INFO)
    logger.info("Starting test data evaluation")

    # Initialize trainer
    trainer = TraderSpecificTrainer(data_dir=data_dir)

    # Evaluate all models on test data
    test_results = trainer.evaluate_all_traders_on_test(trader_ids)

    logger.info(f"Test evaluation completed for {len(test_results)} traders")

    return test_results


if __name__ == "__main__":
    results = run_trader_specific_training(sequence_length=7)
    print(f"Successfully trained models for {len(results)} traders")
