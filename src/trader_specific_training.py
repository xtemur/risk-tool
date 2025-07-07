import pandas as pd
import numpy as np
import lightgbm as lgb
from typing import Dict, Tuple, Optional, List
from pathlib import Path
import logging
import yaml
import pickle
from collections import defaultdict
import warnings
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, mean_absolute_error
import optuna
from datetime import datetime

warnings.filterwarnings('ignore')

from src.data_processing import create_trader_day_panel
from src.feature_engineering import build_features

logger = logging.getLogger(__name__)


class TraderSpecificTrainer:
    """
    Trainer that prepares, tunes, validates and trains models for each trader
    using exactly 80% of their data for training/validation.
    """

    def __init__(self, config_path: str = 'configs/main_config.yaml', sequence_length: int = 7):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

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

        logger.info(f"TraderSpecificTrainer initialized: sequence_length={sequence_length}")

    def calculate_trader_boundaries(self, df: pd.DataFrame, trader_id: str) -> Dict:
        """Calculate 80%/20% data boundaries for a specific trader"""
        trader_data = df[df['trader_id'] == trader_id].sort_values('date')
        dates = trader_data['date'].unique()

        if len(dates) < 100:  # Minimum data requirement
            logger.warning(f"Trader {trader_id} has only {len(dates)} days - insufficient data")
            return None

        # 80% for training/validation, 20% for final holdout test
        split_idx = int(len(dates) * 0.8)
        train_end_date = dates[split_idx - 1]
        test_start_date = dates[split_idx]

        # Within training data, use 80% for training, 20% for validation
        train_dates = dates[:split_idx]
        train_split_idx = int(len(train_dates) * 0.8)
        validation_start_date = train_dates[train_split_idx]

        boundaries = {
            'train_start': dates[0],
            'train_end': train_dates[train_split_idx - 1],
            'validation_start': validation_start_date,
            'validation_end': train_end_date,
            'test_start': test_start_date,
            'test_end': dates[-1],
            'total_days': len(dates),
            'train_days': train_split_idx,
            'validation_days': split_idx - train_split_idx,
            'test_days': len(dates) - split_idx
        }

        logger.info(f"Trader {trader_id} boundaries: "
                   f"Train: {boundaries['train_days']} days, "
                   f"Val: {boundaries['validation_days']} days, "
                   f"Test: {boundaries['test_days']} days")

        return boundaries

    def prepare_trader_features(self, df: pd.DataFrame, trader_id: str) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
        """Prepare combined features for a specific trader"""
        trader_data = df[df['trader_id'] == trader_id].sort_values('date').copy()

        # Get engineered features (exclude metadata and targets)
        exclude_cols = ['trader_id', 'date', 'target_pnl', 'target_large_loss']
        engineered_cols = [col for col in trader_data.columns
                          if col not in exclude_cols + self.raw_feature_cols]

        # Filter out features that are all null for this trader
        valid_engineered_cols = []
        for col in engineered_cols:
            if not trader_data[col].isnull().all():
                valid_engineered_cols.append(col)

        logger.info(f"Trader {trader_id}: Using {len(valid_engineered_cols)} engineered features")

        # Prepare sequential data and aligned engineered features
        combined_features_list = []
        aligned_targets = []
        feature_dates = []

        for i in range(self.sequence_length, len(trader_data)):
            # Raw sequence: last sequence_length days (flattened)
            sequence_data = trader_data.iloc[i-self.sequence_length:i][self.raw_feature_cols].values
            flattened_sequence = sequence_data.flatten()

            # Current day's engineered features
            current_engineered = trader_data.iloc[i][valid_engineered_cols]

            # Target (next day's risk)
            target = trader_data.iloc[i]['target_large_loss']
            current_date = trader_data.iloc[i]['date']

            # Only include if we have valid data
            sequence_valid = not pd.isna(sequence_data).any()
            engineered_valid = not pd.isna(current_engineered.values).any()
            target_valid = not pd.isna(target)

            if sequence_valid and engineered_valid and target_valid:
                # Combine engineered features with flattened sequence
                combined_row = np.concatenate([current_engineered.values, flattened_sequence])
                combined_features_list.append(combined_row)
                aligned_targets.append(target)
                feature_dates.append(current_date)

        if len(combined_features_list) == 0:
            return None, None, None

        # Create feature names
        engineered_feature_names = valid_engineered_cols
        sequence_feature_names = []
        for day in range(self.sequence_length):
            for feat in self.raw_feature_cols:
                sequence_feature_names.append(f"{feat}_lag_{day+1}")

        all_feature_names = engineered_feature_names + sequence_feature_names

        # Convert to DataFrame
        combined_features = pd.DataFrame(combined_features_list, columns=all_feature_names)
        combined_features['date'] = feature_dates
        targets = np.array(aligned_targets, dtype=np.float32)

        # Clean data
        feature_cols = [col for col in combined_features.columns if col != 'date']
        combined_features[feature_cols] = combined_features[feature_cols].replace([np.inf, -np.inf], np.nan)
        combined_features[feature_cols] = combined_features[feature_cols].fillna(
            combined_features[feature_cols].median()
        )
        combined_features[feature_cols] = combined_features[feature_cols].clip(-1e6, 1e6)

        logger.info(f"Trader {trader_id}: Prepared {len(combined_features)} samples with {len(all_feature_names)} features")

        return combined_features, targets, all_feature_names

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

    def _prepare_pnl_targets(self, trader_data: pd.DataFrame, combined_features: pd.DataFrame) -> np.ndarray:
        """Prepare PnL targets aligned with feature data"""
        # Align PnL data with feature dates
        pnl_targets = []

        for _, row in combined_features.iterrows():
            date = row['date']
            # Find corresponding PnL for this date
            pnl_row = trader_data[trader_data['date'] == date]
            if len(pnl_row) > 0:
                pnl_targets.append(pnl_row['daily_pnl'].iloc[0])
            else:
                pnl_targets.append(0.0)  # Default for missing data

        return np.array(pnl_targets, dtype=np.float32)

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

    def train_trader_model(self, df: pd.DataFrame, trader_id: str) -> Optional[Dict]:
        """Train, tune, validate and save both classification and VaR models for a specific trader"""

        # Calculate data boundaries
        boundaries = self.calculate_trader_boundaries(df, trader_id)
        if boundaries is None:
            return None

        # Prepare features
        combined_features, targets, feature_names = self.prepare_trader_features(df, trader_id)
        if combined_features is None:
            logger.warning(f"No valid features for trader {trader_id}")
            return None

        # Prepare PnL targets for VaR model
        trader_data = df[df['trader_id'] == trader_id].sort_values('date').copy()
        pnl_targets = self._prepare_pnl_targets(trader_data, combined_features)

        # Split data according to boundaries
        train_mask = combined_features['date'] <= boundaries['train_end']
        val_mask = (combined_features['date'] >= boundaries['validation_start']) & \
                   (combined_features['date'] <= boundaries['validation_end'])

        feature_cols = [col for col in combined_features.columns if col != 'date']
        X_train = combined_features[train_mask][feature_cols]
        y_train = targets[train_mask]
        X_val = combined_features[val_mask][feature_cols]
        y_val = targets[val_mask]

        # PnL data for VaR model
        pnl_train = pnl_targets[train_mask]
        pnl_val = pnl_targets[val_mask]

        logger.info(f"Trader {trader_id}: Train samples={len(X_train)}, Val samples={len(X_val)}")

        # Check if we have both classes in training data
        if len(np.unique(y_train)) < 2:
            logger.warning(f"Trader {trader_id}: Training data has only one class")
            return None

        # Train Classification Model
        logger.info(f"Training classification model for trader {trader_id}")
        best_params_cls = self.hyperparameter_tuning(X_train, y_train, X_val, y_val, trader_id)

        all_train_mask = combined_features['date'] <= boundaries['validation_end']
        X_all_train = combined_features[all_train_mask][feature_cols]
        y_all_train = targets[all_train_mask]

        final_cls_model = lgb.LGBMClassifier(**best_params_cls)
        final_cls_model.fit(X_all_train, y_all_train)

        cls_validation_metrics = self.validate_model(final_cls_model, X_val, y_val, trader_id)

        # Train VaR Model
        logger.info(f"Training VaR model for trader {trader_id}")
        best_params_var = self.hyperparameter_tuning_var(X_train, pnl_train, X_val, pnl_val, trader_id)

        pnl_all_train = pnl_targets[all_train_mask]
        final_var_model = lgb.LGBMRegressor(**best_params_var)
        final_var_model.fit(X_all_train, pnl_all_train)

        var_validation_metrics = self.validate_var_model(final_var_model, X_val, pnl_val, trader_id)

        # Save both models and metadata
        model_data = {
            'classification_model': final_cls_model,
            'var_model': final_var_model,
            'feature_names': feature_cols,
            'sequence_length': self.sequence_length,
            'boundaries': boundaries,
            'best_params_cls': best_params_cls,
            'best_params_var': best_params_var,
            'cls_validation_metrics': cls_validation_metrics,
            'var_validation_metrics': var_validation_metrics,
            'training_samples': len(X_all_train),
            'trader_id': trader_id,
            'trained_date': datetime.now().isoformat()
        }

        model_path = self.models_dir / f"{trader_id}_tuned_validated.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"Saved classification and VaR models for trader {trader_id} to {model_path}")

        return model_data

    def load_trader_model(self, trader_id: str) -> Optional[Dict]:
        """Load a saved trader model"""
        model_path = self.models_dir / f"{trader_id}_tuned_validated.pkl"
        if not model_path.exists():
            logger.warning(f"No saved model found for trader {trader_id}")
            return None

        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        return model_data

    def train_all_traders(self, df: pd.DataFrame, trader_ids: Optional[List] = None) -> Dict:
        """Train models for all specified traders"""
        if trader_ids is None:
            trader_ids = df['trader_id'].unique()

        results = {}
        successful_trains = 0

        for trader_id in trader_ids:
            logger.info(f"\n=== Training model for trader {trader_id} ===")

            try:
                model_data = self.train_trader_model(df, trader_id)
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


def run_trader_specific_training(sequence_length: int = 7, trader_ids: Optional[List] = None):
    """Main function to run trader-specific training with both classification and VaR models"""

    logging.basicConfig(level=logging.INFO)
    logger.info(f"Starting trader-specific training (Classification + VaR) with sequence_length={sequence_length}")

    # Load and prepare data
    with open('configs/main_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    df = create_trader_day_panel(config)
    df = df.rename(columns={'account_id': 'trader_id', 'trade_date': 'date'})

    # Feature engineering
    df_for_features = df.rename(columns={'trader_id': 'account_id', 'date': 'trade_date'})
    df_for_features = build_features(df_for_features, config)
    df = df_for_features.rename(columns={'account_id': 'trader_id', 'trade_date': 'date'})

    # Use active traders from config if not specified
    if trader_ids is None:
        trader_ids = config['active_traders']

    # Initialize trainer
    trainer = TraderSpecificTrainer(sequence_length=sequence_length)

    # Train all models (both classification and VaR)
    results = trainer.train_all_traders(df, trader_ids)

    logger.info(f"Training completed for {len(results)} traders (Classification + VaR models)")

    return results


def run_classification_only_training(sequence_length: int = 7, trader_ids: Optional[List] = None):
    """Run training with classification models only (backward compatibility)"""

    # For backward compatibility, create a version that only trains classification models
    # This can be done by modifying the trainer to skip VaR training
    logger.warning("Classification-only training mode - consider using full training with both models")

    return run_trader_specific_training(sequence_length, trader_ids)


if __name__ == "__main__":
    results = run_trader_specific_training(sequence_length=7)
    print(f"Successfully trained models for {len(results)} traders")
