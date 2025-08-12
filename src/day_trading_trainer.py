"""
Day Trading Model Trainer

Enhanced trainer that uses the new day trading features and targets for better position sizing
and risk management predictions.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from typing import Dict, Tuple, Optional, List
from pathlib import Path
import logging
import pickle
import json
from datetime import datetime
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
import optuna

logger = logging.getLogger(__name__)


class DayTradingTrainer:
    """
    Enhanced trainer for day trading models focusing on:
    1. Position sizing recommendations (reduce/increase/normal)
    2. Risk management signals (stop trading conditions)
    3. Volatility regime prediction
    4. Performance relative to recent patterns
    """

    def __init__(self, data_dir: str = 'data/processed/trader_splits', model_suffix: str = "_day_trading"):
        self.data_dir = Path(data_dir)
        self.models_dir = Path('models/trader_specific')
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.model_suffix = model_suffix

        # Define which targets to train models for
        self.targets_config = {
            'position_sizing': {
                'target': 'target_reduce_size',
                'model_type': 'classification',
                'importance': 'high',
                'description': 'Predict when to reduce position size'
            },
            'favorable_conditions': {
                'target': 'target_increase_size',
                'model_type': 'classification',
                'importance': 'high',
                'description': 'Predict favorable trading conditions'
            },
            'stop_trading': {
                'target': 'target_stop_trading',
                'model_type': 'classification',
                'importance': 'critical',
                'description': 'Predict when to stop trading for the day'
            },
            'volatility_regime': {
                'target': 'target_vol_regime',
                'model_type': 'multiclass',
                'importance': 'medium',
                'description': 'Predict next day volatility regime'
            },
            'underperformance': {
                'target': 'target_underperform',
                'model_type': 'classification',
                'importance': 'medium',
                'description': 'Predict underperformance vs recent average'
            }
        }

        # Base parameters for different model types
        self.classification_params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'verbose': -1,
            'random_state': 42,
            'force_row_wise': True
        }

        self.multiclass_params = {
            'objective': 'multiclass',
            'metric': 'multi_logloss',
            'num_class': 3,  # 0=low, 1=normal, 2=high volatility
            'boosting_type': 'gbdt',
            'verbose': -1,
            'random_state': 42,
            'force_row_wise': True
        }

        logger.info(f"DayTradingTrainer initialized: {len(self.targets_config)} target models")

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

    def prepare_features_and_targets(self, train_data: pd.DataFrame, trader_id: str) -> Dict:
        """Prepare features and all target variables for training"""
        trader_data = train_data.sort_values('date').copy()

        # Get all feature columns (exclude metadata and all targets)
        exclude_cols = [
            'trader_id', 'date', 'target_pnl', 'target_large_loss',
            'target_reduce_size', 'target_increase_size', 'target_stop_trading',
            'target_vol_regime', 'target_underperform', 'target_outperform'
        ]
        feature_cols = [col for col in trader_data.columns if col not in exclude_cols]

        # Remove features that are all null
        valid_feature_cols = []
        for col in feature_cols:
            if not trader_data[col].isnull().all():
                valid_feature_cols.append(col)

        logger.info(f"Trader {trader_id}: Using {len(valid_feature_cols)} features")

        # Prepare features (skip sequence length for now, use current row)
        features_list = []
        targets_dict = {target_name: [] for target_name in self.targets_config.keys()}
        feature_dates = []

        # Start from beginning but ensure we have valid target data
        for i in range(len(trader_data)):
            current_features = trader_data.iloc[i][valid_feature_cols]
            current_date = trader_data.iloc[i]['date']

            # Check if features are valid
            features_valid = not pd.isna(current_features.values).any()

            if features_valid:
                # Collect all targets for this row
                targets_valid = True
                current_targets = {}

                for target_name, target_config in self.targets_config.items():
                    target_col = target_config['target']
                    if target_col in trader_data.columns:
                        target_value = trader_data.iloc[i][target_col]
                        if not pd.isna(target_value):
                            current_targets[target_name] = target_value
                        else:
                            targets_valid = False
                            break
                    else:
                        targets_valid = False
                        break

                if targets_valid:
                    features_list.append(current_features.values)
                    feature_dates.append(current_date)

                    for target_name, target_value in current_targets.items():
                        targets_dict[target_name].append(target_value)

        if len(features_list) == 0:
            logger.warning(f"No valid samples found for trader {trader_id}")
            return None

        # Convert to arrays/DataFrame
        features_df = pd.DataFrame(features_list, columns=valid_feature_cols)
        features_df['date'] = feature_dates

        # Clean features
        features_df[valid_feature_cols] = features_df[valid_feature_cols].replace([np.inf, -np.inf], np.nan)
        features_df[valid_feature_cols] = features_df[valid_feature_cols].fillna(
            features_df[valid_feature_cols].median()
        )
        features_df[valid_feature_cols] = features_df[valid_feature_cols].clip(-1e6, 1e6)

        # Convert targets to numpy arrays
        targets_arrays = {}
        for target_name in self.targets_config.keys():
            targets_arrays[target_name] = np.array(targets_dict[target_name], dtype=np.float32)

        logger.info(f"Trader {trader_id}: Prepared {len(features_df)} samples")

        return {
            'features': features_df,
            'targets': targets_arrays,
            'feature_names': valid_feature_cols
        }

    def split_train_validation(self, data_dict: Dict, validation_split: float = 0.8) -> Tuple:
        """Split data into train/validation sets"""
        features_df = data_dict['features']
        targets = data_dict['targets']
        feature_names = data_dict['feature_names']

        split_idx = int(len(features_df) * validation_split)

        # Training set
        X_train = features_df.iloc[:split_idx][feature_names]
        y_train_dict = {name: targets[name][:split_idx] for name in targets.keys()}

        # Validation set
        X_val = features_df.iloc[split_idx:][feature_names]
        y_val_dict = {name: targets[name][split_idx:] for name in targets.keys()}

        return X_train, y_train_dict, X_val, y_val_dict

    def tune_hyperparameters(self, X_train: pd.DataFrame, y_train: np.ndarray,
                           X_val: pd.DataFrame, y_val: np.ndarray,
                           model_type: str, trader_id: str, target_name: str,
                           n_trials: int = 30) -> Dict:
        """Tune hyperparameters for a specific target"""

        def objective(trial):
            if model_type == 'classification':
                params = self.classification_params.copy()
            else:
                params = self.multiclass_params.copy()

            params.update({
                'num_leaves': trial.suggest_categorical('num_leaves', [15, 31, 63]),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0)
            })

            # Check for sufficient class diversity
            if model_type == 'classification':
                if len(np.unique(y_train)) < 2:
                    return 0.5
                model = lgb.LGBMClassifier(**params)
            else:
                if len(np.unique(y_train)) < 2:
                    return 1.0  # Return poor loss for multiclass
                model = lgb.LGBMClassifier(**params)

            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )

            if model_type == 'classification':
                if len(np.unique(y_val)) < 2:
                    return 0.5
                y_pred = model.predict_proba(X_val)[:, 1]
                return roc_auc_score(y_val, y_pred)
            else:
                y_pred = model.predict(X_val)
                return accuracy_score(y_val, y_pred)

        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_params = (self.classification_params if model_type == 'classification'
                      else self.multiclass_params).copy()
        best_params.update(study.best_params)

        logger.info(f"Trader {trader_id} {target_name}: Best score={study.best_value:.4f}")

        return best_params

    def train_single_model(self, X_train: pd.DataFrame, y_train: np.ndarray,
                          X_val: pd.DataFrame, y_val: np.ndarray,
                          target_config: Dict, trader_id: str, target_name: str) -> Dict:
        """Train a single model for a specific target"""

        model_type = target_config['model_type']

        # Tune hyperparameters
        best_params = self.tune_hyperparameters(
            X_train, y_train, X_val, y_val, model_type, trader_id, target_name
        )

        # Train final model
        if model_type == 'classification':
            model = lgb.LGBMClassifier(**best_params)
        else:
            model = lgb.LGBMClassifier(**best_params)

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )

        # Calculate validation metrics
        if model_type == 'classification' and len(np.unique(y_val)) >= 2:
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            y_pred = model.predict(X_val)

            metrics = {
                'auc': roc_auc_score(y_val, y_pred_proba),
                'ap': average_precision_score(y_val, y_pred_proba),
                'accuracy': accuracy_score(y_val, y_pred),
                'pos_rate': np.mean(y_val),
                'pred_pos_rate': np.mean(y_pred)
            }
        else:
            y_pred = model.predict(X_val)
            metrics = {
                'accuracy': accuracy_score(y_val, y_pred),
                'unique_classes': len(np.unique(y_val))
            }

        return {
            'model': model,
            'metrics': metrics,
            'best_params': best_params,
            'target_config': target_config
        }

    def train_trader_models(self, trader_id: str) -> Dict:
        """Train all day trading models for a single trader"""

        logger.info(f"\n=== Training Day Trading Models for Trader {trader_id} ===")

        # Load data
        train_data, test_data, metadata = self.load_trader_data(trader_id)
        if train_data is None:
            return None

        # Prepare features and targets
        data_dict = self.prepare_features_and_targets(train_data, trader_id)
        if data_dict is None:
            return None

        # Split train/validation
        X_train, y_train_dict, X_val, y_val_dict = self.split_train_validation(data_dict)

        # Train models for each target
        trained_models = {}

        for target_name, target_config in self.targets_config.items():
            if target_name in y_train_dict:
                logger.info(f"Training {target_name} model for trader {trader_id}")

                model_result = self.train_single_model(
                    X_train, y_train_dict[target_name],
                    X_val, y_val_dict[target_name],
                    target_config, trader_id, target_name
                )

                trained_models[target_name] = model_result

                # Log metrics
                metrics = model_result['metrics']
                logger.info(f"  {target_name}: {metrics}")

        # Package results
        result = {
            'models': trained_models,
            'feature_names': data_dict['feature_names'],
            'metadata': {
                'trader_id': trader_id,
                'training_samples': len(X_train),
                'validation_samples': len(X_val),
                'trained_date': datetime.now().isoformat(),
                'model_type': 'day_trading',
                'targets_trained': list(trained_models.keys())
            }
        }

        # Save models
        model_path = self.models_dir / f'{trader_id}{self.model_suffix}.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(result, f)

        logger.info(f"Saved day trading models for trader {trader_id} to {model_path}")

        return result

    def train_all_traders(self, trader_ids: Optional[List] = None) -> Dict:
        """Train day trading models for all specified traders"""

        if trader_ids is None:
            # Get all available traders
            trader_ids = [d.name for d in self.data_dir.iterdir()
                         if d.is_dir() and d.name.isdigit()]
            trader_ids.sort()

        results = {}
        successful_trains = 0

        logger.info(f"Training day trading models for {len(trader_ids)} traders: {trader_ids}")

        for trader_id in trader_ids:
            try:
                result = self.train_trader_models(trader_id)
                if result is not None:
                    results[trader_id] = {'success': True, 'models': result}
                    successful_trains += 1
                else:
                    results[trader_id] = {'success': False, 'error': 'Training failed'}

            except Exception as e:
                logger.error(f"Error training models for trader {trader_id}: {str(e)}")
                results[trader_id] = {'success': False, 'error': str(e)}

        logger.info(f"Successfully trained day trading models for {successful_trains}/{len(trader_ids)} traders")

        return results


def main():
    """Main function to train day trading models"""
    logging.basicConfig(level=logging.INFO)

    trainer = DayTradingTrainer()
    results = trainer.train_all_traders()

    # Print summary
    successful = [tid for tid, result in results.items() if result['success']]
    failed = [tid for tid, result in results.items() if not result['success']]

    print(f"\nDay Trading Model Training Summary:")
    print(f"Successful: {len(successful)} traders")
    print(f"Failed: {len(failed)} traders")

    if failed:
        print(f"Failed traders: {failed}")

    return results


if __name__ == "__main__":
    main()
