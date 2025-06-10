"""
Model Pipeline
Orchestrates model training, validation, and selection
"""

import pandas as pd
import numpy as np
import optuna
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from pathlib import Path
from datetime import datetime
import json
import warnings

from src.models.base_model import BaseModel
from src.models.risk_model import RiskModel
from src.pipeline.feature_pipeline import FeaturePipeline
from src.pipeline.data_validator import DataValidator
from src.utils.time_series_cv import TimeSeriesSplit, WalkForwardAnalysis
from src.core.constants import ModelConfig as MC, TradingConstants as TC

warnings.filterwarnings('ignore', category=optuna.exceptions.ExperimentalWarning)
logger = logging.getLogger(__name__)


class ModelPipeline:
    """
    End-to-end model training pipeline
    Handles data preparation, model training, hyperparameter tuning, and evaluation
    """

    def __init__(self,
                 model_dir: str = "data/models",
                 experiment_name: str = "risk_model_experiment"):
        """
        Initialize model pipeline

        Args:
            model_dir: Directory to save models
            experiment_name: Name for experiment tracking
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True, parents=True)
        self.experiment_name = experiment_name

        # Components
        self.feature_pipeline = FeaturePipeline()
        self.data_validator = DataValidator()

        # Tracking
        self.experiment_results_ = {}
        self.best_model_ = None
        self.best_params_ = None
        self.selected_features_ = None

    def run_pipeline(self,
                    totals_df: pd.DataFrame,
                    fills_df: Optional[pd.DataFrame] = None,
                    target_col: str = 'net_pnl',
                    target_type: str = 'regression',
                    optimize_hyperparams: bool = True,
                    n_trials: int = 50,
                    cv_splits: int = 5) -> BaseModel:
        """
        Run complete model pipeline

        Args:
            totals_df: Daily totals data
            fills_df: Optional fills data
            target_col: Target column name
            target_type: 'regression' or 'classification'
            optimize_hyperparams: Whether to optimize hyperparameters
            n_trials: Number of Optuna trials
            cv_splits: Number of CV splits

        Returns:
            Trained best model
        """
        logger.info(f"Starting model pipeline: {self.experiment_name}")
        pipeline_start = datetime.now()

        # 1. Validate data
        logger.info("Step 1: Validating data")
        validation_result = self.data_validator.validate_combined(totals_df, fills_df)
        if not validation_result.is_valid:
            raise ValueError(f"Data validation failed: {validation_result.errors}")

        # 2. Generate features
        logger.info("Step 2: Generating features")
        features_df = self.feature_pipeline.generate_features(totals_df, fills_df)

        if features_df.empty:
            raise ValueError("No features generated")

        # 3. Create target variable
        logger.info("Step 3: Creating target variable")
        X, y = self._prepare_modeling_data(features_df, totals_df, target_col, target_type)

        # 4. Feature selection
        logger.info("Step 4: Selecting features")
        selected_features = self.feature_pipeline.select_features(
            X, y, method='importance', top_k=100
        )
        self.selected_features_ = selected_features

        # Transform to selected features
        X_selected = self.feature_pipeline.transform_features(X, selected_features)

        # 5. Hyperparameter optimization
        if optimize_hyperparams:
            logger.info("Step 5: Optimizing hyperparameters")
            best_params = self._optimize_hyperparameters(
                X_selected, y, target_type, n_trials, cv_splits
            )
            self.best_params_ = best_params
        else:
            logger.info("Step 5: Using default hyperparameters")
            self.best_params_ = MC.LIGHTGBM_PARAMS.copy()

        # 6. Train final model with walk-forward validation
        logger.info("Step 6: Training final model with walk-forward validation")
        final_model, wf_results = self._train_walk_forward(
            X_selected, y, target_type, self.best_params_
        )
        self.best_model_ = final_model

        # 7. Evaluate and save results
        logger.info("Step 7: Evaluating and saving results")
        self._evaluate_and_save(final_model, X_selected, y, wf_results)

        # Log completion
        pipeline_time = (datetime.now() - pipeline_start).total_seconds()
        logger.info(f"Pipeline completed in {pipeline_time:.2f} seconds")

        return final_model

    def _prepare_modeling_data(self,
                              features_df: pd.DataFrame,
                              totals_df: pd.DataFrame,
                              target_col: str,
                              target_type: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for modeling"""

        # Merge features with target
        if 'date' in features_df.columns and 'date' in totals_df.columns:
            modeling_df = pd.merge(
                features_df,
                totals_df[['date', 'account_id', target_col]],
                on=['date', 'account_id'],
                how='inner'
            )
        else:
            # Assume aligned
            modeling_df = features_df.copy()
            modeling_df[target_col] = totals_df[target_col]

        # Create forward-looking target
        modeling_df = modeling_df.sort_values(['account_id', 'date'])

        # Shift target by -1 (predict next day)
        modeling_df['target'] = modeling_df.groupby('account_id')[target_col].shift(-1)

        # Drop last row per account (no target)
        modeling_df = modeling_df.dropna(subset=['target'])

        # Handle target type
        if target_type == 'classification':
            # Binary classification: profit or loss
            modeling_df['target'] = (modeling_df['target'] > 0).astype(int)

        # Separate features and target
        feature_cols = [col for col in modeling_df.columns
                       if col not in ['date', 'account_id', 'target', target_col]]

        X = modeling_df[feature_cols]
        y = modeling_df['target']

        # Add date index for time series CV
        if 'date' in modeling_df.columns:
            X.index = pd.to_datetime(modeling_df['date'])
            y.index = pd.to_datetime(modeling_df['date'])

        logger.info(f"Prepared {len(X)} samples with {len(feature_cols)} features")

        return X, y

    def _optimize_hyperparameters(self,
                                 X: pd.DataFrame,
                                 y: pd.Series,
                                 target_type: str,
                                 n_trials: int,
                                 cv_splits: int) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna"""

        # Create CV splitter
        cv_splitter = TimeSeriesSplit(n_splits=cv_splits, mode='expanding')

        def objective(trial):
            # Suggest hyperparameters
            params = {
                'num_leaves': trial.suggest_int('num_leaves',
                                               MC.OPTUNA_SEARCH_SPACE['num_leaves'][0],
                                               MC.OPTUNA_SEARCH_SPACE['num_leaves'][1]),
                'learning_rate': trial.suggest_float('learning_rate',
                                                    MC.OPTUNA_SEARCH_SPACE['learning_rate'][0],
                                                    MC.OPTUNA_SEARCH_SPACE['learning_rate'][1],
                                                    log=True),
                'feature_fraction': trial.suggest_float('feature_fraction',
                                                       MC.OPTUNA_SEARCH_SPACE['feature_fraction'][0],
                                                       MC.OPTUNA_SEARCH_SPACE['feature_fraction'][1]),
                'bagging_fraction': trial.suggest_float('bagging_fraction',
                                                       MC.OPTUNA_SEARCH_SPACE['bagging_fraction'][0],
                                                       MC.OPTUNA_SEARCH_SPACE['bagging_fraction'][1]),
                'lambda_l1': trial.suggest_float('lambda_l1',
                                                MC.OPTUNA_SEARCH_SPACE['lambda_l1'][0],
                                                MC.OPTUNA_SEARCH_SPACE['lambda_l1'][1]),
                'lambda_l2': trial.suggest_float('lambda_l2',
                                                MC.OPTUNA_SEARCH_SPACE['lambda_l2'][0],
                                                MC.OPTUNA_SEARCH_SPACE['lambda_l2'][1]),
                'min_child_samples': trial.suggest_int('min_child_samples',
                                                      MC.OPTUNA_SEARCH_SPACE['min_child_samples'][0],
                                                      MC.OPTUNA_SEARCH_SPACE['min_child_samples'][1]),
                'max_depth': trial.suggest_int('max_depth',
                                              MC.OPTUNA_SEARCH_SPACE['max_depth'][0],
                                              MC.OPTUNA_SEARCH_SPACE['max_depth'][1]),
            }

            # Add fixed parameters
            params.update({
                'objective': 'regression' if target_type == 'regression' else 'binary',
                'metric': 'rmse' if target_type == 'regression' else 'binary_logloss',
                'boosting_type': 'gbdt',
                'verbose': -1,
                'n_estimators': 100,
                'random_state': 42
            })

            # Cross-validate
            scores = []

            for fold in cv_splitter.split(X):
                X_train = X.iloc[fold.train_idx]
                y_train = y.iloc[fold.train_idx]
                X_val = X.iloc[fold.val_idx]
                y_val = y.iloc[fold.val_idx]

                # Train model
                model = RiskModel(
                    model_type=target_type,
                    params=params
                )

                try:
                    model.fit(X_train, y_train,
                             validation_data=(X_val, y_val),
                             early_stopping_rounds=30,
                             verbose=0)

                    # Score
                    if target_type == 'regression':
                        pred = model.predict(X_val)
                        score = np.sqrt(np.mean((y_val - pred) ** 2))
                    else:
                        # Use log loss for classification
                        pred_proba = model.predict_proba(X_val)[:, 1]
                        from sklearn.metrics import log_loss
                        score = log_loss(y_val, pred_proba)

                    scores.append(score)

                except Exception as e:
                    logger.debug(f"Trial failed: {e}")
                    return float('inf')

            return np.mean(scores)

        # Create study
        study = optuna.create_study(
            direction='minimize',
            study_name=self.experiment_name,
            sampler=optuna.samplers.TPESampler(seed=42)
        )

        # Optimize
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        # Get best parameters
        best_params = study.best_params
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best CV score: {study.best_value:.4f}")

        # Add fixed parameters
        best_params.update({
            'objective': 'regression' if target_type == 'regression' else 'binary',
            'boosting_type': 'gbdt',
            'verbose': -1,
            'n_estimators': 200,  # More trees for final model
            'random_state': 42
        })

        return best_params

    def _train_walk_forward(self,
                           X: pd.DataFrame,
                           y: pd.Series,
                           target_type: str,
                           params: Dict[str, Any]) -> Tuple[BaseModel, Dict]:
        """Train model using walk-forward analysis"""

        # Create walk-forward splitter
        wf = WalkForwardAnalysis()
        splits = wf.generate_splits(X)

        if not splits:
            # Fall back to simple train/test split
            logger.warning("Not enough data for walk-forward, using simple split")
            split_idx = int(len(X) * 0.8)
            X_train = X.iloc[:split_idx]
            y_train = y.iloc[:split_idx]
            X_test = X.iloc[split_idx:]
            y_test = y.iloc[split_idx:]

            model = RiskModel(model_type=target_type, params=params)
            model.fit(X_train, y_train, validation_data=(X_test, y_test))

            return model, {}

        # Walk-forward training
        wf_results = {}
        best_model = None
        best_score = float('inf') if target_type == 'regression' else 0

        for split in splits:
            logger.info(f"Training walk-forward split {split.fold_id}")

            # Get data
            X_train = X.iloc[split.train_idx]
            y_train = y.iloc[split.train_idx]
            X_val = X.iloc[split.val_idx]
            y_val = y.iloc[split.val_idx]
            X_test = X.iloc[split.test_idx] if split.test_idx is not None else X_val
            y_test = y.iloc[split.test_idx] if split.test_idx is not None else y_val

            # Train model
            model = RiskModel(
                model_name=f"risk_model_wf{split.fold_id}",
                model_type=target_type,
                params=params
            )

            model.fit(X_train, y_train, validation_data=(X_val, y_val))

            # Evaluate
            test_pred = model.predict(X_test)

            if target_type == 'regression':
                test_score = np.sqrt(np.mean((y_test - test_pred) ** 2))
                if test_score < best_score:
                    best_score = test_score
                    best_model = model
            else:
                test_score = np.mean(y_test == test_pred)
                if test_score > best_score:
                    best_score = test_score
                    best_model = model

            # Store results
            wf_results[split.fold_id] = {
                'train_start': split.train_start,
                'train_end': split.train_end,
                'test_start': split.test_start,
                'test_end': split.test_end,
                'test_score': test_score,
                'n_train': len(split.train_idx),
                'n_test': len(split.test_idx) if split.test_idx is not None else 0
            }

        # Analyze stability
        stability_df = wf.analyze_stability(
            {k: {'score': v['test_score']} for k, v in wf_results.items()}
        )

        logger.info(f"Walk-forward stability:\n{stability_df}")

        return best_model, wf_results

    def _evaluate_and_save(self,
                          model: BaseModel,
                          X: pd.DataFrame,
                          y: pd.Series,
                          wf_results: Dict):
        """Evaluate model and save results"""

        # Final evaluation
        predictions = model.predict(X)

        if model.model_type == 'regression':
            rmse = np.sqrt(np.mean((y - predictions) ** 2))
            mae = np.mean(np.abs(y - predictions))
            r2 = model.score(X, y)

            eval_metrics = {
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'mean_target': float(y.mean()),
                'std_target': float(y.std())
            }

            logger.info(f"Final evaluation - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

        else:
            accuracy = np.mean(y == predictions)

            from sklearn.metrics import classification_report
            report = classification_report(y, predictions, output_dict=True)

            eval_metrics = {
                'accuracy': accuracy,
                'classification_report': report
            }

            logger.info(f"Final evaluation - Accuracy: {accuracy:.4f}")

        # Save experiment results
        self.experiment_results_ = {
            'experiment_name': self.experiment_name,
            'timestamp': datetime.now().isoformat(),
            'model_type': model.model_type,
            'best_params': self.best_params_,
            'selected_features': self.selected_features_,
            'n_features': len(self.selected_features_),
            'n_samples': len(X),
            'evaluation_metrics': eval_metrics,
            'walk_forward_results': wf_results,
            'feature_importance': model.get_feature_importance(top_k=20).to_dict()
        }

        # Save model
        model_path = self.model_dir / f"{self.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        model.save(model_path)

        # Save experiment results
        results_path = self.model_dir / f"{self.experiment_name}_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.experiment_results_, f, indent=2, default=str)

        logger.info(f"Model saved to {model_path}")
        logger.info(f"Results saved to {results_path}")

    def load_best_model(self) -> Optional[BaseModel]:
        """Load the best model from disk"""

        # Find most recent model
        model_files = list(self.model_dir.glob(f"{self.experiment_name}_*.pkl"))

        if not model_files:
            logger.warning("No saved models found")
            return None

        # Sort by modification time
        latest_model = max(model_files, key=lambda p: p.stat().st_mtime)

        logger.info(f"Loading model from {latest_model}")
        return RiskModel.load(latest_model)

    def generate_prediction_report(self,
                                  model: BaseModel,
                                  X: pd.DataFrame,
                                  trader_info: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Generate prediction report for new data

        Args:
            model: Trained model
            X: Features to predict on
            trader_info: Optional trader information

        Returns:
            Prediction report DataFrame
        """
        # Get risk predictions
        risk_report = model.get_risk_report(X)

        # Add trader info if available
        if trader_info is not None and 'account_id' in risk_report.columns:
            risk_report = pd.merge(
                risk_report,
                trader_info[['account_id', 'trader_name']],
                on='account_id',
                how='left'
            )

        # Add recommendation
        risk_report['recommendation'] = risk_report.apply(
            lambda row: self._generate_recommendation(row), axis=1
        )

        return risk_report

    def _generate_recommendation(self, row: pd.Series) -> str:
        """Generate trading recommendation based on risk assessment"""

        risk_score = row.get('risk_score', 0.5)
        confidence = row.get('confidence', 'Medium')

        if risk_score > TC.HIGH_RISK_SCORE:
            if confidence in ['High', 'Very High']:
                return "REDUCE POSITION - High risk detected"
            else:
                return "MONITOR CLOSELY - Elevated risk"
        elif risk_score < TC.LOW_RISK_SCORE:
            if confidence in ['High', 'Very High']:
                return "OPPORTUNITY - Low risk environment"
            else:
                return "PROCEED NORMALLY - Low risk"
        else:
            return "MAINTAIN POSITION - Normal risk levels"
