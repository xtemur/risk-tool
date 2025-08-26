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
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_squared_error
from sklearn.feature_selection import SelectFromModel, mutual_info_regression, mutual_info_classif
import optuna
from datetime import datetime

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class EnhancedTraderModelTrainer:
    """
    Enhanced trainer that handles the new fills-based features and improved risk models.
    """

    def __init__(self, config_path: str = 'configs/main_config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.models_dir = Path(self.config['paths']['model_dir'])
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Enhanced feature categories for better model understanding
        self.feature_categories = {
            'traditional': ['ewm_', 'rolling_vol_', 'rolling_sortino_', 'rolling_profit_factor',
                          'rolling_max_drawdown', 'avg_', 'win_rate', 'cumulative'],
            'fills_based': ['fills_', 'orders_', 'avg_fill_size', 'cost_per_share'],
            'execution_quality': ['price_impact', 'liquidity_', 'trading_aggressiveness',
                                'effective_spread', 'execution_efficiency'],
            'cross_features': ['_ratio', '_trend', '_consistency', '_stress', '_efficiency'],
            'behavioral': ['revenge_trading', 'large_loss', 'execution_stress']
        }

        logger.info(f"EnhancedTraderModelTrainer initialized with enhanced features")

    def load_enhanced_features(self) -> pd.DataFrame:
        """Load enhanced features from the processed features file."""
        features_path = self.config['paths']['processed_features']

        if not Path(features_path).exists():
            # Try alternative paths
            alt_paths = [
                'data/processed/enhanced_features.parquet',
                'data/processed/enhanced_features.pkl'
            ]

            for alt_path in alt_paths:
                if Path(alt_path).exists():
                    features_path = alt_path
                    break
            else:
                raise FileNotFoundError(f"Enhanced features not found at {features_path} or alternative paths")

        # Load data
        if features_path.endswith('.parquet'):
            df = pd.read_parquet(features_path)
        elif features_path.endswith('.pkl'):
            with open(features_path, 'rb') as f:
                df = pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {features_path}")

        logger.info(f"Loaded enhanced features: {df.shape}")
        logger.info(f"Date range: {df['trade_date'].min()} to {df['trade_date'].max()}")

        # Convert trade_date to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df['trade_date']):
            df['trade_date'] = pd.to_datetime(df['trade_date'])

        return df

    def categorize_features(self, feature_names: List[str]) -> Dict[str, List[str]]:
        """Categorize features by type for analysis."""
        categorized = {category: [] for category in self.feature_categories.keys()}
        uncategorized = []

        for feature in feature_names:
            assigned = False
            for category, patterns in self.feature_categories.items():
                if any(pattern in feature for pattern in patterns):
                    categorized[category].append(feature)
                    assigned = True
                    break

            if not assigned:
                uncategorized.append(feature)

        if uncategorized:
            categorized['uncategorized'] = uncategorized

        return categorized

    def prepare_trader_data(self, df: pd.DataFrame, trader_id: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare training and test data for a specific trader."""
        trader_data = df[df['account_id'] == trader_id].copy()

        if len(trader_data) == 0:
            logger.warning(f"No data found for trader {trader_id}")
            return None, None

        # Sort by date
        trader_data = trader_data.sort_values('trade_date').reset_index(drop=True)

        # Remove rows with NaN targets
        trader_data = trader_data.dropna(subset=['target_pnl', 'target_large_loss'])

        if len(trader_data) < 100:
            logger.warning(f"Insufficient data for trader {trader_id}: {len(trader_data)} rows")
            return None, None

        # Split into train/test (80/20)
        split_idx = int(len(trader_data) * 0.8)
        train_data = trader_data.iloc[:split_idx].copy()
        test_data = trader_data.iloc[split_idx:].copy()

        logger.info(f"Trader {trader_id}: Train={len(train_data)}, Test={len(test_data)}")

        return train_data, test_data

    def select_features(self, X_train: pd.DataFrame, y_train_cls: np.ndarray,
                       y_train_reg: np.ndarray, max_features: int = 30) -> List[str]:
        """Enhanced feature selection using multiple methods."""

        # Get feature names (excluding metadata)
        feature_cols = [col for col in X_train.columns
                       if col not in ['account_id', 'trade_date']]

        if len(feature_cols) <= max_features:
            return feature_cols

        X_features = X_train[feature_cols].copy()

        # Handle infinite values and NaNs for feature selection
        X_features = X_features.replace([np.inf, -np.inf], np.nan)
        X_features = X_features.fillna(0)

        # Additional safety checks
        for col in X_features.columns:
            # Convert boolean to numeric
            if X_features[col].dtype == 'bool':
                X_features[col] = X_features[col].astype(int)

            # Ensure numeric and handle zero variance
            if X_features[col].dtype in ['int64', 'float64']:
                if X_features[col].std() == 0 or pd.isna(X_features[col].std()):
                    X_features[col] = 0

        # Method 1: Mutual information for classification
        mi_cls_scores = mutual_info_classif(X_features, y_train_cls, random_state=42)
        mi_cls_features = pd.Series(mi_cls_scores, index=feature_cols).nlargest(max_features//2).index.tolist()

        # Method 2: Mutual information for regression
        mi_reg_scores = mutual_info_regression(X_features, y_train_reg, random_state=42)
        mi_reg_features = pd.Series(mi_reg_scores, index=feature_cols).nlargest(max_features//2).index.tolist()

        # Method 3: LightGBM feature importance
        lgb_model = lgb.LGBMClassifier(n_estimators=50, random_state=42, verbose=-1)
        lgb_model.fit(X_features, y_train_cls)
        lgb_importance = pd.Series(lgb_model.feature_importances_, index=feature_cols)
        lgb_features = lgb_importance.nlargest(max_features//2).index.tolist()

        # Combine and deduplicate
        selected_features = list(set(mi_cls_features + mi_reg_features + lgb_features))

        # Ensure we have a good mix of feature types
        categorized = self.categorize_features(selected_features)

        # If we have too few fills-based features, add some top ones
        if len(categorized['fills_based']) < 3:
            fills_features = [f for f in feature_cols if any(pattern in f for pattern in self.feature_categories['fills_based'])]
            if fills_features:
                fills_importance = lgb_importance[fills_features].nlargest(3)
                selected_features.extend(fills_importance.index.tolist())

        # Remove duplicates and limit to max_features
        selected_features = list(set(selected_features))[:max_features]

        logger.info(f"Selected {len(selected_features)} features from {len(feature_cols)} total")

        # Log feature categories
        final_categorized = self.categorize_features(selected_features)
        for category, features in final_categorized.items():
            if features:
                logger.info(f"  {category}: {len(features)} features")

        return selected_features

    def tune_models(self, X_train: pd.DataFrame, y_train_cls: np.ndarray, y_train_reg: np.ndarray,
                   trader_id: int, n_trials: int = 100) -> Tuple[Dict, Dict]:
        """Tune both classification and regression models."""

        # Split into train/validation for tuning
        split_idx = int(len(X_train) * 0.8)
        X_tune_train = X_train.iloc[:split_idx]
        X_tune_val = X_train.iloc[split_idx:]
        y_cls_tune_train = y_train_cls[:split_idx]
        y_cls_tune_val = y_train_cls[split_idx:]
        y_reg_tune_train = y_train_reg[:split_idx]
        y_reg_tune_val = y_train_reg[split_idx:]

        # Tune classification model
        def cls_objective(trial):
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
                'random_state': 42,
                'verbose': -1
            }

            if len(np.unique(y_cls_tune_train)) < 2:
                return 0.5

            model = lgb.LGBMClassifier(**params)
            model.fit(X_tune_train, y_cls_tune_train,
                     eval_set=[(X_tune_val, y_cls_tune_val)],
                     callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])

            if len(np.unique(y_cls_tune_val)) < 2:
                return 0.5

            y_pred = model.predict_proba(X_tune_val)[:, 1]
            return roc_auc_score(y_cls_tune_val, y_pred)

        # Tune regression model
        def reg_objective(trial):
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
                'random_state': 42,
                'verbose': -1
            }

            model = lgb.LGBMRegressor(**params)
            model.fit(X_tune_train, y_reg_tune_train,
                     eval_set=[(X_tune_val, y_reg_tune_val)],
                     callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])

            y_pred = model.predict(X_tune_val)
            return np.sqrt(mean_squared_error(y_reg_tune_val, y_pred))  # RMSE

        # Run optimization
        logger.info(f"Tuning classification model for trader {trader_id}...")
        cls_study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        cls_study.optimize(cls_objective, n_trials=n_trials//2, show_progress_bar=False)

        logger.info(f"Tuning regression model for trader {trader_id}...")
        reg_study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
        reg_study.optimize(reg_objective, n_trials=n_trials//2, show_progress_bar=False)

        return cls_study.best_params, reg_study.best_params

    def train_trader_models(self, trader_id: int) -> Optional[Dict]:
        """Train enhanced models for a specific trader."""
        logger.info(f"Training enhanced models for trader {trader_id}...")

        try:
            # Load enhanced features
            df = self.load_enhanced_features()

            # Prepare trader data
            train_data, test_data = self.prepare_trader_data(df, trader_id)

            if train_data is None or test_data is None:
                return None

            # Prepare features and targets (exclude all target columns)
            target_cols = [col for col in train_data.columns if col.startswith('target_')]
            feature_cols = [col for col in train_data.columns
                           if col not in ['account_id', 'trade_date'] + target_cols]

            X_train = train_data[['account_id', 'trade_date'] + feature_cols].copy()
            X_test = test_data[['account_id', 'trade_date'] + feature_cols].copy()

            # ENHANCED: Use position sizing as primary targets
            using_position_sizing = 'target_position_size' in train_data.columns

            if using_position_sizing:
                # Primary target: Position sizing (0.0 to 1.5)
                y_train_position = train_data['target_position_size'].values
                y_test_position = test_data['target_position_size'].values

                # Secondary target: High risk classification (position size < 0.7)
                y_train_cls = (train_data['target_position_size'] < 0.7).astype(int).values
                y_test_cls = (test_data['target_position_size'] < 0.7).astype(int).values

                # Regression target: Position size itself
                y_train_reg = y_train_position
                y_test_reg = y_test_position

                logger.info(f"Using position sizing targets:")
                logger.info(f"  Position size range (train): {y_train_position.min():.3f} to {y_train_position.max():.3f}")
                logger.info(f"  High risk days (train): {y_train_cls.sum()} / {len(y_train_cls)} ({y_train_cls.mean()*100:.1f}%)")

            else:
                # Fallback to legacy targets
                logger.warning("No position sizing targets found, using legacy targets")
                y_train_cls = train_data['target_large_loss'].values
                y_train_reg = train_data['target_pnl'].values
                y_test_cls = test_data['target_large_loss'].values
                y_test_reg = test_data['target_pnl'].values

            # Handle missing values and infinite values
            for col in feature_cols:
                # Skip non-numeric columns
                if X_train[col].dtype not in ['int64', 'float64', 'bool']:
                    continue

                # Convert boolean to numeric
                if X_train[col].dtype == 'bool':
                    X_train[col] = X_train[col].astype(int)
                    X_test[col] = X_test[col].astype(int)

                # Replace inf and -inf with NaN first
                X_train[col] = X_train[col].replace([np.inf, -np.inf], np.nan)
                X_test[col] = X_test[col].replace([np.inf, -np.inf], np.nan)

                # Fill NaN values with median
                train_median = X_train[col].median()
                if pd.isna(train_median):
                    train_median = 0

                X_train[col] = X_train[col].fillna(train_median)
                X_test[col] = X_test[col].fillna(train_median)

                # Additional safety: clip extreme values (only for numeric columns with variance)
                if X_train[col].dtype in ['int64', 'float64'] and X_train[col].std() > 0:
                    try:
                        lower_bound = X_train[col].quantile(0.001)
                        upper_bound = X_train[col].quantile(0.999)
                        if not pd.isna(lower_bound) and not pd.isna(upper_bound):
                            X_train[col] = X_train[col].clip(lower_bound, upper_bound)
                            X_test[col] = X_test[col].clip(lower_bound, upper_bound)
                    except:
                        # If quantile fails, just ensure values are reasonable
                        X_train[col] = X_train[col].clip(-1e6, 1e6)
                        X_test[col] = X_test[col].clip(-1e6, 1e6)

            # Feature selection
            max_features = self.config.get('model_quality', {}).get('max_features', 30)
            selected_features = self.select_features(X_train, y_train_cls, y_train_reg, max_features)

            X_train_selected = X_train[selected_features]
            X_test_selected = X_test[selected_features]

            # Hyperparameter tuning
            cls_best_params, reg_best_params = self.tune_models(
                X_train_selected, y_train_cls, y_train_reg, trader_id
            )

            # Train final models
            logger.info(f"Training final models for trader {trader_id}...")

            # Classification model (large loss prediction)
            cls_params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'random_state': 42,
                'verbose': -1
            }
            cls_params.update(cls_best_params)

            cls_model = lgb.LGBMClassifier(**cls_params)
            cls_model.fit(X_train_selected, y_train_cls)

            # Regression model (VaR prediction)
            reg_params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'random_state': 42,
                'verbose': -1
            }
            reg_params.update(reg_best_params)

            reg_model = lgb.LGBMRegressor(**reg_params)
            reg_model.fit(X_train_selected, y_train_reg)

            # Evaluate models
            cls_train_pred = cls_model.predict_proba(X_train_selected)[:, 1]
            cls_test_pred = cls_model.predict_proba(X_test_selected)[:, 1]

            reg_train_pred = reg_model.predict(X_train_selected)
            reg_test_pred = reg_model.predict(X_test_selected)

            # Calculate metrics
            if len(np.unique(y_train_cls)) > 1:
                cls_train_auc = roc_auc_score(y_train_cls, cls_train_pred)
            else:
                cls_train_auc = 0.5

            if len(np.unique(y_test_cls)) > 1:
                cls_test_auc = roc_auc_score(y_test_cls, cls_test_pred)
            else:
                cls_test_auc = 0.5

            reg_train_rmse = np.sqrt(mean_squared_error(y_train_reg, reg_train_pred))
            reg_test_rmse = np.sqrt(mean_squared_error(y_test_reg, reg_test_pred))
            reg_train_mae = mean_absolute_error(y_train_reg, reg_train_pred)
            reg_test_mae = mean_absolute_error(y_test_reg, reg_test_pred)

            # Feature importance analysis
            cls_importance = pd.Series(cls_model.feature_importances_, index=selected_features)
            reg_importance = pd.Series(reg_model.feature_importances_, index=selected_features)

            # Categorize feature importance
            cls_importance_by_category = {}
            reg_importance_by_category = {}

            categorized_features = self.categorize_features(selected_features)
            for category, features in categorized_features.items():
                if features:
                    cls_importance_by_category[category] = cls_importance[features].sum()
                    reg_importance_by_category[category] = reg_importance[features].sum()

            # Save models and metadata
            trader_model_dir = self.models_dir / str(trader_id)
            trader_model_dir.mkdir(exist_ok=True)

            # Save models with appropriate names
            if using_position_sizing:
                cls_model_name = 'enhanced_risk_model.pkl'
                reg_model_name = 'enhanced_position_model.pkl'
            else:
                cls_model_name = 'enhanced_loss_model.pkl'
                reg_model_name = 'enhanced_var_model.pkl'

            with open(trader_model_dir / cls_model_name, 'wb') as f:
                pickle.dump(cls_model, f)

            with open(trader_model_dir / reg_model_name, 'wb') as f:
                pickle.dump(reg_model, f)

            # Prepare metadata with JSON-safe data types
            def make_json_safe(obj):
                """Convert numpy types and handle NaN/inf values for JSON serialization."""
                import numpy as np

                if isinstance(obj, dict):
                    return {k: make_json_safe(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [make_json_safe(v) for v in obj]
                elif isinstance(obj, (np.integer, np.int64, np.int32)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64, np.float32)):
                    if np.isnan(obj) or np.isinf(obj):
                        return 0.0
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif pd.isna(obj) or (isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj))):
                    return 0.0
                else:
                    return obj

            # Save model metadata
            model_metadata = {
                'trader_id': int(trader_id),
                'training_timestamp': datetime.now().isoformat(),
                'enhanced_features': True,
                'feature_categories': make_json_safe(categorized_features),
                'selected_features': list(selected_features),
                'feature_count': len(selected_features),
                'training_data_shape': list(X_train_selected.shape),
                'test_data_shape': list(X_test_selected.shape),
                'models': {
                    'loss_model': {
                        'type': 'LGBMClassifier',
                        'params': make_json_safe(cls_params),
                        'train_auc': float(cls_train_auc) if not pd.isna(cls_train_auc) else 0.0,
                        'test_auc': float(cls_test_auc) if not pd.isna(cls_test_auc) else 0.0,
                        'feature_importance': make_json_safe(cls_importance.to_dict()),
                        'importance_by_category': make_json_safe(cls_importance_by_category)
                    },
                    'var_model': {
                        'type': 'LGBMRegressor',
                        'params': make_json_safe(reg_params),
                        'train_rmse': float(reg_train_rmse) if not pd.isna(reg_train_rmse) else 0.0,
                        'test_rmse': float(reg_test_rmse) if not pd.isna(reg_test_rmse) else 0.0,
                        'train_mae': float(reg_train_mae) if not pd.isna(reg_train_mae) else 0.0,
                        'test_mae': float(reg_test_mae) if not pd.isna(reg_test_mae) else 0.0,
                        'feature_importance': make_json_safe(reg_importance.to_dict()),
                        'importance_by_category': make_json_safe(reg_importance_by_category)
                    }
                },
                'data_info': {
                    'train_period': [train_data['trade_date'].min().isoformat(),
                                   train_data['trade_date'].max().isoformat()],
                    'test_period': [test_data['trade_date'].min().isoformat(),
                                  test_data['trade_date'].max().isoformat()],
                    'large_loss_rate_train': float(y_train_cls.mean()) if not pd.isna(y_train_cls.mean()) else 0.0,
                    'large_loss_rate_test': float(y_test_cls.mean()) if not pd.isna(y_test_cls.mean()) else 0.0
                }
            }

            # Save metadata with proper error handling
            metadata_path = trader_model_dir / 'enhanced_model_metadata.json'
            try:
                with open(metadata_path, 'w') as f:
                    json.dump(model_metadata, f, indent=2)

                # Verify the JSON was saved correctly
                with open(metadata_path, 'r') as f:
                    json.load(f)  # This will raise an exception if JSON is malformed

            except Exception as e:
                logger.error(f"Error saving metadata for trader {trader_id}: {e}")
                # Save a minimal metadata file as fallback
                minimal_metadata = {
                    'trader_id': int(trader_id),
                    'training_timestamp': datetime.now().isoformat(),
                    'enhanced_features': True,
                    'selected_features': list(selected_features),
                    'feature_count': len(selected_features),
                    'error': str(e)
                }
                with open(metadata_path, 'w') as f:
                    json.dump(minimal_metadata, f, indent=2)

            logger.info(f"Enhanced models saved for trader {trader_id}")
            logger.info(f"  Classification AUC: Train={cls_train_auc:.4f}, Test={cls_test_auc:.4f}")
            logger.info(f"  Regression RMSE: Train={reg_train_rmse:.4f}, Test={reg_test_rmse:.4f}")
            logger.info(f"  Top feature categories (Classification): {sorted(cls_importance_by_category.items(), key=lambda x: x[1], reverse=True)[:3]}")
            logger.info(f"  Top feature categories (Regression): {sorted(reg_importance_by_category.items(), key=lambda x: x[1], reverse=True)[:3]}")

            return model_metadata

        except Exception as e:
            logger.error(f"Error training trader {trader_id}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def train_all_traders(self) -> Dict[int, Dict]:
        """Train enhanced models for all active traders."""
        logger.info("Training enhanced models for all active traders...")

        results = {}
        active_traders = self.config['active_traders']

        for trader_id in active_traders:
            logger.info(f"Processing trader {trader_id}...")
            result = self.train_trader_models(trader_id)
            results[trader_id] = result

        # Summary statistics
        successful_traders = [tid for tid, result in results.items() if result is not None]
        logger.info(f"Successfully trained enhanced models for {len(successful_traders)}/{len(active_traders)} traders")

        if successful_traders:
            # Calculate aggregate metrics
            cls_aucs = [results[tid]['models']['loss_model']['test_auc'] for tid in successful_traders]
            reg_rmses = [results[tid]['models']['var_model']['test_rmse'] for tid in successful_traders]

            logger.info(f"Enhanced Model Performance Summary:")
            logger.info(f"  Classification AUC: Mean={np.mean(cls_aucs):.4f}, Std={np.std(cls_aucs):.4f}")
            logger.info(f"  Regression RMSE: Mean={np.mean(reg_rmses):.4f}, Std={np.std(reg_rmses):.4f}")

            # Feature importance analysis across all traders
            all_importance_by_category = defaultdict(list)
            for tid in successful_traders:
                for category, importance in results[tid]['models']['loss_model']['importance_by_category'].items():
                    all_importance_by_category[category].append(importance)

            avg_importance_by_category = {
                category: np.mean(importances)
                for category, importances in all_importance_by_category.items()
            }

            logger.info(f"  Average feature importance by category:")
            for category, avg_importance in sorted(avg_importance_by_category.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"    {category}: {avg_importance:.4f}")

        return results


# Compatibility class for existing code
class TraderSpecificModelTrainer(EnhancedTraderModelTrainer):
    """Alias for backwards compatibility."""
    pass
