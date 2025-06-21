#!/usr/bin/env python3
"""
Enhanced Model Training with Multiple Algorithms
Supports XGBoost, Random Forest, LightGBM, and Neural Networks
"""

import pandas as pd
import numpy as np
import pickle
import json
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Optional imports for enhanced models
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available - skipping LightGBM models")

try:
    from sklearn.neural_network import MLPClassifier
    SKLEARN_NN_AVAILABLE = True
except ImportError:
    SKLEARN_NN_AVAILABLE = False
    print("Scikit-learn neural networks not available")


class EnhancedModelTraining:
    """Enhanced model training with multiple algorithms and improved regularization."""

    def __init__(self, features_path='outputs/signals/target_prepared.pkl'):
        self.feature_df = pd.read_pickle(features_path)
        self.enhanced_models = {}
        self.algorithm_performance = {}

        # Load target strategy info
        try:
            with open('outputs/signals/target_strategy.json', 'r') as f:
                self.target_strategy_info = json.load(f)
        except:
            self.target_strategy_info = {
                'best_strategy': 'Option B: Classification',
                'target_column': 'target_class',
                'model_performance': 0.785,
                'predictability_score': 0.2
            }

    def get_algorithm_configs(self):
        """Get configurations for different algorithms."""
        configs = {
            'xgboost': {
                'name': 'Regularized XGBoost',
                'params': {
                    'objective': 'multi:softprob',
                    'random_state': 42,
                    'n_estimators': 200,
                    'max_depth': 4,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'reg_alpha': 0.1,  # L1 regularization
                    'reg_lambda': 0.1,  # L2 regularization
                    'eval_metric': 'mlogloss'
                }
            },
            'random_forest': {
                'name': 'Random Forest',
                'params': {
                    'n_estimators': 200,
                    'max_depth': 8,
                    'min_samples_split': 10,
                    'min_samples_leaf': 5,
                    'random_state': 42,
                    'class_weight': 'balanced'
                }
            }
        }

        if LIGHTGBM_AVAILABLE:
            configs['lightgbm'] = {
                'name': 'LightGBM',
                'params': {
                    'objective': 'multiclass',
                    'num_class': 3,
                    'random_state': 42,
                    'n_estimators': 200,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'reg_alpha': 0.1,
                    'reg_lambda': 0.1,
                    'verbose': -1
                }
            }

        if SKLEARN_NN_AVAILABLE:
            configs['neural_network'] = {
                'name': 'Neural Networks',
                'params': {
                    'hidden_layer_sizes': (100, 50),
                    'max_iter': 500,
                    'random_state': 42,
                    'alpha': 0.01,  # L2 regularization
                    'early_stopping': True,
                    'validation_fraction': 0.2
                }
            }

        return configs

    def train_enhanced_model(self, trader_id, algorithm, feature_cols, target_col):
        """Train an enhanced model for a specific trader using the specified algorithm."""
        trader_data = self.feature_df[self.feature_df['account_id'] == trader_id].copy()
        trader_data = trader_data.sort_values('trade_date')

        # Remove rows with missing targets
        trader_data = trader_data.dropna(subset=[target_col])

        if len(trader_data) < 30:  # Reduced requirement for enhanced models
            return None

        # Prepare features and target
        X = trader_data[feature_cols].fillna(0)
        y = trader_data[target_col]

        # Split chronologically
        split_idx = int(len(trader_data) * 0.8)
        X_train = X.iloc[:split_idx]
        X_val = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_val = y.iloc[split_idx:]

        # Convert to numpy arrays
        X_train = X_train.select_dtypes(include=[np.number]).values
        X_val = X_val.select_dtypes(include=[np.number]).values
        y_train = y_train.values
        y_val = y_val.values

        if len(X_train) < 20 or len(X_val) < 5:
            return None

        try:
            config = self.get_algorithm_configs()[algorithm]

            if algorithm == 'xgboost':
                # Calculate class weights
                from sklearn.utils.class_weight import compute_class_weight
                classes = np.unique(y_train)
                class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
                weight_dict = dict(zip(classes, class_weights))

                model = xgb.XGBClassifier(**config['params'])
                sample_weights = np.array([weight_dict.get(y, 1.0) for y in y_train])
                model.fit(X_train, y_train, sample_weight=sample_weights)

            elif algorithm == 'random_forest':
                model = RandomForestClassifier(**config['params'])
                model.fit(X_train, y_train)

            elif algorithm == 'lightgbm' and LIGHTGBM_AVAILABLE:
                model = lgb.LGBMClassifier(**config['params'])
                model.fit(X_train, y_train)

            elif algorithm == 'neural_network' and SKLEARN_NN_AVAILABLE:
                model = MLPClassifier(**config['params'])
                model.fit(X_train, y_train)

            else:
                return None

            # Evaluate
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)

            train_score = accuracy_score(y_train, train_pred)
            val_score = accuracy_score(y_val, val_pred)
            train_f1 = f1_score(y_train, train_pred, average='weighted')
            val_f1 = f1_score(y_val, val_pred, average='weighted')

            # Calculate confidence score based on validation performance
            confidence = min(val_score, val_f1)

            # Get feature importance if available
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(feature_cols[:len(model.feature_importances_)],
                                            model.feature_importances_))
            elif hasattr(model, 'coef_'):
                # For neural networks
                feature_importance = dict(zip(feature_cols[:len(model.coef_[0])],
                                            np.abs(model.coef_[0])))
            else:
                feature_importance = {}

            top_features = dict(sorted(feature_importance.items(),
                                     key=lambda x: x[1], reverse=True)[:10])

            return {
                'model': model,
                'algorithm': algorithm,
                'algorithm_name': config['name'],
                'train_score': train_score,
                'val_score': val_score,
                'train_f1': train_f1,
                'val_f1': val_f1,
                'confidence': confidence,
                'train_size': len(X_train),
                'val_size': len(X_val),
                'feature_importance': top_features,
                'target_type': target_col
            }

        except Exception as e:
            print(f"  Error training {algorithm} model for trader {trader_id}: {e}")
            return None

    def train_all_enhanced_models(self):
        """Train enhanced models for each viable trader using multiple algorithms."""
        print("\\n=== ENHANCED MODEL TRAINING ===")
        print("Training multiple algorithms per trader with enhanced regularization")

        # Prepare data
        self.feature_df = self.feature_df.sort_values(['account_id', 'trade_date'])
        target_col = self.target_strategy_info['target_column']

        feature_cols = [col for col in self.feature_df.columns
                       if col not in ['account_id', 'trade_date', 'realized_pnl',
                                     'next_day_pnl', 'target_raw_pnl', 'target_class',
                                     'target_vol_norm', 'target_downside_risk']]

        print(f"✓ Target column: {target_col}")
        print(f"✓ Feature columns: {len(feature_cols)}")

        # Get traders with sufficient data
        trader_counts = self.feature_df.groupby('account_id').size()
        viable_traders = trader_counts[trader_counts >= 30].index

        print(f"✓ Training enhanced models for {len(viable_traders)} viable traders")

        # Test all algorithms
        algorithms = list(self.get_algorithm_configs().keys())
        print(f"✓ Testing algorithms: {[self.get_algorithm_configs()[alg]['name'] for alg in algorithms]}")

        successful_models = 0
        algorithm_counts = {alg: 0 for alg in algorithms}

        for trader_id in viable_traders:
            print(f"\\nTrader {trader_id}: Testing multiple algorithms...")
            trader_results = {}

            # Test each algorithm
            for algorithm in algorithms:
                result = self.train_enhanced_model(trader_id, algorithm, feature_cols, target_col)
                if result:
                    trader_results[algorithm] = result
                    print(f"  {result['algorithm_name']}: {result['val_score']:.3f} accuracy")

            if trader_results:
                # Select best algorithm for this trader
                best_algorithm = max(trader_results.keys(),
                                   key=lambda x: trader_results[x]['val_score'])
                best_result = trader_results[best_algorithm]

                self.enhanced_models[trader_id] = best_result
                algorithm_counts[best_algorithm] += 1
                successful_models += 1

                print(f"  → Selected: {best_result['algorithm_name']} "
                      f"(Val Acc: {best_result['val_score']:.3f})")

        print(f"\\n✓ Successfully trained {successful_models} enhanced models")
        print(f"✓ Algorithm distribution:")
        for alg, count in algorithm_counts.items():
            if count > 0:
                alg_name = self.get_algorithm_configs()[alg]['name']
                print(f"  {alg_name}: {count} traders")

        if successful_models > 0:
            # Calculate statistics
            avg_val_score = np.mean([r['val_score'] for r in self.enhanced_models.values()])
            avg_confidence = np.mean([r['confidence'] for r in self.enhanced_models.values()])

            print(f"✓ Average validation accuracy: {avg_val_score:.3f}")
            print(f"✓ Average confidence score: {avg_confidence:.3f}")

            return True
        else:
            print("❌ No enhanced models trained successfully")
            return False

    def save_enhanced_models(self):
        """Save enhanced models and results."""
        print("\\n=== SAVING ENHANCED MODELS ===")

        if not self.enhanced_models:
            print("❌ No enhanced models to save")
            return False

        # Save models
        with open('outputs/signals/enhanced_models.pkl', 'wb') as f:
            pickle.dump(self.enhanced_models, f)

        # Save algorithm performance summary
        algorithm_summary = {}
        for trader_id, model_info in self.enhanced_models.items():
            alg = model_info['algorithm']
            if alg not in algorithm_summary:
                algorithm_summary[alg] = {
                    'name': model_info['algorithm_name'],
                    'traders': [],
                    'avg_accuracy': 0,
                    'avg_confidence': 0
                }
            algorithm_summary[alg]['traders'].append({
                'trader_id': str(trader_id),
                'val_score': float(model_info['val_score']),
                'confidence': float(model_info['confidence'])
            })

        # Calculate averages
        for alg_info in algorithm_summary.values():
            alg_info['avg_accuracy'] = np.mean([t['val_score'] for t in alg_info['traders']])
            alg_info['avg_confidence'] = np.mean([t['confidence'] for t in alg_info['traders']])
            alg_info['count'] = len(alg_info['traders'])

        with open('outputs/signals/enhanced_algorithm_performance.json', 'w') as f:
            json.dump(algorithm_summary, f, indent=2)

        print(f"✓ Saved {len(self.enhanced_models)} enhanced models")
        print(f"✓ Saved algorithm performance summary")

        return True


def main():
    """Run enhanced model training."""
    trainer = EnhancedModelTraining()

    success = trainer.train_all_enhanced_models()
    if success:
        trainer.save_enhanced_models()
        print("\\n✅ Enhanced model training completed successfully")
    else:
        print("\\n❌ Enhanced model training failed")


if __name__ == "__main__":
    main()
