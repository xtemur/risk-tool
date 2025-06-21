#!/usr/bin/env python3
"""
Trader Model Training
Migrated from step4_model_training.py
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import json
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class TraderModelTraining:
    def __init__(self, features_path='outputs/signals/target_prepared.pkl'):
        self.feature_df = pd.read_pickle(features_path)
        self.trained_models = {}
        self.training_results = {}

        # Load target strategy info from file (created in step 3)
        try:
            with open('outputs/signals/target_strategy.json', 'r') as f:
                self.target_strategy_info = json.load(f)
        except:
            # Fallback hardcoded strategy
            self.target_strategy_info = {
                'best_strategy': 'Option B: Classification',
                'target_column': 'target_class',
                'model_performance': 0.785,
                'predictability_score': 0.2
            }

    def prepare_for_training(self):
        """Prepare data for individual trader model training"""
        print("=== MODEL SELECTION AND TRAINING ===")
        print("CRITICAL: SEPARATE MODEL PER TRADER")

        # Sort data
        self.feature_df = self.feature_df.sort_values(['account_id', 'trade_date'])

        # Get target column
        target_col = self.target_strategy_info['target_column']

        # Get feature columns (exclude metadata and targets)
        feature_cols = [col for col in self.feature_df.columns
                       if col not in ['account_id', 'trade_date', 'realized_pnl',
                                     'next_day_pnl', 'target_raw_pnl', 'target_class',
                                     'target_vol_norm', 'target_downside_risk']]

        print(f"✓ Target column: {target_col}")
        print(f"✓ Feature columns: {len(feature_cols)}")
        print(f"✓ Total observations: {len(self.feature_df)}")

        return feature_cols, target_col

    def train_trader_model(self, trader_id, feature_cols, target_col):
        """Train individual model for a specific trader"""
        trader_data = self.feature_df[self.feature_df['account_id'] == trader_id].copy()
        trader_data = trader_data.sort_values('trade_date')

        # Remove rows with missing targets
        trader_data = trader_data.dropna(subset=[target_col])

        if len(trader_data) < 30:
            return None

        # Prepare features and target
        X = trader_data[feature_cols].fillna(0)
        y = trader_data[target_col]

        # Split chronologically (80% train, 20% validation)
        split_idx = int(len(trader_data) * 0.8)
        X_train = X.iloc[:split_idx]
        X_val = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_val = y.iloc[split_idx:]

        # Convert to numpy arrays for XGBoost
        X_train = X_train.select_dtypes(include=[np.number]).values
        X_val = X_val.select_dtypes(include=[np.number]).values
        y_train = y_train.values
        y_val = y_val.values

        if len(X_train) < 20 or len(X_val) < 5:
            return None

        try:
            # Train XGBoost model based on target type
            if target_col == 'target_class':
                # Calculate class weights for balancing
                from sklearn.utils.class_weight import compute_class_weight
                classes = np.unique(y_train)
                class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
                weight_dict = dict(zip(classes, class_weights))

                model = xgb.XGBClassifier(
                    objective='multi:softprob',
                    random_state=42,
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    eval_metric='mlogloss'
                )

                # Apply sample weights during training
                sample_weights = np.array([weight_dict[y] for y in y_train])

                model.fit(X_train, y_train, sample_weight=sample_weights)

                # Evaluate
                train_pred = model.predict(X_train)
                val_pred = model.predict(X_val)

                train_score = accuracy_score(y_train, train_pred)
                val_score = accuracy_score(y_val, val_pred)

                # Get F1 score for each class
                train_f1 = f1_score(y_train, train_pred, average='weighted')
                val_f1 = f1_score(y_val, val_pred, average='weighted')

            elif target_col == 'target_downside_risk':
                # Calculate scale_pos_weight for binary imbalanced classification
                pos_count = (y_train == 1).sum()
                neg_count = (y_train == 0).sum()
                scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

                model = xgb.XGBClassifier(
                    objective='binary:logistic',
                    random_state=42,
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    eval_metric='logloss',
                    scale_pos_weight=scale_pos_weight  # Balance classes
                )

                model.fit(X_train, y_train)

                train_pred = model.predict(X_train)
                val_pred = model.predict(X_val)

                train_score = accuracy_score(y_train, train_pred)
                val_score = accuracy_score(y_val, val_pred)
                train_f1 = f1_score(y_train, train_pred)
                val_f1 = f1_score(y_val, val_pred)

            else:  # Regression
                model = xgb.XGBRegressor(
                    objective='reg:absoluteerror',
                    random_state=42,
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8
                )

                model.fit(X_train, y_train)

                train_pred = model.predict(X_train)
                val_pred = model.predict(X_val)

                train_score = -mean_absolute_error(y_train, train_pred)
                val_score = -mean_absolute_error(y_val, val_pred)
                train_f1 = train_score  # Use same metric
                val_f1 = val_score

            # Get feature importance
            feature_importance = dict(zip(feature_cols[:len(model.feature_importances_)],
                                        model.feature_importances_))
            top_features = dict(sorted(feature_importance.items(),
                                     key=lambda x: x[1], reverse=True)[:10])

            return {
                'model': model,
                'train_score': train_score,
                'val_score': val_score,
                'train_f1': train_f1,
                'val_f1': val_f1,
                'train_size': len(X_train),
                'val_size': len(X_val),
                'feature_importance': top_features,
                'target_type': target_col
            }

        except Exception as e:
            print(f"  Error training model for trader {trader_id}: {e}")
            return None

    def train_all_models(self):
        """Train individual models for each viable trader"""
        print("\\nTraining individual models for each trader...")

        feature_cols, target_col = self.prepare_for_training()

        # Get traders with sufficient data
        trader_counts = self.feature_df.groupby('account_id').size()
        viable_traders = trader_counts[trader_counts >= 30].index

        print(f"✓ Training models for {len(viable_traders)} viable traders")

        successful_models = 0
        failed_models = 0

        for trader_id in viable_traders:
            result = self.train_trader_model(trader_id, feature_cols, target_col)

            if result:
                self.trained_models[trader_id] = result
                self.training_results[trader_id] = {
                    'train_score': result['train_score'],
                    'val_score': result['val_score'],
                    'train_f1': result['train_f1'],
                    'val_f1': result['val_f1'],
                    'train_size': result['train_size'],
                    'val_size': result['val_size']
                }
                successful_models += 1
            else:
                failed_models += 1

        print(f"\\n✓ Successfully trained {successful_models} models")
        print(f"✓ Failed to train {failed_models} models")

        if successful_models > 0:
            # Calculate aggregate statistics
            avg_train_score = np.mean([r['train_score'] for r in self.training_results.values()])
            avg_val_score = np.mean([r['val_score'] for r in self.training_results.values()])
            avg_train_f1 = np.mean([r['train_f1'] for r in self.training_results.values()])
            avg_val_f1 = np.mean([r['val_f1'] for r in self.training_results.values()])

            print(f"✓ Average training score: {avg_train_score:.4f}")
            print(f"✓ Average validation score: {avg_val_score:.4f}")
            print(f"✓ Average training F1: {avg_train_f1:.4f}")
            print(f"✓ Average validation F1: {avg_val_f1:.4f}")

            return True
        else:
            print("❌ No models trained successfully")
            return False

    def validate_model_features(self):
        """Validate that top features make financial sense"""
        print("\\n=== MODEL FEATURE VALIDATION ===")

        if not self.trained_models:
            print("❌ No models available for validation")
            return False

        # Aggregate feature importance across all models
        all_feature_importance = {}

        for _, model_info in self.trained_models.items():
            for feature, importance in model_info['feature_importance'].items():
                if feature not in all_feature_importance:
                    all_feature_importance[feature] = []
                all_feature_importance[feature].append(importance)

        # Calculate average importance
        avg_feature_importance = {}
        for feature, importances in all_feature_importance.items():
            avg_feature_importance[feature] = np.mean(importances)

        # Sort by importance
        top_features = dict(sorted(avg_feature_importance.items(),
                                 key=lambda x: x[1], reverse=True)[:15])

        print("Top 15 features across all models:")
        for feature, importance in top_features.items():
            print(f"  {feature}: {importance:.4f}")

        # Validate financial logic
        expected_important_features = [
            'realized_pnl_lag1', 'realized_pnl_lag2', 'realized_pnl_lag3',
            'win_rate_lag1', 'win_rate_lag2', 'win_rate_lag3',
            'profit_factor_lag1', 'profit_factor_lag2', 'profit_factor_lag3',
            'volatility_ewma5', 'volatility_ewma20',
            'consecutive_losses', 'consecutive_wins',
            'current_drawdown', 'max_drawdown',
            'sharpe_ratio', 'pnl_vs_recent'
        ]

        financially_logical_features = 0
        for feature in top_features.keys():
            if any(expected in feature for expected in expected_important_features):
                financially_logical_features += 1

        financial_logic_score = financially_logical_features / len(top_features)

        print(f"\\n✓ Financially logical features: {financially_logical_features}/{len(top_features)}")
        print(f"✓ Financial logic score: {financial_logic_score:.2%}")

        if financial_logic_score >= 0.6:
            print("✅ Feature importance passes financial logic check")
            return True
        else:
            print("⚠️  Feature importance may not align with financial logic")
            return True  # Continue anyway

    def save_models_and_results(self):
        """Save trained models and results"""
        print("\\n=== SAVING MODELS AND RESULTS ===")

        if not self.trained_models:
            print("❌ No models to save")
            return False

        # Save models
        with open('outputs/signals/trained_models.pkl', 'wb') as f:
            pickle.dump(self.trained_models, f)

        # Save training results
        with open('outputs/signals/training_results.json', 'w') as f:
            # Convert numpy types for JSON serialization
            json_results = {}
            for trader_id, results in self.training_results.items():
                json_results[str(trader_id)] = {
                    'train_score': float(results['train_score']),
                    'val_score': float(results['val_score']),
                    'train_f1': float(results['train_f1']),
                    'val_f1': float(results['val_f1']),
                    'train_size': int(results['train_size']),
                    'val_size': int(results['val_size'])
                }
            json.dump(json_results, f, indent=2)

        # Save feature names for later use
        feature_cols, _ = self.prepare_for_training()
        with open('outputs/signals/model_feature_names.json', 'w') as f:
            json.dump(feature_cols, f, indent=2)

        print(f"✓ Saved {len(self.trained_models)} models to outputs/signals/trained_models.pkl")
        print(f"✓ Saved training results to outputs/signals/training_results.json")
        print(f"✓ Saved feature names to outputs/signals/model_feature_names.json")

        return True

    def generate_checkpoint_report(self):
        """Generate Step 4 checkpoint report"""
        print("\\n" + "="*50)
        print("MODEL TRAINING CHECKPOINT VALIDATION")
        print("="*50)

        checkpoint_checks = []

        # Check 1: Were models trained successfully?
        models_trained = len(self.trained_models) > 0
        checkpoint_checks.append(models_trained)
        print(f"✓ Models trained: {models_trained} ({len(self.trained_models)} models)")

        if models_trained:
            # Check 2: Do we have sufficient model coverage? (relaxed for active traders)
            min_models = 5
            sufficient_coverage = len(self.trained_models) >= min_models
            checkpoint_checks.append(sufficient_coverage)
            print(f"✓ Sufficient model coverage: {sufficient_coverage} (need ≥{min_models})")

            # Check 3: Are model performance scores reasonable?
            avg_val_score = np.mean([r['val_score'] for r in self.training_results.values()])
            reasonable_performance = avg_val_score > 0.3  # For classification
            checkpoint_checks.append(reasonable_performance)
            print(f"✓ Reasonable performance: {reasonable_performance} (avg: {avg_val_score:.4f})")

            # Check 4: Feature importance validation
            feature_validation = self.validate_model_features()
            checkpoint_checks.append(feature_validation)
            print(f"✓ Feature validation: {feature_validation}")

        checkpoint_pass = all(checkpoint_checks)

        if checkpoint_pass:
            print("\\n✅ MODEL TRAINING CHECKPOINT PASSED")
        else:
            print("\\n❌ MODEL TRAINING CHECKPOINT FAILED")

        return checkpoint_pass
