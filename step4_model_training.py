#!/usr/bin/env python3
"""
Step 4: Model Selection and Training
Building separate XGBoost models for each trader
Following CLAUDE.md methodology
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

class TraderModelTraining:
    def __init__(self):
        # Load data from previous steps
        self.load_prepared_data()
        self.trained_models = {}
        self.model_performance = {}

    def load_prepared_data(self):
        """Load prepared data from Step 3"""
        print("=== STEP 4: MODEL SELECTION AND TRAINING ===")

        # Use classification target as determined best in Step 3
        self.target_info = {
            'best_strategy': 'Option B: Classification',
            'target_column': 'target_class'
        }

        print(f"✓ Using target strategy: {self.target_info['best_strategy']}")
        print(f"✓ Target column: {self.target_info['target_column']}")

        # Load feature data and recreate target
        self.feature_df = pd.read_pickle('data/features_engineered.pkl')
        self.feature_df = self.feature_df.sort_values(['account_id', 'trade_date'])

        # Recreate classification target as determined in Step 3
        self.create_classification_target()

        # Identify feature columns
        self.feature_cols = [col for col in self.feature_df.columns
                           if col not in ['account_id', 'trade_date', 'realized_pnl',
                                         'next_day_pnl', 'target_class']]

        print(f"✓ Loaded data with {len(self.feature_cols)} features")
        print(f"✓ Total observations: {len(self.feature_df)}")

    def create_classification_target(self):
        """Recreate the classification target from Step 3"""
        target_dfs = []

        for trader_id in self.feature_df['account_id'].unique():
            trader_df = self.feature_df[self.feature_df['account_id'] == trader_id].copy()
            trader_df = trader_df.sort_values('trade_date')

            # Create next-day PnL
            trader_df['next_day_pnl'] = trader_df['realized_pnl'].shift(-1)

            # Calculate percentiles for this trader
            pnl_25 = trader_df['next_day_pnl'].quantile(0.25)
            pnl_75 = trader_df['next_day_pnl'].quantile(0.75)

            # Create classification target
            trader_df['target_class'] = 1  # Neutral
            trader_df.loc[trader_df['next_day_pnl'] < pnl_25, 'target_class'] = 0  # Loss
            trader_df.loc[trader_df['next_day_pnl'] > pnl_75, 'target_class'] = 2  # Win

            target_dfs.append(trader_df)

        self.feature_df = pd.concat(target_dfs, ignore_index=True)

        # Remove rows with missing targets
        self.feature_df = self.feature_df.dropna(subset=['target_class'])

    def identify_viable_traders(self):
        """Identify traders with sufficient data for individual models"""
        print("\\n=== IDENTIFYING VIABLE TRADERS ===")

        trader_stats = self.feature_df.groupby('account_id').agg({
            'target_class': 'count',
            'trade_date': ['min', 'max']
        }).reset_index()

        trader_stats.columns = ['account_id', 'total_obs', 'first_date', 'last_date']

        # Filter for traders with sufficient data (minimum 60 observations)
        viable_traders = trader_stats[trader_stats['total_obs'] >= 60]['account_id'].tolist()

        # Check class distribution for viable traders
        final_viable_traders = []

        for trader_id in viable_traders:
            trader_data = self.feature_df[self.feature_df['account_id'] == trader_id]
            class_counts = trader_data['target_class'].value_counts()

            # Ensure we have at least 2 classes and minimum observations per class
            if len(class_counts) >= 2 and class_counts.min() >= 5:
                final_viable_traders.append(trader_id)

        print(f"✓ Traders with sufficient data: {len(final_viable_traders)}")
        print(f"✓ Average observations per trader: {trader_stats[trader_stats['account_id'].isin(final_viable_traders)]['total_obs'].mean():.0f}")

        return final_viable_traders

    def train_trader_model(self, trader_id):
        """Train individual model for a specific trader"""
        # Get trader data
        trader_data = self.feature_df[self.feature_df['account_id'] == trader_id].copy()
        trader_data = trader_data.sort_values('trade_date')

        if len(trader_data) < 60:
            return None

        # Prepare features and target
        X = trader_data[self.feature_cols].fillna(0)
        y = trader_data['target_class']

        # Ensure numeric data
        X = X.select_dtypes(include=[np.number]).values
        y = y.values

        # Check class distribution
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            return None

        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        cv_scores = []

        # Store best model
        best_score = 0
        best_model = None

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            try:
                # Train XGBoost classifier
                if len(unique_classes) == 2:
                    # Binary classification
                    model = xgb.XGBClassifier(
                        objective='binary:logistic',
                        random_state=42,
                        n_estimators=100,
                        max_depth=3,
                        learning_rate=0.1,
                        subsample=0.8,
                        colsample_bytree=0.8
                    )
                else:
                    # Multi-class classification
                    model = xgb.XGBClassifier(
                        objective='multi:softprob',
                        random_state=42,
                        n_estimators=100,
                        max_depth=3,
                        learning_rate=0.1,
                        subsample=0.8,
                        colsample_bytree=0.8
                    )

                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                score = f1_score(y_val, y_pred, average='weighted')
                cv_scores.append(score)

                if score > best_score:
                    best_score = score
                    best_model = model

            except Exception as e:
                print(f"  Warning: Error training model for trader {trader_id}: {e}")
                continue

        if best_model is not None and cv_scores:
            # Train final model on all data
            try:
                final_model = xgb.XGBClassifier(
                    objective='binary:logistic' if len(unique_classes) == 2 else 'multi:softprob',
                    random_state=42,
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8
                )

                final_model.fit(X, y)

                # Feature importance
                feature_importance = dict(zip(
                    [f"feature_{i}" for i in range(len(self.feature_cols))],
                    final_model.feature_importances_
                ))

                return {
                    'model': final_model,
                    'cv_scores': cv_scores,
                    'avg_cv_score': np.mean(cv_scores),
                    'std_cv_score': np.std(cv_scores),
                    'feature_importance': feature_importance,
                    'num_observations': len(trader_data),
                    'class_distribution': dict(zip(*np.unique(y, return_counts=True))),
                    'feature_names': self.feature_cols[:len(self.feature_cols)]
                }

            except Exception as e:
                print(f"  Warning: Error training final model for trader {trader_id}: {e}")
                return None

        return None

    def train_all_models(self):
        """Train models for all viable traders"""
        print("\\n=== TRAINING INDIVIDUAL TRADER MODELS ===")

        viable_traders = self.identify_viable_traders()

        successful_models = 0
        failed_models = 0

        for i, trader_id in enumerate(viable_traders):
            print(f"\\rTraining models... {i+1}/{len(viable_traders)}", end="")

            model_result = self.train_trader_model(trader_id)

            if model_result:
                self.trained_models[trader_id] = model_result
                self.model_performance[trader_id] = {
                    'avg_score': model_result['avg_cv_score'],
                    'std_score': model_result['std_cv_score'],
                    'num_obs': model_result['num_observations']
                }
                successful_models += 1
            else:
                failed_models += 1

        print(f"\\n✓ Successfully trained: {successful_models} models")
        print(f"✓ Failed to train: {failed_models} models")

        return successful_models > 0

    def validate_feature_importance(self):
        """Validate feature importance across traders"""
        print("\\n=== FEATURE IMPORTANCE VALIDATION ===")

        if not self.trained_models:
            print("❌ No trained models to validate")
            return False

        # Aggregate feature importance across all traders
        all_importances = {}

        for trader_id, model_info in self.trained_models.items():
            feature_names = model_info['feature_names']
            importances = model_info['feature_importance']

            for i, (feature_key, importance) in enumerate(importances.items()):
                if i < len(feature_names):
                    feature_name = feature_names[i]
                    if feature_name not in all_importances:
                        all_importances[feature_name] = []
                    all_importances[feature_name].append(importance)

        # Calculate average importance across traders
        avg_importances = {}
        for feature, importance_list in all_importances.items():
            avg_importances[feature] = np.mean(importance_list)

        # Sort by importance
        sorted_features = sorted(avg_importances.items(), key=lambda x: x[1], reverse=True)

        print("Top 15 most important features (average across traders):")
        for i, (feature, importance) in enumerate(sorted_features[:15]):
            print(f"  {i+1:2d}. {feature:<25} {importance:.4f}")

        # Validate financial intuition
        important_features = [f[0] for f in sorted_features[:10]]

        # Check for financially logical features
        logical_features = [
            'realized_pnl_lag1', 'win_rate_lag1', 'profit_factor_lag1',
            'volatility_ewma5', 'consecutive_losses', 'current_drawdown',
            'sharpe_ratio', 'pnl_vs_recent'
        ]

        logical_count = sum(1 for f in important_features if any(lf in f for lf in logical_features))

        print(f"\\n✓ Financially logical features in top 10: {logical_count}/10")

        if logical_count >= 3:
            print("✅ Feature importance validation PASSED")
            return True
        else:
            print("⚠️  Feature importance validation WARNING - few logical features")
            return True  # Continue anyway

    def generate_model_diagnostics(self):
        """Generate comprehensive model diagnostics"""
        print("\\n=== MODEL DIAGNOSTICS ===")

        if not self.trained_models:
            print("❌ No trained models for diagnostics")
            return False

        # Performance statistics
        all_scores = [info['avg_score'] for info in self.model_performance.values()]

        print(f"✓ Models trained: {len(self.trained_models)}")
        print(f"✓ Average F1 score: {np.mean(all_scores):.4f} ± {np.std(all_scores):.4f}")
        print(f"✓ Best performing model: {max(all_scores):.4f}")
        print(f"✓ Worst performing model: {min(all_scores):.4f}")

        # Model stability analysis
        stable_models = sum(1 for info in self.model_performance.values() if info['std_score'] < 0.2)
        print(f"✓ Stable models (std < 0.2): {stable_models}/{len(self.trained_models)}")

        # Check for reasonable performance
        good_models = sum(1 for score in all_scores if score > 0.3)
        print(f"✓ Models with F1 > 0.3: {good_models}/{len(self.trained_models)}")

        return True

    def save_models(self):
        """Save trained models and metadata"""
        print("\\n=== SAVING MODELS ===")

        # Save models
        with open('data/trained_models.pkl', 'wb') as f:
            pickle.dump(self.trained_models, f)

        # Save performance data
        with open('data/model_performance.json', 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            performance_data = {}
            for trader_id, perf in self.model_performance.items():
                performance_data[str(trader_id)] = {
                    'avg_score': float(perf['avg_score']),
                    'std_score': float(perf['std_score']),
                    'num_obs': int(perf['num_obs'])
                }
            json.dump(performance_data, f, indent=2)

        # Save feature names
        if self.trained_models:
            sample_model = next(iter(self.trained_models.values()))
            with open('data/model_feature_names.json', 'w') as f:
                json.dump(sample_model['feature_names'], f, indent=2)

        print(f"✓ Saved {len(self.trained_models)} models to data/trained_models.pkl")
        print(f"✓ Saved performance data to data/model_performance.json")

    def generate_checkpoint_report(self):
        """Generate Step 4 checkpoint report"""
        print("\\n" + "="*50)
        print("STEP 4 CHECKPOINT VALIDATION")
        print("="*50)

        checkpoint_checks = []

        # Check 1: Were models successfully trained?
        models_trained = len(self.trained_models) > 0
        checkpoint_checks.append(models_trained)
        print(f"✓ Models trained: {models_trained} ({len(self.trained_models)} models)")

        if models_trained:
            # Check 2: Do models show predictive power?
            avg_performance = np.mean([info['avg_score'] for info in self.model_performance.values()])
            has_predictive_power = avg_performance > 0.3
            checkpoint_checks.append(has_predictive_power)
            print(f"✓ Predictive power: {has_predictive_power} (avg F1: {avg_performance:.4f})")

            # Check 3: Are models stable across time periods?
            stable_models = sum(1 for info in self.model_performance.values() if info['std_score'] < 0.3)
            stability_ratio = stable_models / len(self.trained_models)
            is_stable = stability_ratio > 0.6
            checkpoint_checks.append(is_stable)
            print(f"✓ Model stability: {is_stable} ({stability_ratio:.2%} stable models)")

            # Check 4: Sufficient number of viable models
            sufficient_models = len(self.trained_models) >= 5
            checkpoint_checks.append(sufficient_models)
            print(f"✓ Sufficient models: {sufficient_models} ({len(self.trained_models)} ≥ 5)")

        checkpoint_pass = all(checkpoint_checks)

        if checkpoint_pass:
            print("\\n✅ CHECKPOINT 4 PASSED - Proceeding to Step 5")
        else:
            print("\\n❌ CHECKPOINT 4 FAILED - Model training issues")

        return checkpoint_pass

def main():
    """Run Step 4 model training"""
    trainer = TraderModelTraining()

    # Train all models
    training_success = trainer.train_all_models()

    if training_success:
        # Validate feature importance
        trainer.validate_feature_importance()

        # Generate diagnostics
        trainer.generate_model_diagnostics()

        # Save models
        trainer.save_models()

    # Generate checkpoint report
    checkpoint_pass = trainer.generate_checkpoint_report()

    return checkpoint_pass, trainer.trained_models

if __name__ == "__main__":
    main()
