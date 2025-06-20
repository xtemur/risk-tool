#!/usr/bin/env python3
"""
Enhanced Model Training with Multiple Algorithms and Ensemble Methods
"""

import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFECV
import xgboost as xgb

# Try to import additional models
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available - install with: pip install lightgbm")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost not available - install with: pip install catboost")

try:
    from sklearn.neural_network import MLPClassifier
    NEURAL_NETWORK_AVAILABLE = True
except ImportError:
    NEURAL_NETWORK_AVAILABLE = False

class EnhancedModelTraining:
    def __init__(self):
        self.load_data()
        self.enhanced_models = {}
        self.ensemble_models = {}
        self.feature_importances = {}

    def load_data(self):
        """Load feature data and prepare for enhanced modeling"""
        print("=== ENHANCED MODEL TRAINING ===")

        # Load the prepared feature data
        self.feature_df = pd.read_pickle('data/target_prepared.pkl')

        # Load feature names
        with open('data/model_feature_names.json', 'r') as f:
            self.feature_names = json.load(f)

        print(f"✓ Loaded feature data: {len(self.feature_df)} observations")
        print(f"✓ Feature count: {len(self.feature_names)}")

    def create_enhanced_features(self):
        """Create additional advanced features"""
        print("\n=== CREATING ENHANCED FEATURES ===")

        enhanced_features = []

        for trader_id in self.feature_df['account_id'].unique():
            trader_data = self.feature_df[self.feature_df['account_id'] == trader_id].copy()
            trader_data = trader_data.sort_values('trade_date')

            # Advanced technical indicators
            trader_data['pnl_momentum_3d'] = trader_data['realized_pnl'].rolling(3).mean()
            trader_data['pnl_momentum_7d'] = trader_data['realized_pnl'].rolling(7).mean()
            trader_data['pnl_volatility_7d'] = trader_data['realized_pnl'].rolling(7).std()
            trader_data['pnl_skewness_7d'] = trader_data['realized_pnl'].rolling(7).skew()

            # Performance regime indicators
            trader_data['high_performance_regime'] = (trader_data['realized_pnl'] > trader_data['realized_pnl'].rolling(20).quantile(0.8)).astype(int)
            trader_data['low_performance_regime'] = (trader_data['realized_pnl'] < trader_data['realized_pnl'].rolling(20).quantile(0.2)).astype(int)

            # Trend indicators
            trader_data['pnl_trend_5d'] = np.where(
                trader_data['realized_pnl'].rolling(5).mean() > trader_data['realized_pnl'].rolling(10).mean(), 1, 0
            )

            # Risk-adjusted metrics
            trader_data['sharpe_ratio_7d'] = trader_data['realized_pnl'].rolling(7).mean() / (trader_data['realized_pnl'].rolling(7).std() + 1e-8)
            trader_data['max_drawdown_7d'] = trader_data['realized_pnl'].rolling(7).apply(
                lambda x: (x.cumsum() - x.cumsum().expanding().max()).min(), raw=False
            )

            # Day-of-week effects
            trader_data['day_of_week'] = pd.to_datetime(trader_data['trade_date']).dt.dayofweek
            for dow in range(5):  # Monday=0 to Friday=4
                trader_data[f'is_dow_{dow}'] = (trader_data['day_of_week'] == dow).astype(int)

            enhanced_features.append(trader_data)

        self.enhanced_feature_df = pd.concat(enhanced_features, ignore_index=True)

        # Update feature names
        new_features = [col for col in self.enhanced_feature_df.columns
                       if col not in self.feature_df.columns and
                       col not in ['account_id', 'trade_date', 'target_class', 'next_day_pnl']]

        self.enhanced_feature_names = self.feature_names + new_features
        print(f"✓ Added {len(new_features)} enhanced features")
        print(f"✓ Total features: {len(self.enhanced_feature_names)}")

    def prepare_model_configs(self):
        """Prepare different model configurations"""
        self.model_configs = {}

        # 1. Regularized XGBoost (less overfitting)
        self.model_configs['xgb_regularized'] = {
            'model': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=1.0,
                reg_lambda=1.0,
                random_state=42,
                eval_metric='mlogloss'
            ),
            'name': 'Regularized XGBoost'
        }

        # 2. Random Forest with balanced parameters
        self.model_configs['rf_balanced'] = {
            'model': RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
                class_weight='balanced',
                random_state=42
            ),
            'name': 'Balanced Random Forest'
        }

        # 3. LightGBM if available
        if LIGHTGBM_AVAILABLE:
            self.model_configs['lgb'] = {
                'model': lgb.LGBMClassifier(
                    n_estimators=150,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.5,
                    reg_lambda=0.5,
                    class_weight='balanced',
                    random_state=42,
                    verbose=-1
                ),
                'name': 'LightGBM'
            }

        # 4. CatBoost if available
        if CATBOOST_AVAILABLE:
            self.model_configs['catboost'] = {
                'model': cb.CatBoostClassifier(
                    iterations=150,
                    depth=6,
                    learning_rate=0.05,
                    class_weights='Balanced',
                    random_seed=42,
                    verbose=False
                ),
                'name': 'CatBoost'
            }

        # 5. Neural Network if available
        if NEURAL_NETWORK_AVAILABLE:
            self.model_configs['neural_net'] = {
                'model': MLPClassifier(
                    hidden_layer_sizes=(100, 50),
                    alpha=0.01,
                    learning_rate='adaptive',
                    max_iter=500,
                    random_state=42
                ),
                'name': 'Neural Network'
            }

        print(f"✓ Prepared {len(self.model_configs)} model configurations")

    def feature_selection(self, X, y, trader_id):
        """Perform feature selection for each trader"""

        # Remove features with low variance
        feature_vars = X.var()
        high_var_features = feature_vars[feature_vars > 0.001].index.tolist()

        if len(high_var_features) == 0:
            # Fallback: use all features if none have sufficient variance
            high_var_features = X.columns.tolist()

        X_filtered = X[high_var_features]

        # Select top K features using statistical tests
        if len(high_var_features) > 20:
            k_best = min(20, len(high_var_features))
            try:
                selector = SelectKBest(score_func=f_classif, k=k_best)
                X_selected = selector.fit_transform(X_filtered, y)
                feature_indices = selector.get_support(indices=True)
                selected_features = [high_var_features[i] for i in feature_indices]
            except (ValueError, IndexError) as e:
                # Fallback: use all high variance features
                X_selected = X_filtered.values
                selected_features = high_var_features
        else:
            X_selected = X_filtered.values
            selected_features = high_var_features

        return X_selected, selected_features

    def train_enhanced_models(self):
        """Train enhanced models for each trader"""
        print("\n=== TRAINING ENHANCED MODELS ===")

        # Prepare enhanced features
        self.create_enhanced_features()
        self.prepare_model_configs()

        training_cutoff = pd.to_datetime('2025-04-01')

        enhanced_results = {}

        for trader_id in self.enhanced_feature_df['account_id'].unique():
            print(f"\nTraining enhanced models for trader {trader_id}...")

            trader_data = self.enhanced_feature_df[self.enhanced_feature_df['account_id'] == trader_id].copy()
            trainer_data = trader_data[trader_data['trade_date'] < training_cutoff].copy()

            if len(trainer_data) < 100:
                print(f"  ⚠️ Insufficient training data for trader {trader_id}")
                continue

            # Prepare features and target
            X = trainer_data[self.enhanced_feature_names].fillna(0)
            y = trainer_data['target_class']

            # Remove samples with missing targets
            valid_mask = ~y.isna()
            X = X[valid_mask]
            y = y[valid_mask]

            if len(y.unique()) < 2:
                print(f"  ⚠️ Insufficient class diversity for trader {trader_id}")
                continue

            # Feature selection
            X_selected, selected_features = self.feature_selection(X, y, trader_id)

            # Scale features for some models
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X_selected)

            # Time series split for validation
            tscv = TimeSeriesSplit(n_splits=3)

            trader_results = {}
            trader_models = {}

            # Test each model configuration
            for model_key, config in self.model_configs.items():
                try:
                    model = config['model']

                    # Use scaled features for neural networks, raw for tree-based
                    if 'neural' in model_key:
                        X_train = X_scaled
                    else:
                        X_train = X_selected

                    # Cross-validation scores
                    cv_scores = cross_val_score(model, X_train, y, cv=tscv, scoring='accuracy')

                    # Train final model
                    model.fit(X_train, y)

                    # Calculate training score
                    train_pred = model.predict(X_train)
                    train_score = accuracy_score(y, train_pred)

                    trader_results[model_key] = {
                        'model_name': config['name'],
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std(),
                        'train_score': train_score,
                        'feature_count': len(selected_features)
                    }

                    trader_models[model_key] = {
                        'model': model,
                        'scaler': scaler if 'neural' in model_key else None,
                        'selected_features': selected_features
                    }

                    print(f"  ✓ {config['name']}: CV={cv_scores.mean():.3f}±{cv_scores.std():.3f}")

                except Exception as e:
                    print(f"  ❌ {config['name']} failed: {e}")
                    continue

            if trader_results:
                # Select best model based on CV score
                best_model_key = max(trader_results.keys(), key=lambda k: trader_results[k]['cv_mean'])
                best_cv_score = trader_results[best_model_key]['cv_mean']

                print(f"  🏆 Best model: {trader_results[best_model_key]['model_name']} (CV: {best_cv_score:.3f})")

                enhanced_results[str(trader_id)] = {
                    'best_model': best_model_key,
                    'best_cv_score': best_cv_score,
                    'all_results': trader_results
                }

                self.enhanced_models[str(trader_id)] = trader_models

        # Save enhanced models and results
        with open('data/enhanced_models.pkl', 'wb') as f:
            pickle.dump(self.enhanced_models, f)

        with open('data/enhanced_training_results.json', 'w') as f:
            json.dump(enhanced_results, f, indent=2)

        print(f"\n✓ Enhanced models trained for {len(enhanced_results)} traders")
        return enhanced_results

    def create_ensemble_models(self):
        """Create ensemble models combining multiple algorithms"""
        print("\n=== CREATING ENSEMBLE MODELS ===")

        training_cutoff = pd.to_datetime('2025-04-01')
        ensemble_results = {}

        for trader_id in self.enhanced_feature_df['account_id'].unique():
            print(f"\nCreating ensemble for trader {trader_id}...")

            trader_data = self.enhanced_feature_df[self.enhanced_feature_df['account_id'] == trader_id].copy()
            trainer_data = trader_data[trader_data['trade_date'] < training_cutoff].copy()

            if len(trainer_data) < 100:
                continue

            # Prepare data
            X = trainer_data[self.enhanced_feature_names].fillna(0)
            y = trainer_data['target_class']
            valid_mask = ~y.isna()
            X = X[valid_mask]
            y = y[valid_mask]

            if len(y.unique()) < 2:
                continue

            X_selected, selected_features = self.feature_selection(X, y, trader_id)

            try:
                # Create individual models for ensemble
                base_models = []

                # Regularized XGBoost
                xgb_model = xgb.XGBClassifier(
                    n_estimators=100, max_depth=3, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    reg_alpha=1.0, reg_lambda=1.0, random_state=42
                )
                base_models.append(('xgb', xgb_model))

                # Random Forest
                rf_model = RandomForestClassifier(
                    n_estimators=100, max_depth=6, min_samples_split=10,
                    class_weight='balanced', random_state=42
                )
                base_models.append(('rf', rf_model))

                # Add LightGBM if available
                if LIGHTGBM_AVAILABLE:
                    lgb_model = lgb.LGBMClassifier(
                        n_estimators=100, max_depth=4, learning_rate=0.05,
                        class_weight='balanced', random_state=42, verbose=-1
                    )
                    base_models.append(('lgb', lgb_model))

                # Voting Classifier (Hard Voting)
                voting_clf = VotingClassifier(estimators=base_models, voting='hard')

                # Bagging Ensemble
                bagging_clf = BaggingClassifier(
                    base_estimator=xgb.XGBClassifier(
                        n_estimators=50, max_depth=3, learning_rate=0.1,
                        random_state=42
                    ),
                    n_estimators=10,
                    random_state=42
                )

                # Test ensemble methods
                ensemble_methods = {
                    'voting': voting_clf,
                    'bagging': bagging_clf
                }

                best_ensemble = None
                best_score = 0

                for method_name, ensemble_model in ensemble_methods.items():
                    # Cross-validation
                    tscv = TimeSeriesSplit(n_splits=3)
                    cv_scores = cross_val_score(ensemble_model, X_selected, y, cv=tscv, scoring='accuracy')

                    print(f"  {method_name.title()}: CV={cv_scores.mean():.3f}±{cv_scores.std():.3f}")

                    if cv_scores.mean() > best_score:
                        best_score = cv_scores.mean()
                        best_ensemble = ensemble_model
                        best_method = method_name

                if best_ensemble is not None:
                    # Train best ensemble
                    best_ensemble.fit(X_selected, y)

                    ensemble_results[str(trader_id)] = {
                        'method': best_method,
                        'cv_score': best_score,
                        'feature_count': len(selected_features)
                    }

                    self.ensemble_models[str(trader_id)] = {
                        'model': best_ensemble,
                        'selected_features': selected_features,
                        'method': best_method
                    }

                    print(f"  🏆 Best ensemble: {best_method} (CV: {best_score:.3f})")

            except Exception as e:
                print(f"  ❌ Ensemble creation failed: {e}")
                continue

        # Save ensemble models
        with open('data/ensemble_models.pkl', 'wb') as f:
            pickle.dump(self.ensemble_models, f)

        with open('data/ensemble_results.json', 'w') as f:
            json.dump(ensemble_results, f, indent=2)

        print(f"\n✓ Ensemble models created for {len(ensemble_results)} traders")
        return ensemble_results

    def compare_model_performance(self):
        """Compare performance of different model approaches"""
        print("\n=== MODEL PERFORMANCE COMPARISON ===")

        # Load original results
        with open('data/training_results.json', 'r') as f:
            original_results = json.load(f)

        # Load enhanced results
        try:
            with open('data/enhanced_training_results.json', 'r') as f:
                enhanced_results = json.load(f)
        except FileNotFoundError:
            enhanced_results = {}

        # Load ensemble results
        try:
            with open('data/ensemble_results.json', 'r') as f:
                ensemble_results = json.load(f)
        except FileNotFoundError:
            ensemble_results = {}

        comparison = []

        for trader_id in original_results.keys():
            original_val_score = original_results[trader_id]['val_score']

            enhanced_score = None
            if trader_id in enhanced_results:
                enhanced_score = enhanced_results[trader_id]['best_cv_score']

            ensemble_score = None
            if trader_id in ensemble_results:
                ensemble_score = ensemble_results[trader_id]['cv_score']

            comparison.append({
                'trader_id': trader_id,
                'original_xgb': original_val_score,
                'enhanced_best': enhanced_score,
                'ensemble': ensemble_score
            })

        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison)
        comparison_df.to_csv('outputs/reports/model_comparison.csv', index=False)

        # Print summary
        print("\nModel Performance Summary:")
        print(f"{'Trader':<8} {'Original':<10} {'Enhanced':<10} {'Ensemble':<10} {'Best':<10}")
        print("-" * 55)

        improvements = {'enhanced': 0, 'ensemble': 0, 'original': 0}

        for _, row in comparison_df.iterrows():
            trader_id = row['trader_id']
            original = row['original_xgb']
            enhanced = row['enhanced_best'] if pd.notna(row['enhanced_best']) else 0
            ensemble = row['ensemble'] if pd.notna(row['ensemble']) else 0

            best_score = max(original, enhanced, ensemble)
            if best_score == enhanced and enhanced > original:
                best_method = 'Enhanced'
                improvements['enhanced'] += 1
            elif best_score == ensemble and ensemble > original:
                best_method = 'Ensemble'
                improvements['ensemble'] += 1
            else:
                best_method = 'Original'
                improvements['original'] += 1

            print(f"{trader_id:<8} {original:<10.3f} {enhanced:<10.3f} {ensemble:<10.3f} {best_method:<10}")

        print(f"\nImprovement Summary:")
        print(f"  Enhanced models better: {improvements['enhanced']} traders")
        print(f"  Ensemble models better: {improvements['ensemble']} traders")
        print(f"  Original models better: {improvements['original']} traders")

        # Calculate average improvements
        avg_original = comparison_df['original_xgb'].mean()
        avg_enhanced = comparison_df['enhanced_best'].mean()
        avg_ensemble = comparison_df['ensemble'].mean()

        print(f"\nAverage Scores:")
        print(f"  Original XGBoost: {avg_original:.3f}")
        print(f"  Enhanced Models: {avg_enhanced:.3f}")
        print(f"  Ensemble Models: {avg_ensemble:.3f}")

        return comparison_df

def main():
    """Main function to run enhanced model training"""
    print("=" * 80)
    print("ENHANCED MODEL TRAINING & ENSEMBLE METHODS")
    print("=" * 80)

    trainer = EnhancedModelTraining()

    # Train enhanced models
    enhanced_results = trainer.train_enhanced_models()

    # Create ensemble models
    ensemble_results = trainer.create_ensemble_models()

    # Compare performance
    comparison = trainer.compare_model_performance()

    print("\n✅ ENHANCED MODEL TRAINING COMPLETE!")
    print(f"📊 Enhanced models trained for {len(enhanced_results)} traders")
    print(f"🔄 Ensemble models created for {len(ensemble_results)} traders")
    print(f"📈 Performance comparison saved to outputs/reports/model_comparison.csv")

if __name__ == "__main__":
    main()
