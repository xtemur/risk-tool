#!/usr/bin/env python3
"""
Target Variable Strategy
Migrated from step3_target_strategy.py
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class TargetVariableStrategy:
    def __init__(self, features_path='data/features_engineered.pkl'):
        self.feature_df = pd.read_pickle(features_path)
        self.target_results = {}
        self.best_target_strategy = None

    def prepare_data(self):
        """Prepare data for target variable testing"""
        print("=== TARGET VARIABLE STRATEGY ===")

        # Sort data
        self.feature_df = self.feature_df.sort_values(['account_id', 'trade_date'])

        # Remove non-feature columns for modeling
        feature_cols = [col for col in self.feature_df.columns
                       if col not in ['account_id', 'trade_date', 'realized_pnl']]

        print(f"✓ Prepared data with {len(feature_cols)} features")
        print(f"✓ Total observations: {len(self.feature_df)}")

        return feature_cols

    def create_target_option_a(self):
        """Option A: Raw PnL Prediction"""
        print("\\nTesting Option A: Raw PnL Prediction...")

        target_dfs = []

        for trader_id in self.feature_df['account_id'].unique():
            trader_df = self.feature_df[self.feature_df['account_id'] == trader_id].copy()
            trader_df = trader_df.sort_values('trade_date')

            # Create next-day PnL target
            trader_df['target_raw_pnl'] = trader_df['realized_pnl'].shift(-1)

            target_dfs.append(trader_df)

        target_df_a = pd.concat(target_dfs, ignore_index=True)

        # Remove rows with missing targets
        target_df_a = target_df_a.dropna(subset=['target_raw_pnl'])

        print(f"✓ Option A: {len(target_df_a)} observations with targets")

        return target_df_a, 'target_raw_pnl'

    def create_target_option_b(self):
        """Option B: Classification Approach (Loss/Neutral/Win)"""
        print("Testing Option B: Classification Approach...")

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

        target_df_b = pd.concat(target_dfs, ignore_index=True)

        # Remove rows with missing targets
        target_df_b = target_df_b.dropna(subset=['target_class'])

        print(f"✓ Option B: {len(target_df_b)} observations with targets")
        print(f"  - Loss class: {(target_df_b['target_class'] == 0).sum()}")
        print(f"  - Neutral class: {(target_df_b['target_class'] == 1).sum()}")
        print(f"  - Win class: {(target_df_b['target_class'] == 2).sum()}")

        return target_df_b, 'target_class'

    def create_target_option_c(self):
        """Option C: Volatility-Normalized Returns"""
        print("Testing Option C: Volatility-Normalized Returns...")

        target_dfs = []

        for trader_id in self.feature_df['account_id'].unique():
            trader_df = self.feature_df[self.feature_df['account_id'] == trader_id].copy()
            trader_df = trader_df.sort_values('trade_date')

            # Create next-day PnL
            trader_df['next_day_pnl'] = trader_df['realized_pnl'].shift(-1)

            # Calculate rolling volatility (use existing volatility_ewma20)
            if 'volatility_ewma20' in trader_df.columns:
                rolling_vol = trader_df['volatility_ewma20']
            else:
                rolling_vol = trader_df['realized_pnl'].rolling(window=20, min_periods=5).std()

            # Create volatility-normalized target
            trader_df['target_vol_norm'] = trader_df['next_day_pnl'] / np.maximum(rolling_vol, 1)

            # Cap extreme values
            trader_df['target_vol_norm'] = trader_df['target_vol_norm'].clip(-10, 10)

            target_dfs.append(trader_df)

        target_df_c = pd.concat(target_dfs, ignore_index=True)

        # Remove rows with missing targets
        target_df_c = target_df_c.dropna(subset=['target_vol_norm'])

        print(f"✓ Option C: {len(target_df_c)} observations with targets")

        return target_df_c, 'target_vol_norm'

    def create_target_option_d(self):
        """Option D: Risk-Adjusted Targets (Focus on downside risk)"""
        print("Testing Option D: Risk-Adjusted Targets (Downside Risk)...")

        target_dfs = []

        for trader_id in self.feature_df['account_id'].unique():
            trader_df = self.feature_df[self.feature_df['account_id'] == trader_id].copy()
            trader_df = trader_df.sort_values('trade_date')

            # Create next-day PnL
            trader_df['next_day_pnl'] = trader_df['realized_pnl'].shift(-1)

            # Focus on downside risk - binary target for large losses
            # Define "large loss" as bottom 10th percentile for this trader
            loss_threshold = trader_df['next_day_pnl'].quantile(0.10)

            trader_df['target_downside_risk'] = (trader_df['next_day_pnl'] < loss_threshold).astype(int)

            target_dfs.append(trader_df)

        target_df_d = pd.concat(target_dfs, ignore_index=True)

        # Remove rows with missing targets
        target_df_d = target_df_d.dropna(subset=['target_downside_risk'])

        print(f"✓ Option D: {len(target_df_d)} observations with targets")
        print(f"  - High risk days: {target_df_d['target_downside_risk'].sum()}")
        print(f"  - Normal days: {(target_df_d['target_downside_risk'] == 0).sum()}")

        return target_df_d, 'target_downside_risk'

    def evaluate_target_option(self, target_df, target_col, option_name):
        """Evaluate a target option using simple XGBoost models"""
        print(f"\\nEvaluating {option_name}...")

        # Select features
        feature_cols = [col for col in target_df.columns
                       if col not in ['account_id', 'trade_date', 'realized_pnl',
                                     'next_day_pnl', 'target_raw_pnl', 'target_class',
                                     'target_vol_norm', 'target_downside_risk']]

        results = []

        # Test on a sample of traders with sufficient data
        trader_counts = target_df.groupby('account_id').size()
        viable_traders = trader_counts[trader_counts >= 100].index[:10]  # Top 10 traders

        for trader_id in viable_traders:
            trader_data = target_df[target_df['account_id'] == trader_id].copy()
            trader_data = trader_data.sort_values('trade_date')

            if len(trader_data) < 100:
                continue

            # Prepare features and target
            X = trader_data[feature_cols].fillna(0)
            y = trader_data[target_col]

            # Ensure X is numeric numpy array
            X = X.select_dtypes(include=[np.number]).values
            y = y.values

            # Time series split for validation
            tscv = TimeSeriesSplit(n_splits=3)
            trader_scores = []

            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                try:
                    # Choose model based on target type
                    if target_col == 'target_class':
                        model = xgb.XGBClassifier(
                            objective='multi:softprob',
                            random_state=42,
                            n_estimators=100,
                            max_depth=3
                        )
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_val)
                        score = accuracy_score(y_val, y_pred)

                    elif target_col == 'target_downside_risk':
                        model = xgb.XGBClassifier(
                            objective='binary:logistic',
                            random_state=42,
                            n_estimators=100,
                            max_depth=3
                        )
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_val)
                        score = f1_score(y_val, y_pred)

                    else:  # Regression targets
                        model = xgb.XGBRegressor(
                            objective='reg:absoluteerror',
                            random_state=42,
                            n_estimators=100,
                            max_depth=3
                        )
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_val)
                        score = -mean_absolute_error(y_val, y_pred)  # Negative for "higher is better"

                    trader_scores.append(score)

                except Exception as e:
                    print(f"  Warning: Error training model for trader {trader_id}: {e}")
                    continue

            if trader_scores:
                results.append({
                    'trader_id': trader_id,
                    'avg_score': np.mean(trader_scores),
                    'std_score': np.std(trader_scores),
                    'num_folds': len(trader_scores)
                })

        # Summarize results
        if results:
            avg_scores = [r['avg_score'] for r in results]
            overall_score = np.mean(avg_scores)
            score_std = np.std(avg_scores)

            print(f"  ✓ Tested on {len(results)} traders")
            print(f"  ✓ Average score: {overall_score:.4f} ± {score_std:.4f}")

            return {
                'option_name': option_name,
                'target_column': target_col,
                'overall_score': overall_score,
                'score_std': score_std,
                'num_traders': len(results),
                'trader_results': results
            }
        else:
            print(f"  ❌ No valid results for {option_name}")
            return None

    def validate_target_predictability(self, target_df, target_col):
        """Validate that target is predictable and not purely random"""
        print(f"\\nValidating predictability for {target_col}...")

        # Test autocorrelation of target
        autocorr_results = []

        for trader_id in target_df['account_id'].unique()[:10]:
            trader_data = target_df[target_df['account_id'] == trader_id].copy()
            trader_data = trader_data.sort_values('trade_date')

            if len(trader_data) < 50:
                continue

            target_series = trader_data[target_col]

            # Calculate lag-1 autocorrelation
            autocorr = target_series.corr(target_series.shift(1))
            if not pd.isna(autocorr):
                autocorr_results.append(autocorr)

        avg_autocorr = np.mean(autocorr_results) if autocorr_results else 0

        # Check feature-target relationships
        feature_cols = [col for col in target_df.columns
                       if col not in ['account_id', 'trade_date', 'realized_pnl',
                                     'next_day_pnl', 'target_raw_pnl', 'target_class',
                                     'target_vol_norm', 'target_downside_risk']]

        # Sample correlation analysis
        sample_data = target_df.sample(min(10000, len(target_df)), random_state=42)

        strong_correlations = 0
        for feature in feature_cols[:20]:  # Test top 20 features
            corr = sample_data[feature].corr(sample_data[target_col])
            if abs(corr) > 0.05:  # Arbitrary threshold for "meaningful"
                strong_correlations += 1

        print(f"  ✓ Average autocorrelation: {avg_autocorr:.4f}")
        print(f"  ✓ Features with >0.05 correlation: {strong_correlations}/{min(20, len(feature_cols))}")

        # Predictability score (combination of low autocorr and feature relationships)
        predictability_score = strong_correlations / min(20, len(feature_cols)) - abs(avg_autocorr)

        return {
            'avg_autocorrelation': avg_autocorr,
            'strong_correlations': strong_correlations,
            'predictability_score': predictability_score
        }

    def compare_target_options(self):
        """Compare all target options and select the best"""
        print("\\n=== COMPARING TARGET OPTIONS ===")

        # Test all options
        options = []

        # Option A: Raw PnL
        try:
            target_df_a, target_col_a = self.create_target_option_a()
            result_a = self.evaluate_target_option(target_df_a, target_col_a, "Option A: Raw PnL")
            if result_a:
                predictability = self.validate_target_predictability(target_df_a, target_col_a)
                result_a['predictability'] = predictability
                options.append(result_a)
        except Exception as e:
            print(f"❌ Option A failed: {e}")

        # Option B: Classification
        try:
            target_df_b, target_col_b = self.create_target_option_b()
            result_b = self.evaluate_target_option(target_df_b, target_col_b, "Option B: Classification")
            if result_b:
                predictability = self.validate_target_predictability(target_df_b, target_col_b)
                result_b['predictability'] = predictability
                options.append(result_b)
        except Exception as e:
            print(f"❌ Option B failed: {e}")

        # Option C: Volatility-Normalized
        try:
            target_df_c, target_col_c = self.create_target_option_c()
            result_c = self.evaluate_target_option(target_df_c, target_col_c, "Option C: Vol-Normalized")
            if result_c:
                predictability = self.validate_target_predictability(target_df_c, target_col_c)
                result_c['predictability'] = predictability
                options.append(result_c)
        except Exception as e:
            print(f"❌ Option C failed: {e}")

        # Option D: Downside Risk
        try:
            target_df_d, target_col_d = self.create_target_option_d()
            result_d = self.evaluate_target_option(target_df_d, target_col_d, "Option D: Downside Risk")
            if result_d:
                predictability = self.validate_target_predictability(target_df_d, target_col_d)
                result_d['predictability'] = predictability
                options.append(result_d)
        except Exception as e:
            print(f"❌ Option D failed: {e}")

        # Select best option
        if options:
            # Score based on model performance and predictability
            for option in options:
                pred_score = option['predictability']['predictability_score']
                model_score = option['overall_score']

                # Combined score (weight predictability and model performance)
                option['combined_score'] = 0.6 * model_score + 0.4 * pred_score

            # Sort by combined score
            options.sort(key=lambda x: x['combined_score'], reverse=True)

            self.best_target_strategy = options[0]
            self.target_results = {opt['option_name']: opt for opt in options}

            print(f"\\n✅ BEST TARGET STRATEGY: {self.best_target_strategy['option_name']}")
            print(f"   Combined Score: {self.best_target_strategy['combined_score']:.4f}")
            print(f"   Model Performance: {self.best_target_strategy['overall_score']:.4f}")
            print(f"   Predictability: {self.best_target_strategy['predictability']['predictability_score']:.4f}")

            return self.best_target_strategy
        else:
            print("❌ No viable target strategies found")
            return None

    def prepare_final_target_data(self):
        """Prepare final dataset with the best target strategy"""
        if not self.best_target_strategy:
            raise ValueError("No target strategy selected")

        print(f"\\nPreparing final dataset with {self.best_target_strategy['option_name']}...")

        # Recreate the best target
        target_col = self.best_target_strategy['target_column']

        if target_col == 'target_raw_pnl':
            final_df, _ = self.create_target_option_a()
        elif target_col == 'target_class':
            final_df, _ = self.create_target_option_b()
        elif target_col == 'target_vol_norm':
            final_df, _ = self.create_target_option_c()
        elif target_col == 'target_downside_risk':
            final_df, _ = self.create_target_option_d()
        else:
            raise ValueError(f"Unknown target column: {target_col}")

        print(f"✓ Final dataset prepared with {len(final_df)} observations")

        return final_df, target_col
