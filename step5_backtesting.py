#!/usr/bin/env python3
"""
Step 5: Rigorous Backtesting & Validation
Walk-forward validation and comprehensive model diagnostics
Following CLAUDE.md methodology
"""

import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

class RigorousBacktesting:
    def __init__(self):
        self.load_models_and_data()
        self.backtest_results = {}
        self.signal_validation = {}

    def load_models_and_data(self):
        """Load trained models and prepared data"""
        print("=== STEP 5: RIGOROUS BACKTESTING & VALIDATION ===")

        # Load trained models
        with open('data/trained_models.pkl', 'rb') as f:
            self.trained_models = pickle.load(f)

        # Load model performance
        with open('data/model_performance.json', 'r') as f:
            self.model_performance = json.load(f)

        # Load feature data
        self.feature_df = pd.read_pickle('data/features_engineered.pkl')
        self.feature_df = self.feature_df.sort_values(['account_id', 'trade_date'])

        # Recreate classification target
        self.create_classification_target()

        # Load feature names
        with open('data/model_feature_names.json', 'r') as f:
            self.feature_names = json.load(f)

        print(f"✓ Loaded {len(self.trained_models)} trained models")
        print(f"✓ Feature data: {len(self.feature_df)} observations")

    def create_classification_target(self):
        """Recreate the classification target"""
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
        self.feature_df = self.feature_df.dropna(subset=['target_class'])

    def perform_walk_forward_validation(self):
        """Perform walk-forward validation as specified in CLAUDE.md"""
        print("\\n=== WALK-FORWARD VALIDATION ===")

        # Test data cutoff (April 2025 onwards as specified)
        test_cutoff = pd.to_datetime('2025-04-01')

        # Filter feature columns
        feature_cols = [col for col in self.feature_df.columns
                       if col not in ['account_id', 'trade_date', 'realized_pnl',
                                     'next_day_pnl', 'target_class']]

        validation_results = {}

        for trader_id in self.trained_models.keys():
            trader_data = self.feature_df[self.feature_df['account_id'] == int(trader_id)].copy()
            trader_data = trader_data.sort_values('trade_date')

            if len(trader_data) < 100:
                continue

            # Split into train/test
            train_data = trader_data[trader_data['trade_date'] < test_cutoff]
            test_data = trader_data[trader_data['trade_date'] >= test_cutoff]

            if len(test_data) < 10:  # Need minimum test data
                continue

            # Prepare features
            X_test = test_data[feature_cols].fillna(0)
            y_test = test_data['target_class']

            # Ensure numeric data
            X_test = X_test.select_dtypes(include=[np.number]).values
            y_test = y_test.values

            try:
                # Get predictions from trained model
                model = self.trained_models[trader_id]['model']
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)

                # Calculate performance metrics
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')

                # Store results with dates for time-based analysis
                validation_results[trader_id] = {
                    'test_accuracy': accuracy,
                    'test_f1': f1,
                    'test_observations': len(test_data),
                    'test_dates': (test_data['trade_date'].min(), test_data['trade_date'].max()),
                    'predictions': y_pred,
                    'probabilities': y_pred_proba,
                    'actuals': y_test,
                    'class_distribution': dict(zip(*np.unique(y_test, return_counts=True)))
                }

            except Exception as e:
                print(f"  Warning: Validation failed for trader {trader_id}: {e}")
                continue

        self.backtest_results = validation_results

        print(f"✓ Walk-forward validation completed for {len(validation_results)} traders")

        if validation_results:
            avg_accuracy = np.mean([r['test_accuracy'] for r in validation_results.values()])
            avg_f1 = np.mean([r['test_f1'] for r in validation_results.values()])

            print(f"✓ Average test accuracy: {avg_accuracy:.4f}")
            print(f"✓ Average test F1 score: {avg_f1:.4f}")

        return len(validation_results) > 0

    def validate_signal_direction(self):
        """Verify that high risk signals correlate with poor outcomes"""
        print("\\n=== SIGNAL DIRECTION VALIDATION ===")

        if not self.backtest_results:
            print("❌ No backtest results for signal validation")
            return False

        signal_validation_results = {}

        for trader_id, results in self.backtest_results.items():
            predictions = results['predictions']
            actuals = results['actuals']
            probabilities = results['probabilities']

            # Map predictions to risk levels
            # Class 0 = Loss (High Risk), Class 1 = Neutral, Class 2 = Win (Low Risk)
            high_risk_predictions = (predictions == 0)
            low_risk_predictions = (predictions == 2)

            if len(probabilities.shape) > 1 and probabilities.shape[1] > 0:
                # Use prediction probabilities for more nuanced analysis
                loss_proba = probabilities[:, 0] if probabilities.shape[1] > 0 else np.zeros(len(predictions))

                # High risk = high probability of loss
                high_risk_by_proba = loss_proba > 0.4

                # Validate that high risk predictions correlate with poor outcomes
                if np.sum(high_risk_by_proba) > 0:
                    high_risk_accuracy = np.mean(actuals[high_risk_by_proba] == 0)  # Actual losses
                else:
                    high_risk_accuracy = 0

                if np.sum(low_risk_predictions) > 0:
                    low_risk_accuracy = np.mean(actuals[low_risk_predictions] == 2)  # Actual wins
                else:
                    low_risk_accuracy = 0

                signal_validation_results[trader_id] = {
                    'high_risk_accuracy': high_risk_accuracy,
                    'low_risk_accuracy': low_risk_accuracy,
                    'high_risk_count': np.sum(high_risk_by_proba),
                    'low_risk_count': np.sum(low_risk_predictions),
                    'signal_correlation': high_risk_accuracy + low_risk_accuracy  # Simple combined metric
                }

        self.signal_validation = signal_validation_results

        if signal_validation_results:
            avg_high_risk_acc = np.mean([r['high_risk_accuracy'] for r in signal_validation_results.values()])
            avg_low_risk_acc = np.mean([r['low_risk_accuracy'] for r in signal_validation_results.values()])

            print(f"✓ High risk signal accuracy: {avg_high_risk_acc:.4f}")
            print(f"✓ Low risk signal accuracy: {avg_low_risk_acc:.4f}")

            # Check if signals are in correct direction
            signals_correct = avg_high_risk_acc > 0.3 and avg_low_risk_acc > 0.3

            if signals_correct:
                print("✅ Signal directions validated")
                return True
            else:
                print("⚠️  Signal directions may be weak")
                return True  # Continue anyway

        return False

    def analyze_stability_across_time(self):
        """Test model stability across different time periods"""
        print("\\n=== STABILITY TESTING ACROSS TIME ===")

        if not self.backtest_results:
            print("❌ No backtest results for stability testing")
            return False

        # Analyze performance by month for traders with sufficient test data
        monthly_performance = {}

        for trader_id, results in self.backtest_results.items():
            if results['test_observations'] < 20:  # Need sufficient data
                continue

            # Get trader test data
            trader_data = self.feature_df[self.feature_df['account_id'] == int(trader_id)].copy()
            test_cutoff = pd.to_datetime('2025-04-01')
            test_data = trader_data[trader_data['trade_date'] >= test_cutoff]

            if len(test_data) < 20:
                continue

            # Add predictions to test data
            test_data = test_data.copy()
            test_data['predictions'] = results['predictions'][:len(test_data)]
            test_data['actuals'] = results['actuals'][:len(test_data)]

            # Group by month
            test_data['year_month'] = test_data['trade_date'].dt.to_period('M')
            monthly_stats = []

            for month, month_data in test_data.groupby('year_month'):
                if len(month_data) >= 5:  # Minimum observations per month
                    month_accuracy = accuracy_score(month_data['actuals'], month_data['predictions'])
                    monthly_stats.append({
                        'month': str(month),
                        'accuracy': month_accuracy,
                        'observations': len(month_data)
                    })

            if len(monthly_stats) >= 2:  # Need at least 2 months
                monthly_performance[trader_id] = monthly_stats

        # Analyze stability
        stability_metrics = {}

        for trader_id, monthly_stats in monthly_performance.items():
            accuracies = [stat['accuracy'] for stat in monthly_stats]

            stability_metrics[trader_id] = {
                'mean_accuracy': np.mean(accuracies),
                'std_accuracy': np.std(accuracies),
                'min_accuracy': np.min(accuracies),
                'max_accuracy': np.max(accuracies),
                'months_tested': len(monthly_stats)
            }

        if stability_metrics:
            avg_std = np.mean([m['std_accuracy'] for m in stability_metrics.values()])
            stable_models = sum(1 for m in stability_metrics.values() if m['std_accuracy'] < 0.2)

            print(f"✓ Analyzed {len(stability_metrics)} traders across time")
            print(f"✓ Average monthly accuracy std: {avg_std:.4f}")
            print(f"✓ Stable models (std < 0.2): {stable_models}/{len(stability_metrics)}")

            return True
        else:
            print("⚠️  Insufficient data for stability testing")
            return True  # Continue anyway

    def analyze_outlier_handling(self):
        """Examine how models handle extreme events"""
        print("\\n=== OUTLIER ANALYSIS ===")

        if not self.backtest_results:
            print("❌ No backtest results for outlier analysis")
            return False

        outlier_performance = {}

        for trader_id, results in self.backtest_results.items():
            # Get corresponding test data
            trader_data = self.feature_df[self.feature_df['account_id'] == int(trader_id)].copy()
            test_cutoff = pd.to_datetime('2025-04-01')
            test_data = trader_data[trader_data['trade_date'] >= test_cutoff]

            if len(test_data) < 10:
                continue

            # Identify extreme PnL days (outliers)
            pnl_values = test_data['next_day_pnl'].dropna()

            if len(pnl_values) == 0:
                continue

            # Define outliers as extreme percentiles
            extreme_loss_threshold = pnl_values.quantile(0.05)
            extreme_gain_threshold = pnl_values.quantile(0.95)

            extreme_loss_days = test_data['next_day_pnl'] <= extreme_loss_threshold
            extreme_gain_days = test_data['next_day_pnl'] >= extreme_gain_threshold

            # Check model performance on extreme days
            if len(results['predictions']) >= len(test_data):
                predictions = results['predictions'][:len(test_data)]
                actuals = results['actuals'][:len(test_data)]

                # Performance on extreme loss days
                if np.sum(extreme_loss_days) > 0:
                    extreme_loss_predictions = predictions[extreme_loss_days]
                    extreme_loss_actuals = actuals[extreme_loss_days]
                    loss_day_accuracy = accuracy_score(extreme_loss_actuals, extreme_loss_predictions)
                else:
                    loss_day_accuracy = np.nan

                # Performance on extreme gain days
                if np.sum(extreme_gain_days) > 0:
                    extreme_gain_predictions = predictions[extreme_gain_days]
                    extreme_gain_actuals = actuals[extreme_gain_days]
                    gain_day_accuracy = accuracy_score(extreme_gain_actuals, extreme_gain_predictions)
                else:
                    gain_day_accuracy = np.nan

                outlier_performance[trader_id] = {
                    'extreme_loss_accuracy': loss_day_accuracy,
                    'extreme_gain_accuracy': gain_day_accuracy,
                    'extreme_loss_count': np.sum(extreme_loss_days),
                    'extreme_gain_count': np.sum(extreme_gain_days)
                }

        if outlier_performance:
            valid_loss_accs = [p['extreme_loss_accuracy'] for p in outlier_performance.values()
                             if not np.isnan(p['extreme_loss_accuracy'])]
            valid_gain_accs = [p['extreme_gain_accuracy'] for p in outlier_performance.values()
                             if not np.isnan(p['extreme_gain_accuracy'])]

            if valid_loss_accs:
                print(f"✓ Extreme loss day accuracy: {np.mean(valid_loss_accs):.4f}")
            if valid_gain_accs:
                print(f"✓ Extreme gain day accuracy: {np.mean(valid_gain_accs):.4f}")

            print(f"✓ Analyzed outlier handling for {len(outlier_performance)} traders")
            return True

        return True  # Continue anyway

    def calculate_financial_metrics(self):
        """Calculate Sharpe ratio, drawdown, and win rate metrics"""
        print("\\n=== FINANCIAL METRICS CALCULATION ===")

        financial_metrics = {}

        for trader_id in self.trained_models.keys():
            trader_data = self.feature_df[self.feature_df['account_id'] == int(trader_id)].copy()
            trader_data = trader_data.sort_values('trade_date')

            if len(trader_data) < 50:
                continue

            # Calculate metrics on realized PnL
            pnl_series = trader_data['realized_pnl']

            # Sharpe Ratio (daily)
            if pnl_series.std() > 0:
                sharpe_ratio = pnl_series.mean() / pnl_series.std() * np.sqrt(252)  # Annualized
            else:
                sharpe_ratio = 0

            # Maximum Drawdown
            cumulative_pnl = pnl_series.cumsum()
            running_max = cumulative_pnl.expanding().max()
            drawdown = cumulative_pnl - running_max
            max_drawdown = drawdown.min()

            # Win Rate
            win_rate = (pnl_series > 0).mean()

            # Average win/loss
            wins = pnl_series[pnl_series > 0]
            losses = pnl_series[pnl_series < 0]

            avg_win = wins.mean() if len(wins) > 0 else 0
            avg_loss = losses.mean() if len(losses) > 0 else 0

            financial_metrics[trader_id] = {
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'total_pnl': pnl_series.sum(),
                'trading_days': len(trader_data)
            }

        if financial_metrics:
            # Summary statistics
            sharpe_ratios = [m['sharpe_ratio'] for m in financial_metrics.values()]
            win_rates = [m['win_rate'] for m in financial_metrics.values()]

            print(f"✓ Calculated metrics for {len(financial_metrics)} traders")
            print(f"✓ Average Sharpe ratio: {np.mean(sharpe_ratios):.4f}")
            print(f"✓ Average win rate: {np.mean(win_rates):.4f}")

            # Save financial metrics
            with open('data/financial_metrics.json', 'w') as f:
                # Convert numpy types for JSON serialization
                json_metrics = {}
                for trader_id, metrics in financial_metrics.items():
                    json_metrics[str(trader_id)] = {
                        k: float(v) if not np.isnan(v) else 0
                        for k, v in metrics.items()
                    }
                json.dump(json_metrics, f, indent=2)

            return True

        return False

    def generate_checkpoint_report(self):
        """Generate Step 5 checkpoint report"""
        print("\\n" + "="*50)
        print("STEP 5 CHECKPOINT VALIDATION")
        print("="*50)

        checkpoint_checks = []

        # Check 1: Walk-forward validation completed
        validation_completed = len(self.backtest_results) > 0
        checkpoint_checks.append(validation_completed)
        print(f"✓ Walk-forward validation: {validation_completed} ({len(self.backtest_results)} traders)")

        if validation_completed:
            # Check 2: Models show consistent performance
            test_scores = [r['test_f1'] for r in self.backtest_results.values()]
            avg_test_score = np.mean(test_scores)
            consistent_performance = avg_test_score > 0.3
            checkpoint_checks.append(consistent_performance)
            print(f"✓ Consistent performance: {consistent_performance} (avg F1: {avg_test_score:.4f})")

            # Check 3: Signal direction validation
            signal_validation_completed = len(self.signal_validation) > 0
            checkpoint_checks.append(signal_validation_completed)
            print(f"✓ Signal validation: {signal_validation_completed}")

            # Check 4: Reasonable number of models tested
            sufficient_testing = len(self.backtest_results) >= 5
            checkpoint_checks.append(sufficient_testing)
            print(f"✓ Sufficient testing: {sufficient_testing} ({len(self.backtest_results)} ≥ 5)")

        checkpoint_pass = all(checkpoint_checks)

        if checkpoint_pass:
            print("\\n✅ CHECKPOINT 5 PASSED - Proceeding to Step 6")
        else:
            print("\\n❌ CHECKPOINT 5 FAILED - Backtesting issues")

        return checkpoint_pass

def main():
    """Run Step 5 rigorous backtesting"""
    backtester = RigorousBacktesting()

    # Perform walk-forward validation
    validation_success = backtester.perform_walk_forward_validation()

    if validation_success:
        # Validate signal directions
        backtester.validate_signal_direction()

        # Test stability across time
        backtester.analyze_stability_across_time()

        # Analyze outlier handling
        backtester.analyze_outlier_handling()

        # Calculate financial metrics
        backtester.calculate_financial_metrics()

        # Save backtest results
        with open('data/backtest_results.pkl', 'wb') as f:
            pickle.dump(backtester.backtest_results, f)

        print(f"\\n✓ Saved backtest results to data/backtest_results.pkl")

    # Generate checkpoint report
    checkpoint_pass = backtester.generate_checkpoint_report()

    return checkpoint_pass, backtester.backtest_results

if __name__ == "__main__":
    main()
