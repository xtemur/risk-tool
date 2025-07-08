#!/usr/bin/env python3
"""
Test model integrity, feature engineering, and prediction quality
Checks for lookahead bias, feature validity, and model behavior
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
import sys
sys.path.append('/Users/temurbekkhujaev/Repos/risk-tool')

# Note: feature_engineering uses functions, not a class
from src.feature_engineering import build_features

class ModelIntegrityTester:
    """Test model and feature engineering integrity."""

    def __init__(self, base_path="/Users/temurbekkhujaev/Repos/risk-tool"):
        self.base_path = Path(base_path)
        self.models_path = self.base_path / "models" / "trader_specific"
        self.data_path = self.base_path / "data" / "processed" / "trader_splits"

        self.integrity_issues = []
        self.test_results = {}

    def run_all_tests(self):
        """Execute all model integrity tests."""
        print("="*60)
        print("MODEL AND FEATURE INTEGRITY TESTING")
        print("="*60)

        # 1. Test feature engineering for lookahead bias
        self.test_feature_lookahead_bias()

        # 2. Test model predictions distribution
        self.test_prediction_distributions()

        # 3. Test threshold optimization logic
        self.test_threshold_optimization()

        # 4. Test model complexity vs data size
        self.test_model_complexity()

        # 5. Generate report
        self.generate_integrity_report()

    def test_feature_lookahead_bias(self):
        """Test if features properly avoid lookahead bias."""
        print("\n1. TESTING FEATURE ENGINEERING FOR LOOKAHEAD BIAS")
        print("-" * 40)

        # Load sample data
        trader_ids = ['3950', '3956', '4396']  # Top performers to check

        for trader_id in trader_ids:
            train_path = self.data_path / trader_id / "train_data.parquet"

            if train_path.exists():
                train_data = pd.read_parquet(train_path)

                # Check feature columns in the data
                feature_cols = [col for col in train_data.columns if col not in
                               ['trader_id', 'date', 'daily_pnl', 'cumulative_pnl']]

                print(f"Trader {trader_id}: Found {len(feature_cols)} feature columns")

                # Check for common lookahead indicators
                risky_patterns = ['future', 'next', 'forward', 'lead']
                for col in feature_cols:
                    for pattern in risky_patterns:
                        if pattern in col.lower():
                            self.integrity_issues.append(
                                f"Trader {trader_id}: Suspicious feature name '{col}' contains '{pattern}'"
                            )

                # Check if lagged features exist
                lag_features = [col for col in feature_cols if 'lag' in col.lower() or 'ma_' in col]
                if len(lag_features) == 0:
                    self.integrity_issues.append(
                        f"Trader {trader_id}: No lagged features found - possible lookahead bias"
                    )

    def test_prediction_distributions(self):
        """Test if model predictions show realistic distributions."""
        print("\n2. TESTING MODEL PREDICTION DISTRIBUTIONS")
        print("-" * 40)

        # Load test evaluation results
        eval_path = self.models_path / "test_evaluation_results.pkl"

        if eval_path.exists():
            with open(eval_path, 'rb') as f:
                eval_results = pickle.load(f)

            for trader_id, results in eval_results.items():
                if 'predictions' in results:
                    preds = results['predictions']

                    # Check VaR predictions
                    var_preds = preds.get('var_predictions', [])
                    if len(var_preds) > 0:
                        var_array = np.array(var_preds)

                        # Check if all predictions are negative (VaR should be negative)
                        positive_vars = np.sum(var_array > 0)
                        if positive_vars > 0:
                            self.integrity_issues.append(
                                f"Trader {trader_id}: {positive_vars} positive VaR predictions (should be negative)"
                            )

                        # Check prediction variance
                        var_std = np.std(var_array)
                        var_mean = np.mean(var_array)
                        cv = abs(var_std / var_mean) if var_mean != 0 else 0

                        if cv < 0.1:  # Very low variation
                            self.integrity_issues.append(
                                f"Trader {trader_id}: VaR predictions show low variance (CV={cv:.3f})"
                            )

                    # Check loss probabilities
                    loss_probs = preds.get('loss_probabilities', [])
                    if len(loss_probs) > 0:
                        prob_array = np.array(loss_probs)

                        # Check if probabilities are in [0, 1]
                        invalid_probs = np.sum((prob_array < 0) | (prob_array > 1))
                        if invalid_probs > 0:
                            self.integrity_issues.append(
                                f"Trader {trader_id}: {invalid_probs} invalid probability predictions"
                            )

                        # Check if all probabilities are extreme (near 0 or 1)
                        extreme_probs = np.sum((prob_array < 0.1) | (prob_array > 0.9))
                        extreme_ratio = extreme_probs / len(prob_array)

                        if extreme_ratio > 0.8:
                            self.integrity_issues.append(
                                f"Trader {trader_id}: {extreme_ratio:.1%} extreme probability predictions"
                            )

                    self.test_results[trader_id] = {
                        'var_predictions_count': len(var_preds),
                        'loss_prob_count': len(loss_probs)
                    }

    def test_threshold_optimization(self):
        """Test if threshold optimization makes sense."""
        print("\n3. TESTING THRESHOLD OPTIMIZATION")
        print("-" * 40)

        # Load optimal thresholds
        threshold_path = self.base_path / "configs" / "optimal_thresholds" / "optimal_thresholds.json"

        with open(threshold_path, 'r') as f:
            threshold_data = json.load(f)

        thresholds = threshold_data['thresholds']

        # Check for suspicious patterns
        var_thresholds = [t['var_threshold'] for t in thresholds]
        prob_thresholds = [t['loss_prob_threshold'] for t in thresholds]

        # Check 1: VaR thresholds should be negative
        positive_var_thresholds = sum(1 for v in var_thresholds if v > 0)
        if positive_var_thresholds > 0:
            self.integrity_issues.append(
                f"{positive_var_thresholds} traders have positive VaR thresholds"
            )

        # Check 2: Loss probability thresholds should be in [0, 1]
        invalid_prob_thresholds = sum(1 for p in prob_thresholds if p < 0 or p > 1)
        if invalid_prob_thresholds > 0:
            self.integrity_issues.append(
                f"{invalid_prob_thresholds} traders have invalid probability thresholds"
            )

        # Check 3: Look for extreme thresholds
        extreme_var = sum(1 for v in var_thresholds if abs(v) > 50000)
        extreme_prob = sum(1 for p in prob_thresholds if p < 0.05 or p > 0.95)

        print(f"VaR thresholds range: [{min(var_thresholds):.2f}, {max(var_thresholds):.2f}]")
        print(f"Prob thresholds range: [{min(prob_thresholds):.3f}, {max(prob_thresholds):.3f}]")
        print(f"Extreme thresholds: {extreme_var} VaR, {extreme_prob} probability")

        # Check 4: Correlation between thresholds and intervention rates
        # If thresholds are optimized properly, traders with extreme thresholds
        # should have different intervention rates

    def test_model_complexity(self):
        """Test if model complexity is appropriate for data size."""
        print("\n4. TESTING MODEL COMPLEXITY VS DATA SIZE")
        print("-" * 40)

        # Load training summary
        summary_path = self.models_path / "training_summary.pkl"

        if summary_path.exists():
            with open(summary_path, 'rb') as f:
                training_summary = pickle.load(f)

            for trader_id, summary in training_summary.items():
                # Get model parameters
                var_params = summary.get('var_best_params', {})
                class_params = summary.get('classification_best_params', {})

                # Get data size
                metadata_path = self.data_path / trader_id / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)

                    train_days = metadata.get('train_days', 0)
                    n_features = len(summary.get('feature_names', []))

                    # Check complexity indicators
                    var_n_estimators = var_params.get('n_estimators', 100)
                    var_max_depth = var_params.get('max_depth', -1)

                    # Rule of thumb: need at least 10 samples per feature
                    samples_per_feature = train_days / n_features if n_features > 0 else 0

                    if samples_per_feature < 10:
                        self.integrity_issues.append(
                            f"Trader {trader_id}: Only {samples_per_feature:.1f} samples per feature"
                        )

                    # Check if model is too complex
                    if var_n_estimators > train_days:
                        self.integrity_issues.append(
                            f"Trader {trader_id}: More trees ({var_n_estimators}) than training samples ({train_days})"
                        )

                    self.test_results[f"{trader_id}_complexity"] = {
                        'train_days': train_days,
                        'n_features': n_features,
                        'samples_per_feature': samples_per_feature,
                        'n_estimators': var_n_estimators,
                        'max_depth': var_max_depth
                    }

    def generate_integrity_report(self):
        """Generate final integrity report."""
        print("\n" + "="*60)
        print("MODEL INTEGRITY TEST SUMMARY")
        print("="*60)

        if not self.integrity_issues:
            print("✅ No major integrity issues found")
        else:
            print(f"❌ Found {len(self.integrity_issues)} integrity issues:")
            for i, issue in enumerate(self.integrity_issues[:15], 1):
                print(f"{i}. {issue}")

        print("\nKEY OBSERVATIONS:")

        # Summarize findings
        lookahead_issues = sum(1 for i in self.integrity_issues if 'insufficient data' in i)
        prediction_issues = sum(1 for i in self.integrity_issues if 'prediction' in i)
        complexity_issues = sum(1 for i in self.integrity_issues if 'samples per feature' in i)

        print(f"1. Lookahead bias issues: {lookahead_issues}")
        print(f"2. Prediction distribution issues: {prediction_issues}")
        print(f"3. Model complexity issues: {complexity_issues}")

        print("\nRECOMMENDATIONS:")
        if complexity_issues > 0:
            print("- Models are likely overfitted due to high complexity relative to data size")
        if prediction_issues > 0:
            print("- Prediction distributions suggest possible overfitting or calibration issues")
        if lookahead_issues > 0:
            print("- Some features may be using insufficient historical data")

        # Save results
        results = {
            "integrity_issues": self.integrity_issues,
            "test_results": self.test_results,
            "summary": {
                "total_issues": len(self.integrity_issues),
                "lookahead_issues": lookahead_issues,
                "prediction_issues": prediction_issues,
                "complexity_issues": complexity_issues
            }
        }

        with open(self.base_path / "tests" / "model_integrity_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\nDetailed results saved to: tests/model_integrity_results.json")


if __name__ == "__main__":
    tester = ModelIntegrityTester()
    tester.run_all_tests()
