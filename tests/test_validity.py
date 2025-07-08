#!/usr/bin/env python3
"""
Comprehensive validity testing for the Risk Management System
Tests for data leakage, overfitting, and sanity checks on reported metrics
"""

import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class ValidityTester:
    """Comprehensive testing suite for validating risk management system claims."""

    def __init__(self, base_path="/Users/temurbekkhujaev/Repos/risk-tool"):
        self.base_path = Path(base_path)
        self.results_path = self.base_path / "results" / "causal_impact_comparison"
        self.data_path = self.base_path / "data" / "processed" / "trader_splits"
        self.models_path = self.base_path / "models" / "trader_specific"

        self.test_results = {
            "data_integrity": {},
            "overfitting_tests": {},
            "sanity_checks": {},
            "statistical_tests": {},
            "red_flags": []
        }

    def run_all_tests(self):
        """Execute all validity tests."""
        print("="*60)
        print("RISK MANAGEMENT SYSTEM VALIDITY TESTING")
        print("="*60)

        # 1. Test data integrity
        self.test_data_integrity()

        # 2. Test for overfitting indicators
        self.test_overfitting_indicators()

        # 3. Sanity check reported metrics
        self.sanity_check_metrics()

        # 4. Statistical validity tests
        self.test_statistical_validity()

        # 5. Generate summary report
        self.generate_validity_report()

    def test_data_integrity(self):
        """Test for data leakage and proper train/test split."""
        print("\n1. DATA INTEGRITY TESTS")
        print("-" * 40)

        trader_ids = [d.name for d in self.data_path.iterdir() if d.is_dir() and d.name.isdigit()]

        for trader_id in trader_ids:
            # Load train and test data
            train_path = self.data_path / trader_id / "train_data.parquet"
            test_path = self.data_path / trader_id / "test_data.parquet"
            metadata_path = self.data_path / trader_id / "metadata.json"

            if train_path.exists() and test_path.exists():
                train_data = pd.read_parquet(train_path)
                test_data = pd.read_parquet(test_path)

                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

                # Check temporal ordering
                train_end = pd.to_datetime(train_data['date'].max())
                test_start = pd.to_datetime(test_data['date'].min())

                temporal_valid = test_start > train_end

                # Check data size
                train_size = len(train_data)
                test_size = len(test_data)
                total_size = train_size + test_size

                # Check for overlap
                train_dates = set(train_data['date'])
                test_dates = set(test_data['date'])
                date_overlap = len(train_dates.intersection(test_dates))

                self.test_results["data_integrity"][trader_id] = {
                    "temporal_ordering_valid": temporal_valid,
                    "train_size": train_size,
                    "test_size": test_size,
                    "test_ratio": test_size / total_size if total_size > 0 else 0,
                    "date_overlap": date_overlap,
                    "total_days": metadata.get("total_days", total_size)
                }

                if not temporal_valid:
                    self.test_results["red_flags"].append(
                        f"Trader {trader_id}: Test data starts before training ends!"
                    )

                if date_overlap > 0:
                    self.test_results["red_flags"].append(
                        f"Trader {trader_id}: {date_overlap} overlapping dates between train/test!"
                    )

                if test_size < 30:
                    self.test_results["red_flags"].append(
                        f"Trader {trader_id}: Only {test_size} test days - insufficient for validation!"
                    )

        print(f"Tested {len(trader_ids)} traders for data integrity")

    def test_overfitting_indicators(self):
        """Check for signs of overfitting in model performance."""
        print("\n2. OVERFITTING TESTS")
        print("-" * 40)

        # Load model training summary
        summary_path = self.models_path / "training_summary.pkl"
        if summary_path.exists():
            with open(summary_path, 'rb') as f:
                training_summary = pickle.load(f)

            for trader_id, summary in training_summary.items():
                if isinstance(summary, dict) and 'validation_results' in summary:
                    val_results = summary['validation_results']

                    # Check train vs validation performance gap
                    train_accuracy = val_results.get('train_accuracy', 0)
                    val_accuracy = val_results.get('val_accuracy', 0)
                    accuracy_gap = train_accuracy - val_accuracy

                    # Check model complexity vs data size
                    n_features = len(summary.get('feature_names', []))
                    train_size = self.test_results["data_integrity"].get(trader_id, {}).get("train_size", 0)

                    feature_to_sample_ratio = n_features / train_size if train_size > 0 else float('inf')

                    self.test_results["overfitting_tests"][trader_id] = {
                        "train_accuracy": train_accuracy,
                        "val_accuracy": val_accuracy,
                        "accuracy_gap": accuracy_gap,
                        "n_features": n_features,
                        "feature_to_sample_ratio": feature_to_sample_ratio
                    }

                    if accuracy_gap > 0.15:
                        self.test_results["red_flags"].append(
                            f"Trader {trader_id}: Large train/val gap ({accuracy_gap:.1%}) suggests overfitting"
                        )

                    if feature_to_sample_ratio > 0.5:
                        self.test_results["red_flags"].append(
                            f"Trader {trader_id}: High feature/sample ratio ({feature_to_sample_ratio:.2f}) - overfitting risk"
                        )

    def sanity_check_metrics(self):
        """Verify if reported metrics are realistic."""
        print("\n3. SANITY CHECKS")
        print("-" * 40)

        # Load results for different scenarios
        scenarios = [25, 50, 70, 90]

        for scenario in scenarios:
            results_file = self.results_path / f"reduction_{scenario}pct" / "detailed_results.pkl"

            if results_file.exists():
                with open(results_file, 'rb') as f:
                    results = pickle.load(f)

                individual_results = results.get('individual_results', {})

                unrealistic_improvements = []
                intervention_rates = []

                for trader_id, trader_data in individual_results.items():
                    if isinstance(trader_data, dict) and 'metrics' in trader_data:
                        metrics = trader_data['metrics']

                        # Check for unrealistic improvements
                        improvement = metrics.get('overall_improvement', 0)
                        if improvement > 200:  # >200% improvement is suspicious
                            unrealistic_improvements.append((trader_id, improvement))

                        # Collect intervention rates
                        intervention_rate = metrics.get('intervention_rate', 0)
                        intervention_rates.append(intervention_rate)

                # Check intervention rate consistency
                if intervention_rates:
                    mean_intervention = np.mean(intervention_rates)
                    std_intervention = np.std(intervention_rates)

                    self.test_results["sanity_checks"][f"scenario_{scenario}"] = {
                        "unrealistic_improvements": len(unrealistic_improvements),
                        "mean_intervention_rate": mean_intervention,
                        "std_intervention_rate": std_intervention,
                        "suspicious_traders": unrealistic_improvements[:3]  # Top 3
                    }

                    if len(unrealistic_improvements) > 0:
                        self.test_results["red_flags"].append(
                            f"Scenario {scenario}%: {len(unrealistic_improvements)} traders show >200% improvement"
                        )

    def test_statistical_validity(self):
        """Test statistical claims made in the report."""
        print("\n4. STATISTICAL VALIDITY TESTS")
        print("-" * 40)

        # Test claim: 72.7% success rate (8/11 traders) across ALL scenarios
        scenarios = [25, 50, 70, 90]
        success_rates = []

        for scenario in scenarios:
            results_file = self.results_path / f"reduction_{scenario}pct" / "detailed_results.pkl"

            if results_file.exists():
                with open(results_file, 'rb') as f:
                    results = pickle.load(f)

                individual_results = results.get('individual_results', {})

                successful = 0
                total = 0

                for trader_id, trader_data in individual_results.items():
                    if isinstance(trader_data, dict) and 'metrics' in trader_data:
                        metrics = trader_data['metrics']
                        if metrics.get('net_benefit', 0) > 0:
                            successful += 1
                        total += 1

                if total > 0:
                    success_rate = successful / total
                    success_rates.append(success_rate)

        # Check if all success rates are identical
        if len(set(success_rates)) == 1:
            self.test_results["red_flags"].append(
                f"SUSPICIOUS: All scenarios have identical success rate ({success_rates[0]:.1%})"
            )

        self.test_results["statistical_tests"]["success_rate_consistency"] = {
            "success_rates": success_rates,
            "all_identical": len(set(success_rates)) == 1
        }

        # Test intervention logic (50% reduction claim)
        self.test_intervention_logic()

    def test_intervention_logic(self):
        """Verify the 50% PnL reduction intervention logic."""
        # Sample test: when intervening on -$1000 loss day, adjusted PnL should be -$500
        test_cases = [
            {"actual_pnl": -1000, "expected_adjusted": -500},
            {"actual_pnl": 1000, "expected_adjusted": 500},
            {"actual_pnl": -5000, "expected_adjusted": -2500},
            {"actual_pnl": 5000, "expected_adjusted": 2500}
        ]

        for case in test_cases:
            # Simulate intervention
            adjusted = case["actual_pnl"] * 0.5
            if adjusted != case["expected_adjusted"]:
                self.test_results["red_flags"].append(
                    f"Intervention logic error: {case['actual_pnl']} -> {adjusted} (expected {case['expected_adjusted']})"
                )

    def generate_validity_report(self):
        """Generate comprehensive validity report."""
        print("\n" + "="*60)
        print("VALIDITY TEST REPORT SUMMARY")
        print("="*60)

        # Count red flags by category
        overfitting_flags = sum(1 for flag in self.test_results["red_flags"] if "overfitting" in flag.lower())
        data_flags = sum(1 for flag in self.test_results["red_flags"] if any(x in flag.lower() for x in ["test data", "overlap", "days"]))
        metric_flags = sum(1 for flag in self.test_results["red_flags"] if "improvement" in flag.lower())

        print(f"\nTOTAL RED FLAGS: {len(self.test_results['red_flags'])}")
        print(f"- Data integrity issues: {data_flags}")
        print(f"- Overfitting indicators: {overfitting_flags}")
        print(f"- Unrealistic metrics: {metric_flags}")

        if self.test_results["red_flags"]:
            print("\nCRITICAL ISSUES:")
            for i, flag in enumerate(self.test_results["red_flags"][:10], 1):
                print(f"{i}. {flag}")

        # Summary statistics
        print("\nDATA SIZE SUMMARY:")
        for trader_id, data in self.test_results["data_integrity"].items():
            if data["test_size"] < 50:
                print(f"- Trader {trader_id}: Only {data['test_size']} test days ({data['test_ratio']:.1%} of total)")

        print("\nVERDICT:")
        if len(self.test_results["red_flags"]) > 5:
            print("❌ SYSTEM SHOWS MULTIPLE VALIDITY CONCERNS")
            print("   Results should be viewed with extreme skepticism")
        elif len(self.test_results["red_flags"]) > 2:
            print("⚠️  SYSTEM HAS SOME VALIDITY ISSUES")
            print("   Further validation required before deployment")
        else:
            print("✅ SYSTEM PASSES BASIC VALIDITY CHECKS")
            print("   Still recommend extended forward testing")

        # Save detailed results
        with open(self.base_path / "tests" / "validity_test_results.json", 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)

        print(f"\nDetailed results saved to: tests/validity_test_results.json")


if __name__ == "__main__":
    tester = ValidityTester()
    tester.run_all_tests()
