"""
Tests for walk-forward validator.
"""

import unittest
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.models.walk_forward_validator import WalkForwardValidator


class TestWalkForwardValidator(unittest.TestCase):
    """Test walk-forward validation functionality."""

    def setUp(self):
        """Set up test data."""
        # Create synthetic time series data
        np.random.seed(42)
        n_days = 365
        n_accounts = 5

        dates = []
        features = []
        targets = []

        start_date = datetime(2023, 1, 1)

        for i in range(n_days):
            date = start_date + timedelta(days=i)
            for account in range(n_accounts):
                dates.append(date)

                # Create some features that evolve over time
                base_feature = np.sin(i / 30) + np.random.normal(0, 0.1)
                features.append([
                    base_feature,
                    base_feature * 2 + np.random.normal(0, 0.1),
                    np.random.normal(0, 1),
                ])

                # Target: high risk if feature is extreme
                is_high_risk = abs(base_feature) > 0.8
                targets.append(int(is_high_risk))

        self.dates = pd.Series(dates)
        self.X = pd.DataFrame(features, columns=['feature1', 'feature2', 'feature3'])
        self.y = pd.Series(targets)

        self.validator = WalkForwardValidator(
            min_train_days=60,
            test_days=30,
            step_days=15,
            purge_days=2
        )

    def test_split_generation(self):
        """Test that splits are generated correctly."""
        splits = self.validator.split(self.X, self.y, self.dates)

        # Should have multiple splits
        self.assertGreater(len(splits), 0)

        # Check each split
        for train_idx, test_idx in splits:
            # Train and test should not overlap
            self.assertEqual(len(set(train_idx) & set(test_idx)), 0)

            # Train should come before test
            max_train_date = self.dates.iloc[train_idx].max()
            min_test_date = self.dates.iloc[test_idx].min()
            self.assertLess(max_train_date, min_test_date)

            # Should have purge gap
            gap_days = (min_test_date - max_train_date).days
            self.assertGreaterEqual(gap_days, self.validator.purge_days)

            # Check minimum training size
            self.assertGreaterEqual(len(train_idx), self.validator.min_train_days)

    def test_temporal_order(self):
        """Test that temporal order is maintained."""
        splits = self.validator.split(self.X, self.y, self.dates)

        # Check that test start dates advance by step_days
        prev_test_start = None
        for train_idx, test_idx in splits:
            test_dates = self.dates.iloc[test_idx]
            test_start = test_dates.min()

            if prev_test_start is not None:
                # Test periods should advance by step_days
                days_advanced = (test_start - prev_test_start).days
                self.assertEqual(days_advanced, self.validator.step_days)

            prev_test_start = test_start

    def test_validation_with_model(self):
        """Test full validation with a model."""
        model = RandomForestClassifier(n_estimators=10, random_state=42)

        results = self.validator.validate(
            model=model,
            X=self.X,
            y=self.y,
            dates=self.dates
        )

        # Check results structure
        self.assertIn('overall_metrics', results)
        self.assertIn('fold_metrics', results)
        self.assertIn('predictions', results)
        self.assertIn('n_folds', results)

        # Check metrics
        metrics = results['overall_metrics']
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            self.assertIn(metric, metrics)
            self.assertGreaterEqual(metrics[metric], 0)
            self.assertLessEqual(metrics[metric], 1)

        # Check fold metrics
        fold_metrics = results['fold_metrics']
        self.assertEqual(len(fold_metrics), results['n_folds'])

        # Check predictions
        predictions = results['predictions']
        self.assertGreater(len(predictions), 0)
        self.assertIn('date', predictions.columns)
        self.assertIn('actual', predictions.columns)
        self.assertIn('predicted', predictions.columns)
        self.assertIn('probability', predictions.columns)

    def test_no_data_leakage(self):
        """Test that there's no data leakage between folds."""
        # Create data with clear time dependency
        n_days = 200
        dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')

        # Feature that perfectly predicts next day
        X = pd.DataFrame({
            'leaky_feature': range(n_days),
            'normal_feature': np.random.randn(n_days)
        })

        # Target is just shifted feature (would be perfect if leaked)
        y = pd.Series([0] + [int(x > 100) for x in range(n_days-1)])

        validator = WalkForwardValidator(
            min_train_days=50,
            test_days=10,
            step_days=5,
            purge_days=1
        )

        splits = validator.split(X, y, pd.Series(dates))

        # With proper validation, we shouldn't be able to use future data
        for train_idx, test_idx in splits:
            train_dates = dates[train_idx]
            test_dates = dates[test_idx]

            # No test date should appear in training
            self.assertEqual(len(set(test_dates) & set(train_dates)), 0)

            # Test dates should be after train dates
            self.assertGreater(test_dates.min(), train_dates.max())

    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        # Create very small dataset
        small_X = self.X.iloc[:30]
        small_y = self.y.iloc[:30]
        small_dates = self.dates.iloc[:30]

        splits = self.validator.split(small_X, small_y, small_dates)

        # Should return empty list as we need min_train_days=60
        self.assertEqual(len(splits), 0)

    def test_sample_weights(self):
        """Test validation with sample weights."""
        model = RandomForestClassifier(n_estimators=10, random_state=42)

        # Create sample weights (higher for high-risk days)
        weights = pd.Series([2.0 if y == 1 else 1.0 for y in self.y])

        results = self.validator.validate(
            model=model,
            X=self.X,
            y=self.y,
            dates=self.dates,
            sample_weights=weights
        )

        # Should complete successfully
        self.assertIn('overall_metrics', results)
        self.assertGreater(results['n_folds'], 0)


if __name__ == '__main__':
    unittest.main()
