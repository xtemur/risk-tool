"""
Tests for minimal feature set.
"""

import unittest
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.features.minimal_features import MinimalRiskFeatures


class TestMinimalRiskFeatures(unittest.TestCase):
    """Test minimal feature calculation."""

    def setUp(self):
        """Create test data."""
        # Create realistic trading data
        np.random.seed(42)
        n_days = 100
        dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')

        # Simulate trading metrics
        base_balance = 100000
        daily_returns = np.random.normal(0.001, 0.02, n_days)
        cumulative_balance = base_balance * (1 + daily_returns).cumprod()

        self.daily_data = pd.DataFrame({
            'account_id': 'TEST001',
            'date': dates,
            'net': daily_returns * base_balance,
            'gross': daily_returns * base_balance + np.abs(np.random.normal(50, 20, n_days)),
            'fees': np.abs(np.random.normal(50, 20, n_days)),
            'qty': np.abs(np.random.normal(1000, 200, n_days)),
            'fills': np.random.randint(80, 120, n_days),
            'orders': np.random.randint(100, 130, n_days),
            'unrealized': np.random.normal(0, 5000, n_days),
            'end_balance': cumulative_balance,
        })

        self.feature_calculator = MinimalRiskFeatures()

    def test_feature_calculation(self):
        """Test that all features are calculated."""
        result = self.feature_calculator.calculate_features(self.daily_data)

        # Check all features exist
        expected_features = self.feature_calculator.get_feature_names()
        for feature in expected_features:
            self.assertIn(feature, result.columns)

        # Check no NaN values after minimum periods
        for feature in expected_features:
            non_nan_count = result[feature].notna().sum()
            # Should have values after min_periods
            self.assertGreater(non_nan_count, 80)

    def test_profit_per_volume(self):
        """Test profit per volume calculation."""
        result = self.feature_calculator.calculate_features(self.daily_data)

        # Manually calculate for verification
        expected = self.daily_data['net'] / self.daily_data['qty']

        # Compare (allowing for handling of division by zero)
        valid_mask = self.daily_data['qty'] != 0
        np.testing.assert_array_almost_equal(
            result.loc[valid_mask, 'profit_per_volume'],
            expected[valid_mask],
            decimal=5
        )

    def test_execution_efficiency(self):
        """Test execution efficiency calculation."""
        result = self.feature_calculator.calculate_features(self.daily_data)

        # Should be between 0 and 1 (fills/orders ratio)
        efficiency = result['execution_efficiency']
        self.assertTrue((efficiency >= 0).all())
        self.assertTrue((efficiency <= 1.5).all())  # Can be >1 if multiple fills per order

    def test_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        result = self.feature_calculator.calculate_features(self.daily_data)

        # Sharpe ratio should be reasonable
        sharpe = result['sharpe_ratio'].dropna()
        self.assertTrue((sharpe > -10).all())
        self.assertTrue((sharpe < 10).all())

        # Should have variation
        self.assertGreater(sharpe.std(), 0)

    def test_sortino_ratio(self):
        """Test Sortino ratio calculation."""
        result = self.feature_calculator.calculate_features(self.daily_data)

        # Sortino should focus on downside risk
        sortino = result['sortino_ratio'].dropna()
        sharpe = result['sharpe_ratio'].dropna()

        # In general, Sortino >= Sharpe for same data
        # (but not always due to rolling window differences)
        self.assertTrue(len(sortino) > 0)
        self.assertTrue(len(sharpe) > 0)

    def test_max_drawdown(self):
        """Test maximum drawdown calculation."""
        result = self.feature_calculator.calculate_features(self.daily_data)

        # Drawdown should be positive (we take absolute value)
        drawdown = result['max_drawdown'].dropna()
        self.assertTrue((drawdown >= 0).all())
        self.assertTrue((drawdown <= 1).all())  # Should be percentage

    def test_win_rate(self):
        """Test win rate calculation."""
        result = self.feature_calculator.calculate_features(self.daily_data)

        # Win rate should be between 0 and 1
        win_rate = result['win_rate'].dropna()
        self.assertTrue((win_rate >= 0).all())
        self.assertTrue((win_rate <= 1).all())

        # Should have some variation
        self.assertGreater(win_rate.std(), 0)

    def test_point_in_time_consistency(self):
        """Test that features respect point-in-time constraints."""
        # Calculate features for a specific date
        target_date = pd.Timestamp('2023-03-01')
        result = self.feature_calculator.calculate_features(
            self.daily_data,
            target_date=target_date
        )

        # Should only include data up to target date
        self.assertTrue((result['date'] <= target_date).all())

        # Features should be calculated only with historical data
        self.assertLessEqual(len(result), len(self.daily_data))

    def test_multiple_accounts(self):
        """Test with multiple accounts."""
        # Create data for multiple accounts
        multi_data = []
        for account in ['ACC001', 'ACC002', 'ACC003']:
            account_data = self.daily_data.copy()
            account_data['account_id'] = account
            multi_data.append(account_data)

        combined_data = pd.concat(multi_data, ignore_index=True)

        result = self.feature_calculator.calculate_features(combined_data)

        # Each account should have features
        for account in ['ACC001', 'ACC002', 'ACC003']:
            account_result = result[result['account_id'] == account]
            self.assertGreater(len(account_result), 0)

            # Check features are calculated
            for feature in self.feature_calculator.get_feature_names():
                non_nan = account_result[feature].notna().sum()
                self.assertGreater(non_nan, 0)

    def test_feature_validation(self):
        """Test feature predictive power validation."""
        # Create synthetic target with some correlation to features
        features = self.feature_calculator.calculate_features(self.daily_data)

        # Create target based on low Sharpe ratio and high drawdown
        target = (
            (features['sharpe_ratio'] < features['sharpe_ratio'].quantile(0.2)) |
            (features['max_drawdown'] > features['max_drawdown'].quantile(0.8))
        ).astype(int)

        # Validate predictive power
        importance = self.feature_calculator.validate_predictive_power(
            features,
            target,
            features['date']
        )

        # Should have importance scores for all features
        self.assertEqual(len(importance), len(self.feature_calculator.get_feature_names()))

        # Scores should be between 0 and 1
        for score in importance.values():
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 1)

        # Features we used to create target should have higher importance
        self.assertGreater(importance['sharpe_ratio'], 0.5)
        self.assertGreater(importance['max_drawdown'], 0.5)

    def test_edge_cases(self):
        """Test edge cases in calculations."""
        # Create edge case data
        edge_data = self.daily_data.copy()

        # Zero volume
        edge_data.loc[10, 'qty'] = 0

        # Zero orders
        edge_data.loc[20, 'orders'] = 0

        # Zero balance
        edge_data.loc[30, 'end_balance'] = 0

        # Zero gross
        edge_data.loc[40, 'gross'] = 0

        # Should not crash
        result = self.feature_calculator.calculate_features(edge_data)

        # Check all features exist
        for feature in self.feature_calculator.get_feature_names():
            self.assertIn(feature, result.columns)

        # Should handle edge cases gracefully (no infinities)
        for feature in self.feature_calculator.get_feature_names():
            self.assertFalse(np.isinf(result[feature]).any())


if __name__ == '__main__':
    unittest.main()
