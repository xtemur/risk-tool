"""
Tests for target variable definition.
"""

import unittest
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.features.target_definition import RiskTargetDefinition


class TestRiskTargetDefinition(unittest.TestCase):
    """Test risk target calculation."""

    def setUp(self):
        """Create test data."""
        # Create synthetic daily data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        n_days = len(dates)

        # Normal trading pattern with some anomalies
        np.random.seed(42)
        normal_pnl = np.random.normal(1000, 500, n_days)

        # Insert some high-risk days
        # Large drawdowns
        normal_pnl[50] = -5000
        normal_pnl[150] = -4000
        normal_pnl[250] = -6000

        self.daily_data = pd.DataFrame({
            'account_id': 'TEST001',
            'date': dates,
            'net': normal_pnl,
            'gross': normal_pnl + np.abs(np.random.normal(50, 20, n_days)),
            'fees': np.abs(np.random.normal(50, 20, n_days)),
            'qty': np.abs(np.random.normal(1000, 200, n_days)),
            'fills': np.random.randint(50, 150, n_days),
            'orders': np.random.randint(60, 160, n_days),
        })

        # Add some high fee days
        self.daily_data.loc[100, 'fees'] = 2000
        self.daily_data.loc[100, 'gross'] = 2500

        # Add overleveraged day
        self.daily_data.loc[200, 'qty'] = 10000

        self.target_def = RiskTargetDefinition()

    def test_calculate_targets(self):
        """Test basic target calculation."""
        result = self.target_def.calculate_targets(self.daily_data)

        # Check all columns are present
        expected_cols = [
            'is_large_drawdown', 'is_high_fee_ratio',
            'is_overleveraged', 'is_poor_execution',
            'is_high_risk', 'risk_score'
        ]
        for col in expected_cols:
            self.assertIn(col, result.columns)

        # Check that we have some high-risk days
        self.assertGreater(result['is_high_risk'].sum(), 0)
        self.assertLess(result['is_high_risk'].sum(), len(result))

        # Check specific high-risk days we inserted
        self.assertTrue(result.iloc[50]['is_large_drawdown'])
        self.assertTrue(result.iloc[100]['is_high_fee_ratio'])
        self.assertTrue(result.iloc[200]['is_overleveraged'])

    def test_drawdown_detection(self):
        """Test large drawdown detection."""
        result = self.target_def.calculate_targets(self.daily_data)

        # Days with large negative P&L should be flagged
        drawdown_days = result[result['is_large_drawdown']]

        # Check that flagged days have low P&L
        for idx in drawdown_days.index:
            day_pnl = result.loc[idx, 'net']
            threshold = result.loc[idx, 'drawdown_threshold']

            # Skip NaN thresholds (insufficient history)
            if pd.notna(threshold):
                self.assertLess(day_pnl, threshold)

    def test_fee_ratio_detection(self):
        """Test high fee ratio detection."""
        result = self.target_def.calculate_targets(self.daily_data)

        # Check the day we set with high fees
        high_fee_day = result.iloc[100]
        self.assertTrue(high_fee_day['is_high_fee_ratio'])
        self.assertGreater(high_fee_day['fee_ratio'], 0.5)

    def test_overleveraging_detection(self):
        """Test overleveraging detection."""
        result = self.target_def.calculate_targets(self.daily_data)

        # Check the day we set with high position size
        overleveraged_day = result.iloc[200]
        self.assertTrue(overleveraged_day['is_overleveraged'])

        # Position should be much larger than average
        self.assertGreater(
            overleveraged_day['qty'],
            3 * overleveraged_day['avg_position_size']
        )

    def test_risk_score(self):
        """Test risk score calculation."""
        result = self.target_def.calculate_targets(self.daily_data)

        # Risk score should be sum of indicators
        for idx, row in result.iterrows():
            expected_score = (
                int(row['is_large_drawdown']) +
                int(row['is_high_fee_ratio']) +
                int(row['is_overleveraged']) +
                int(row['is_poor_execution'])
            )
            self.assertEqual(row['risk_score'], expected_score)

        # Score should be between 0 and 4
        self.assertGreaterEqual(result['risk_score'].min(), 0)
        self.assertLessEqual(result['risk_score'].max(), 4)

    def test_forward_looking_target(self):
        """Test forward-looking target calculation."""
        # First calculate regular targets
        data_with_targets = self.target_def.calculate_targets(self.daily_data)

        # Calculate forward-looking
        result = self.target_def.calculate_forward_looking_target(
            data_with_targets,
            forward_days=1
        )

        # Check that forward target is shifted correctly
        for i in range(len(result) - 1):
            current_risk_next = result.iloc[i]['is_high_risk_next']
            actual_next_risk = result.iloc[i + 1]['is_high_risk']
            self.assertEqual(current_risk_next, actual_next_risk)

    def test_multiple_accounts(self):
        """Test with multiple accounts."""
        # Create data for multiple accounts
        account_data = []

        for account in ['ACC001', 'ACC002', 'ACC003']:
            account_df = self.daily_data.copy()
            account_df['account_id'] = account

            # Add different risk patterns for each account
            if account == 'ACC002':
                account_df['net'] = account_df['net'] * 0.5  # Lower P&L
            elif account == 'ACC003':
                account_df['fees'] = account_df['fees'] * 3  # Higher fees

            account_data.append(account_df)

        multi_account_data = pd.concat(account_data, ignore_index=True)

        result = self.target_def.calculate_targets(multi_account_data)

        # Each account should have been processed
        for account in ['ACC001', 'ACC002', 'ACC003']:
            account_result = result[result['account_id'] == account]
            self.assertGreater(len(account_result), 0)

            # Each account should have some risk days
            self.assertGreater(account_result['is_high_risk'].sum(), 0)

    def test_target_statistics(self):
        """Test statistics calculation."""
        data_with_targets = self.target_def.calculate_targets(self.daily_data)
        stats = self.target_def.get_target_statistics(data_with_targets)

        # Check expected statistics
        self.assertIn('total_days', stats)
        self.assertIn('high_risk_days', stats)
        self.assertIn('high_risk_pct', stats)
        self.assertIn('risk_score_mean', stats)

        # Validate statistics
        self.assertEqual(stats['total_days'], len(data_with_targets))
        self.assertEqual(stats['high_risk_days'], data_with_targets['is_high_risk'].sum())

        # Check percentage calculation
        expected_pct = (stats['high_risk_days'] / stats['total_days']) * 100
        self.assertAlmostEqual(stats['high_risk_pct'], expected_pct, places=5)

    def test_edge_cases(self):
        """Test edge cases."""
        # Test with zero gross P&L
        edge_data = self.daily_data.copy()
        edge_data.loc[10, 'gross'] = 0
        edge_data.loc[10, 'fees'] = 50

        result = self.target_def.calculate_targets(edge_data)

        # Should handle division by zero gracefully
        self.assertTrue(result.iloc[10]['is_high_fee_ratio'])

        # Test with zero orders
        edge_data.loc[20, 'orders'] = 0
        edge_data.loc[20, 'fills'] = 0

        result = self.target_def.calculate_targets(edge_data)
        # Should not crash
        self.assertIn('fill_rate', result.columns)


if __name__ == '__main__':
    unittest.main()
