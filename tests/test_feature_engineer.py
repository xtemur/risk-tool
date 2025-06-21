"""Tests for feature engineering module."""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from core.features.feature_engineer import FeatureEngineer


class TestFeatureEngineer(unittest.TestCase):
    """Test feature engineering functionality."""

    def setUp(self):
        """Create sample data for testing."""
        # Generate sample daily data
        dates = pd.date_range(start='2024-01-01', end='2024-06-30', freq='D')

        # Create data for two traders
        data = []
        for trader_id in ['trader1', 'trader2']:
            for date in dates:
                # Skip some days to simulate non-trading days
                if np.random.random() > 0.8:
                    continue

                # Generate realistic trading data
                daily_pnl = np.random.normal(0, 1000)
                trade_count = np.random.poisson(5) + 1
                wins = np.random.binomial(trade_count, 0.55)

                data.append({
                    'account_id': trader_id,
                    'date': date,
                    'daily_pnl': daily_pnl,
                    'trade_count': trade_count,
                    'winning_trades': wins,
                    'losing_trades': trade_count - wins,
                    'win_rate': wins / trade_count if trade_count > 0 else 0
                })

        self.test_df = pd.DataFrame(data)

        # Initialize feature engineer with test data
        self.engineer = FeatureEngineer()
        self.engineer.daily_df = self.test_df.copy()
        self.engineer.feature_df = self.test_df.copy()

    def test_create_basic_features(self):
        """Test basic feature creation."""
        self.engineer.create_basic_features()

        # Check new features exist
        expected_features = [
            'cumulative_pnl',
            'consecutive_wins',
            'consecutive_losses',
            'days_since_last_trade'
        ]

        for feature in expected_features:
            self.assertIn(feature, self.engineer.feature_df.columns)

        # Validate cumulative PnL calculation
        for trader_id in ['trader1', 'trader2']:
            trader_data = self.engineer.feature_df[
                self.engineer.feature_df['account_id'] == trader_id
            ].sort_values('date')

            # Check cumulative sum
            manual_cumsum = trader_data['daily_pnl'].cumsum()
            np.testing.assert_array_almost_equal(
                trader_data['cumulative_pnl'].values,
                manual_cumsum.values
            )

    def test_create_rolling_features_ewma(self):
        """Test EWMA rolling feature creation."""
        # Need basic features first
        self.engineer.create_basic_features()
        self.engineer.create_rolling_features_ewma()

        # Check EWMA features exist
        ewma_features = [
            'pnl_ewma_5',
            'pnl_ewma_20',
            'win_rate_ewma_5',
            'win_rate_ewma_20',
            'trade_count_ewma_5',
            'trade_count_ewma_20'
        ]

        for feature in ewma_features:
            self.assertIn(feature, self.engineer.feature_df.columns)

        # Validate EWMA calculation
        trader_data = self.engineer.feature_df[
            self.engineer.feature_df['account_id'] == 'trader1'
        ].sort_values('date')

        # Check 5-day EWMA manually
        manual_ewma = trader_data['daily_pnl'].ewm(span=5, adjust=False).mean()
        np.testing.assert_array_almost_equal(
            trader_data['pnl_ewma_5'].values,
            manual_ewma.values,
            decimal=5
        )

    def test_create_lagged_features(self):
        """Test lagged feature creation."""
        # Need basic features first
        self.engineer.create_basic_features()
        self.engineer.create_lagged_features()

        # Check lagged features exist
        lag_features = [
            'daily_pnl_lag_1',
            'daily_pnl_lag_2',
            'daily_pnl_lag_3',
            'win_rate_lag_1',
            'trade_count_lag_1'
        ]

        for feature in lag_features:
            self.assertIn(feature, self.engineer.feature_df.columns)

        # Validate lag calculation
        trader_data = self.engineer.feature_df[
            self.engineer.feature_df['account_id'] == 'trader1'
        ].sort_values('date')

        # Check 1-day lag
        expected_lag = trader_data['daily_pnl'].shift(1)
        pd.testing.assert_series_equal(
            trader_data['daily_pnl_lag_1'],
            expected_lag,
            check_names=False
        )

    def test_create_advanced_features(self):
        """Test advanced feature creation."""
        # Need all previous features
        self.engineer.create_basic_features()
        self.engineer.create_rolling_features_ewma()
        self.engineer.create_advanced_features()

        # Check advanced features exist
        advanced_features = [
            'max_drawdown_20d',
            'volatility_20d',
            'sharpe_ratio_20d',
            'avg_win_loss_ratio_20d'
        ]

        for feature in advanced_features:
            self.assertIn(feature, self.engineer.feature_df.columns)

        # Validate volatility calculation
        trader_data = self.engineer.feature_df[
            self.engineer.feature_df['account_id'] == 'trader1'
        ].sort_values('date')

        # Check 20-day volatility
        manual_vol = trader_data['daily_pnl'].rolling(20).std()
        pd.testing.assert_series_equal(
            trader_data['volatility_20d'],
            manual_vol,
            check_names=False,
            check_exact=False,
            rtol=1e-5
        )

    def test_validate_features(self):
        """Test feature validation."""
        # Create all features
        self.engineer.create_basic_features()
        self.engineer.create_rolling_features_ewma()
        self.engineer.create_lagged_features()
        self.engineer.create_advanced_features()

        # Validate features
        validation_results = self.engineer.validate_features()

        # Check validation performed
        self.assertIsInstance(validation_results, dict)
        self.assertIn('missing_values', validation_results)
        self.assertIn('infinite_values', validation_results)
        self.assertIn('feature_stats', validation_results)

    def test_finalize_features(self):
        """Test feature finalization."""
        # Create all features
        self.engineer.create_basic_features()
        self.engineer.create_rolling_features_ewma()
        self.engineer.create_lagged_features()
        self.engineer.create_advanced_features()
        self.engineer.validate_features()

        # Finalize features
        feature_list = self.engineer.finalize_features()

        # Check feature list
        self.assertIsInstance(feature_list, list)
        self.assertGreater(len(feature_list), 20)  # Should have many features

        # Check no date/id columns in features
        self.assertNotIn('date', feature_list)
        self.assertNotIn('account_id', feature_list)

        # Check data is complete (no NaN in finalized data)
        final_df = self.engineer.feature_df[feature_list]
        self.assertEqual(final_df.isnull().sum().sum(), 0)

    def test_handle_missing_days(self):
        """Test handling of missing trading days."""
        # Create data with gaps
        sparse_df = self.test_df[self.test_df['date'] < '2024-03-01'].copy()

        # Add a gap
        gap_start = pd.Timestamp('2024-02-10')
        gap_end = pd.Timestamp('2024-02-20')
        sparse_df = sparse_df[
            (sparse_df['date'] < gap_start) | (sparse_df['date'] > gap_end)
        ]

        self.engineer.daily_df = sparse_df
        self.engineer.feature_df = sparse_df.copy()

        # Create features
        self.engineer.create_basic_features()

        # Check consecutive calculations handle gaps properly
        self.assertIn('consecutive_wins', self.engineer.feature_df.columns)
        self.assertIn('days_since_last_trade', self.engineer.feature_df.columns)


if __name__ == '__main__':
    unittest.main()
