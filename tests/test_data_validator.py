"""Tests for data validation module."""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import sqlite3
import os

from core.data.data_validator import DataValidator


class TestDataValidator(unittest.TestCase):
    """Test data validation functionality."""

    def setUp(self):
        """Create test database with sample data."""
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.conn = sqlite3.connect(self.temp_db.name)

        # Create test tables
        self._create_test_tables()
        self._insert_test_data()

        # Mock the database path in validator
        self.validator = DataValidator()
        self.validator.db_path = self.temp_db.name

    def tearDown(self):
        """Clean up test database."""
        self.conn.close()
        os.unlink(self.temp_db.name)

    def _create_test_tables(self):
        """Create test database schema."""
        cursor = self.conn.cursor()

        # Create accounts table
        cursor.execute('''
            CREATE TABLE accounts (
                account_id TEXT PRIMARY KEY,
                account_name TEXT,
                active INTEGER DEFAULT 1,
                created_at TEXT
            )
        ''')

        # Create trades table
        cursor.execute('''
            CREATE TABLE trades (
                trade_id TEXT PRIMARY KEY,
                account_id TEXT,
                date TEXT,
                pnl REAL,
                asset TEXT,
                side TEXT,
                quantity REAL,
                price REAL,
                FOREIGN KEY (account_id) REFERENCES accounts(account_id)
            )
        ''')

        self.conn.commit()

    def _insert_test_data(self):
        """Insert sample test data."""
        cursor = self.conn.cursor()

        # Insert test accounts
        accounts = [
            ('acc1', 'Test Trader 1', 1, '2024-01-01'),
            ('acc2', 'Test Trader 2', 1, '2024-01-01'),
            ('acc3', 'Inactive Trader', 0, '2024-01-01')
        ]
        cursor.executemany(
            'INSERT INTO accounts VALUES (?, ?, ?, ?)',
            accounts
        )

        # Generate trades for active accounts
        trades = []
        base_date = datetime(2024, 1, 1)

        # Account 1: 90 days of trading
        for i in range(90):
            date = base_date + timedelta(days=i)
            # Multiple trades per day
            for j in range(np.random.randint(1, 5)):
                pnl = np.random.normal(0, 1000)  # Random PnL
                trades.append((
                    f'trade1_{i}_{j}',
                    'acc1',
                    date.strftime('%Y-%m-%d'),
                    pnl,
                    'AAPL',
                    'BUY' if j % 2 == 0 else 'SELL',
                    100,
                    150.0
                ))

        # Account 2: 45 days of trading (below threshold)
        for i in range(45):
            date = base_date + timedelta(days=i*2)  # Every other day
            pnl = np.random.normal(0, 500)
            trades.append((
                f'trade2_{i}',
                'acc2',
                date.strftime('%Y-%m-%d'),
                pnl,
                'GOOGL',
                'BUY',
                50,
                2800.0
            ))

        cursor.executemany(
            'INSERT INTO trades VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
            trades
        )

        self.conn.commit()

    def test_load_and_validate_data(self):
        """Test data loading and validation."""
        # Load data
        self.validator.load_and_validate_data(active_only=True)

        # Check accounts loaded
        self.assertIsNotNone(self.validator.accounts_df)
        self.assertEqual(len(self.validator.accounts_df), 2)  # Only active

        # Check trades loaded
        self.assertIsNotNone(self.validator.trades_df)
        self.assertGreater(len(self.validator.trades_df), 0)

        # Check date parsing
        self.assertEqual(self.validator.trades_df['date'].dtype, 'datetime64[ns]')

    def test_create_daily_aggregations(self):
        """Test daily aggregation creation."""
        # Load data first
        self.validator.load_and_validate_data(active_only=True)

        # Create aggregations
        self.validator.create_daily_aggregations()

        # Check daily DataFrame created
        self.assertIsNotNone(self.validator.daily_df)

        # Check required columns exist
        required_cols = ['account_id', 'date', 'daily_pnl', 'trade_count', 'win_rate']
        for col in required_cols:
            self.assertIn(col, self.validator.daily_df.columns)

        # Check aggregation logic
        acc1_data = self.validator.daily_df[
            self.validator.daily_df['account_id'] == 'acc1'
        ]
        self.assertGreater(len(acc1_data), 0)

        # Verify trade counts are positive where trades exist
        trades_days = acc1_data[acc1_data['trade_count'] > 0]
        self.assertGreater(len(trades_days), 0)

    def test_validate_data_quality(self):
        """Test data quality validation."""
        # Load and aggregate data
        self.validator.load_and_validate_data(active_only=True)
        self.validator.create_daily_aggregations()

        # Validate quality
        quality_issues = self.validator.validate_data_quality()

        # Check validation performed
        self.assertIsInstance(quality_issues, dict)

        # Check specific validations
        self.assertIn('missing_values', quality_issues)
        self.assertIn('data_gaps', quality_issues)
        self.assertIn('outliers', quality_issues)

    def test_analyze_predictability(self):
        """Test predictability analysis."""
        # Load and prepare data
        self.validator.load_and_validate_data(active_only=True)
        self.validator.create_daily_aggregations()

        # Analyze predictability
        predictability_results = self.validator.analyze_predictability()

        # Check results structure
        self.assertIsInstance(predictability_results, dict)

        # Check key metrics exist
        expected_keys = ['autocorrelation', 'stationarity', 'trend_analysis']
        for key in expected_keys:
            self.assertIn(key, predictability_results)

    def test_trader_filtering(self):
        """Test filtering traders by minimum days."""
        # Load data
        self.validator.load_and_validate_data(active_only=True)
        self.validator.create_daily_aggregations()

        # Get trader summaries
        trader_summary = self.validator.daily_df.groupby('account_id').agg({
            'date': 'count',
            'daily_pnl': ['sum', 'mean', 'std']
        })

        # Check acc1 has enough days
        acc1_days = trader_summary.loc['acc1', ('date', 'count')]
        self.assertGreaterEqual(acc1_days, 60)

        # Check acc2 has fewer days
        if 'acc2' in trader_summary.index:
            acc2_days = trader_summary.loc['acc2', ('date', 'count')]
            self.assertLess(acc2_days, 60)

    def test_generate_summary_report(self):
        """Test summary report generation."""
        # Load and process all data
        self.validator.load_and_validate_data(active_only=True)
        self.validator.create_daily_aggregations()
        self.validator.validate_data_quality()
        self.validator.analyze_predictability()

        # Generate summary
        checkpoint_pass = self.validator.generate_summary_report()

        # Check checkpoint result
        self.assertIsInstance(checkpoint_pass, bool)

        # For our test data, we expect at least one trader to pass
        viable_traders = self.validator.daily_df.groupby('account_id').size()
        viable_traders = viable_traders[viable_traders >= 60]

        if len(viable_traders) > 0:
            self.assertTrue(checkpoint_pass)


if __name__ == '__main__':
    unittest.main()
