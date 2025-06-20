#!/usr/bin/env python3
"""
Step 1: Data Validation & Exploration
Following CLAUDE.md methodology
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DataValidator:
    def __init__(self, db_path='data/risk_tool.db'):
        self.db_path = db_path
        self.trades_df = None
        self.daily_df = None

    def load_and_validate_data(self):
        """Load trades data and perform basic validation"""
        print("=== STEP 1: DATA VALIDATION & EXPLORATION ===")

        conn = sqlite3.connect(self.db_path)

        # Load trades data with all relevant fields
        query = """
        SELECT
            account_id,
            trade_date,
            net as realized_pnl,
            gross,
            qty,
            symbol,
            type,
            entry,
            exit,
            held,
            CASE WHEN net > 0 THEN 1 ELSE 0 END as is_winner
        FROM trades
        WHERE trade_date IS NOT NULL
        ORDER BY account_id, trade_date
        """

        self.trades_df = pd.read_sql_query(query, conn)
        conn.close()

        # Convert dates and parse holding period
        self.trades_df['trade_date'] = pd.to_datetime(self.trades_df['trade_date'])
        self.trades_df['holding_days'] = pd.to_numeric(
            self.trades_df['held'].str.replace('d', ''), errors='coerce'
        )

        print(f"✓ Loaded {len(self.trades_df)} trades")
        print(f"✓ Date range: {self.trades_df.trade_date.min()} to {self.trades_df.trade_date.max()}")
        print(f"✓ Unique traders: {self.trades_df.account_id.nunique()}")

        return self.trades_df

    def create_daily_aggregations(self):
        """Create daily aggregations for each trader"""
        print("\n=== CREATING DAILY AGGREGATIONS ===")

        # Group by trader and date
        daily_agg = self.trades_df.groupby(['account_id', 'trade_date']).agg({
            'realized_pnl': ['sum', 'count', 'mean', 'std'],
            'is_winner': ['sum', 'mean'],
            'qty': 'sum',
            'holding_days': 'mean'
        }).reset_index()

        # Flatten column names
        daily_agg.columns = [
            'account_id', 'trade_date', 'realized_pnl', 'num_trades',
            'avg_trade_pnl', 'pnl_std', 'num_wins', 'win_rate',
            'total_qty', 'avg_holding_days'
        ]

        # Fill NaN std with 0 (single trade days)
        daily_agg['pnl_std'] = daily_agg['pnl_std'].fillna(0)

        # Create complete daily timeline for each trader
        all_traders = self.trades_df['account_id'].unique()
        date_range = pd.date_range(
            start=self.trades_df['trade_date'].min(),
            end=self.trades_df['trade_date'].max(),
            freq='D'
        )

        # Create complete grid
        complete_grid = []
        for trader in all_traders:
            for date in date_range:
                complete_grid.append({'account_id': trader, 'trade_date': date})

        complete_df = pd.DataFrame(complete_grid)

        # Merge with actual data
        self.daily_df = complete_df.merge(daily_agg, on=['account_id', 'trade_date'], how='left')

        # Fill missing values (non-trading days)
        activity_cols = ['realized_pnl', 'num_trades', 'avg_trade_pnl', 'pnl_std',
                        'num_wins', 'win_rate', 'total_qty', 'avg_holding_days']

        for col in activity_cols:
            self.daily_df[col] = self.daily_df[col].fillna(0)

        print(f"✓ Created daily aggregations: {len(self.daily_df)} trader-days")
        print(f"✓ Active trading days: {(self.daily_df['num_trades'] > 0).sum()}")

        return self.daily_df

    def validate_data_quality(self):
        """Perform data quality checks"""
        print("\n=== DATA QUALITY VALIDATION ===")

        # Check for outliers
        pnl_99 = self.trades_df['realized_pnl'].quantile(0.99)
        pnl_1 = self.trades_df['realized_pnl'].quantile(0.01)

        extreme_trades = self.trades_df[
            (self.trades_df['realized_pnl'] > pnl_99) |
            (self.trades_df['realized_pnl'] < pnl_1)
        ]

        print(f"✓ Extreme trades (1%/99% percentiles): {len(extreme_trades)}")
        print(f"  - 99th percentile: ${pnl_99:,.2f}")
        print(f"  - 1st percentile: ${pnl_1:,.2f}")

        # Check for data leakage (future dates)
        future_trades = self.trades_df[self.trades_df['trade_date'] > datetime.now()]
        if len(future_trades) > 0:
            print(f"⚠️  WARNING: {len(future_trades)} trades with future dates")
        else:
            print("✓ No future-dated trades found")

        # Validate high PnL days
        top_pnl_days = self.daily_df.nlargest(10, 'realized_pnl')
        print(f"✓ Top 10 PnL days range: ${top_pnl_days['realized_pnl'].min():,.2f} to ${top_pnl_days['realized_pnl'].max():,.2f}")

        return True

    def analyze_predictability(self):
        """Check if yesterday's performance predicts tomorrow's"""
        print("\n=== PREDICTABILITY ANALYSIS ===")

        results = {}

        for trader_id in self.trades_df['account_id'].unique()[:10]:  # Sample of traders
            trader_daily = self.daily_df[self.daily_df['account_id'] == trader_id].copy()
            trader_daily = trader_daily[trader_daily['num_trades'] > 0]  # Only active days

            if len(trader_daily) < 20:  # Need minimum data
                continue

            trader_daily = trader_daily.sort_values('trade_date')

            # Create lagged features
            trader_daily['pnl_lag1'] = trader_daily['realized_pnl'].shift(1)
            trader_daily['pnl_lag2'] = trader_daily['realized_pnl'].shift(2)
            trader_daily['win_rate_lag1'] = trader_daily['win_rate'].shift(1)

            # Calculate correlations
            corr_lag1 = trader_daily['realized_pnl'].corr(trader_daily['pnl_lag1'])
            corr_lag2 = trader_daily['realized_pnl'].corr(trader_daily['pnl_lag2'])

            results[trader_id] = {
                'lag1_corr': corr_lag1,
                'lag2_corr': corr_lag2,
                'trading_days': len(trader_daily)
            }

        # Summarize results
        lag1_corrs = [r['lag1_corr'] for r in results.values() if not pd.isna(r['lag1_corr'])]
        lag2_corrs = [r['lag2_corr'] for r in results.values() if not pd.isna(r['lag2_corr'])]

        print(f"✓ Average lag-1 correlation: {np.mean(lag1_corrs):.3f}")
        print(f"✓ Average lag-2 correlation: {np.mean(lag2_corrs):.3f}")
        print(f"✓ Analyzed {len(results)} traders")

        if abs(np.mean(lag1_corrs)) > 0.1:
            print("⚠️  WARNING: Strong autocorrelation detected - investigate further")
        else:
            print("✓ Autocorrelation within expected range")

        return results

    def identify_viable_traders(self):
        """Identify traders with sufficient data for modeling"""
        print("\n=== VIABLE TRADERS FOR MODELING ===")

        # Calculate trader statistics
        trader_stats = self.daily_df.groupby('account_id').agg({
            'realized_pnl': ['sum', 'count', 'std'],
            'num_trades': 'sum',
            'trade_date': ['min', 'max']
        }).reset_index()

        trader_stats.columns = ['account_id', 'total_pnl', 'active_days', 'pnl_std',
                               'total_trades', 'first_date', 'last_date']

        # Filter for viable traders (>60 trades, >30 active days)
        viable_traders = trader_stats[
            (trader_stats['total_trades'] >= 60) &
            (trader_stats['active_days'] >= 30)
        ].copy()

        # Calculate trading days span
        viable_traders['trading_span_days'] = (
            viable_traders['last_date'] - viable_traders['first_date']
        ).dt.days

        print(f"✓ Viable traders for modeling: {len(viable_traders)}")
        print(f"✓ Average trades per viable trader: {viable_traders['total_trades'].mean():.0f}")
        print(f"✓ Average active days per viable trader: {viable_traders['active_days'].mean():.0f}")

        # Check test data availability
        test_cutoff = pd.to_datetime('2025-04-01')
        test_data = self.daily_df[
            (self.daily_df['trade_date'] >= test_cutoff) &
            (self.daily_df['num_trades'] > 0)
        ]

        test_traders = test_data['account_id'].unique()
        viable_test_traders = set(viable_traders['account_id']).intersection(set(test_traders))

        print(f"✓ Viable traders with test data: {len(viable_test_traders)}")

        return viable_traders, list(viable_test_traders)

    def generate_summary_report(self):
        """Generate final validation summary"""
        print("\n" + "="*50)
        print("STEP 1 VALIDATION SUMMARY")
        print("="*50)

        total_trades = len(self.trades_df)
        total_traders = self.trades_df['account_id'].nunique()
        date_range = (self.trades_df['trade_date'].max() - self.trades_df['trade_date'].min()).days

        viable_traders, test_traders = self.identify_viable_traders()

        print(f"✓ Total trades: {total_trades:,}")
        print(f"✓ Total traders: {total_traders}")
        print(f"✓ Date range: {date_range} days")
        print(f"✓ Viable traders: {len(viable_traders)}")
        print(f"✓ Test traders: {len(test_traders)}")

        # Checkpoint validation
        checkpoint_pass = (
            len(viable_traders) >= 10 and  # Minimum viable traders
            len(test_traders) >= 5 and     # Minimum test traders
            total_trades >= 1000           # Minimum total trades
        )

        if checkpoint_pass:
            print("\n✅ CHECKPOINT 1 PASSED - Proceeding to Step 2")
        else:
            print("\n❌ CHECKPOINT 1 FAILED - Insufficient data quality")

        return checkpoint_pass

def main():
    """Run Step 1 validation"""
    validator = DataValidator()

    # Load and validate data
    validator.load_and_validate_data()

    # Create daily aggregations
    validator.create_daily_aggregations()

    # Validate data quality
    validator.validate_data_quality()

    # Analyze predictability
    validator.analyze_predictability()

    # Generate summary
    checkpoint_pass = validator.generate_summary_report()

    # Save processed data
    validator.daily_df.to_pickle('data/daily_aggregated.pkl')
    print(f"\n✓ Saved daily aggregated data to data/daily_aggregated.pkl")

    return checkpoint_pass

if __name__ == "__main__":
    main()
