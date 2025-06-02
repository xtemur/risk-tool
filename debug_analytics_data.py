#!/usr/bin/env python
"""
Debug Analytics Data - Identify why metrics are 0.0
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append('src')

from database import Database
from analytics import TraderAnalytics

def debug_analytics_data():
    """Debug analytics data and calculations"""

    print("="*60)
    print("ANALYTICS DATA DEBUG")
    print("="*60)

    # Initialize
    db = Database()
    analytics = TraderAnalytics()

    # Get all traders
    traders_df = db.get_all_traders()
    print(f"\n1. TRADERS OVERVIEW")
    print(f"Total traders in database: {len(traders_df)}")

    if traders_df.empty:
        print("❌ No traders found!")
        return

    # Debug each trader
    for i, (_, trader) in enumerate(traders_df.iterrows()):
        if i >= 3:  # Only check first 3 traders
            break

        account_id = str(trader['account_id'])
        trader_name = trader['trader_name']

        print(f"\n2. TRADER DEBUG: {trader_name} ({account_id})")
        print("-" * 40)

        # Get recent data (30 days)
        from datetime import date, timedelta
        end_date = date.today()
        start_date = end_date - timedelta(days=30)

        totals_df, fills_df = db.get_trader_data(
            account_id,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )

        print(f"Data availability:")
        print(f"  Totals records: {len(totals_df)}")
        print(f"  Fills records: {len(fills_df)}")

        if totals_df.empty:
            print("  ❌ No totals data - skipping")
            continue

        # Data quality check
        print(f"\nData quality:")
        print(f"  Date range: {totals_df['date'].min()} to {totals_df['date'].max()}")
        print(f"  P&L statistics:")
        print(f"    Sum: ${totals_df['net_pnl'].sum():,.2f}")
        print(f"    Mean: ${totals_df['net_pnl'].mean():.2f}")
        print(f"    Std: ${totals_df['net_pnl'].std():.2f}")
        print(f"    Min: ${totals_df['net_pnl'].min():.2f}")
        print(f"    Max: ${totals_df['net_pnl'].max():.2f}")
        print(f"    Non-zero days: {len(totals_df[totals_df['net_pnl'] != 0])}")
        print(f"    Positive days: {len(totals_df[totals_df['net_pnl'] > 0])}")
        print(f"    Negative days: {len(totals_df[totals_df['net_pnl'] < 0])}")

        # Show sample data
        print(f"\nSample data (last 5 days):")
        sample_cols = ['date', 'net_pnl', 'orders_count', 'fills_count']
        if all(col in totals_df.columns for col in sample_cols):
            sample_data = totals_df[sample_cols].tail(5)
            for _, row in sample_data.iterrows():
                print(f"  {row['date']}: P&L=${row['net_pnl']:>8.2f}, Orders={row['orders_count']:>3}, Fills={row['fills_count']:>3}")

        # Test metric calculations
        print(f"\nTesting metric calculations:")

        # Test performance metrics
        try:
            perf_metrics = analytics.calculate_performance_metrics(totals_df)
            if perf_metrics:
                print(f"  ✓ Performance metrics calculated")
                print(f"    Win rate: {perf_metrics.get('win_rate', 0):.1f}%")
                print(f"    Profit factor: {perf_metrics.get('profit_factor', 0):.2f}")
                print(f"    Sharpe ratio: {perf_metrics.get('sharpe_ratio', 0):.3f}")
                print(f"    Trading days: {perf_metrics.get('trading_days', 0)}")
            else:
                print(f"  ❌ Performance metrics empty")
        except Exception as e:
            print(f"  ❌ Performance metrics failed: {str(e)}")

        # Test risk metrics
        try:
            risk_metrics = analytics.calculate_risk_metrics(totals_df)
            if risk_metrics:
                print(f"  ✓ Risk metrics calculated")
                print(f"    VaR (5%): ${risk_metrics.get('var_5_percent', 0):.2f}")
                print(f"    Max losing streak: {risk_metrics.get('max_losing_streak', 0)}")
            else:
                print(f"  ❌ Risk metrics empty")
        except Exception as e:
            print(f"  ❌ Risk metrics failed: {str(e)}")

        # Test advanced metrics
        try:
            advanced_metrics = analytics.calculate_advanced_trading_metrics(totals_df)
            if advanced_metrics:
                print(f"  ✓ Advanced metrics calculated")
                print(f"    Omega ratio: {advanced_metrics.get('omega_ratio', 0):.2f}")
                print(f"    Hurst exponent: {advanced_metrics.get('hurst_exponent', 0):.3f}")
            else:
                print(f"  ❌ Advanced metrics empty")
        except Exception as e:
            print(f"  ❌ Advanced metrics failed: {str(e)}")

        # Full analytics test
        print(f"\nFull analytics test:")
        try:
            full_analytics = analytics.generate_trader_analytics(account_id, 30)
            if 'error' in full_analytics:
                print(f"  ❌ Full analytics error: {full_analytics['error']}")
            else:
                print(f"  ✓ Full analytics generated")
                perf = full_analytics.get('performance', {})
                print(f"    Total P&L: ${perf.get('total_pnl', 0):,.2f}")
                print(f"    Win rate: {perf.get('win_rate', 0):.1f}%")
                print(f"    Sharpe: {perf.get('sharpe_ratio', 0):.3f}")
        except Exception as e:
            print(f"  ❌ Full analytics failed: {str(e)}")

    # Database health check
    print(f"\n3. DATABASE HEALTH CHECK")
    print("-" * 40)

    stats = db.get_database_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Check for common issues
    print(f"\n4. COMMON ISSUES CHECK")
    print("-" * 40)

    # Check if all P&L is zero
    total_query = "SELECT SUM(net_pnl) as total_pnl, COUNT(*) as days FROM daily_totals"
    try:
        import sqlite3
        with sqlite3.connect(db.db_path) as conn:
            result = conn.execute(total_query).fetchone()
            total_pnl, total_days = result

            print(f"  Portfolio total P&L: ${total_pnl:,.2f} over {total_days} trading days")

            if total_pnl == 0:
                print(f"  ⚠️  All P&L is zero - check data download")

            if total_days < 100:
                print(f"  ⚠️  Limited data available - consider downloading more history")

    except Exception as e:
        print(f"  ❌ Database query failed: {str(e)}")

    print(f"\n" + "="*60)
    print("DEBUG COMPLETE")
    print("="*60)

    print(f"\nTROUBLESHOoting RECOMMENDATIONS:")
    print(f"1. If 'No totals data': Run python scripts/setup_database.py")
    print(f"2. If 'All P&L is zero': Check API_TOKEN and data download")
    print(f"3. If 'Limited data': Download more historical data")
    print(f"4. If 'Metrics failed': Check data format and column names")


if __name__ == "__main__":
    debug_analytics_data()
