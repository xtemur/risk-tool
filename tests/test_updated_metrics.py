#!/usr/bin/env python3
"""Test the updated 30d metrics calculations."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference.signal_generator import SignalGenerator
import logging

logging.basicConfig(level=logging.WARNING, format='%(message)s')

def test_updated_metrics():
    """Test all 30d metrics calculations."""

    print("Testing Updated 30d Metrics Calculations")
    print("=" * 80)

    # Initialize signal generator
    generator = SignalGenerator()

    # Get trader metrics from database
    print("Fetching updated trader metrics...")
    trader_metrics = generator.get_trader_metrics_from_db()

    # Test with several traders
    test_traders = [3942, 3950, 3951, 4004]

    print(f"\n{'Trader':<8} {'30d Avg':<12} {'All Avg':<12} {'30d Sharpe':<12} {'All Sharpe':<12}")
    print(f"{'ID':<8} {'Daily PnL':<12} {'Daily PnL':<12} {'Ratio':<12} {'Ratio':<12}")
    print("-" * 68)

    for trader_id in test_traders:
        if trader_id in trader_metrics:
            m = trader_metrics[trader_id]
            print(f"{trader_id:<8} ${m['avg_daily_pnl']:>10,.0f} ${m['all_time_avg_daily_pnl']:>10,.0f} "
                  f"{m['sharpe_30d']:>10.2f} {m['all_time_sharpe']:>10.2f}")

    print("\nWinning/Losing Trade Analysis:")
    print(f"{'Trader':<8} {'30d Avg Win':<12} {'All Avg Win':<12} {'30d Avg Loss':<12} {'All Avg Loss':<12}")
    print("-" * 68)

    for trader_id in test_traders:
        if trader_id in trader_metrics:
            m = trader_metrics[trader_id]
            print(f"{trader_id:<8} ${m['avg_winning_trade']:>10,.0f} ${m['all_time_avg_winning_trade']:>10,.0f} "
                  f"${m['avg_losing_trade']:>10,.0f} ${m['all_time_avg_losing_trade']:>10,.0f}")

    print("\nDetailed Analysis for Trader 3950:")
    print("-" * 40)
    if 3950 in trader_metrics:
        m = trader_metrics[3950]
        print(f"30-day period:")
        print(f"  Average Daily PnL: ${m['avg_daily_pnl']:,.2f}")
        print(f"  Total PnL: ${m['total_pnl']:,.2f}")
        print(f"  Trading Days: {m['trading_days']}")
        print(f"  Sharpe Ratio: {m['sharpe_30d']:.3f}")
        print(f"  Avg Winning Trade: ${m['avg_winning_trade']:,.2f}")
        print(f"  Avg Losing Trade: ${m['avg_losing_trade']:,.2f}")
        print(f"  Highest Daily PnL: ${m['highest_pnl']:,.2f}")
        print(f"  Lowest Daily PnL: ${m['lowest_pnl']:,.2f}")

        print(f"\nAll-time comparison:")
        print(f"  All-time Avg Daily PnL: ${m['all_time_avg_daily_pnl']:,.2f}")
        print(f"  All-time Sharpe: {m['all_time_sharpe']:.3f}")
        print(f"  All-time Avg Win: ${m['all_time_avg_winning_trade']:,.2f}")
        print(f"  All-time Avg Loss: ${m['all_time_avg_losing_trade']:,.2f}")
        print(f"  All-time Highest: ${m['all_time_highest_pnl']:,.2f}")
        print(f"  All-time Lowest: ${m['all_time_lowest_pnl']:,.2f}")

        # Check for logical consistency
        print(f"\nLogical Checks:")
        print(f"  30d vs All-time performance comparison:")
        print(f"    Daily PnL: {'Better' if m['avg_daily_pnl'] > m['all_time_avg_daily_pnl'] else 'Worse'}")
        print(f"    Sharpe Ratio: {'Better' if m['sharpe_30d'] > m['all_time_sharpe'] else 'Worse'}")
        print(f"    Winning Trades: {'Better' if m['avg_winning_trade'] > m['all_time_avg_winning_trade'] else 'Worse'}")
        print(f"    Losing Trades: {'Better' if m['avg_losing_trade'] > m['all_time_avg_losing_trade'] else 'Worse'} (closer to 0)")

    print("\n" + "=" * 80)
    print("✅ Updated metrics calculation test completed!")

if __name__ == '__main__':
    test_updated_metrics()
