#!/usr/bin/env python3
"""Test the PnL metrics and color grading fixes."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference.signal_generator import SignalGenerator
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

def test_pnl_metrics():
    """Test PnL metrics calculation and color grading."""

    print("Testing PnL metrics and color grading fixes...")
    print("=" * 80)

    # Initialize signal generator
    generator = SignalGenerator()

    # Get trader metrics from database
    print("\nFetching trader metrics from database...")
    trader_metrics = generator.get_trader_metrics_from_db()

    # Focus on trader 3950 as requested
    trader_id = 3950
    if trader_id in trader_metrics:
        metrics = trader_metrics[trader_id]
        print(f"\nTrader {trader_id} Metrics:")
        print(f"  30-day window:")
        print(f"    Highest daily PnL: ${metrics['highest_pnl']:,.2f}")
        print(f"    Lowest daily PnL: ${metrics['lowest_pnl']:,.2f}")
        print(f"    Average daily PnL: ${metrics['avg_daily_pnl']:,.2f}")
        print(f"  All-time:")
        print(f"    Highest daily PnL: ${metrics['all_time_highest_pnl']:,.2f}")
        print(f"    Lowest daily PnL: ${metrics['all_time_lowest_pnl']:,.2f}")
        print(f"    Average daily PnL: ${metrics['all_time_avg_daily_pnl']:,.2f}")

        # Test color calculation
        print("\nColor Grading Tests:")

        # Highest PnL color
        highest_color = generator.calculate_heatmap_color(
            metrics['highest_pnl'],
            metrics['all_time_highest_pnl'],
            'higher_better'
        )
        print(f"  Highest PnL color: {highest_color[0]} (background), {highest_color[1]} (text)")
        print(f"    Interpretation: {'Green/Good' if 'green' in highest_color[0].lower() or int(highest_color[0][3:5], 16) > 200 else 'Red/Poor'}")

        # Lowest PnL color
        lowest_color = generator.calculate_heatmap_color(
            metrics['lowest_pnl'],
            metrics['all_time_lowest_pnl'],
            'higher_better'
        )
        print(f"  Lowest PnL color: {lowest_color[0]} (background), {lowest_color[1]} (text)")
        print(f"    Interpretation: {'Green/Good' if 'green' in lowest_color[0].lower() or int(lowest_color[0][3:5], 16) > 200 else 'Red/Poor'}")

        # Additional traders for comparison
        print("\n\nOther Traders Summary:")
        print("-" * 60)
        print(f"{'Trader':<8} {'30d High':<12} {'All High':<12} {'30d Low':<12} {'All Low':<12}")
        print("-" * 60)

        for tid in [3942, 3943, 3946, 3950, 3951]:
            if tid in trader_metrics:
                m = trader_metrics[tid]
                print(f"{tid:<8} ${m['highest_pnl']:>10,.0f} ${m['all_time_highest_pnl']:>10,.0f} "
                      f"${m['lowest_pnl']:>10,.0f} ${m['all_time_lowest_pnl']:>10,.0f}")

    else:
        print(f"No metrics found for trader {trader_id}")

    print("\n" + "=" * 80)
    print("✅ PnL metrics test completed!")

if __name__ == '__main__':
    test_pnl_metrics()
