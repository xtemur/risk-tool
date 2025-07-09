#!/usr/bin/env python3
"""
Test the corrected Sharpe ratio calculation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference.signal_generator import SignalGenerator
from src.trader_metrics import TraderMetricsProvider
from src.utils import load_config
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_corrected_sharpe():
    """Test the corrected Sharpe ratio calculation in the signal generator."""
    print("=" * 80)
    print("TESTING CORRECTED SHARPE RATIO CALCULATION")
    print("=" * 80)

    # Test metrics provider directly
    config = load_config('configs/main_config.yaml')
    metrics_provider = TraderMetricsProvider(config)

    metrics = metrics_provider.get_comprehensive_trader_metrics(30)

    print("📊 CORRECTED SHARPE RATIOS:")
    print(f"{'Trader':<8} {'30d Sharpe':<12} {'All-time Sharpe':<15} {'Status':<15}")
    print("-" * 65)

    reasonable_count = 0
    total_count = 0

    for trader_id, trader_metrics in metrics.items():
        sharpe_30d = trader_metrics['sharpe_30d']
        sharpe_all = trader_metrics['all_time_sharpe']

        # Determine status
        if sharpe_30d == 0:
            status = "Insufficient data"
        elif -3 <= sharpe_30d <= 3:
            status = "Reasonable"
            reasonable_count += 1
        elif -5 <= sharpe_30d <= 5:
            status = "High but capped"
        else:
            status = "ERROR"

        total_count += 1
        print(f"{trader_id:<8} {sharpe_30d:<12.2f} {sharpe_all:<15.2f} {status:<15}")

    print(f"\n📈 SUMMARY:")
    print(f"  Total traders: {total_count}")
    print(f"  Reasonable Sharpe ratios: {reasonable_count} ({reasonable_count/total_count*100:.1f}%)")
    print(f"  Zero/insufficient data: {sum(1 for m in metrics.values() if m['sharpe_30d'] == 0)}")
    print(f"  Capped values: {sum(1 for m in metrics.values() if abs(m['sharpe_30d']) == 5)}")

    # Test signal generation
    print(f"\n" + "=" * 60)
    print("TESTING SIGNAL GENERATION WITH CORRECTED SHARPE")
    print("=" * 60)

    generator = SignalGenerator('configs/main_config.yaml')
    signal_data = generator.generate_daily_signals()

    print(f"Generated signals for {len(signal_data['trader_signals'])} traders")

    # Show sample with corrected Sharpe ratios
    print(f"\n📊 Sample trader signals with corrected Sharpe ratios:")
    for i, signal in enumerate(signal_data['trader_signals'][:5]):
        print(f"  {signal['trader_label']}: Sharpe 30d = {signal['sharpe_30d']:.2f}")

    # Verify no extreme values
    extreme_sharpe = [s for s in signal_data['trader_signals'] if abs(s['sharpe_30d']) > 5]
    if extreme_sharpe:
        print(f"\n⚠️  WARNING: Found {len(extreme_sharpe)} traders with extreme Sharpe ratios!")
        for s in extreme_sharpe:
            print(f"    {s['trader_label']}: {s['sharpe_30d']:.2f}")
    else:
        print(f"\n✅ SUCCESS: No extreme Sharpe ratios found (all within -5 to 5 range)")

    return True

if __name__ == '__main__':
    success = test_corrected_sharpe()
    sys.exit(0 if success else 1)
