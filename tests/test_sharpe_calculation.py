#!/usr/bin/env python3
"""
Test script to investigate the Sharpe ratio calculation issue.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.trader_metrics import TraderMetricsProvider
from src.utils import load_config
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_sharpe_calculation():
    """Test and debug the Sharpe ratio calculation."""
    print("=" * 80)
    print("INVESTIGATING SHARPE RATIO CALCULATION")
    print("=" * 80)

    config = load_config('configs/main_config.yaml')
    metrics_provider = TraderMetricsProvider(config)

    # Get current metrics
    current_metrics = metrics_provider.get_comprehensive_trader_metrics(30)

    # Focus on trader 5580 which has the suspicious Sharpe ratio
    trader_id = 5580

    if trader_id in current_metrics:
        current_sharpe = current_metrics[trader_id]['sharpe_30d']
        print(f"Current Sharpe ratio for trader {trader_id}: {current_sharpe:.2f}")

        # Manual calculation to verify
        db_path = config['paths']['db_path']
        conn = sqlite3.connect(db_path)

        # Get 30-day daily PnL data
        cutoff_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

        query = """
        SELECT trade_date, SUM(net) as daily_pnl
        FROM trades
        WHERE account_id = ? AND trade_date >= ?
        GROUP BY trade_date
        ORDER BY trade_date
        """

        cursor = conn.cursor()
        cursor.execute(query, (trader_id, cutoff_date))

        daily_pnl_data = []
        for row in cursor.fetchall():
            daily_pnl_data.append(row[1])

        conn.close()

        if daily_pnl_data:
            # Convert to numpy array
            daily_pnl = np.array(daily_pnl_data)

            # Calculate proper Sharpe ratio
            mean_daily_pnl = np.mean(daily_pnl)
            std_daily_pnl = np.std(daily_pnl, ddof=1)  # Sample standard deviation

            print(f"\nManual calculation for trader {trader_id}:")
            print(f"  Number of trading days: {len(daily_pnl)}")
            print(f"  Daily PnL values: {daily_pnl[:5]}... (showing first 5)")
            print(f"  Mean daily PnL: ${mean_daily_pnl:.2f}")
            print(f"  Std dev of daily PnL: ${std_daily_pnl:.2f}")

            # Correct Sharpe ratio calculation
            if std_daily_pnl > 0:
                # Without risk-free rate (assuming 0)
                daily_sharpe = mean_daily_pnl / std_daily_pnl
                annualized_sharpe = daily_sharpe * np.sqrt(252)

                print(f"  Daily Sharpe ratio: {daily_sharpe:.4f}")
                print(f"  Annualized Sharpe ratio: {annualized_sharpe:.4f}")

                # Check what the current SQL is calculating
                variance = np.var(daily_pnl, ddof=1)
                sql_result = mean_daily_pnl / np.sqrt(variance) * np.sqrt(252)
                print(f"  Current SQL calculation: {sql_result:.4f}")

                print(f"\n🔍 Analysis:")
                print(f"  Current SQL result: {current_sharpe:.4f}")
                print(f"  Correct calculation: {annualized_sharpe:.4f}")
                print(f"  Difference: {abs(current_sharpe - annualized_sharpe):.4f}")

                if abs(current_sharpe - annualized_sharpe) > 0.01:
                    print(f"  ⚠️  ISSUE DETECTED: Significant difference in calculations!")
                else:
                    print(f"  ✅ Calculations match closely")

                # Additional checks
                if annualized_sharpe > 3:
                    print(f"  ⚠️  WARNING: Annualized Sharpe > 3 is extremely high!")
                    print(f"  This suggests either:")
                    print(f"    1. Exceptional performance (rare)")
                    print(f"    2. Calculation error")
                    print(f"    3. Insufficient data or outliers")

                    # Check for outliers
                    q75, q25 = np.percentile(daily_pnl, [75, 25])
                    iqr = q75 - q25
                    lower_bound = q25 - 1.5 * iqr
                    upper_bound = q75 + 1.5 * iqr
                    outliers = daily_pnl[(daily_pnl < lower_bound) | (daily_pnl > upper_bound)]

                    print(f"    Outliers detected: {len(outliers)} out of {len(daily_pnl)} days")
                    if len(outliers) > 0:
                        print(f"    Outlier values: {outliers}")

            else:
                print(f"  ⚠️  Zero standard deviation - cannot calculate Sharpe ratio")
        else:
            print(f"No data found for trader {trader_id}")
    else:
        print(f"Trader {trader_id} not found in metrics")

    # Test a few more traders
    print(f"\n" + "=" * 60)
    print("SHARPE RATIOS FOR ALL TRADERS")
    print("=" * 60)

    print(f"{'Trader':<8} {'30d Sharpe':<12} {'All-time Sharpe':<15} {'Status':<10}")
    print("-" * 55)

    for tid, metrics in current_metrics.items():
        sharpe_30d = metrics['sharpe_30d']
        sharpe_all = metrics['all_time_sharpe']

        status = "Normal"
        if abs(sharpe_30d) > 5:
            status = "HIGH"
        elif abs(sharpe_30d) > 3:
            status = "Elevated"

        print(f"{tid:<8} {sharpe_30d:<12.2f} {sharpe_all:<15.2f} {status:<10}")

if __name__ == '__main__':
    test_sharpe_calculation()
