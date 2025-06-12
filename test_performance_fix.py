#!/usr/bin/env python3
"""
Test the performance calculation fixes
"""
import sys
sys.path.append('src')
from email_service.signal_command import SignalCommand
import pandas as pd

def main():
    # Create a signal command instance
    signal_cmd = SignalCommand()

    # Load some sample data
    data = pd.read_csv('data/processed/features_demo.csv')

    print('=== TESTING RECENT PERFORMANCE CALCULATION ===')
    print()

    # Test the recent performance calculation for each trader
    traders = [3942, 3946, 3950, 3951, 3956, 3957, 3978, 4004, 5093]

    for trader_id in traders:
        recent_pnl = signal_cmd._calculate_recent_performance(trader_id, data)
        print(f'Trader {trader_id}: Recent 7-day PnL = ${recent_pnl:.2f}')

    print()
    print('=== TESTING MODEL PERFORMANCE METRICS ===')
    print()

    # Test the model performance calculation
    performance = signal_cmd._extract_model_performance()
    if performance:
        print(f'Hit Rate: {performance.get("hit_rate", 0):.1%}')
        print(f'Sharpe Ratio: {performance.get("sharpe_ratio", 0):.3f}')
        print(f'R² Score: {performance.get("r2_score", 0):.3f}')
        print(f'Model Version: {performance.get("model_version", "unknown")}')
    else:
        print('No performance metrics available')

if __name__ == "__main__":
    main()
