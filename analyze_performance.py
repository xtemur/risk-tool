#!/usr/bin/env python3
"""
Analyze why recent 7-day performance and Sharpe ratio are 0
"""
import pandas as pd
import numpy as np

def main():
    # Load the feature data
    data = pd.read_csv('data/processed/features_demo.csv')
    data['date'] = pd.to_datetime(data['date'])

    print('=== ANALYSIS OF RECENT 7-DAY PERFORMANCE ISSUE ===')
    print()

    # Check if 'net' column exists and what it contains
    if 'net' in data.columns:
        print('1. NET COLUMN ANALYSIS:')
        print(f'   - Total rows: {len(data)}')
        print(f'   - Non-null values: {data["net"].notna().sum()}')
        print(f'   - Null values: {data["net"].isnull().sum()}')
        print(f'   - Zero values: {(data["net"] == 0).sum()}')
        print(f'   - Non-zero values: {(data["net"] != 0).sum()}')
        print(f'   - Min: {data["net"].min()}')
        print(f'   - Max: {data["net"].max()}')
        print(f'   - Sum: {data["net"].sum()}')
        print()

        # Check last 7 days for each trader
        print('2. RECENT 7-DAY NET PNL BY TRADER:')
        traders = data['account_id'].unique()
        for trader_id in sorted(traders):
            trader_data = data[data['account_id'] == trader_id].tail(7)
            if len(trader_data) > 0:
                net_sum = trader_data['net'].sum()
                dates = f'{trader_data["date"].min().strftime("%m-%d")} to {trader_data["date"].max().strftime("%m-%d")}'
                print(f'   Trader {trader_id}: ${net_sum:8.2f} ({dates})')
    else:
        print('1. NET COLUMN: NOT FOUND!')

    print()

    # Check alternative columns
    print('3. ALTERNATIVE PERFORMANCE COLUMNS:')
    perf_columns = ['target_next_pnl', 'gross', 'cumulative_pnl', 'rolling_pnl_7d']
    for col in perf_columns:
        if col in data.columns:
            non_null = data[col].notna().sum()
            non_zero = (data[col] != 0).sum()
            total_sum = data[col].sum()
            print(f'   {col}: {non_null} non-null, {non_zero} non-zero values, sum: ${total_sum:.2f}')
        else:
            print(f'   {col}: NOT FOUND')

    print()

    # Check the most recent dates per trader
    print('4. MOST RECENT DATA DATES BY TRADER:')
    traders = data['account_id'].unique()
    for trader_id in sorted(traders):
        trader_data = data[data['account_id'] == trader_id]
        latest_date = trader_data['date'].max()
        latest_row = trader_data[trader_data['date'] == latest_date].iloc[0]
        latest_net = latest_row['net'] if 'net' in data.columns else 'N/A'
        latest_target = latest_row['target_next_pnl'] if 'target_next_pnl' in data.columns else 'N/A'
        print(f'   Trader {trader_id}: {latest_date.strftime("%Y-%m-%d")} (net: {latest_net}, target: {latest_target:.2f})')

    print()

    # Let's check the source database directly
    print('5. CHECKING ACTUAL DATABASE DATA:')
    print('   Looking at actual daily summary data...')

    # Let's also check what the real source data looks like
    import sys
    sys.path.append('src')

    try:
        from data.database_manager import DatabaseManager

        db = DatabaseManager()

        # Get recent data for one trader
        sample_trader = traders[0]
        print(f'   Checking recent data for trader {sample_trader}:')

        recent_data = db.get_account_daily_summary(sample_trader).tail(7)
        if not recent_data.empty:
            print(f'   Database has {len(recent_data)} recent records')
            print(f'   Columns: {list(recent_data.columns)}')
            if 'net' in recent_data.columns:
                net_sum = recent_data['net'].sum()
                print(f'   Recent 7-day net sum from DB: ${net_sum:.2f}')
                print('   Recent net values:', recent_data['net'].tolist())
        else:
            print('   No recent data found in database')

    except Exception as e:
        print(f'   Error accessing database: {e}')

if __name__ == "__main__":
    main()
