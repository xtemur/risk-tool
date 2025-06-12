#!/usr/bin/env python3
"""
Check database vs feature data mismatch
"""
import sys
sys.path.append('src')
from data.database_manager import DatabaseManager
import pandas as pd

def main():
    db = DatabaseManager()

    print('=== DATABASE ANALYSIS ===')
    print()

    # Check what's in the database vs feature data
    print('1. DATABASE SUMMARY RECORDS:')
    accounts = db.get_accounts()
    for _, account in accounts.iterrows():
        account_id = account['account_id']
        summary = db.get_account_daily_summary(account_id)
        print(f'   Trader {account_id}: {len(summary)} records in DB')
        if not summary.empty:
            latest_date = summary['date'].max()
            recent_7 = summary.tail(7)
            net_sum = recent_7['net'].sum()
            print(f'     Latest date: {latest_date}, Recent 7-day net: ${net_sum:.2f}')
            print(f'     Recent dates: {recent_7["date"].dt.strftime("%Y-%m-%d").tolist()}')
            print(f'     Recent net values: {recent_7["net"].tolist()}')
        else:
            print(f'     No data found for trader {account_id}')

        # Just check first trader in detail
        if account_id == 3942:
            break

    print()
    print('2. FEATURE DATA VS DATABASE ISSUE IDENTIFIED:')
    print('   - Feature data (features_demo.csv) has mostly NaN in "net" column for recent dates')
    print('   - Database has actual net values')
    print('   - This indicates feature engineering is not preserving recent net values')
    print()
    print('3. SOLUTIONS:')
    print('   A) Fix the recent performance calculation to use database directly')
    print('   B) Fix the feature engineering to preserve net values')
    print('   C) Use alternative performance metrics from available data')

if __name__ == "__main__":
    main()
