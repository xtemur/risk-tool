#!/usr/bin/env python
"""
Test script to verify database functionality
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.database_manager import DatabaseManager
from src.data.propreports_parser import PropreReportsParser
import pandas as pd
from datetime import datetime, date

def test_database():
    """Test database functionality"""
    print("Testing DatabaseManager...")

    # Initialize
    db = DatabaseManager()
    parser = PropreReportsParser()

    # 1. Test account creation
    print("\n1. Testing account creation...")
    db.save_account("TEST001", "Test Trader 001")
    accounts = db.get_accounts()
    print(f"   Accounts in database: {len(accounts)}")

    # 2. Test parsing sample data
    print("\n2. Testing data parsing...")

    # Check if sample files exist
    totals_file = Path("tbd_3976_2025_04.csv")
    fills_file = Path("fills_6973_2025_04.csv")

    if totals_file.exists():
        print(f"   Parsing totals file: {totals_file}")
        totals_df, report_type = parser.parse_csv_file(totals_file)
        print(f"   Parsed {len(totals_df)} rows, type: {report_type}")

        # Validate data
        validation = parser.validate_data(totals_df, report_type)
        print(f"   Validation: {'PASSED' if validation['is_valid'] else 'FAILED'}")
        if validation['errors']:
            print(f"   Errors: {validation['errors']}")

    if fills_file.exists():
        print(f"   Parsing fills file: {fills_file}")
        fills_df, report_type = parser.parse_csv_file(fills_file)
        print(f"   Parsed {len(fills_df)} rows, type: {report_type}")

        # Validate data
        validation = parser.validate_data(fills_df, report_type)
        print(f"   Validation: {'PASSED' if validation['is_valid'] else 'FAILED'}")
        if validation['errors']:
            print(f"   Errors: {validation['errors']}")

    # 3. Test duplicate handling
    print("\n3. Testing duplicate handling...")

    # Create test data
    test_daily = pd.DataFrame({
        'Date': [date(2024, 1, 1), date(2024, 1, 1), date(2024, 1, 2)],
        'Symbol': ['AAPL', 'AAPL', 'AAPL'],
        'Net P&L': [100, 200, 150],
        'Trades': [5, 10, 7]
    })

    # Save twice to test duplicate handling
    records1 = db.save_daily_summary(test_daily, "TEST001", handle_duplicates='replace')
    print(f"   First save: {records1} records")

    records2 = db.save_daily_summary(test_daily, "TEST001", handle_duplicates='replace')
    print(f"   Second save (replace): {records2} records")

    # Check for duplicates
    duplicates = db.check_duplicates()
    print(f"   Duplicates found: {list(duplicates.keys())}")

    # 4. Test time ordering
    print("\n4. Testing time ordering...")

    # Insert data out of order
    out_of_order = pd.DataFrame({
        'Date': [date(2024, 1, 5), date(2024, 1, 3), date(2024, 1, 4)],
        'Symbol': ['TSLA', 'TSLA', 'TSLA'],
        'Net P&L': [500, 300, 400],
        'Trades': [3, 2, 4]
    })

    db.save_daily_summary(out_of_order, "TEST001")

    # Retrieve and check ordering
    retrieved = db.get_daily_summary(account_id="TEST001")
    print(f"   Retrieved {len(retrieved)} records")
    print("   Date order check:", retrieved['date'].is_monotonic_increasing)

    # 5. Test time series functionality
    print("\n5. Testing time series generation...")
    ts = db.get_trader_time_series("TEST001")
    if not ts.empty:
        print(f"   Time series: {len(ts)} days")
        print(f"   Date range: {ts.index.min()} to {ts.index.max()}")
        print(f"   Total P&L: {ts['net_pl'].sum()}")

    # 6. Test validation
    print("\n6. Testing time consistency validation...")
    validation = db.validate_time_consistency("TEST001")
    print(f"   Validation: {'PASSED' if validation['is_valid'] else 'FAILED'}")
    if validation['issues']:
        print(f"   Issues: {validation['issues']}")

    # 7. Test database stats
    print("\n7. Database statistics:")
    stats = db.get_database_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    # 8. Create views
    print("\n8. Creating time-ordered views...")
    db.create_time_ordered_views()
    print("   Views created successfully")

    # 9. Clean up test data
    print("\n9. Cleaning up test data...")
    with db.get_connection() as conn:
        conn.execute("DELETE FROM daily_summary WHERE account_id = 'TEST001'")
        conn.execute("DELETE FROM accounts WHERE account_id = 'TEST001'")
        conn.commit()
    print("   Test data cleaned up")

    print("\n✅ All tests completed!")


if __name__ == "__main__":
    test_database()
