#!/usr/bin/env python
"""
Test script to verify handling of both Equities and Options accounts
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.database_manager import DatabaseManager
from src.data.propreports_parser import PropreReportsParser
import pandas as pd
from datetime import datetime, date

def test_account_types():
    """Test handling of different account types"""
    print("Testing Account Type Handling...")

    # Initialize
    db = DatabaseManager()
    parser = PropreReportsParser()

    print("\n1. Creating test data for both account types...")

    # Create test Equities account data
    equities_data = pd.DataFrame({
        'Date': ['2024-01-01', '2024-01-02'],
        'Type': ['Eq', 'Eq'],
        'Orders': [10, 15],
        'Fills': [8, 12],
        'Qty': [1000, 1500],
        'Gross': [500.00, 750.00],
        'Comm': [10.00, 15.00],
        'Ecn Fee': [2.00, 3.00],
        'SEC': [0.50, 0.75],
        'ORF': [0.25, 0.35],
        'CAT': [0.10, 0.15],
        'TAF': [0.05, 0.08],
        'FTT': [0.00, 0.00],
        'NSCC': [0.15, 0.20],
        'Acc': [0.00, 0.00],
        'Clr': [1.00, 1.50],
        'Misc': [0.00, 0.00],
        'Trade Fees': [14.05, 21.03],
        'Net': [485.95, 728.97],
        'Fee: Software & MD': [50.00, 50.00],  # Equities specific
        'Fee: VAT': [10.00, 10.00],            # Equities specific
        'Adj Fees': [74.05, 81.03],
        'Adj Net': [425.95, 668.97],
        'Unrealized Δ': [0.00, 0.00],
        'Total Δ': [425.95, 668.97],
        'Transfer: Deposit': [0.00, 0.00],
        'Transfers': [0.00, 0.00],
        'Cash': [10425.95, 11094.92],
        'Unrealized': [0.00, 0.00],
        'End Balance': [10425.95, 11094.92]
    })

    # Create test Options account data
    options_data = pd.DataFrame({
        'Date': ['2024-01-01', '2024-01-02'],
        'Type': ['Op', 'Op'],
        'Orders': [5, 8],
        'Fills': [4, 6],
        'Qty': [10, 15],
        'Gross': [1500.00, 2250.00],
        'Comm': [20.00, 30.00],
        'Ecn Fee': [0.00, 0.00],
        'SEC': [0.00, 0.00],
        'ORF': [1.00, 1.50],
        'CAT': [0.50, 0.75],
        'TAF': [0.00, 0.00],
        'FTT': [0.00, 0.00],
        'NSCC': [0.00, 0.00],
        'Acc': [0.00, 0.00],
        'Clr': [5.00, 7.50],
        'Misc': [0.00, 0.00],
        'Trade Fees': [26.50, 39.75],
        'Net': [1473.50, 2210.25],
        'Fee: Daily Interest': [25.00, 25.00],  # Options specific
        'Adj Fees': [51.50, 64.75],
        'Adj Net': [1448.50, 2185.25],
        'Unrealized Δ': [100.00, -50.00],
        'Total Δ': [1548.50, 2135.25],
        'Transfer: Deposit': [0.00, 0.00],
        'Transfers': [0.00, 0.00],
        'Cash': [51548.50, 53683.75],
        'Unrealized': [100.00, 50.00],
        'End Balance': [51648.50, 53733.75]
    })

    # Save test accounts
    db.save_account("TEST_EQ_001", "Test Equities Trader")
    db.save_account("TEST_OP_001", "Test Options Trader")

    # Save data
    print("\n2. Saving test data to database...")
    eq_records = db.save_account_daily_summary(equities_data, "TEST_EQ_001")
    print(f"   Saved {eq_records} equities records")

    op_records = db.save_account_daily_summary(options_data, "TEST_OP_001")
    print(f"   Saved {op_records} options records")

    print("\n3. Testing account type detection...")

    # Get account summaries
    eq_summary = db.get_account_summary("TEST_EQ_001")
    op_summary = db.get_account_summary("TEST_OP_001")

    print(f"\n   Equities Account Type: {eq_summary['account_type']}")
    print(f"   Options Account Type: {op_summary['account_type']}")

    # Verify data retrieval
    print("\n4. Verifying data retrieval...")

    eq_data = db.get_account_daily_summary(account_id="TEST_EQ_001")
    op_data = db.get_account_daily_summary(account_id="TEST_OP_001")

    print(f"\n   Equities account columns: {list(eq_data.columns)[:10]}...")
    print(f"   Options account columns: {list(op_data.columns)[:10]}...")

    # Check for optional columns
    print("\n5. Checking optional columns...")

    if 'fee_software_md' in eq_data.columns:
        print(f"   ✓ Equities has fee_software_md: ${eq_data['fee_software_md'].sum():.2f}")
    if 'fee_vat' in eq_data.columns:
        print(f"   ✓ Equities has fee_vat: ${eq_data['fee_vat'].sum():.2f}")

    if 'fee_daily_interest' in op_data.columns:
        print(f"   ✓ Options has fee_daily_interest: ${op_data['fee_daily_interest'].sum():.2f}")

    # Test parser with sample CSV content
    print("\n6. Testing parser with sample CSV content...")

    # Create sample equities CSV
    eq_csv_content = """Date,Type,Orders,Fills,Qty,Gross,Comm,Ecn Fee,SEC,ORF,CAT,TAF,FTT,NSCC,Acc,Clr,Misc,Trade Fees,Net,Fee: Software & MD,Fee: VAT,Adj Fees,Adj Net,Unrealized Δ,Total Δ,Transfer: Deposit,Transfers,Cash,Unrealized,End Balance
2024-01-03,Eq,20,18,2000,1000.00,20.00,4.00,1.00,0.50,0.20,0.10,0.00,0.30,0.00,2.00,0.00,28.10,971.90,50.00,10.00,88.10,911.90,0.00,911.90,0.00,0.00,12006.82,0.00,12006.82
Equities"""

    eq_temp_file = Path("temp_eq_test.csv")
    with open(eq_temp_file, 'w') as f:
        f.write(eq_csv_content)

    eq_df, eq_type = parser.parse_csv_file(eq_temp_file)
    print(f"   Parsed equities file: {len(eq_df)} rows, type: {eq_type}")
    print(f"   Detected account type: {parser.detect_account_type(eq_df)}")

    eq_temp_file.unlink()

    # Create sample options CSV
    op_csv_content = """Date,Type,Orders,Fills,Qty,Gross,Comm,Ecn Fee,SEC,ORF,CAT,TAF,FTT,NSCC,Acc,Clr,Misc,Trade Fees,Net,Fee: Daily Interest,Adj Fees,Adj Net,Unrealized Δ,Total Δ,Transfer: Deposit,Transfers,Cash,Unrealized,End Balance
2024-01-03,Op,10,8,20,3000.00,40.00,0.00,0.00,2.00,1.00,0.00,0.00,0.00,0.00,10.00,0.00,53.00,2947.00,25.00,78.00,2872.00,-100.00,2772.00,0.00,0.00,56505.75,-50.00,56455.75
Options"""

    op_temp_file = Path("temp_op_test.csv")
    with open(op_temp_file, 'w') as f:
        f.write(op_csv_content)

    op_df, op_type = parser.parse_csv_file(op_temp_file)
    print(f"   Parsed options file: {len(op_df)} rows, type: {op_type}")
    print(f"   Detected account type: {parser.detect_account_type(op_df)}")

    op_temp_file.unlink()

    # Database stats
    print("\n7. Database statistics...")
    stats = db.get_database_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    print("\n✓ Account type handling test completed successfully!")

    # Cleanup test data
    print("\n8. Cleaning up test data...")
    with db.get_connection() as conn:
        conn.execute("DELETE FROM account_daily_summary WHERE account_id LIKE 'TEST_%'")
        conn.execute("DELETE FROM accounts WHERE account_id LIKE 'TEST_%'")
        conn.commit()
    print("   Test data cleaned up")


if __name__ == "__main__":
    test_account_types()
