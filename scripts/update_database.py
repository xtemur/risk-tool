#!/usr/bin/env python3
"""
Update risk-tool database with latest data from all sources.
This script orchestrates the execution of all data collection scripts.

Usage:
    python update_database.py [--days-back N]

Options:
    --days-back N    Fetch data for the last N days (default: all data since 2023-04-01)
"""

import sys
import subprocess
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import os

def run_script(script_name, args=None):
    """
    Run a Python script with optional arguments.

    Args:
        script_name (str): Name of the script to run
        args (list): Optional list of arguments to pass to the script

    Returns:
        bool: True if successful, False otherwise
    """
    script_path = Path('scripts') / script_name
    cmd = [sys.executable, str(script_path)]

    if args:
        cmd.extend(args)

    print(f"\n{'='*60}")
    print(f"Running {script_name}...")
    print(f"{'='*60}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)

        # Print output
        if result.stdout:
            print(result.stdout)

        # Print errors if any
        if result.stderr:
            print(f"Warnings/Errors:\n{result.stderr}", file=sys.stderr)

        # Check return code
        if result.returncode != 0:
            print(f"❌ {script_name} failed with return code {result.returncode}")
            return False

        print(f"✅ {script_name} completed successfully")
        return True

    except Exception as e:
        print(f"❌ Error running {script_name}: {e}")
        return False

def main():
    """Main function to update all database tables."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Update risk-tool database with latest data from all sources'
    )
    parser.add_argument(
        '--days-back',
        type=int,
        default=None,
        help='Fetch data for the last N days (default: all data since 2023-04-01)'
    )

    args = parser.parse_args()

    # Change to the project root directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    # Calculate date range based on --days-back argument
    if args.days_back:
        # If days-back is specified, fetch last N days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.days_back)
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        print("Risk Tool Database Update (Recent Data)")
        print(f"Fetching last {args.days_back} days of data")
    else:
        # Default: fetch all data since project start
        start_date_str = "2023-04-01"
        end_date_str = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        print("Risk Tool Database Update (Full Sync)")
        print("Fetching all historical data")

    print(f"Start Date: {start_date_str}")
    print(f"End Date: {end_date_str}")

    # Track success of each step
    all_success = True

    # Step 1: Authenticate (this validates credentials)
    print("\nStep 1: Validating authentication...")
    if not run_script('authenticate.py'):
        print("❌ Authentication failed. Please check your credentials.")
        return 1

    # Step 2: Update accounts (always fetch all accounts)
    print("\nStep 2: Updating accounts...")
    if not run_script('save_accounts.py'):
        print("⚠️  Failed to update accounts, but continuing...")
        all_success = False

    # Step 3: Update trades with calculated date range
    print(f"\nStep 3: Updating trades ({start_date_str} to {end_date_str})...")
    if not run_script('save_trades.py', [start_date_str, end_date_str]):
        print("⚠️  Failed to update trades, but continuing...")
        all_success = False


    # Summary
    print(f"\n{'='*60}")
    print("Database Update Summary")
    print(f"{'='*60}")

    if all_success:
        if args.days_back:
            print(f"✅ Last {args.days_back} days update completed successfully!")
        else:
            print("✅ Full database sync completed successfully!")

        # Show database statistics
        try:
            import sqlite3
            conn = sqlite3.connect('data/risk_tool.db')
            cursor = conn.cursor()

            print("\nDatabase Statistics:")

            # Count accounts
            cursor.execute("SELECT COUNT(*) FROM accounts WHERE is_active = 1")
            active_accounts = cursor.fetchone()[0]
            print(f"  - Active accounts: {active_accounts}")

            # Count trades
            cursor.execute("SELECT COUNT(*) FROM trades")
            total_trades = cursor.fetchone()[0]
            cursor.execute("SELECT MIN(trade_date), MAX(trade_date) FROM trades")
            date_range = cursor.fetchone()
            print(f"  - Total trades: {total_trades:,}")
            if date_range[0] and date_range[1]:
                print(f"  - Trade date range: {date_range[0]} to {date_range[1]}")

            # Show recent trades if using days-back
            if args.days_back:
                cursor.execute(
                    "SELECT COUNT(*) FROM trades WHERE trade_date >= ?",
                    (start_date_str,)
                )
                recent_trades = cursor.fetchone()[0]
                print(f"  - Trades from last {args.days_back} days: {recent_trades:,}")

            conn.close()
        except Exception as e:
            print(f"Could not retrieve database statistics: {e}")
    else:
        print("⚠️  Some updates failed. Check the logs above for details.")
        return 1

    if args.days_back:
        print(f"\n✨ Last {args.days_back} days database update complete!")
    else:
        print("\n✨ Full database update complete!")
    return 0

if __name__ == "__main__":
    exit(main())
