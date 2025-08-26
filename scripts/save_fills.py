import os
import sys
import sqlite3
import requests
import re
from datetime import datetime
from dotenv import load_dotenv
from authenticate import authenticate

def create_fills_table(cursor):
    """Create fills table if it doesn't exist."""
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS fills (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            account_id INTEGER NOT NULL,
            datetime TEXT NOT NULL,
            account TEXT,
            side TEXT,
            qty INTEGER,
            symbol TEXT,
            price REAL,
            route TEXT,
            liq TEXT,
            comm REAL,
            ecn_fee REAL,
            sec REAL,
            taf REAL,
            nscc REAL,
            clr REAL,
            misc REAL,
            order_id TEXT,
            fill_id TEXT,
            currency TEXT,
            isin TEXT,
            cusip TEXT,
            status TEXT,
            prop_reports_id TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (account_id) REFERENCES accounts (account_id),
            UNIQUE(account_id, datetime, symbol, qty, price, fill_id) ON CONFLICT REPLACE
        )
    ''')

def get_active_accounts(db_path='data/risk_tool.db'):
    """Get list of active account IDs from database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        cursor.execute('SELECT account_id FROM accounts WHERE is_active = 1')
        accounts = [row[0] for row in cursor.fetchall()]
        return accounts
    finally:
        conn.close()

def fetch_fills(token, account_id, start_date, end_date):
    """
    Fetch fills data from API for a specific account.

    Args:
        token (str): Authentication token
        account_id (int): Account ID
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format

    Returns:
        str: Response text
    """
    load_dotenv()
    api_url = os.getenv('API_URL')

    payload = {
        'action': 'fills',
        'token': token,
        'accountId': account_id,
        'startDate': start_date,
        'endDate': end_date
    }

    try:
        response = requests.post(
            api_url,
            data=payload,
            timeout=30,
            headers={'Content-Type': 'application/x-www-form-urlencoded'}
        )
        response.raise_for_status()
        return response.text

    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Failed to fetch fills for account {account_id}: {e}")

def parse_fills_data(response_text, account_id):
    """
    Parse fills response text and extract fill data.

    Args:
        response_text (str): Raw response text
        account_id (int): Account ID

    Returns:
        list: List of fill dictionaries
    """
    fills = []
    lines = response_text.strip().split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Skip header lines
        if line.startswith('Date/Time,Account,B/S'):
            continue

        # Parse fill data lines
        if ',' in line:
            parts = line.split(',')
            if len(parts) >= 21:  # Ensure we have all expected columns
                try:
                    fill = {
                        'account_id': account_id,
                        'datetime': parts[0] if parts[0] else None,
                        'account': parts[1] if parts[1] else None,
                        'side': parts[2] if parts[2] else None,  # B/S (Buy/Sell) or T
                        'qty': int(parts[3]) if parts[3] else None,
                        'symbol': parts[4] if parts[4] else None,
                        'price': float(parts[5]) if parts[5] else None,
                        'route': parts[6] if parts[6] else None,
                        'liq': parts[7] if parts[7] else None,
                        'comm': float(parts[8]) if parts[8] else None,
                        'ecn_fee': float(parts[9]) if parts[9] else None,
                        'sec': float(parts[10]) if parts[10] else None,
                        'taf': float(parts[11]) if parts[11] else None,
                        'nscc': float(parts[12]) if parts[12] else None,
                        'clr': float(parts[13]) if parts[13] else None,
                        'misc': float(parts[14]) if parts[14] else None,
                        'order_id': parts[15] if parts[15] else None,
                        'fill_id': parts[16] if parts[16] else None,
                        'currency': parts[17] if parts[17] else None,
                        'isin': parts[18] if parts[18] else None,
                        'cusip': parts[19] if parts[19] else None,
                        'status': parts[20] if len(parts) > 20 and parts[20] else None,
                        'prop_reports_id': parts[21] if len(parts) > 21 and parts[21] else None,
                    }

                    # Only add if we have essential fill data
                    if fill['symbol'] and fill['qty'] is not None and fill['price'] is not None:
                        fills.append(fill)

                except (ValueError, IndexError) as e:
                    print(f"Warning: Could not parse fill line: {line[:50]}... Error: {e}")
                    continue

    return fills

def save_fills_to_db(fills, db_path='data/risk_tool.db'):
    """
    Save or update fills in database.

    Args:
        fills (list): List of fill dictionaries
        db_path (str): Path to SQLite database
    """
    if not fills:
        print("No fills to save")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        create_fills_table(cursor)

        fills_processed = 0

        for fill in fills:
            # Insert or replace based on UNIQUE constraint
            cursor.execute('''
                INSERT OR REPLACE INTO fills
                (account_id, datetime, account, side, qty, symbol, price, route, liq,
                 comm, ecn_fee, sec, taf, nscc, clr, misc, order_id, fill_id, currency,
                 isin, cusip, status, prop_reports_id, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (
                fill['account_id'], fill['datetime'], fill['account'], fill['side'],
                fill['qty'], fill['symbol'], fill['price'], fill['route'], fill['liq'],
                fill['comm'], fill['ecn_fee'], fill['sec'], fill['taf'], fill['nscc'],
                fill['clr'], fill['misc'], fill['order_id'], fill['fill_id'], fill['currency'],
                fill['isin'], fill['cusip'], fill['status'], fill['prop_reports_id']
            ))

            fills_processed += 1

        conn.commit()
        print(f"Successfully processed {len(fills)} fills: {fills_processed} inserted/updated")

    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def main():
    """Main function to fetch and save fills."""
    if len(sys.argv) != 3:
        print("Usage: python save_fills.py <start_date> <end_date>")
        print("Date format: YYYY-MM-DD")
        print("Example: python save_fills.py 2023-12-01 2023-12-31")
        return 1

    start_date = sys.argv[1]
    end_date = sys.argv[2]

    # Validate date format
    try:
        datetime.strptime(start_date, '%Y-%m-%d')
        datetime.strptime(end_date, '%Y-%m-%d')
    except ValueError:
        print("Error: Invalid date format. Use YYYY-MM-DD")
        return 1

    try:
        # Authenticate and get token
        print("Authenticating...")
        token = authenticate()
        print("Authentication successful")

        # Get active accounts
        print("Getting active accounts...")
        accounts = get_active_accounts()
        print(f"Found {len(accounts)} active accounts: {accounts}")

        total_fills = 0

        # Fetch fills for each active account
        for account_id in accounts:
            print(f"\nFetching fills for account {account_id} ({start_date} to {end_date})...")

            try:
                response_text = fetch_fills(token, account_id, start_date, end_date)

                # Parse fills data
                fills = parse_fills_data(response_text, account_id)
                print(f"Found {len(fills)} fills for account {account_id}")

                # Save to database
                if fills:
                    save_fills_to_db(fills)
                    total_fills += len(fills)

            except Exception as e:
                print(f"Error processing account {account_id}: {e}")
                continue

        print(f"\nTotal fills processed: {total_fills}")
        print("Fills data successfully saved/updated!")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
