import os
import sys
import sqlite3
import requests
import re
from datetime import datetime
from dotenv import load_dotenv
from authenticate import authenticate

def create_trades_table(cursor):
    """Create trades table if it doesn't exist."""
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            account_id INTEGER NOT NULL,
            trade_date TEXT NOT NULL,
            opened TEXT,
            closed TEXT,
            held TEXT,
            symbol TEXT,
            type TEXT,
            entry REAL,
            exit REAL,
            qty INTEGER,
            gross REAL,
            comm REAL,
            ecn_fee REAL,
            sec REAL,
            orf REAL,
            cat REAL,
            taf REAL,
            ftt REAL,
            nscc REAL,
            acc REAL,
            clr REAL,
            misc REAL,
            net REAL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (account_id) REFERENCES accounts (account_id),
            UNIQUE(account_id, trade_date, opened, symbol, qty, entry) ON CONFLICT REPLACE
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

def fetch_trades(token, account_id, start_date, end_date):
    """
    Fetch trades data from API for a specific account.

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
        'action': 'report',
        'type': 'trades',
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
        raise ConnectionError(f"Failed to fetch trades for account {account_id}: {e}")

def parse_trades_data(response_text, account_id):
    """
    Parse trades response text and extract trade data.

    Args:
        response_text (str): Raw response text
        account_id (int): Account ID

    Returns:
        list: List of trade dictionaries
    """
    trades = []
    lines = response_text.strip().split('\n')

    current_date = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check if line is a date (MM/DD/YY or MM/DD/YYYY format)
        date_match = re.match(r'^(\d{1,2})/(\d{1,2})/(\d{2,4})$', line)
        if date_match:
            month, day, year = date_match.groups()
            # Convert 2-digit year to 4-digit
            if len(year) == 2:
                year = '20' + year
            current_date = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
            continue

        # Skip header lines
        if line.startswith('Opened,Closed') or line.startswith('Equities,'):
            # Check if header format matches expected columns
            if line.startswith('Opened,Closed') and line != 'Opened,Closed,Held,Symbol,Type,Entry,Exit,Qty,Gross,Comm,Ecn Fee,SEC,ORF,CAT,TAF,FTT,NSCC,Acc,Clr,Misc,Net':
                print(f"WARNING: Column headers may have changed. Expected format but got: {line}")
            continue

        # Parse trade data lines
        if current_date and ',' in line:
            parts = line.split(',')
            if len(parts) >= 21:  # Ensure we have all expected columns
                try:
                    trade = {
                        'account_id': account_id,
                        'trade_date': current_date,
                        'opened': parts[0] if parts[0] else None,
                        'closed': parts[1] if parts[1] else None,
                        'held': parts[2] if parts[2] else None,
                        'symbol': parts[3] if parts[3] else None,
                        'type': parts[4] if parts[4] else None,
                        'entry': float(parts[5]) if parts[5] else None,
                        'exit': float(parts[6]) if parts[6] else None,
                        'qty': int(parts[7]) if parts[7] else None,
                        'gross': float(parts[8]) if parts[8] else None,
                        'comm': float(parts[9]) if parts[9] else None,
                        'ecn_fee': float(parts[10]) if parts[10] else None,
                        'sec': float(parts[11]) if parts[11] else None,
                        'orf': float(parts[12]) if parts[12] else None,
                        'cat': float(parts[13]) if parts[13] else None,
                        'taf': float(parts[14]) if parts[14] else None,
                        'ftt': float(parts[15]) if parts[15] else None,
                        'nscc': float(parts[16]) if parts[16] else None,
                        'acc': float(parts[17]) if parts[17] else None,
                        'clr': float(parts[18]) if parts[18] else None,
                        'misc': float(parts[19]) if parts[19] else None,
                        'net': float(parts[20]) if parts[20] else None,
                    }

                    # Only add if we have essential trade data
                    if trade['symbol'] and trade['qty'] is not None:
                        trades.append(trade)

                except (ValueError, IndexError) as e:
                    print(f"Warning: Could not parse trade line: {line[:50]}... Error: {e}")
                    continue

    return trades

def save_trades_to_db(trades, db_path='data/risk_tool.db'):
    """
    Save or update trades in database.

    Args:
        trades (list): List of trade dictionaries
        db_path (str): Path to SQLite database
    """
    if not trades:
        print("No trades to save")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        create_trades_table(cursor)

        trades_processed = 0

        for trade in trades:
            # Insert or replace based on UNIQUE constraint
            cursor.execute('''
                INSERT OR REPLACE INTO trades
                (account_id, trade_date, opened, closed, held, symbol, type, entry, exit, qty,
                 gross, comm, ecn_fee, sec, orf, cat, taf, ftt, nscc, acc, clr, misc, net, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (
                trade['account_id'], trade['trade_date'], trade['opened'], trade['closed'],
                trade['held'], trade['symbol'], trade['type'], trade['entry'], trade['exit'],
                trade['qty'], trade['gross'], trade['comm'], trade['ecn_fee'], trade['sec'],
                trade['orf'], trade['cat'], trade['taf'], trade['ftt'], trade['nscc'],
                trade['acc'], trade['clr'], trade['misc'], trade['net']
            ))

            trades_processed += 1

        conn.commit()
        print(f"Successfully processed {len(trades)} trades: {trades_processed} inserted/updated")

    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def main():
    """Main function to fetch and save trades."""
    if len(sys.argv) != 3:
        print("Usage: python save_trades.py <start_date> <end_date>")
        print("Date format: YYYY-MM-DD")
        print("Example: python save_trades.py 2023-12-01 2023-12-31")
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

        total_trades = 0

        # Fetch trades for each active account
        for account_id in accounts:
            print(f"\nFetching trades for account {account_id} ({start_date} to {end_date})...")

            try:
                response_text = fetch_trades(token, account_id, start_date, end_date)

                # Parse trades data
                trades = parse_trades_data(response_text, account_id)
                print(f"Found {len(trades)} trades for account {account_id}")

                # Save to database
                if trades:
                    save_trades_to_db(trades)
                    total_trades += len(trades)

            except Exception as e:
                print(f"Error processing account {account_id}: {e}")
                continue

        print(f"\nTotal trades processed: {total_trades}")
        print("Trades data successfully saved/updated!")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
