import os
import sys
import sqlite3
import requests
import re
from datetime import datetime
from dotenv import load_dotenv
from authenticate import authenticate

def create_open_positions_table(cursor):
    """Create open_positions table if it doesn't exist."""
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS open_positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            account_id INTEGER NOT NULL,
            position_date TEXT NOT NULL,
            symbol TEXT NOT NULL,
            qty INTEGER NOT NULL,
            avg_price REAL,
            close_price REAL,
            ccy TEXT,
            spot REAL,
            cost REAL,
            market_value REAL,
            unrealized_delta REAL,
            unrealized REAL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (account_id) REFERENCES accounts (account_id),
            UNIQUE(account_id, position_date, symbol) ON CONFLICT REPLACE
        )
    ''')

def create_position_summaries_table(cursor):
    """Create position_summaries table if it doesn't exist."""
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS position_summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            account_id INTEGER NOT NULL,
            summary_date TEXT NOT NULL,
            position_type TEXT NOT NULL,  -- 'Long', 'Short', or 'Closed'
            cost REAL,
            market_value REAL,
            unrealized_delta REAL,
            unrealized REAL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (account_id) REFERENCES accounts (account_id),
            UNIQUE(account_id, summary_date, position_type) ON CONFLICT REPLACE
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

def fetch_open_positions(token, account_id, start_date, end_date):
    """
    Fetch open positions data from API for a specific account.

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
        'type': 'openPositions',
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
        raise ConnectionError(f"Failed to fetch open positions for account {account_id}: {e}")

def parse_open_positions_data(response_text, account_id):
    """
    Parse open positions response text and extract position data.

    Args:
        response_text (str): Raw response text
        account_id (int): Account ID

    Returns:
        tuple: (positions list, summaries list)
    """
    positions = []
    summaries = []

    # Handle bytes response
    if isinstance(response_text, bytes):
        response_text = response_text.decode('utf-8', errors='replace')

    # Check for no data response
    if 'No data available' in response_text:
        return positions, summaries

    lines = response_text.strip().split('\n')

    current_date = None
    in_summary_section = False

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check if line is a date (M/D/YYYY format)
        date_match = re.match(r'^(\d{1,2})/(\d{1,2})/(\d{4})$', line)
        if date_match:
            month, day, year = date_match.groups()
            current_date = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
            in_summary_section = False
            continue

        # Skip header line
        if line.startswith('Symbol,Qty,Avg Price'):
            continue

        # Parse position data lines
        if current_date and ',' in line:
            parts = line.split(',')

            # Summary lines (Long:, Short:, Closed:)
            if parts[0] in ['Long:', 'Short:', 'Closed:'] and len(parts) >= 10:
                in_summary_section = True
                position_type = parts[0].rstrip(':')

                try:
                    summary = {
                        'account_id': account_id,
                        'summary_date': current_date,
                        'position_type': position_type
                    }

                    if position_type in ['Long', 'Short']:
                        summary['cost'] = float(parts[6]) if parts[6] else 0
                        summary['market_value'] = float(parts[7]) if parts[7] else 0
                        summary['unrealized_delta'] = float(parts[8]) if parts[8] else 0
                        summary['unrealized'] = float(parts[9]) if parts[9] else 0
                    elif position_type == 'Closed':
                        # For closed positions, the PnL is in column 8
                        summary['cost'] = None
                        summary['market_value'] = None
                        summary['unrealized_delta'] = float(parts[8]) if parts[8] else 0
                        summary['unrealized'] = float(parts[8]) if parts[8] else 0

                    summaries.append(summary)

                except (ValueError, IndexError) as e:
                    print(f"Warning: Could not parse summary line: {line[:50]}... Error: {e}")
                continue

            # Skip total summary line (starts with empty field)
            if in_summary_section and not parts[0]:
                continue

            # Individual position lines
            if not in_summary_section and len(parts) >= 10:  # Changed from 11 to 10
                try:
                    symbol = parts[0]
                    if symbol and symbol not in ['Long:', 'Short:', 'Closed:']:
                        qty = int(parts[1]) if parts[1] else 0
                        if qty != 0:  # Only save positions with non-zero quantity
                            position = {
                                'account_id': account_id,
                                'position_date': current_date,
                                'symbol': symbol,
                                'qty': qty,
                                'avg_price': float(parts[2]) if parts[2] else None,
                                'close_price': float(parts[3]) if parts[3] else None,
                                'ccy': parts[4] if parts[4] else 'USD',
                                'spot': float(parts[5]) if parts[5] else 1,
                                'cost': float(parts[6]) if parts[6] else None,
                                'market_value': float(parts[7]) if parts[7] else None,
                                'unrealized_delta': float(parts[8]) if parts[8] else None,
                                'unrealized': float(parts[9]) if parts[9] else None
                            }
                            positions.append(position)

                except (ValueError, IndexError) as e:
                    print(f"Warning: Could not parse position line: {line[:50]}... Error: {e}")
                    continue

    return positions, summaries

def save_positions_to_db(positions, summaries, db_path='data/risk_tool.db'):
    """
    Save or update open positions and position summaries in database.

    Args:
        positions (list): List of position dictionaries
        summaries (list): List of summary dictionaries
        db_path (str): Path to SQLite database
    """
    if not positions and not summaries:
        print("No positions or summaries to save")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        create_open_positions_table(cursor)
        create_position_summaries_table(cursor)

        positions_processed = 0
        summaries_processed = 0

        # Save individual positions
        for position in positions:
            cursor.execute('''
                INSERT OR REPLACE INTO open_positions
                (account_id, position_date, symbol, qty, avg_price, close_price, ccy, spot,
                 cost, market_value, unrealized_delta, unrealized, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (
                position['account_id'], position['position_date'], position['symbol'],
                position['qty'], position['avg_price'], position['close_price'],
                position['ccy'], position['spot'], position['cost'], position['market_value'],
                position['unrealized_delta'], position['unrealized']
            ))
            positions_processed += 1

        # Save position summaries
        for summary in summaries:
            cursor.execute('''
                INSERT OR REPLACE INTO position_summaries
                (account_id, summary_date, position_type, cost, market_value,
                 unrealized_delta, unrealized, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (
                summary['account_id'], summary['summary_date'], summary['position_type'],
                summary.get('cost'), summary.get('market_value'),
                summary.get('unrealized_delta'), summary.get('unrealized')
            ))
            summaries_processed += 1

        conn.commit()
        print(f"Successfully processed {positions_processed} positions and {summaries_processed} summaries")

    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def main():
    """Main function to fetch and save open positions."""
    if len(sys.argv) != 3:
        print("Usage: python save_open_positions.py <start_date> <end_date>")
        print("Date format: YYYY-MM-DD")
        print("Example: python save_open_positions.py 2025-06-01 2025-06-10")
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

        total_positions = 0
        total_summaries = 0

        # Fetch open positions for each active account
        for account_id in accounts:
            print(f"\nFetching open positions for account {account_id} ({start_date} to {end_date})...")

            try:
                response_text = fetch_open_positions(token, account_id, start_date, end_date)


                # Parse positions data
                positions, summaries = parse_open_positions_data(response_text, account_id)
                print(f"Found {len(positions)} positions and {len(summaries)} summaries for account {account_id}")

                # Save to database
                if positions or summaries:
                    save_positions_to_db(positions, summaries)
                    total_positions += len(positions)
                    total_summaries += len(summaries)

            except Exception as e:
                print(f"Error processing account {account_id}: {e}")
                continue

        print(f"\nTotal positions processed: {total_positions}")
        print(f"Total summaries processed: {total_summaries}")
        print("Open positions data successfully saved/updated!")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
