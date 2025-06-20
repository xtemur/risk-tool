import os
import sqlite3
import requests
import csv
from io import StringIO
from dotenv import load_dotenv
from authenticate import authenticate

def create_accounts_table(cursor):
    """Create accounts table if it doesn't exist."""
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS accounts (
            account_id INTEGER PRIMARY KEY,
            account_name TEXT NOT NULL,
            first_traded TEXT,
            last_traded TEXT,
            currency TEXT,
            cash REAL,
            unrealized REAL,
            is_active BOOLEAN DEFAULT 1,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

def fetch_accounts(token):
    """
    Fetch accounts data from API.

    Args:
        token (str): Authentication token

    Returns:
        str: CSV response text
    """
    load_dotenv()
    api_url = os.getenv('API_URL')

    payload = {
        'action': 'accounts',
        'page': '1',
        'groupId': '-2',
        'token': token
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
        raise ConnectionError(f"Failed to fetch accounts: {e}")

def parse_accounts_csv(csv_text):
    """
    Parse CSV response and extract account data.

    Args:
        csv_text (str): CSV response text

    Returns:
        list: List of account dictionaries
    """
    accounts = []
    lines = csv_text.strip().split('\n')

    # Find the header line (starts with "Account Id")
    header_line_idx = None
    for i, line in enumerate(lines):
        if line.startswith('Account Id'):
            header_line_idx = i
            break

    if header_line_idx is None:
        raise ValueError("Could not find CSV header in response")

    # Extract CSV data (header + data lines, excluding page info)
    csv_lines = []
    for i in range(header_line_idx, len(lines)):
        line = lines[i].strip()
        if line and not line.startswith('Page'):
            csv_lines.append(line)

    # Parse CSV
    csv_data = '\n'.join(csv_lines)
    reader = csv.DictReader(StringIO(csv_data))

    for row in reader:
        account = {
            'account_id': int(row['Account Id']),
            'account_name': row['Account Name'],
            'first_traded': row['First Traded'] if row['First Traded'] else None,
            'last_traded': row['Last Traded'] if row['Last Traded'] else None,
            'currency': row['Currency'],
            'cash': float(row['Cash']) if row['Cash'] else 0.0,
            'unrealized': float(row['Unrealized']) if row['Unrealized'] else 0.0
        }
        accounts.append(account)

    return accounts

def save_accounts_to_db(accounts, db_path='data/risk_tool.db'):
    """
    Save or update accounts in database.
    On first run: inserts all account data
    On subsequent runs: updates only last_traded, currency, cash, unrealized, updated_at

    Args:
        accounts (list): List of account dictionaries
        db_path (str): Path to SQLite database
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        create_accounts_table(cursor)

        new_accounts = 0
        updated_accounts = 0

        for account in accounts:
            # Check if account exists
            cursor.execute('SELECT account_id FROM accounts WHERE account_id = ?', (account['account_id'],))
            exists = cursor.fetchone()

            if exists:
                # Update existing account - only update specified columns
                cursor.execute('''
                    UPDATE accounts
                    SET last_traded = ?, currency = ?, cash = ?, unrealized = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE account_id = ?
                ''', (
                    account['last_traded'],
                    account['currency'],
                    account['cash'],
                    account['unrealized'],
                    account['account_id']
                ))
                updated_accounts += 1
            else:
                # Insert new account with all data (is_active defaults to 1)
                cursor.execute('''
                    INSERT INTO accounts
                    (account_id, account_name, first_traded, last_traded, currency, cash, unrealized, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (
                    account['account_id'],
                    account['account_name'],
                    account['first_traded'],
                    account['last_traded'],
                    account['currency'],
                    account['cash'],
                    account['unrealized']
                ))
                new_accounts += 1

        conn.commit()
        print(f"Successfully processed {len(accounts)} accounts: {new_accounts} new, {updated_accounts} updated")

    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def main():
    """Main function to fetch and save accounts."""
    try:
        # Authenticate and get token
        print("Authenticating...")
        token = authenticate()
        print("Authentication successful")

        # Fetch accounts data
        print("Fetching accounts data...")
        csv_response = fetch_accounts(token)

        # Parse CSV response
        print("Parsing accounts data...")
        accounts = parse_accounts_csv(csv_response)
        print(f"Found {len(accounts)} accounts")

        # Save to database
        print("Saving to database...")
        save_accounts_to_db(accounts)

        print("Accounts data successfully saved/updated!")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
