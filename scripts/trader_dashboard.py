import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Trader Dashboard",
    page_icon="📊",
    layout="wide"
)

def get_db_connection():
    """Get database connection."""
    return sqlite3.connect('data/risk_tool.db')

def load_accounts():
    """Load accounts from database."""
    conn = get_db_connection()
    query = '''
        SELECT account_id, account_name, first_traded, last_traded,
               currency, cash, unrealized, is_active, updated_at
        FROM accounts
        ORDER BY account_name
    '''
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def update_account_status(account_id, is_active):
    """Update account active status."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        'UPDATE accounts SET is_active = ?, updated_at = CURRENT_TIMESTAMP WHERE account_id = ?',
        (is_active, account_id)
    )
    conn.commit()
    conn.close()

def bulk_update_accounts(account_ids, is_active):
    """Bulk update multiple accounts."""
    conn = get_db_connection()
    cursor = conn.cursor()
    for account_id in account_ids:
        cursor.execute(
            'UPDATE accounts SET is_active = ?, updated_at = CURRENT_TIMESTAMP WHERE account_id = ?',
            (is_active, account_id)
        )
    conn.commit()
    conn.close()

# Main dashboard
st.title("🎯 Trader Dashboard")
st.markdown("Toggle traders on/off for experiments and forecasting")

# Load data
try:
    df = load_accounts()

    if df.empty:
        st.warning("No accounts found. Run `python scripts/save_accounts.py` first to populate the database.")
        st.stop()

    # Summary metrics
    total_accounts = len(df)
    active_accounts = len(df[df['is_active'] == 1])
    inactive_accounts = total_accounts - active_accounts

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Accounts", total_accounts)
    with col2:
        st.metric("Active Accounts", active_accounts, delta=None)
    with col3:
        st.metric("Inactive Accounts", inactive_accounts)

    st.divider()

    # Bulk operations
    st.subheader("🔧 Bulk Operations")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("🟢 Enable All", use_container_width=True):
            bulk_update_accounts(df['account_id'].tolist(), 1)
            st.rerun()

    with col2:
        if st.button("🔴 Disable All", use_container_width=True):
            bulk_update_accounts(df['account_id'].tolist(), 0)
            st.rerun()

    with col3:
        selected_accounts = st.multiselect(
            "Select accounts",
            options=df['account_id'].tolist(),
            format_func=lambda x: f"{x} - {df[df['account_id']==x]['account_name'].iloc[0]}"
        )

    with col4:
        if selected_accounts:
            col_enable, col_disable = st.columns(2)
            with col_enable:
                if st.button("✅ Enable Selected", use_container_width=True):
                    bulk_update_accounts(selected_accounts, 1)
                    st.rerun()
            with col_disable:
                if st.button("❌ Disable Selected", use_container_width=True):
                    bulk_update_accounts(selected_accounts, 0)
                    st.rerun()

    st.divider()

    # Search and filter
    st.subheader("🔍 Account Management")

    # Search box
    search_term = st.text_input("Search accounts by name or ID", placeholder="Enter account name or ID...")

    # Filter by status
    col1, col2 = st.columns(2)
    with col1:
        status_filter = st.selectbox("Filter by status", ["All", "Active", "Inactive"])
    with col2:
        currency_filter = st.selectbox("Filter by currency", ["All"] + sorted(df['currency'].unique().tolist()))

    # Apply filters
    filtered_df = df.copy()

    if search_term:
        filtered_df = filtered_df[
            (filtered_df['account_name'].str.contains(search_term, case=False, na=False)) |
            (filtered_df['account_id'].astype(str).str.contains(search_term, na=False))
        ]

    if status_filter == "Active":
        filtered_df = filtered_df[filtered_df['is_active'] == 1]
    elif status_filter == "Inactive":
        filtered_df = filtered_df[filtered_df['is_active'] == 0]

    if currency_filter != "All":
        filtered_df = filtered_df[filtered_df['currency'] == currency_filter]

    st.write(f"Showing {len(filtered_df)} of {total_accounts} accounts")

    # Display accounts with toggle switches
    if not filtered_df.empty:
        for _, account in filtered_df.iterrows():
            with st.container():
                col1, col2, col3, col4, col5, col6, col7 = st.columns([1, 3, 2, 2, 1, 2, 2])

                with col1:
                    # Toggle switch for each account
                    current_status = bool(account['is_active'])
                    new_status = st.checkbox(
                        "",
                        value=current_status,
                        key=f"toggle_{account['account_id']}",
                        help=f"Toggle {account['account_name']}"
                    )

                    # Update database if status changed
                    if new_status != current_status:
                        update_account_status(account['account_id'], new_status)
                        st.rerun()

                with col2:
                    status_color = "🟢" if account['is_active'] else "🔴"
                    st.write(f"{status_color} **{account['account_name']}**")

                with col3:
                    st.write(f"ID: {account['account_id']}")

                with col4:
                    st.write(f"Currency: {account['currency']}")

                with col5:
                    st.write(f"Cash: ${account['cash']:,.0f}")

                with col6:
                    st.write(f"Last Traded: {account['last_traded'] or 'Never'}")

                with col7:
                    st.write(f"Updated: {account['updated_at'][:10] if account['updated_at'] else 'N/A'}")

    else:
        st.info("No accounts match your search criteria.")

except sqlite3.OperationalError as e:
    if "no such table" in str(e):
        st.error("Database not found or accounts table doesn't exist.")
        st.info("Run `python scripts/save_accounts.py` first to create the database and populate accounts.")
    else:
        st.error(f"Database error: {e}")
except Exception as e:
    st.error(f"An error occurred: {e}")

# Footer
st.divider()
st.markdown("💡 **Tip**: Changes are saved automatically when you toggle switches. Use bulk operations for faster management.")
