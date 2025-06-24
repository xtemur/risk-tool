# src/data_processing.py

import pandas as pd
import numpy as np
import sqlite3
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_trader_day_panel(config: Dict) -> pd.DataFrame:
    """
    Create a clean, regular time-series panel for every active trader.

    Args:
        config: Configuration dictionary containing paths and parameters

    Returns:
        pd.DataFrame: Complete panel DataFrame with trader-day observations
    """
    logger.info("Starting trader-day panel creation...")

    # Load data from SQLite database
    con = sqlite3.connect(config['paths']['raw_data'])

    # Query only active traders
    active_traders = config['active_traders']
    placeholders = ','.join('?' * len(active_traders))
    query = f"SELECT * FROM trades WHERE account_id IN ({placeholders})"

    df = pd.read_sql_query(query, con, params=active_traders)
    con.close()

    logger.info(f"Loaded {len(df)} trades for {len(active_traders)} active traders")

    # Basic type conversions
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df['pnl'] = df['net']  # Alias net to pnl

    # Aggregate fee columns
    fee_cols = ['comm', 'ecn_fee', 'sec', 'orf', 'cat', 'taf', 'ftt', 'nscc', 'acc', 'clr', 'misc']
    existing_fee_cols = [col for col in fee_cols if col in df.columns]
    df['total_fees'] = df[existing_fee_cols].sum(axis=1)

    # Aggregate to daily level
    daily_agg = df.groupby(['account_id', 'trade_date']).agg({
        'pnl': 'sum',
        'gross': 'sum',
        'total_fees': 'sum',
        'qty': 'sum',
        'id': 'count'  # Count of trades
    }).rename(columns={
        'pnl': 'daily_pnl',
        'gross': 'daily_gross',
        'total_fees': 'daily_fees',
        'qty': 'daily_volume',
        'id': 'n_trades'
    })

    # Calculate gross profit and loss
    daily_agg['gross_profit'] = daily_agg['daily_gross'].clip(lower=0)
    daily_agg['gross_loss'] = daily_agg['daily_gross'].clip(upper=0).abs()

    # Create master calendar
    min_date = df['trade_date'].min()
    max_date = df['trade_date'].max()

    # Create business day calendar
    business_days = pd.bdate_range(start=min_date, end=max_date)

    logger.info(f"Creating panel from {min_date} to {max_date} ({len(business_days)} business days)")

    # Build panel data with MultiIndex
    panel_index = pd.MultiIndex.from_product(
        [active_traders, business_days],
        names=['account_id', 'trade_date']
    )

    # Create empty panel
    panel_df = pd.DataFrame(index=panel_index)

    # Join aggregated daily data
    panel_df = panel_df.join(daily_agg, how='left')

    # Reset index to make it easier to work with
    panel_df = panel_df.reset_index()

    # Sort by account_id and trade_date
    panel_df = panel_df.sort_values(['account_id', 'trade_date'])

    logger.info(f"Created panel with shape: {panel_df.shape}")
    logger.info(f"Panel contains {panel_df['account_id'].nunique()} traders")

    return panel_df
