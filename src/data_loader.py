import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

#Symbol,Orders,Fills,Qty,Gross,Comm,Ecn Fee,SEC,ORF,CAT,TAF,FTT,NSCC,Acc,Clr,Misc,Net,Unrealized δ,Total δ,Unrealized,Date,account_id,trader_name
#Date/Time,Account,B/S,Qty,Symbol,Price,Route,Liq,Comm,Ecn Fee,SEC,ORF,CAT,TAF,FTT,NSCC,Acc,Clr,Misc,Order Id,Fill Id,Currency,ISIN,CUSIP,Status,PropReports Id

class DataLoader:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        with open("config/trader_accounts.yaml", "r") as f:
            self.traders = yaml.safe_load(f)["traders"]

        self.data_path = Path("data")
        self.setup_logging()
        self.fee_columns = [
            "Comm", "Ecn Fee", "SEC", "ORF", "CAT", "TAF", "FTT", "NSCC", "Acc", "Clr", "Misc"
        ]
        self.column_mapping = {
            # Common columns
            'Symbol': 'symbol',
            'Qty': 'qty',

            # Totals-specific columns
            'Date': 'date',
            'Orders': 'orders_count',
            'Fills': 'fills_count',
            'Gross': 'gross_pnl',
            'Net': 'net_pnl',
            'Unrealized δ': 'unrealized_delta',
            'Total δ': 'total_delta',
            'Unrealized': 'unrealized_pnl',
            'account_id': 'account_id',
            'trader_name': 'trader_name',

            # Fills-specific columns (only useful ones)
            'Date/Time': 'datetime',
            'Account': 'trader_name',
            'B/S': 'trade_side',
            'Price': 'price',
            'Order Id': 'order_id'
        }

        # Columns to drop immediately (never used)
        self.columns_to_drop = [
            'Route', 'Liq', 'Fill Id', 'Currency', 'ISIN', 'CUSIP',
            'Status', 'PropReports Id'
        ]


    def standardize_totals_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize totals (daily aggregate) data"""
        # Drop unnecessary columns immediately
        columns_to_drop = [col for col in self.columns_to_drop if col in df.columns]
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)
            self.logger.debug(f"Dropped unnecessary columns from totals: {columns_to_drop}")

        # Rename columns to standard format
        df = df.rename(columns=self.column_mapping)

        # Convert date column
        df['date'] = pd.to_datetime(df['date'])
        df['symbol'] = df['symbol'].astype(str)

        # Create total fees column
        existing_fee_columns = [col for col in self.fee_columns if col in df.columns]
        if existing_fee_columns:
            df['total_fees'] = df[existing_fee_columns].fillna(0).sum(axis=1)
        else:
            df['total_fees'] = 0

        df = df.drop(columns=existing_fee_columns, errors='ignore')


        # Calculate net PnL if missing (gross - total fees)
        if 'net_pnl' not in df.columns and 'gross_pnl' in df.columns:
            df['net_pnl'] = df['gross_pnl'] - df['total_fees']

        # Fill missing numeric columns with 0
        numeric_columns = [
            'orders_count', 'fills_count', 'quantity', 'gross_pnl',
            'net_pnl', 'total_fees', 'unrealized_delta', 'total_delta', 'unrealized_pnl'
        ]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    def standardize_fills_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize fills (transaction-level) data"""

        # Drop unnecessary columns immediately
        columns_to_drop = [col for col in self.columns_to_drop if col in df.columns]
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)
            self.logger.debug(f"Dropped unnecessary columns from fills: {columns_to_drop}")

        # Rename columns to standard format
        df = df.rename(columns=self.column_mapping)

        # Convert date column
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        df['symbol'] = df['symbol'].astype(str)

        # Convert numeric columns
        numeric_columns = ['quantity', 'price'] + [
            col for col in self.fee_columns if col in df.columns
        ]

        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Create total fees for fills
        existing_fee_columns = [col for col in self.fee_columns if col in df.columns]
        if existing_fee_columns:
            df['total_fees'] = df[existing_fee_columns].sum(axis=1)
        else:
            df['total_fees'] = 0

        df = df.drop(columns=existing_fee_columns, errors='ignore')

        # Calculate trade value
        if 'quantity' in df.columns and 'price' in df.columns:
            df['trade_value'] = df['quantity'] * df['price']

        return df

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("logs/data_loader.log"),
                logging.StreamHandler()
            ],
        )
        self.logger = logging.getLogger(__name__)

    def load_trader_data( self, account_id: str, start_date: str = "", end_date: str = "") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load fills and totals data for a specific trader"""
        try:
            # Load fills data
            fills_path = self.data_path / "raw" / f"{account_id}_fills.csv"
            fills_df = pd.read_csv(fills_path)
            fills_df = self.standardize_fills_data(fills_df)

            # Load totals data
            totals_path = self.data_path / "raw" / f"{account_id}_totals.csv"
            totals_df = pd.read_csv(totals_path)
            totals_df = self.standardize_totals_data(totals_df)

            # Sort data by date
            fills_df = fills_df.sort_values("datetime").reset_index(drop=True)
            totals_df = totals_df.sort_values("date").reset_index(drop=True)

            # Filter by date range if provided
            if start_date:
                start_date_dt = pd.to_datetime(start_date)
                fills_df = fills_df.loc[fills_df["datetime"] >= start_date_dt].copy()
                totals_df = totals_df.loc[totals_df["date"] >= start_date_dt].copy()

            if end_date:
                end_date_dt = pd.to_datetime(end_date)
                fills_df = fills_df.loc[fills_df["datetime"] <= end_date_dt].copy()
                totals_df = totals_df.loc[totals_df["date"] <= end_date_dt].copy()

            self.logger.info(
                f"Loaded data for {account_id}: {len(totals_df['date'].unique())} trading days"
            )
            return fills_df, totals_df

        except Exception as e:
            self.logger.error(f"Error loading data for {account_id}: {str(e)}")
            return pd.DataFrame(), pd.DataFrame()

    def load_all_traders_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Load data for all traders"""
        all_data = {}

        for trader in self.traders:
            account_id = trader["account_id"]
            fills_df, totals_df = self.load_trader_data(account_id)

            if not totals_df.empty:
                all_data[account_id] = {
                    "fills": fills_df,
                    "totals": totals_df,
                    "name": trader.get("name", account_id),
                    "strategy": trader.get("strategy", "Unknown"),
                }

        self.logger.info(f"Successfully loaded data for {len(all_data)} traders")
        return all_data

    def create_master_dataset(self, all_data: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Combine all trader data into master dataset"""
        master_totals = []
        master_fills = []

        for account_id, data in all_data.items():
            totals_df = data["totals"].copy()
            totals_df["account_id"] = account_id
            totals_df["trader_name"] = data["name"]

            fills_df = data["fills"].copy()
            fills_df["account_id"] = account_id
            fills_df["trader_name"] = data["name"]

            master_totals.append(totals_df)
            master_fills.append(fills_df)

        master_totals_df = pd.concat(master_totals, ignore_index=True)
        master_fills_df = pd.concat(master_fills, ignore_index=True)

        master_totals_df = master_totals_df.sort_values(["account_id", "date"]).reset_index(drop=True)
        master_fills_df = master_fills_df.sort_values(["account_id", "datetime"]).reset_index(drop=True)

        # Save processed data
        processed_path = self.data_path / "processed"
        processed_path.mkdir(exist_ok=True)
        master_fills_df.to_csv(processed_path / "master_fills_dataset.csv", index=False)
        master_totals_df.to_csv(processed_path / "master_totals_dataset.csv", index=False)

        self.logger.info(f"Created master dataset with {len(master_fills_df)} fill records and {len(master_totals_df)} total records")
        return master_totals_df, master_fills_df
