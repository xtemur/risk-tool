import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml


class DataLoader:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        with open("config/trader_accounts.yaml", "r") as f:
            self.traders = yaml.safe_load(f)["traders"]

        self.data_path = Path("data")
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("logs/data_loader.log"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def load_trader_data(
        self, account_id: str, start_date: str = None, end_date: str = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load fills and totals data for a specific trader"""
        try:
            # Load fills data
            fills_path = self.data_path / "raw" / f"{account_id}_fills.csv"
            fills_df = pd.read_csv(fills_path)
            fills_df["Date"] = pd.to_datetime(fills_df["Date"])

            # Load totals data
            totals_path = self.data_path / "raw" / f"{account_id}_totals.csv"
            totals_df = pd.read_csv(totals_path)
            totals_df["Date"] = pd.to_datetime(totals_df["Date"])

            # Filter by date range if provided
            if start_date:
                start_date = pd.to_datetime(start_date)
                fills_df = fills_df[fills_df["Date"] >= start_date]
                totals_df = totals_df[totals_df["Date"] >= start_date]

            if end_date:
                end_date = pd.to_datetime(end_date)
                fills_df = fills_df[fills_df["Date"] <= end_date]
                totals_df = totals_df[totals_df["Date"] <= end_date]

            self.logger.info(
                f"Loaded data for {account_id}: {len(totals_df)} trading days"
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

    def create_master_dataset(self, all_data: Dict) -> pd.DataFrame:
        """Combine all trader data into master dataset"""
        master_data = []

        for account_id, data in all_data.items():
            totals_df = data["totals"].copy()
            totals_df["account_id"] = account_id
            totals_df["trader_name"] = data["name"]
            master_data.append(totals_df)

        master_df = pd.concat(master_data, ignore_index=True)
        master_df = master_df.sort_values(["account_id", "Date"]).reset_index(drop=True)

        # Save processed data
        processed_path = self.data_path / "processed"
        processed_path.mkdir(exist_ok=True)
        master_df.to_csv(processed_path / "master_dataset.csv", index=False)

        self.logger.info(f"Created master dataset with {len(master_df)} records")
        return master_df
