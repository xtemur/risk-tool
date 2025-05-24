import logging
from typing import Dict, List

import numpy as np
import pandas as pd


class FeatureEngineer:
    def __init__(self, config: Dict):
        self.config = config
        self.feature_windows = config["data"]["feature_windows"]
        self.logger = logging.getLogger(__name__)
        self.last_engineered_df = pd.DataFrame()

    def create_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variable: 1 if next day Net PnL is negative"""
        df = df.copy()
        df = df.sort_values(["account_id", "Date"])

        # Create next day Net PnL
        df["next_day_net"] = df.groupby("account_id")["Net"].shift(-1)

        # Target: 1 if next day is negative, 0 otherwise
        df["target"] = (df["next_day_net"] < 0).astype(int)

        # Remove last day for each trader (no target available)
        df = df.dropna(subset=["target"])

        return df

    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        df = df.copy()

        # Day of week (0=Monday, 6=Sunday)
        df["day_of_week"] = df["Date"].dt.dayofweek

        # Month
        df["month"] = df["Date"].dt.month

        # Is Monday (often volatile)
        df["is_monday"] = (df["day_of_week"] == 0).astype(int)

        # Is Friday (often different behavior)
        df["is_friday"] = (df["day_of_week"] == 4).astype(int)

        return df

    def create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling window features"""
        df = df.copy()
        df = df.sort_values(["account_id", "Date"])

        for window in self.feature_windows:
            # Rolling PnL features
            df[f"net_rolling_{window}d"] = df.groupby("account_id")["Net"].transform(
                lambda x: x.rolling(window, min_periods=1).sum()
            )

            df[f"net_mean_{window}d"] = df.groupby("account_id")["Net"].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )

            df[f"net_std_{window}d"] = df.groupby("account_id")["Net"].transform(
                lambda x: x.rolling(window, min_periods=1).std()
            )

            # Rolling volume features
            df[f"qty_mean_{window}d"] = df.groupby("account_id")["Qty"].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )

            # Rolling trade frequency
            df[f"orders_mean_{window}d"] = df.groupby("account_id")["Orders"].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )

            # Win rate
            df[f"win_rate_{window}d"] = df.groupby("account_id")["Net"].transform(
                lambda x: (
                    x.rolling(window, min_periods=1).apply(lambda y: (y > 0).sum())
                    / window
                )
            )

        return df

    def create_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create risk-based features"""
        df = df.copy()

        # Sharpe ratio approximation (rolling)
        for window in [10, 20]:
            mean_ret = df.groupby("account_id")["Net"].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            std_ret = df.groupby("account_id")["Net"].transform(
                lambda x: x.rolling(window, min_periods=1).std()
            )
            df[f"sharpe_{window}d"] = mean_ret / (std_ret + 1e-8)

        # Recent performance (last 3 days)
        df["recent_performance"] = df.groupby("account_id")["Net"].transform(
            lambda x: x.rolling(3, min_periods=1).sum()
        )

        # Consecutive losing days
        df["is_loss"] = (df["Net"] < 0).astype(int)
        df["consecutive_losses"] = df.groupby("account_id")["is_loss"].transform(
            lambda x: x.groupby((x != x.shift()).cumsum()).cumsum()
        )
        df["consecutive_losses"] = df["consecutive_losses"] * df["is_loss"]

        return df

    def create_trader_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create trader-specific features"""
        df = df.copy()

        # Trading experience (days since start)
        df["trading_days"] = df.groupby("account_id").cumcount() + 1

        # Historical average performance
        df["historical_avg_net"] = df.groupby("account_id")["Net"].transform(
            lambda x: x.expanding().mean()
        )
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main feature engineering pipeline"""
        self.logger.info("Starting feature engineering...")

        # Create target variable
        df = self.create_target_variable(df)

        # Create all feature sets
        df = self.create_temporal_features(df)
        df = self.create_rolling_features(df)
        df = self.create_risk_features(df)
        df = self.create_trader_features(df)

        # Fill NaN values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)

        self.logger.info(f"Feature engineering completed. Dataset shape: {df.shape}")
        self.last_engineered_df = df.copy()

        return df

    def get_feature_columns(self) -> List[str]:
        """Get list of feature columns (excluding target and metadata)"""
        exclude_cols = [
            "Date",
            "Symbol",
            "account_id",
            "trader_name",
            "target",
            "next_day_net",
            "Orders",
            "Fills",
            "Qty",
            "Gross",
            "All Trade Fees",
            "Net",
            "Adj Fees",
            "Unrealized delta",
            "Total delta",
            "Transfers",
            "End Balance",
        ]

        # This will be populated after feature engineering
        return [
            col for col in self.last_engineered_df.columns if col not in exclude_cols
        ]
