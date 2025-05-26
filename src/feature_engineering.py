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
        """Create target variable: next day's actual P&L value"""
        df = df.copy()
        df = df.sort_values(["account_id", "date"])

        # Create next day P&L as target (continuous value)
        df["target"] = df.groupby("account_id")["total_delta"].shift(-1)

        # Remove last day for each trader (no target available)
        df = df.dropna(subset=["target"])

        # Optional: Create additional binary target for risk classification
        # df["target_negative"] = (df["target"] < 0).astype(int)

        return df

    def aggregate_fills_to_daily(self, fills: pd.DataFrame) -> pd.DataFrame:

        fills['datetime'] = pd.to_datetime(fills['datetime'])
        fills['date'] = fills['datetime'].dt.date
        fills['hour'] = fills['datetime'].dt.hour
        fills['minute'] = fills['datetime'].dt.minute

        # Group by date and account
        daily_agg = fills.groupby(['date', 'account_id']).agg({
            # Trading activity metrics
            'order_id': 'nunique',  # Number of unique orders
            'symbol': 'nunique',    # Number of unique symbols traded
            'qty': ['sum', 'mean', 'std', 'max', 'count'],  # Volume patterns
            'price': ['mean', 'std', 'min', 'max'],  # Price volatility
            'total_fees': ['sum', 'mean'],     # Total fees paid
        })
        daily_agg.columns = ['_'.join(col).strip() for col in daily_agg.columns]
        daily_agg = daily_agg.reset_index()


        return daily_agg

    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        df = df.copy()

        # Day of week (0=Monday, 6=Sunday)
        df["day_of_week"] = df["date"].dt.dayofweek

        # Month
        df["month"] = df["date"].dt.month

        # Is Monday (often volatile)
        df["is_monday"] = (df["day_of_week"] == 0).astype(int)

        # Is Friday (often different behavior)
        df["is_friday"] = (df["day_of_week"] == 4).astype(int)

        return df

    def create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling window features"""
        df = df.copy()
        df = df.sort_values(["account_id", "date"])

        for window in self.feature_windows:
            # Rolling PnL features
            df[f"net_rolling_{window}d"] = df.groupby("account_id")["net_pnl"].transform(
                lambda x: x.rolling(window, min_periods=1).sum()
            )

            df[f"net_mean_{window}d"] = df.groupby("account_id")["net_pnl"].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )

            df[f"net_std_{window}d"] = df.groupby("account_id")["net_pnl"].transform(
                lambda x: x.rolling(window, min_periods=1).std()
            )

            # Rolling volume features
            df[f"qty_mean_{window}d"] = df.groupby("account_id")["qty"].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )

            # Rolling trade frequency
            df[f"orders_mean_{window}d"] = df.groupby("account_id")["orders_count"].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )

            # Win rate
            df[f"win_rate_{window}d"] = df.groupby("account_id")["net_pnl"].transform(
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
            mean_ret = df.groupby("account_id")["total_delta"].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            std_ret = df.groupby("account_id")["total_delta"].transform(
                lambda x: x.rolling(window, min_periods=1).std()
            )
            df[f"sharpe_{window}d"] = mean_ret / (std_ret + 1e-8)

        # Recent performance (last 3 days)
        df["recent_performance"] = df.groupby("account_id")["total_delta"].transform(
            lambda x: x.rolling(3, min_periods=1).sum()
        )

        # Consecutive losing days
        df["is_loss"] = (df["total_delta"] < 0).astype(int)
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
        df["historical_avg_total_delta"] = df.groupby("account_id")["total_delta"].transform(
            lambda x: x.expanding().mean()
        )
        return df

    def engineer_features(self, totals: pd.DataFrame, fills: pd.DataFrame) -> pd.DataFrame:
        """Main feature engineering pipeline"""
        self.logger.info("Starting feature engineering...")

        if fills is not None and not fills.empty:
            self.logger.info("Aggregating fills data to daily behavioral metrics...")

            daily_fills = self.aggregate_fills_to_daily(fills)

            totals['date'] = pd.to_datetime(totals['date'])
            daily_fills['date'] = pd.to_datetime(daily_fills['date'])
            # Merge with totals
            totals = totals.merge(
                daily_fills,
                on=['date', 'account_id'],
                how='left'
            )

            # Create behavioral change features
            # totals = self.create_behavioral_change_features(totals)

        # Ensure date column is datetime
        totals['date'] = pd.to_datetime(totals['date'])

        # Create target variable (for regression)
        df = self.create_target_variable(totals)

        # Create all feature sets
        df = self.create_temporal_features(df)
        df = self.create_rolling_features(df)
        df = self.create_risk_features(df)
        df = self.create_trader_features(df)

        numeric_columns = df.select_dtypes(include=[np.number]).columns

        # Different fill strategies for different feature types
        for col in numeric_columns:
            if 'lag' in col or 'rolling' in col or 'ma_' in col:
                # Forward fill for lagged features
                df[col] = df.groupby('account_id')[col].ffill()
            elif 'zscore' in col or 'distance' in col:
                # Fill with 0 for normalized features
                df[col] = df[col].fillna(0)
            else:
                # Fill with median for other features
                df[col] = df.groupby('account_id')[col].transform(
                    lambda x: x.fillna(x.median())
                )

        # Final cleanup
        df[numeric_columns] = df[numeric_columns].fillna(0)

        # Remove infinite values
        df = df.replace([np.inf, -np.inf], 0)

        self.logger.info(f"Feature engineering completed. Dataset shape: {df.shape}")
        self.last_engineered_df = df.copy()

        # self.logger.info(f"Feature columns: {df.columns.tolist()}")
        # # Log df head
        # self.logger.info(f"Feature data head:\n{df.head()}")

        # self.logger.info(f"Feature columns: {self.get_feature_columns(df)}")

        return df


    def get_feature_columns(self, df: pd.DataFrame = None) -> List[str]:
        """Get list of feature columns (excluding target and metadata)"""

        if df is None:
            df = self.last_engineered_df

        # Columns to exclude
        exclude_cols = {
            'date', 'datetime', 'account_id', 'trader_name', 'target', 'target_total_delta',
            'target_gross', 'target_direction', 'target_large_loss', 'target_large_gain',
            'next_day_volatility', 'order_id', 'fills', 'qty', 'gross', 'unrealized_delta', 'total_delta'
        }

        # Return feature columns
        return [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['int64', 'float64']]
