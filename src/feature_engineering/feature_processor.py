"""
Feature Processor for Trading Data

Handles data preprocessing, aggregation, and basic feature engineering
for trader PnL prediction models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class FeatureProcessor:
    """
    Main feature processor for trading data.

    Handles:
    - Column aggregation and dropping
    - Daily aggregation of fills data
    - Target variable creation (tomorrow's PnL)
    - Basic preprocessing
    """

    def __init__(self,
                 columns_to_drop: Optional[List[str]] = None,
                 sum_columns: Optional[List[str]] = None):
        """
        Initialize feature processor

        Args:
            columns_to_drop: List of columns to drop (cashbalance, transfer, etc.)
            sum_columns: List of columns that should be summed into total_sum
        """
        # Default columns to drop
        self.columns_to_drop = columns_to_drop or [
            'cash', 'transfers', 'transfer_deposit', 'end_balance',
            'fee_software_md', 'fee_vat', 'fee_daily_interest',
            'id', 'created_at', 'unrealized_delta', 'total_delta',
            'adj_fees', 'adj_net'
        ]

        # Default sum columns (fee-related columns)
        self.sum_columns = sum_columns or [
            'comm', 'ecn_fee', 'sec', 'orf', 'cat', 'taf',
            'ftt', 'nscc', 'acc', 'clr', 'misc', 'trade_fees'
        ]

    def process_daily_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process daily summary data

        Args:
            df: Daily summary dataframe from database

        Returns:
            Processed dataframe
        """
        if df.empty:
            return df

        df = df.copy()

        # Ensure date is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])

        # Combine sum columns into total_sum
        df = self._combine_sum_columns(df)

        # Drop specified columns
        df = self._drop_columns(df)

        # Sort by account and date
        if 'account_id' in df.columns and 'date' in df.columns:
            df = df.sort_values(['account_id', 'date']).reset_index(drop=True)

        return df

    def _combine_sum_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Combine specified sum columns into total_sum and drop original columns

        Args:
            df: Input dataframe

        Returns:
            Dataframe with combined sum column
        """
        existing_sum_cols = [col for col in self.sum_columns if col in df.columns]

        if existing_sum_cols:
            # Create total_sum column
            df['total_sum'] = df[existing_sum_cols].fillna(0).sum(axis=1)

            # Drop original sum columns
            df = df.drop(columns=existing_sum_cols)

            logger.info(f"Combined {len(existing_sum_cols)} sum columns into total_sum")
        else:
            # Create empty total_sum column if no sum columns exist
            df['total_sum'] = 0.0

        return df

    def _drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop specified columns from dataframe

        Args:
            df: Input dataframe

        Returns:
            Dataframe with dropped columns
        """
        existing_drop_cols = [col for col in self.columns_to_drop if col in df.columns]

        if existing_drop_cols:
            df = df.drop(columns=existing_drop_cols)
            logger.info(f"Dropped {len(existing_drop_cols)} columns: {existing_drop_cols}")

        return df

    def aggregate_fills_daily(self, fills_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate fills data to daily level

        Args:
            fills_df: Fills dataframe from database

        Returns:
            Daily aggregated fills data
        """
        if fills_df.empty:
            return pd.DataFrame()

        fills_df = fills_df.copy()

        # Ensure datetime is properly formatted
        if 'datetime' in fills_df.columns:
            fills_df['datetime'] = pd.to_datetime(fills_df['datetime'])
            fills_df['date'] = fills_df['datetime'].dt.date

        # Calculate additional metrics per fill
        fills_df['trade_value'] = abs(fills_df['quantity'] * fills_df['price'])
        fills_df['is_buy'] = (fills_df['side'] == 'B').astype(int)
        fills_df['is_sell'] = (fills_df['side'] == 'S').astype(int)

        # Define aggregation functions
        agg_funcs = {
            'quantity': ['sum', 'count'],  # Total quantity and number of fills
            'trade_value': ['sum', 'mean', 'std'],  # Total, average, std of trade values
            'price': ['mean', 'min', 'max'],  # Price statistics
            'total_fees': 'sum',  # Total fees
            'is_buy': 'sum',  # Number of buy trades
            'is_sell': 'sum',  # Number of sell trades
            'symbol': 'nunique',  # Number of unique symbols
        }

        # Group by account and date
        daily_fills = fills_df.groupby(['account_id', 'date']).agg(agg_funcs).reset_index()

        # Flatten column names
        daily_fills.columns = [
            '_'.join(col).strip('_') if col[1] else col[0]
            for col in daily_fills.columns
        ]

        # Rename columns for clarity
        column_renames = {
            'quantity_sum': 'total_quantity',
            'quantity_count': 'num_fills',
            'trade_value_sum': 'total_trade_value',
            'trade_value_mean': 'avg_trade_value',
            'trade_value_std': 'std_trade_value',
            'price_mean': 'avg_price',
            'price_min': 'min_price',
            'price_max': 'max_price',
            'is_buy_sum': 'num_buys',
            'is_sell_sum': 'num_sells',
            'symbol_nunique': 'unique_symbols'
        }

        daily_fills = daily_fills.rename(columns=column_renames)

        # Fill NaN values in std columns
        daily_fills['std_trade_value'] = daily_fills['std_trade_value'].fillna(0)

        # Calculate additional metrics
        daily_fills['buy_sell_ratio'] = (
            daily_fills['num_buys'] / (daily_fills['num_sells'] + 1e-6)
        )

        # Convert date back to datetime for consistency
        daily_fills['date'] = pd.to_datetime(daily_fills['date'])

        return daily_fills

    def create_target_variable(self, df: pd.DataFrame,
                             target_column: str = 'net') -> pd.DataFrame:
        """
        Create target variable (tomorrow's PnL) for prediction

        Args:
            df: Daily summary dataframe (must be sorted by account_id, date)
            target_column: Column to use as target (default: 'net')

        Returns:
            Dataframe with target variable added
        """
        if df.empty or target_column not in df.columns:
            return df

        df = df.copy()

        # Ensure proper sorting
        df = df.sort_values(['account_id', 'date']).reset_index(drop=True)

        # Create target variable (tomorrow's PnL)
        df['target_next_pnl'] = df.groupby('account_id')[target_column].shift(-1)

        # Drop the last row for each trader (no target available)
        df = df.dropna(subset=['target_next_pnl']).reset_index(drop=True)

        logger.info(f"Created target variable from {target_column}, {len(df)} samples available")

        return df

    def join_daily_data(self, daily_summary: pd.DataFrame,
                       daily_fills: pd.DataFrame) -> pd.DataFrame:
        """
        Join daily summary and aggregated fills data

        Args:
            daily_summary: Daily summary dataframe
            daily_fills: Daily aggregated fills dataframe

        Returns:
            Combined dataframe
        """
        if daily_summary.empty:
            return pd.DataFrame()

        if daily_fills.empty:
            # If no fills data, just return summary with zeros for fills columns
            logger.warning("No fills data available, returning summary only")
            return daily_summary

        # Ensure date columns are the same type
        daily_summary['date'] = pd.to_datetime(daily_summary['date'])
        daily_fills['date'] = pd.to_datetime(daily_fills['date'])

        # Merge on account_id and date
        combined = daily_summary.merge(
            daily_fills,
            on=['account_id', 'date'],
            how='left'
        )

        # Fill missing fills data with zeros
        fills_columns = [col for col in daily_fills.columns
                        if col not in ['account_id', 'date']]

        for col in fills_columns:
            if col in combined.columns:
                combined[col] = combined[col].fillna(0)

        logger.info(f"Joined daily data: {len(combined)} rows, {len(combined.columns)} columns")

        return combined

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for machine learning

        Args:
            df: Combined daily dataframe

        Returns:
            Feature-ready dataframe
        """
        if df.empty:
            return df

        df = df.copy()

        # Ensure proper sorting
        df = df.sort_values(['account_id', 'date']).reset_index(drop=True)

        # Basic feature engineering
        df = self._add_basic_features(df)

        # Remove any remaining non-numeric columns except essentials
        essential_cols = ['account_id', 'date', 'target_next_pnl']
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Keep essential + numeric columns
        cols_to_keep = essential_cols + [col for col in numeric_cols
                                       if col not in essential_cols]
        df = df[cols_to_keep]

        return df

    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add basic engineered features

        Args:
            df: Input dataframe

        Returns:
            Dataframe with basic features added
        """
        # Trading intensity features
        if 'orders' in df.columns and 'fills' in df.columns:
            df['fill_ratio'] = df['fills'] / (df['orders'] + 1e-6)

        # Profitability features
        if 'gross' in df.columns and 'total_sum' in df.columns:
            df['net_profit'] = df['gross'] - df['total_sum']
            df['profit_margin'] = df['net_profit'] / (abs(df['gross']) + 1e-6)

        # Volume features
        if 'qty' in df.columns:
            df['qty_abs'] = abs(df['qty'])

        # Unrealized features
        if 'unrealized' in df.columns:
            df['unrealized_abs'] = abs(df['unrealized'])

        return df

    def get_feature_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary statistics of processed features

        Args:
            df: Processed dataframe

        Returns:
            Summary statistics
        """
        if df.empty:
            return {}

        summary = {
            'total_rows': len(df),
            'num_traders': df['account_id'].nunique() if 'account_id' in df.columns else 0,
            'date_range': {
                'start': df['date'].min() if 'date' in df.columns else None,
                'end': df['date'].max() if 'date' in df.columns else None
            },
            'features': {
                'total_features': len(df.select_dtypes(include=[np.number]).columns),
                'feature_names': df.select_dtypes(include=[np.number]).columns.tolist()
            }
        }

        # Target variable stats
        if 'target_next_pnl' in df.columns:
            target_stats = df['target_next_pnl'].describe()
            summary['target_stats'] = {
                'mean': target_stats['mean'],
                'std': target_stats['std'],
                'min': target_stats['min'],
                'max': target_stats['max'],
                'positive_days': (df['target_next_pnl'] > 0).sum(),
                'negative_days': (df['target_next_pnl'] < 0).sum()
            }

        return summary
