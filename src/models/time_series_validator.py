import pandas as pd
from typing import Generator, Tuple, List

class TimeSeriesValidator:
    """
    Walk-forward validation for time series data to prevent lookahead bias.
    Critical for financial forecasting where data leakage must be avoided.
    """

    def __init__(self, min_train_days: int = 60, step_days: int = 1):
        """
        Initialize time series validator.

        Args:
            min_train_days: Minimum number of days needed for initial training
            step_days: Number of days to step forward in each iteration
        """
        self.min_train_days = min_train_days
        self.step_days = step_days

    def walk_forward_split(self,
                          df: pd.DataFrame,
                          date_col: str = 'trade_date',
                          start_date: str = None) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
        """
        Generate walk-forward train/validation splits.

        Args:
            df: DataFrame with time series data
            date_col: Name of the date column
            start_date: Start date for walk-forward validation (if None, auto-calculate)

        Yields:
            Tuple of (train_df, val_df) for each time step
        """
        # Ensure data is sorted by date
        df_sorted = df.sort_values([date_col]).reset_index(drop=True)

        # Get unique dates
        unique_dates = sorted(df_sorted[date_col].unique())

        # Determine start date for validation
        if start_date is None:
            start_idx = self.min_train_days
        else:
            start_date = pd.to_datetime(start_date)
            start_idx = next(i for i, date in enumerate(unique_dates) if date >= start_date)

        # Walk forward through time
        for val_idx in range(start_idx, len(unique_dates), self.step_days):
            val_date = unique_dates[val_idx]

            # Training data: all data before validation date
            train_mask = df_sorted[date_col] < val_date
            train_df = df_sorted[train_mask].copy()

            # Validation data: single day
            val_mask = df_sorted[date_col] == val_date
            val_df = df_sorted[val_mask].copy()

            if len(train_df) > 0 and len(val_df) > 0:
                yield train_df, val_df

    def create_feature_target_split(self,
                                   df: pd.DataFrame,
                                   target_col: str = 'target',
                                   feature_cols: List[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Split features and target, removing rows with missing targets.

        Args:
            df: DataFrame with features and target
            target_col: Name of target column
            feature_cols: List of feature column names (if None, auto-detect)

        Returns:
            Tuple of (features_df, target_series)
        """
        # Remove rows where target is NaN (can't predict without target)
        clean_df = df.dropna(subset=[target_col]).copy()

        if len(clean_df) == 0:
            return pd.DataFrame(), pd.Series()

        # Auto-detect feature columns if not provided
        if feature_cols is None:
            exclude_cols = [target_col, 'account_id', 'trade_date', 'next_day_realized_pnl']
            feature_cols = [col for col in clean_df.columns if col not in exclude_cols]

        # Select features and handle missing values
        X = clean_df[feature_cols].copy()
        y = clean_df[target_col].copy()

        # Fill remaining NaN values in features (forward fill, then backward fill, then 0)
        X = X.ffill().bfill().fillna(0)

        return X, y
