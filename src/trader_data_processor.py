import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from pathlib import Path
import logging
import yaml
import json
from datetime import datetime

from src.data_processing import create_trader_day_panel

logger = logging.getLogger(__name__)


class ImprovedTraderProcessor:
    """
    Improved trader data processor that:
    1. Drops missing trading days instead of forward-filling
    2. Adds 'days_since_last_trade' feature
    3. Adds day of week features
    4. Maintains data quality
    """

    def __init__(self, config_path: str = 'configs/main_config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.output_dir = Path('data/processed/trader_splits')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"ImprovedTraderProcessor initialized, output dir: {self.output_dir}")

    def create_improved_features(self, trader_data: pd.DataFrame) -> pd.DataFrame:
        """Create features with proper handling of missing days"""

        df = trader_data.copy()

        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)

        # IMPORTANT: Drop rows with no trading activity (NaN in daily_pnl)
        initial_rows = len(df)
        df = df.dropna(subset=['daily_pnl'])
        dropped_rows = initial_rows - len(df)

        if dropped_rows > 0:
            logger.info(f"Dropped {dropped_rows} days with no trading activity")

        # Calculate days since last trade
        df['days_since_last_trade'] = 0

        # For each row, calculate business days since previous row
        for i in range(1, len(df)):
            current_date = df.iloc[i]['date']
            previous_date = df.iloc[i-1]['date']

            # Calculate business days between dates
            business_days = pd.bdate_range(start=previous_date, end=current_date)
            days_between = len(business_days) - 1  # Subtract 1 because range includes both dates

            df.iloc[i, df.columns.get_loc('days_since_last_trade')] = max(0, days_between)

        # Add day of week features (0 = Monday, 4 = Friday)
        df['day_of_week'] = df['date'].dt.dayofweek

        # Create one-hot encoded day of week features
        for day_num, day_name in enumerate(['monday', 'tuesday', 'wednesday', 'thursday', 'friday']):
            df[f'is_{day_name}'] = (df['day_of_week'] == day_num).astype(int)

        # Basic rolling statistics (only on actual trading days)
        for window in [7, 21]:
            # Use min_periods=1 to handle beginning of series
            df[f'pnl_mean_{window}d'] = df['daily_pnl'].rolling(window=window, min_periods=1).mean()
            df[f'pnl_std_{window}d'] = df['daily_pnl'].rolling(window=window, min_periods=1).std()
            df[f'volume_mean_{window}d'] = df['daily_volume'].rolling(window=window, min_periods=1).mean()
            df[f'trades_mean_{window}d'] = df['n_trades'].rolling(window=window, min_periods=1).mean()

        # Cumulative features (only from actual trading days)
        df['cumulative_pnl'] = df['daily_pnl'].cumsum()
        df['cumulative_volume'] = df['daily_volume'].cumsum()
        df['cumulative_trades'] = df['n_trades'].cumsum()

        # Days traded so far
        df['days_traded'] = range(1, len(df) + 1)

        # Basic ratios
        df['pnl_to_volume'] = df['daily_pnl'] / (df['daily_volume'] + 1e-6)
        df['gross_to_fees'] = df['daily_gross'] / (df['daily_fees'] + 1e-6)

        # Lag features (from actual previous trading days)
        for lag in [1, 2, 3]:
            df[f'pnl_lag_{lag}'] = df['daily_pnl'].shift(lag)
            df[f'volume_lag_{lag}'] = df['daily_volume'].shift(lag)
            df[f'trades_lag_{lag}'] = df['n_trades'].shift(lag)

        # Target variables
        # Large loss indicator (bottom 10% of actual trading days)
        loss_threshold = df['daily_pnl'].quantile(0.1)
        df['target_large_loss'] = (df['daily_pnl'] < loss_threshold).astype(int)

        # Next day PnL target for VaR (next actual trading day)
        df['target_pnl'] = df['daily_pnl'].shift(-1)

        # Fill NaN values with 0 (not forward fill!)
        # This only affects the first few rows for lag features
        df = df.fillna(0)

        # Add data quality check
        self._check_data_quality(df)

        return df

    def _check_data_quality(self, df: pd.DataFrame) -> None:
        """Check for data quality issues like excessive repetition"""

        # Check for excessive PnL repetition
        if len(df) > 0:
            max_repetition_rate = df['daily_pnl'].value_counts().iloc[0] / len(df)
            if max_repetition_rate > 0.3:
                most_common_pnl = df['daily_pnl'].value_counts().index[0]
                logger.warning(f"Data quality warning: {max_repetition_rate:.1%} of days have PnL = {most_common_pnl:.2f}")

        # Check for long gaps in trading
        if 'days_since_last_trade' in df.columns:
            max_gap = df['days_since_last_trade'].max()
            if max_gap > 10:
                logger.warning(f"Data quality warning: Maximum gap between trades is {max_gap} days")

    def split_trader_data(self, trader_data: pd.DataFrame, trader_id: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split trader data into 80% train and 20% test based on chronological order"""

        # Sort by date to ensure chronological order
        trader_data = trader_data.sort_values('date').reset_index(drop=True)

        # Get number of actual trading days
        total_days = len(trader_data)

        if total_days < 50:  # Minimum requirement
            logger.warning(f"Trader {trader_id} has only {total_days} trading days - insufficient for train/test split")
            return None, None

        # Calculate split point (80% for training)
        split_idx = int(total_days * 0.8)

        # Split data
        train_data = trader_data.iloc[:split_idx].copy()
        test_data = trader_data.iloc[split_idx:].copy()

        train_days = len(train_data)
        test_days = len(test_data)

        logger.info(f"Trader {trader_id}: Split into {train_days} train days and {test_days} test days")

        return train_data, test_data

    def process_single_trader(self, df: pd.DataFrame, trader_id: int) -> Dict:
        """Process a single trader with improved features"""

        logger.info(f"\n=== Processing Trader {trader_id} ===")

        # Extract trader data
        trader_data = df[df['trader_id'] == trader_id].copy()

        if len(trader_data) == 0:
            logger.warning(f"No data found for trader {trader_id}")
            return None

        # Get initial statistics
        initial_rows = len(trader_data)
        date_range = (trader_data['date'].min(), trader_data['date'].max())

        # Create improved features (this will drop missing days)
        try:
            trader_features = self.create_improved_features(trader_data)
            actual_trading_days = len(trader_features)

            logger.info(f"Trader {trader_id}: {initial_rows} calendar days -> {actual_trading_days} actual trading days")
            logger.info(f"Date range: {date_range[0]} to {date_range[1]}")

        except Exception as e:
            logger.error(f"Feature creation failed for trader {trader_id}: {str(e)}")
            return None

        # Split into train/test
        train_data, test_data = self.split_trader_data(trader_features, trader_id)

        if train_data is None or test_data is None:
            return None

        # Save individual trader datasets
        trader_output_dir = self.output_dir / str(trader_id)
        trader_output_dir.mkdir(exist_ok=True)

        # Save train set
        train_path = trader_output_dir / 'train_data.parquet'
        train_data.to_parquet(train_path, index=False)

        # Save test set
        test_path = trader_output_dir / 'test_data.parquet'
        test_data.to_parquet(test_path, index=False)

        # Calculate data quality metrics
        quality_metrics = {
            'dropped_days': int(initial_rows - actual_trading_days),
            'drop_rate': float((initial_rows - actual_trading_days) / initial_rows) if initial_rows > 0 else 0,
            'max_gap_days': int(trader_features['days_since_last_trade'].max()) if 'days_since_last_trade' in trader_features.columns else 0,
            'avg_gap_days': float(trader_features['days_since_last_trade'].mean()) if 'days_since_last_trade' in trader_features.columns else 0,
            'most_common_day': int(trader_features['day_of_week'].mode()[0]) if len(trader_features) > 0 else None
        }

        # Save metadata
        metadata = {
            'trader_id': int(trader_id),
            'initial_calendar_days': int(initial_rows),
            'actual_trading_days': int(actual_trading_days),
            'date_range': {
                'start': date_range[0].strftime('%Y-%m-%d'),
                'end': date_range[1].strftime('%Y-%m-%d')
            },
            'train_data': {
                'records': int(len(train_data)),
                'date_range': {
                    'start': train_data['date'].min().strftime('%Y-%m-%d'),
                    'end': train_data['date'].max().strftime('%Y-%m-%d')
                }
            },
            'test_data': {
                'records': int(len(test_data)),
                'date_range': {
                    'start': test_data['date'].min().strftime('%Y-%m-%d'),
                    'end': test_data['date'].max().strftime('%Y-%m-%d')
                }
            },
            'features': {
                'total_features': int(len(trader_features.columns)),
                'feature_names': trader_features.columns.tolist()
            },
            'quality_metrics': quality_metrics,
            'processed_date': datetime.now().isoformat()
        }

        metadata_path = trader_output_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Trader {trader_id}: Saved to {trader_output_dir}")

        return metadata

    def process_all_traders(self) -> Dict:
        """Process all traders with improved features"""

        logger.info("Starting improved trader processing...")

        # Load raw data
        df = create_trader_day_panel(self.config)
        df = df.rename(columns={'account_id': 'trader_id', 'trade_date': 'date'})

        # Get available traders
        available_traders = df['trader_id'].unique()
        logger.info(f"Found {len(available_traders)} traders: {sorted(available_traders)}")

        # Process each trader
        results = {}
        successful_processors = 0

        for trader_id in sorted(available_traders):
            try:
                metadata = self.process_single_trader(df, trader_id)
                if metadata is not None:
                    results[trader_id] = metadata
                    successful_processors += 1
                else:
                    logger.warning(f"Failed to process trader {trader_id}")

            except Exception as e:
                logger.error(f"Error processing trader {trader_id}: {str(e)}")
                continue

        # Save summary
        summary = {
            'total_traders': int(len(available_traders)),
            'successful_processors': int(successful_processors),
            'processed_traders': [int(k) for k in results.keys()],
            'processing_date': datetime.now().isoformat(),
            'output_directory': str(self.output_dir),
            'processing_improvements': [
                'Dropped days with no trading activity',
                'Added days_since_last_trade feature',
                'Added day of week features',
                'No forward-filling of missing values'
            ],
            'individual_metadata': results
        }

        summary_path = self.output_dir / 'processing_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"\n=== Processing Summary ===")
        logger.info(f"Successfully processed: {successful_processors}/{len(available_traders)} traders")
        logger.info(f"Summary saved to: {summary_path}")

        return summary

    def verify_anomaly_fixed(self, trader_id: int = 3943) -> bool:
        """Verify that the anomaly is fixed for a specific trader"""

        trader_dir = self.output_dir / str(trader_id)
        test_path = trader_dir / 'test_data.parquet'

        if not test_path.exists():
            logger.warning(f"No test data found for trader {trader_id}")
            return False

        test_data = pd.read_parquet(test_path)

        # Check May-June 2025 period
        test_data['date'] = pd.to_datetime(test_data['date'])
        period_data = test_data[(test_data['date'] >= '2025-05-01') & (test_data['date'] <= '2025-06-30')]

        if len(period_data) == 0:
            logger.info(f"Trader {trader_id}: No data in May-June 2025 (correctly dropped)")
            return True

        # Check for repetition
        max_repetition = period_data['daily_pnl'].value_counts().iloc[0] if len(period_data) > 0 else 0
        repetition_rate = max_repetition / len(period_data) if len(period_data) > 0 else 0

        logger.info(f"Trader {trader_id} May-June 2025: {len(period_data)} days, max repetition rate: {repetition_rate:.1%}")

        return repetition_rate < 0.3  # Less than 30% repetition is acceptable


def run_improved_processing():
    """Main function to run improved trader processing"""

    logging.basicConfig(level=logging.INFO)
    logger.info("Starting improved trader data processing...")

    processor = ImprovedTraderProcessor()

    # Process all traders
    summary = processor.process_all_traders()

    # Verify anomaly is fixed
    print(f"\n=== Verifying Anomaly Fix ===")
    for trader_id in [3943, 4004]:  # Check both problematic traders
        is_fixed = processor.verify_anomaly_fixed(trader_id)
        status = "✅ FIXED" if is_fixed else "❌ STILL PRESENT"
        print(f"Trader {trader_id}: {status}")

    return summary


if __name__ == "__main__":
    summary = run_improved_processing()
