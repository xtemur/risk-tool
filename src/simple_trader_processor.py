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


class SimpleTraderProcessor:
    """
    Simple trader data processor that splits each trader's data into 80/20 train/test
    without complex feature engineering to avoid compatibility issues.
    """

    def __init__(self, config_path: str = 'configs/main_config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.output_dir = Path('data/processed/trader_splits')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"SimpleTraderProcessor initialized, output dir: {self.output_dir}")

    def create_basic_features(self, trader_data: pd.DataFrame) -> pd.DataFrame:
        """Create basic features for a single trader without complex engineering"""

        df = trader_data.copy()

        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)

        # Basic rolling statistics (7 and 21 day windows)
        for window in [7, 21]:
            df[f'pnl_mean_{window}d'] = df['daily_pnl'].rolling(window=window, min_periods=1).mean()
            df[f'pnl_std_{window}d'] = df['daily_pnl'].rolling(window=window, min_periods=1).std()
            df[f'volume_mean_{window}d'] = df['daily_volume'].rolling(window=window, min_periods=1).mean()
            df[f'trades_mean_{window}d'] = df['n_trades'].rolling(window=window, min_periods=1).mean()

        # Cumulative features
        df['cumulative_pnl'] = df['daily_pnl'].cumsum()
        df['cumulative_volume'] = df['daily_volume'].cumsum()
        df['cumulative_trades'] = df['n_trades'].cumsum()

        # Basic ratios
        df['pnl_to_volume'] = df['daily_pnl'] / (df['daily_volume'] + 1e-6)  # Avoid division by zero
        df['gross_to_fees'] = df['daily_gross'] / (df['daily_fees'] + 1e-6)

        # Lag features (previous day values)
        for lag in [1, 2, 3]:
            df[f'pnl_lag_{lag}'] = df['daily_pnl'].shift(lag)
            df[f'volume_lag_{lag}'] = df['daily_volume'].shift(lag)
            df[f'trades_lag_{lag}'] = df['n_trades'].shift(lag)

        # Target variables
        # Large loss indicator (top 10% of losses)
        loss_threshold = df['daily_pnl'].quantile(0.1)  # Bottom 10%
        df['target_large_loss'] = (df['daily_pnl'] < loss_threshold).astype(int)

        # Next day PnL target for VaR
        df['target_pnl'] = df['daily_pnl'].shift(-1)  # Next day's PnL

        # Fill NaN values
        df = df.ffill().fillna(0)

        return df

    def split_trader_data(self, trader_data: pd.DataFrame, trader_id: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split trader data into 80% train and 20% test based on chronological order"""

        # Sort by date to ensure chronological order
        trader_data = trader_data.sort_values('date').reset_index(drop=True)

        # Get unique dates
        unique_dates = trader_data['date'].unique()
        total_days = len(unique_dates)

        if total_days < 50:  # Minimum requirement
            logger.warning(f"Trader {trader_id} has only {total_days} days - insufficient for train/test split")
            return None, None

        # Calculate split point (80% for training)
        split_idx = int(total_days * 0.8)
        train_end_date = unique_dates[split_idx - 1]
        test_start_date = unique_dates[split_idx]

        # Split data
        train_data = trader_data[trader_data['date'] <= train_end_date].copy()
        test_data = trader_data[trader_data['date'] >= test_start_date].copy()

        train_days = len(train_data['date'].unique())
        test_days = len(test_data['date'].unique())

        logger.info(f"Trader {trader_id}: Split into {train_days} train days ({len(train_data)} records) "
                   f"and {test_days} test days ({len(test_data)} records)")

        return train_data, test_data

    def process_single_trader(self, df: pd.DataFrame, trader_id: int) -> Dict:
        """Process a single trader: basic features + train/test split"""

        logger.info(f"\n=== Processing Trader {trader_id} ===")

        # Extract trader data
        trader_data = df[df['trader_id'] == trader_id].copy()

        if len(trader_data) == 0:
            logger.warning(f"No data found for trader {trader_id}")
            return None

        # Get date range
        date_range = (trader_data['date'].min(), trader_data['date'].max())
        total_days = len(trader_data['date'].unique())

        logger.info(f"Trader {trader_id}: {len(trader_data)} records from {date_range[0]} to {date_range[1]} ({total_days} days)")

        # Create basic features
        try:
            trader_features = self.create_basic_features(trader_data)
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

        # Save metadata
        metadata = {
            'trader_id': int(trader_id),
            'total_records': int(len(trader_features)),
            'total_days': int(total_days),
            'date_range': {
                'start': date_range[0].strftime('%Y-%m-%d'),
                'end': date_range[1].strftime('%Y-%m-%d')
            },
            'train_data': {
                'records': int(len(train_data)),
                'days': int(len(train_data['date'].unique())),
                'date_range': {
                    'start': train_data['date'].min().strftime('%Y-%m-%d'),
                    'end': train_data['date'].max().strftime('%Y-%m-%d')
                }
            },
            'test_data': {
                'records': int(len(test_data)),
                'days': int(len(test_data['date'].unique())),
                'date_range': {
                    'start': test_data['date'].min().strftime('%Y-%m-%d'),
                    'end': test_data['date'].max().strftime('%Y-%m-%d')
                }
            },
            'features': {
                'total_features': int(len(trader_features.columns)),
                'feature_names': trader_features.columns.tolist()
            },
            'processed_date': datetime.now().isoformat()
        }

        metadata_path = trader_output_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Trader {trader_id}: Saved to {trader_output_dir}")
        logger.info(f"  - Train: {train_path}")
        logger.info(f"  - Test: {test_path}")
        logger.info(f"  - Metadata: {metadata_path}")

        return metadata

    def process_all_traders(self) -> Dict:
        """Process all traders individually"""

        logger.info("Starting individual trader processing...")

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
                    logger.info(f"Successfully processed trader {trader_id}")
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
            'individual_metadata': results
        }

        summary_path = self.output_dir / 'processing_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"\n=== Processing Summary ===")
        logger.info(f"Successfully processed: {successful_processors}/{len(available_traders)} traders")
        logger.info(f"Summary saved to: {summary_path}")

        return summary

    def load_trader_data(self, trader_id: int, split: str = 'train') -> Optional[pd.DataFrame]:
        """Load processed data for a specific trader"""

        if split not in ['train', 'test']:
            raise ValueError("split must be 'train' or 'test'")

        trader_dir = self.output_dir / str(trader_id)
        data_path = trader_dir / f'{split}_data.parquet'

        if not data_path.exists():
            logger.warning(f"No {split} data found for trader {trader_id} at {data_path}")
            return None

        return pd.read_parquet(data_path)

    def get_data_summary(self) -> Dict:
        """Get summary of all processed data"""

        summary_path = self.output_dir / 'processing_summary.json'
        if not summary_path.exists():
            logger.warning("No processing summary found")
            return {}

        with open(summary_path, 'r') as f:
            summary = json.load(f)

        # Add detailed statistics
        detailed_stats = {}

        for trader_id in summary.get('processed_traders', []):
            try:
                train_data = self.load_trader_data(trader_id, 'train')
                test_data = self.load_trader_data(trader_id, 'test')

                if train_data is not None and test_data is not None:
                    detailed_stats[trader_id] = {
                        'train_samples': len(train_data),
                        'test_samples': len(test_data),
                        'total_features': len(train_data.columns),
                        'train_date_range': [train_data['date'].min().strftime('%Y-%m-%d'),
                                           train_data['date'].max().strftime('%Y-%m-%d')],
                        'test_date_range': [test_data['date'].min().strftime('%Y-%m-%d'),
                                          test_data['date'].max().strftime('%Y-%m-%d')],
                        'train_loss_rate': train_data['target_large_loss'].mean(),
                        'test_loss_rate': test_data['target_large_loss'].mean(),
                        'train_avg_pnl': train_data['daily_pnl'].mean(),
                        'test_avg_pnl': test_data['daily_pnl'].mean()
                    }
            except Exception as e:
                logger.warning(f"Error getting stats for trader {trader_id}: {e}")

        summary['detailed_stats'] = detailed_stats
        return summary


def run_simple_trader_processing():
    """Main function to run simple trader-specific data processing"""

    logging.basicConfig(level=logging.INFO)
    logger.info("Starting simple trader-specific data processing...")

    processor = SimpleTraderProcessor()

    # Process all traders
    summary = processor.process_all_traders()

    # Get detailed summary
    detailed_summary = processor.get_data_summary()

    # Print summary
    print(f"\n=== Processing Complete ===")
    print(f"Processed traders: {summary['successful_processors']}/{summary['total_traders']}")

    if detailed_summary.get('detailed_stats'):
        print(f"\n=== Trader Statistics ===")
        for trader_id, stats in detailed_summary['detailed_stats'].items():
            print(f"Trader {trader_id}:")
            print(f"  Train: {stats['train_samples']} samples, Loss rate: {stats['train_loss_rate']:.3f}")
            print(f"  Test:  {stats['test_samples']} samples, Loss rate: {stats['test_loss_rate']:.3f}")

    return summary


if __name__ == "__main__":
    summary = run_simple_trader_processing()
