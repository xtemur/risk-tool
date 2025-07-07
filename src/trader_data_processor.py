import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from pathlib import Path
import logging
import yaml
import pickle
from datetime import datetime

from src.data_processing import create_trader_day_panel
from src.feature_engineering import build_features

logger = logging.getLogger(__name__)


class TraderDataProcessor:
    """
    Process each trader individually with feature engineering and 80/20 train/test split
    based on their individual trading days.
    """

    def __init__(self, config_path: str = 'configs/main_config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.output_dir = Path('data/processed/trader_specific')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"TraderDataProcessor initialized, output dir: {self.output_dir}")

    def create_trader_features(self, trader_data: pd.DataFrame, trader_id: int) -> pd.DataFrame:
        """Apply feature engineering to a single trader's data"""

        # Rename columns for feature engineering compatibility
        trader_features = trader_data.rename(columns={'trader_id': 'account_id', 'date': 'trade_date'})

        # Apply feature engineering
        trader_features = build_features(trader_features, self.config)

        # Rename back
        trader_features = trader_features.rename(columns={'account_id': 'trader_id', 'trade_date': 'date'})

        logger.info(f"Trader {trader_id}: Created {len(trader_features)} records with {len(trader_features.columns)} features")

        return trader_features

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
        """Process a single trader: feature engineering + train/test split"""

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

        # Apply feature engineering
        try:
            trader_features = self.create_trader_features(trader_data, trader_id)
        except Exception as e:
            logger.error(f"Feature engineering failed for trader {trader_id}: {str(e)}")
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
            'trader_id': trader_id,
            'total_records': len(trader_features),
            'total_days': total_days,
            'date_range': {
                'start': date_range[0].strftime('%Y-%m-%d'),
                'end': date_range[1].strftime('%Y-%m-%d')
            },
            'train_data': {
                'records': len(train_data),
                'days': len(train_data['date'].unique()),
                'date_range': {
                    'start': train_data['date'].min().strftime('%Y-%m-%d'),
                    'end': train_data['date'].max().strftime('%Y-%m-%d')
                }
            },
            'test_data': {
                'records': len(test_data),
                'days': len(test_data['date'].unique()),
                'date_range': {
                    'start': test_data['date'].min().strftime('%Y-%m-%d'),
                    'end': test_data['date'].max().strftime('%Y-%m-%d')
                }
            },
            'features': {
                'total_features': len(trader_features.columns),
                'feature_names': trader_features.columns.tolist()
            },
            'processed_date': datetime.now().isoformat()
        }

        metadata_path = trader_output_dir / 'metadata.json'
        import json
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
            'total_traders': len(available_traders),
            'successful_processors': successful_processors,
            'processed_traders': list(results.keys()),
            'processing_date': datetime.now().isoformat(),
            'output_directory': str(self.output_dir),
            'individual_metadata': results
        }

        summary_path = self.output_dir / 'processing_summary.json'
        import json
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

    def load_trader_metadata(self, trader_id: int) -> Optional[Dict]:
        """Load metadata for a specific trader"""

        trader_dir = self.output_dir / str(trader_id)
        metadata_path = trader_dir / 'metadata.json'

        if not metadata_path.exists():
            logger.warning(f"No metadata found for trader {trader_id}")
            return None

        import json
        with open(metadata_path, 'r') as f:
            return json.load(f)

    def get_processed_traders(self) -> list:
        """Get list of successfully processed traders"""

        summary_path = self.output_dir / 'processing_summary.json'
        if not summary_path.exists():
            logger.warning("No processing summary found. Run process_all_traders() first.")
            return []

        import json
        with open(summary_path, 'r') as f:
            summary = json.load(f)

        return summary.get('processed_traders', [])

    def validate_processed_data(self) -> Dict:
        """Validate all processed trader datasets"""

        processed_traders = self.get_processed_traders()
        validation_results = {}

        for trader_id in processed_traders:
            try:
                # Load train and test data
                train_data = self.load_trader_data(trader_id, 'train')
                test_data = self.load_trader_data(trader_id, 'test')
                metadata = self.load_trader_metadata(trader_id)

                if train_data is None or test_data is None or metadata is None:
                    validation_results[trader_id] = {'status': 'failed', 'error': 'Missing data'}
                    continue

                # Validation checks
                validation = {
                    'status': 'passed',
                    'train_records': len(train_data),
                    'test_records': len(test_data),
                    'train_days': len(train_data['date'].unique()),
                    'test_days': len(test_data['date'].unique()),
                    'total_features': len(train_data.columns),
                    'date_continuity': train_data['date'].max() < test_data['date'].min(),
                    'target_present': 'target_large_loss' in train_data.columns,
                    'pnl_present': 'daily_pnl' in train_data.columns
                }

                # Check for data quality issues
                issues = []
                if validation['train_records'] < 100:
                    issues.append('Insufficient training data')
                if validation['test_records'] < 20:
                    issues.append('Insufficient test data')
                if not validation['date_continuity']:
                    issues.append('Date overlap between train/test')
                if not validation['target_present']:
                    issues.append('Missing target variable')
                if not validation['pnl_present']:
                    issues.append('Missing PnL data')

                validation['issues'] = issues
                if len(issues) > 0:
                    validation['status'] = 'warning'

                validation_results[trader_id] = validation

            except Exception as e:
                validation_results[trader_id] = {'status': 'error', 'error': str(e)}

        # Save validation results
        validation_path = self.output_dir / 'validation_results.json'
        import json
        with open(validation_path, 'w') as f:
            json.dump(validation_results, f, indent=2)

        logger.info(f"Validation completed for {len(validation_results)} traders")
        logger.info(f"Results saved to: {validation_path}")

        return validation_results


def run_trader_processing():
    """Main function to run trader-specific data processing"""

    logging.basicConfig(level=logging.INFO)
    logger.info("Starting trader-specific data processing...")

    processor = TraderDataProcessor()

    # Process all traders
    summary = processor.process_all_traders()

    # Validate processed data
    validation = processor.validate_processed_data()

    # Print summary
    print(f"\n=== Processing Complete ===")
    print(f"Processed traders: {summary['successful_processors']}/{summary['total_traders']}")

    passed_validations = sum(1 for v in validation.values() if v.get('status') == 'passed')
    print(f"Validation passed: {passed_validations}/{len(validation)}")

    return summary, validation


if __name__ == "__main__":
    summary, validation = run_trader_processing()
