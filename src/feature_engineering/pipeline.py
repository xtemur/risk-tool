"""
Feature Engineering Pipeline

Main pipeline that orchestrates the entire feature engineering process
from raw database data to model-ready features.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path

from .feature_processor import FeatureProcessor
from .feature_extractor import FeatureExtractor

# Handle imports for both package and script usage
try:
    from ..data.database_manager import DatabaseManager
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from data.database_manager import DatabaseManager

logger = logging.getLogger(__name__)


class FeatureEngineeringPipeline:
    """
    Complete feature engineering pipeline for trader PnL prediction
    """

    def __init__(self,
                 db_path: str = "data/trading_risk.db",
                 lookback_window: int = 10,
                 n_features: int = 10):
        """
        Initialize the pipeline

        Args:
            db_path: Path to the trading database
            lookback_window: Days to look back for rolling features
            n_features: Number of top features to select
        """
        self.db_manager = DatabaseManager(db_path)
        self.processor = FeatureProcessor()
        self.extractor = FeatureExtractor(lookback_window=lookback_window)
        self.n_features = n_features

    def process_trader_data(self,
                           account_id: str,
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Process data for a single trader

        Args:
            account_id: Trader account ID
            start_date: Start date (YYYY-MM-DD) or None for all data
            end_date: End date (YYYY-MM-DD) or None for all data

        Returns:
            Tuple of (processed dataframe, summary statistics)
        """
        logger.info(f"Processing data for trader {account_id}")

        # Get daily summary data
        daily_summary = self.db_manager.get_account_daily_summary(
            account_id=account_id,
            start_date=start_date,
            end_date=end_date
        )

        if daily_summary.empty:
            logger.warning(f"No daily summary data found for trader {account_id}")
            return pd.DataFrame(), {}

        # Get fills data
        fills_data = self.db_manager.get_fills(
            account_id=account_id,
            start_date=start_date,
            end_date=end_date
        )

        # Process daily summary
        processed_summary = self.processor.process_daily_summary(daily_summary)

        # Aggregate fills to daily
        daily_fills = self.processor.aggregate_fills_daily(fills_data)

        # Join daily data
        combined_data = self.processor.join_daily_data(processed_summary, daily_fills)

        if combined_data.empty:
            logger.warning(f"No combined data for trader {account_id}")
            return pd.DataFrame(), {}

        # Create target variable
        combined_data = self.processor.create_target_variable(combined_data)

        # Extract advanced features
        feature_data = self.extractor.extract_all_features(combined_data)

        # Prepare features
        final_data = self.processor.prepare_features(feature_data)

        # Select top features
        selected_data, selected_features = self.extractor.select_top_features(
            final_data, n_features=self.n_features
        )

        # Generate summary
        summary = self._generate_summary(selected_data, selected_features, account_id)

        logger.info(f"Processed {len(selected_data)} samples for trader {account_id}")

        return selected_data, summary

    def process_all_traders(self,
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None,
                           min_samples: int = 30) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Process data for all traders

        Args:
            start_date: Start date (YYYY-MM-DD) or None for all data
            end_date: End date (YYYY-MM-DD) or None for all data
            min_samples: Minimum samples required per trader

        Returns:
            Tuple of (combined dataframe, summary statistics)
        """
        logger.info("Processing data for all traders")

        # Get all accounts
        accounts = self.db_manager.get_accounts()

        if accounts.empty:
            logger.error("No accounts found in database")
            return pd.DataFrame(), {}

        all_data = []
        trader_summaries = {}

        for _, account in accounts.iterrows():
            account_id = account['account_id']

            try:
                trader_data, trader_summary = self.process_trader_data(
                    account_id=account_id,
                    start_date=start_date,
                    end_date=end_date
                )

                if len(trader_data) >= min_samples:
                    all_data.append(trader_data)
                    trader_summaries[account_id] = trader_summary
                    logger.info(f"Added {len(trader_data)} samples for trader {account_id}")
                else:
                    logger.warning(f"Trader {account_id} has only {len(trader_data)} samples, skipping")

            except Exception as e:
                logger.error(f"Error processing trader {account_id}: {e}")
                continue

        if not all_data:
            logger.error("No valid trader data found")
            return pd.DataFrame(), {}

        # Combine all trader data
        combined_data = pd.concat(all_data, ignore_index=True)

        # Generate overall summary
        overall_summary = self._generate_overall_summary(combined_data, trader_summaries)

        logger.info(f"Processed {len(combined_data)} total samples from {len(all_data)} traders")

        return combined_data, overall_summary

    def get_feature_ready_data(self,
                              account_id: Optional[str] = None,
                              start_date: Optional[str] = None,
                              end_date: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Get data ready for machine learning (X, y format)

        Args:
            account_id: Specific trader or None for all traders
            start_date: Start date or None
            end_date: End date or None

        Returns:
            Tuple of (X features, y target, feature names)
        """
        if account_id:
            data, _ = self.process_trader_data(account_id, start_date, end_date)
        else:
            data, _ = self.process_all_traders(start_date, end_date)

        if data.empty:
            return np.array([]), np.array([]), []

        # Separate features and target
        exclude_cols = ['account_id', 'date', 'target_next_pnl']
        feature_cols = [col for col in data.columns if col not in exclude_cols]

        X = data[feature_cols].fillna(0).values
        y = data['target_next_pnl'].fillna(0).values

        return X, y, feature_cols

    def save_processed_data(self,
                           data: pd.DataFrame,
                           output_path: str = "data/processed/features.csv") -> None:
        """
        Save processed data to CSV

        Args:
            data: Processed dataframe
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data.to_csv(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")

    def load_processed_data(self,
                           input_path: str = "data/processed/features.csv") -> pd.DataFrame:
        """
        Load processed data from CSV

        Args:
            input_path: Input file path

        Returns:
            Loaded dataframe
        """
        input_path = Path(input_path)

        if not input_path.exists():
            logger.error(f"File not found: {input_path}")
            return pd.DataFrame()

        data = pd.read_csv(input_path)
        logger.info(f"Loaded processed data from {input_path}, shape: {data.shape}")

        return data

    def _generate_summary(self,
                         data: pd.DataFrame,
                         selected_features: List[str],
                         account_id: str) -> Dict[str, Any]:
        """
        Generate summary for a single trader
        """
        if data.empty:
            return {}

        base_summary = self.processor.get_feature_summary(data)
        importance_summary = self.extractor.get_feature_importance_summary(data)

        return {
            'account_id': account_id,
            'data_summary': base_summary,
            'selected_features': selected_features,
            'feature_importance': importance_summary
        }

    def _generate_overall_summary(self,
                                 combined_data: pd.DataFrame,
                                 trader_summaries: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate overall summary for all traders
        """
        if combined_data.empty:
            return {}

        # Overall statistics
        overall_stats = {
            'total_samples': len(combined_data),
            'num_traders': combined_data['account_id'].nunique(),
            'date_range': {
                'start': combined_data['date'].min(),
                'end': combined_data['date'].max()
            },
            'target_distribution': {
                'mean': combined_data['target_next_pnl'].mean(),
                'std': combined_data['target_next_pnl'].std(),
                'positive_ratio': (combined_data['target_next_pnl'] > 0).mean()
            }
        }

        # Feature usage across traders
        all_features = set()
        for summary in trader_summaries.values():
            if 'selected_features' in summary:
                all_features.update(summary['selected_features'])

        feature_stats = {
            'unique_features_used': len(all_features),
            'common_features': list(all_features)[:10]  # Top 10 most common
        }

        return {
            'overall_stats': overall_stats,
            'feature_stats': feature_stats,
            'trader_summaries': trader_summaries
        }

    def run_diagnostic(self) -> Dict[str, Any]:
        """
        Run diagnostic on the pipeline

        Returns:
            Diagnostic information
        """
        logger.info("Running pipeline diagnostic")

        # Database stats
        db_stats = self.db_manager.get_database_stats()

        # Sample one trader for testing
        accounts = self.db_manager.get_accounts()

        if accounts.empty:
            return {'error': 'No accounts found in database'}

        sample_account = accounts.iloc[0]['account_id']

        try:
            sample_data, sample_summary = self.process_trader_data(sample_account)

            return {
                'database_stats': db_stats,
                'sample_processing': {
                    'account_id': sample_account,
                    'samples_generated': len(sample_data),
                    'features_selected': len(sample_summary.get('selected_features', [])),
                    'summary': sample_summary
                },
                'pipeline_status': 'healthy'
            }

        except Exception as e:
            return {
                'database_stats': db_stats,
                'pipeline_status': 'error',
                'error': str(e)
            }
