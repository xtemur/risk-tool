# src/model_training.py

import pandas as pd
import numpy as np
import lightgbm as lgb
from typing import Dict, Tuple, Optional
from pathlib import Path
import logging
import yaml
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from src.data_processing import create_trader_day_panel
from src.feature_engineering import build_features

logger = logging.getLogger(__name__)


class FeaturePreparer:
    """Prepares datasets with both engineered features and raw sequences"""

    def __init__(self, sequence_length: int = 5):
        self.sequence_length = sequence_length
        self.raw_feature_cols = [
            'daily_pnl', 'daily_gross', 'daily_fees', 'daily_volume',
            'n_trades', 'gross_profit', 'gross_loss'
        ]

    def prepare_features(self, df: pd.DataFrame, trader_id: str) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Prepare features for a single trader

        Returns:
            features: DataFrame with engineered + flattened raw features
            targets: Array with target values
        """
        # Filter trader data and sort by date
        trader_data = df[df['trader_id'] == trader_id].sort_values('date').copy()

        # Get engineered features (exclude metadata and targets)
        exclude_cols = ['trader_id', 'date', 'target_pnl', 'target_large_loss']
        engineered_cols = [col for col in trader_data.columns if col not in exclude_cols + self.raw_feature_cols]

        # Filter out features that are all null for this trader
        valid_engineered_cols = []
        for col in engineered_cols:
            if not trader_data[col].isnull().all():
                valid_engineered_cols.append(col)

        logger.info(f"Using {len(valid_engineered_cols)} valid engineered features for trader {trader_id}")

        # Prepare sequential data and aligned engineered features
        combined_features_list = []
        aligned_targets = []

        for i in range(self.sequence_length, len(trader_data)):
            # Raw sequence: last sequence_length days (flattened)
            sequence_data = trader_data.iloc[i-self.sequence_length:i][self.raw_feature_cols].values

            # Flatten the sequence into a single vector
            flattened_sequence = sequence_data.flatten()

            # Current day's engineered features
            current_engineered = trader_data.iloc[i][valid_engineered_cols]

            # Target (next day's risk)
            target = trader_data.iloc[i]['target_large_loss']

            # Only include if we have valid data
            sequence_valid = not pd.isna(sequence_data).any()
            engineered_valid = not pd.isna(current_engineered.values).any()
            target_valid = not pd.isna(target)

            if sequence_valid and engineered_valid and target_valid:
                # Combine engineered features with flattened sequence
                combined_row = np.concatenate([current_engineered.values, flattened_sequence])
                combined_features_list.append(combined_row)
                aligned_targets.append(target)

        if len(combined_features_list) == 0:
            return None, None

        # Create feature names
        engineered_feature_names = valid_engineered_cols
        sequence_feature_names = []
        for day in range(self.sequence_length):
            for feat in self.raw_feature_cols:
                sequence_feature_names.append(f"{feat}_lag_{day+1}")

        all_feature_names = engineered_feature_names + sequence_feature_names

        # Convert to DataFrame
        combined_features = pd.DataFrame(combined_features_list, columns=all_feature_names)
        targets = np.array(aligned_targets, dtype=np.float32)

        # Clean data - replace inf and very large values
        combined_features = combined_features.replace([np.inf, -np.inf], np.nan)
        combined_features = combined_features.fillna(combined_features.median())  # Use median for better stability

        # Clip extreme values to reasonable range
        combined_features = combined_features.clip(-1e6, 1e6)

        logger.info(f"Trader {trader_id}: {len(combined_features)} samples, "
                   f"combined features shape: {combined_features.shape}")
        logger.info(f"Feature breakdown: {len(engineered_feature_names)} engineered + "
                   f"{len(sequence_feature_names)} sequential = {len(all_feature_names)} total")

        return combined_features, targets


class ExpandingWindowTrainer:
    """Expanding window trainer using LightGBM with combined features"""

    def __init__(self, config_path: str = 'configs/main_config.yaml', sequence_length: int = 5):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.sequence_length = sequence_length
        self.models_dir = Path('models/expanding_window')
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.preparer = FeaturePreparer(sequence_length)
        self.performance_history = defaultdict(list)

        # LightGBM parameters
        self.lgb_params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42,
            'force_row_wise': True
        }

        logger.info(f"Trainer initialized: sequence_length={sequence_length}")

    def train_trader_model(self, df: pd.DataFrame, trader_id: str, boundary: Dict) -> Optional[lgb.LGBMClassifier]:
        """Train model for a single trader"""

        # Prepare training data
        train_data = df[
            (df['trader_id'] == trader_id) &
            (df['date'] >= boundary['train_start']) &
            (df['date'] <= boundary['validation_end'])  # Use all available for training
        ].copy()

        combined_features, targets = self.preparer.prepare_features(train_data, trader_id)

        if combined_features is None:
            logger.warning(f"No valid data for trader {trader_id}")
            return None

        # Check if we have both classes
        if len(np.unique(targets)) < 2:
            logger.warning(f"Trader {trader_id} has only one class in targets")
            return None

        # Split into train/validation for early stopping
        train_size = int(0.8 * len(combined_features))
        X_train = combined_features.iloc[:train_size]
        y_train = targets[:train_size]
        X_val = combined_features.iloc[train_size:]
        y_val = targets[train_size:]

        # Train LightGBM model
        model = lgb.LGBMClassifier(**self.lgb_params)

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(0)  # Silent training
            ]
        )

        # Save model
        import pickle
        model_path = self.models_dir / f"{trader_id}_final.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': model,
                'feature_names': combined_features.columns.tolist(),
                'sequence_length': self.sequence_length
            }, f)

        logger.info(f"Trained model for trader {trader_id}, saved to {model_path}")
        return model

    def predict_with_model(self, model: lgb.LGBMClassifier, df: pd.DataFrame,
                          trader_id: str, boundary: Dict) -> pd.DataFrame:
        """Generate predictions on test set"""

        test_data = df[
            (df['trader_id'] == trader_id) &
            (df['date'] >= boundary['test_start']) &
            (df['date'] <= boundary['test_end'])
        ].copy()

        combined_features, targets = self.preparer.prepare_features(test_data, trader_id)

        if combined_features is None:
            return pd.DataFrame()

        # Generate predictions
        predictions = model.predict_proba(combined_features)[:, 1]  # Probability of positive class

        # Create results DataFrame
        # Note: We need to align dates properly since we skip first sequence_length days
        test_dates = test_data['date'].iloc[self.sequence_length:].reset_index(drop=True)

        results = pd.DataFrame({
            'trader_id': trader_id,
            'date': test_dates[:len(predictions)],
            'prediction': predictions,
            'actual': targets
        })

        return results


def run_expanding_window_training(sequence_length: int = 7, cutoff_date: str = None):
    """
    Main function to run expanding window training with LightGBM
    """
    logger.info(f"Starting expanding window training: LightGBM with seq_len={sequence_length}")

    # Load and prepare data
    with open('configs/main_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    df = create_trader_day_panel(config)
    df = df.rename(columns={'account_id': 'trader_id', 'trade_date': 'date'})

    # Feature engineering
    df_for_features = df.rename(columns={'trader_id': 'account_id', 'date': 'trade_date'})
    df_for_features = build_features(df_for_features, config)
    df = df_for_features.rename(columns={'account_id': 'trader_id', 'trade_date': 'date'})

    # Apply cutoff date if specified
    if cutoff_date:
        logger.info(f"Applying training cutoff date: {cutoff_date}")
        df = df[df['date'] <= cutoff_date].copy()

    # Calculate boundaries - simple implementation
    boundaries = {}
    for trader_id in df['trader_id'].unique():
        trader_data = df[df['trader_id'] == trader_id].sort_values('date')
        dates = trader_data['date'].unique()

        # 80% train, 20% test split
        split_idx = int(len(dates) * 0.8)
        train_end = dates[split_idx-1] if split_idx > 0 else dates[-1]
        test_start = dates[split_idx] if split_idx < len(dates) else dates[-1]

        boundaries[trader_id] = {
            'train_start': dates[0],
            'validation_end': train_end,
            'test_start': test_start,
            'test_end': dates[-1]
        }

    # Initialize trainer
    trainer = ExpandingWindowTrainer(sequence_length=sequence_length)

    all_predictions = []

    for trader_id in df['trader_id'].unique():
        logger.info(f"Training model for trader {trader_id}")

        # Train model
        model = trainer.train_trader_model(df, trader_id, boundaries[trader_id])

        if model is not None:
            # Generate predictions
            predictions = trainer.predict_with_model(
                model, df, trader_id, boundaries[trader_id]
            )

            if len(predictions) > 0:
                all_predictions.append(predictions)

    # Save results
    if all_predictions:
        all_predictions_df = pd.concat(all_predictions, ignore_index=True)
        results_path = trainer.models_dir / f'predictions_seq{sequence_length}.csv'
        all_predictions_df.to_csv(results_path, index=False)

        logger.info(f"Training completed. Results saved to: {results_path}")

        # Calculate overall performance
        from sklearn.metrics import roc_auc_score, average_precision_score

        if len(all_predictions_df['actual'].unique()) > 1:
            overall_auc = roc_auc_score(all_predictions_df['actual'], all_predictions_df['prediction'])
            overall_ap = average_precision_score(all_predictions_df['actual'], all_predictions_df['prediction'])

            logger.info(f"Overall AUC: {overall_auc:.4f}")
            logger.info(f"Overall AP: {overall_ap:.4f}")

        return all_predictions_df
    else:
        logger.warning("No predictions generated")
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Run with optimal sequence length
    print("=== Running expanding window training ===")
    try:
        results = run_expanding_window_training(sequence_length=7)
        if results is not None:
            print(f"Successfully trained model with {len(results)} predictions")
        else:
            print("Failed to train model")
    except Exception as e:
        print(f"Error training model: {str(e)}")
