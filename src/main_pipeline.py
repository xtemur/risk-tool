#!/usr/bin/env python3
"""
Trader Risk Management MVP - Main Pipeline
Implements walk-forward validation for predicting realized PNL.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from data.data_loader import DataLoader
from features.feature_engineer import FeatureEngineer
from models.time_series_validator import TimeSeriesValidator
from models.risk_model import RiskModel
from evaluation.evaluator import ModelEvaluator

class RiskManagementPipeline:
    """Main pipeline for trader risk management system."""

    def __init__(self, test_cutoff_date: str = '2025-04-01'):
        self.test_cutoff_date = test_cutoff_date
        self.data_loader = DataLoader()
        self.feature_engineer = FeatureEngineer()
        self.validator = TimeSeriesValidator()
        self.model = RiskModel()
        self.evaluator = ModelEvaluator()

        # Store results
        self.train_features = None
        self.test_features = None
        self.walk_forward_results = []
        self.final_predictions = None
        self.final_metrics = None

    def load_and_prepare_data(self):
        """Load data and create features."""
        print("Loading and preparing data...")

        # Load trades data with train/test split
        train_trades, test_trades = self.data_loader.load_trades_data(self.test_cutoff_date)

        # Create features for training data
        print("Creating training features...")
        self.train_features = self.feature_engineer.create_daily_features(train_trades)
        self.train_features = self.feature_engineer.create_target_variable(self.train_features)

        # Create features for test data
        print("Creating test features...")
        self.test_features = self.feature_engineer.create_daily_features(test_trades)
        self.test_features = self.feature_engineer.create_target_variable(self.test_features)

        print(f"Training features shape: {self.train_features.shape}")
        print(f"Test features shape: {self.test_features.shape}")

        # Check for data quality
        print(f"Training data date range: {self.train_features['trade_date'].min()} to {self.train_features['trade_date'].max()}")
        print(f"Test data date range: {self.test_features['trade_date'].min()} to {self.test_features['trade_date'].max()}")

        return self.train_features, self.test_features

    def tune_model_parameters(self):
        """Tune model hyperparameters on training data."""
        print("Tuning model hyperparameters...")

        # Prepare training data for hyperparameter tuning
        X_train, y_train = self.validator.create_feature_target_split(
            self.train_features, target_col='target'
        )

        if len(X_train) == 0:
            raise ValueError("No valid training data available for hyperparameter tuning")

        print(f"Hyperparameter tuning on {len(X_train)} samples with {len(X_train.columns)} features")

        # Tune hyperparameters
        best_params = self.model.tune_hyperparameters(X_train, y_train, cv_folds=3)

        return best_params

    def run_walk_forward_validation(self):
        """Run walk-forward validation on training data."""
        print("Running walk-forward validation...")

        self.walk_forward_results = []

        # Use last 3 months of training data for validation
        validation_start_date = '2025-01-01'  # Adjust based on your data

        fold_count = 0
        for train_fold, val_fold in self.validator.walk_forward_split(
            self.train_features,
            date_col='trade_date',
            start_date=validation_start_date
        ):
            fold_count += 1
            if fold_count > 30:  # Limit to prevent excessive computation
                break

            # Prepare features and targets
            X_train_fold, y_train_fold = self.validator.create_feature_target_split(
                train_fold, target_col='target'
            )
            X_val_fold, y_val_fold = self.validator.create_feature_target_split(
                val_fold, target_col='target'
            )

            if len(X_train_fold) == 0 or len(X_val_fold) == 0:
                continue

            # Train model on fold
            self.model.train(X_train_fold, y_train_fold)

            # Make predictions
            y_pred_fold = self.model.predict(X_val_fold)

            # Evaluate fold
            fold_metrics = self.model.evaluate(X_val_fold, y_val_fold)

            # Store results
            fold_result = {
                'fold': fold_count,
                'val_date': val_fold['trade_date'].iloc[0],
                'train_size': len(X_train_fold),
                'val_size': len(X_val_fold),
                'metrics': fold_metrics,
                'predictions': y_pred_fold,
                'actuals': y_val_fold.values,
                'actual_pnl': val_fold['next_day_realized_pnl'].values
            }

            self.walk_forward_results.append(fold_result)

            if fold_count % 5 == 0:
                print(f"Completed fold {fold_count}, MAE: {fold_metrics['mae']:.4f}")

        print(f"Walk-forward validation completed: {len(self.walk_forward_results)} folds")

        return self._summarize_walk_forward_results()

    def train_final_model(self):
        """Train final model on all training data."""
        print("Training final model on all training data...")

        # Prepare all training data
        X_train_all, y_train_all = self.validator.create_feature_target_split(
            self.train_features, target_col='target'
        )

        if len(X_train_all) == 0:
            raise ValueError("No valid training data available")

        print(f"Training final model on {len(X_train_all)} samples")

        # Train final model
        self.model.train(X_train_all, y_train_all)

        # Show feature importance
        importance_df = self.model.get_feature_importance(top_n=15)
        print("\nTop 15 Most Important Features:")
        print(importance_df.to_string(index=False))

        return self.model

    def evaluate_on_test_set(self):
        """CRITICAL: One-time evaluation on isolated test set."""
        print("="*60)
        print("EVALUATING ON ISOLATED TEST SET (ONE TIME ONLY)")
        print("="*60)

        # Prepare test data
        X_test, y_test = self.validator.create_feature_target_split(
            self.test_features, target_col='target'
        )

        if len(X_test) == 0:
            print("ERROR: No valid test data available")
            return None

        print(f"Test set size: {len(X_test)} samples")
        print(f"Test date range: {self.test_features['trade_date'].min()} to {self.test_features['trade_date'].max()}")

        # Make predictions
        y_pred_test = self.model.predict(X_test)

        # Generate risk signals
        risk_signals = self.model.generate_risk_signals(y_pred_test)

        # Get actual PNL values for financial evaluation
        test_with_target = self.test_features.dropna(subset=['target'])
        actual_pnl = test_with_target['next_day_realized_pnl'].values

        # Comprehensive evaluation
        self.final_metrics = self.evaluator.generate_evaluation_report(
            y_true=y_test.values,
            y_pred=y_pred_test,
            actual_pnl=actual_pnl,
            risk_signals=risk_signals,
            dates=test_with_target['trade_date']
        )

        # Print results
        self.evaluator.print_evaluation_summary(self.final_metrics)

        # Store predictions for further analysis
        self.final_predictions = pd.DataFrame({
            'account_id': test_with_target['account_id'].values,
            'trade_date': test_with_target['trade_date'].values,
            'actual_target': y_test.values,
            'predicted_target': y_pred_test,
            'actual_pnl': actual_pnl,
            'risk_signal': risk_signals
        })

        return self.final_metrics

    def _summarize_walk_forward_results(self):
        """Summarize walk-forward validation results."""
        if not self.walk_forward_results:
            return None

        # Calculate average metrics
        mae_scores = [result['metrics']['mae'] for result in self.walk_forward_results]
        r2_scores = [result['metrics']['r2'] for result in self.walk_forward_results]

        summary = {
            'num_folds': len(self.walk_forward_results),
            'avg_mae': np.mean(mae_scores),
            'std_mae': np.std(mae_scores),
            'avg_r2': np.mean(r2_scores),
            'std_r2': np.std(r2_scores),
            'mae_scores': mae_scores,
            'r2_scores': r2_scores
        }

        print(f"\nWalk-Forward Validation Summary:")
        print(f"  Average MAE: {summary['avg_mae']:.4f} ± {summary['std_mae']:.4f}")
        print(f"  Average R²: {summary['avg_r2']:.4f} ± {summary['std_r2']:.4f}")

        return summary

    def generate_daily_risk_signals(self):
        """Generate actionable daily risk signals."""
        if self.final_predictions is None:
            print("No predictions available. Run evaluation first.")
            return None

        # Create risk signal mapping
        signal_map = {0: 'HIGH RISK', 1: 'NEUTRAL', 2: 'LOW RISK'}
        action_map = {
            0: 'Reduce position sizes, avoid new positions',
            1: 'Trade normally',
            2: 'Favorable conditions for trading'
        }

        # Generate daily signals
        signals_df = self.final_predictions.copy()
        signals_df['risk_level'] = signals_df['risk_signal'].map(signal_map)
        signals_df['recommended_action'] = signals_df['risk_signal'].map(action_map)

        # Show sample signals
        print("\nSample Daily Risk Signals:")
        print(signals_df[['account_id', 'trade_date', 'risk_level', 'recommended_action']].head(10))

        return signals_df

    def run_full_pipeline(self):
        """Run the complete risk management pipeline."""
        print("Starting Trader Risk Management Pipeline...")
        print("="*60)

        try:
            # Step 1: Load and prepare data
            self.load_and_prepare_data()

            # Step 2: Tune hyperparameters
            self.tune_model_parameters()

            # Step 3: Walk-forward validation
            self.run_walk_forward_validation()

            # Step 4: Train final model
            self.train_final_model()

            # Step 5: Evaluate on test set (ONE TIME ONLY)
            self.evaluate_on_test_set()

            # Step 6: Generate actionable signals
            self.generate_daily_risk_signals()

            print("\nPipeline completed successfully!")

        except Exception as e:
            print(f"Pipeline failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

        return True

def main():
    """Main execution function."""
    # Initialize and run pipeline
    pipeline = RiskManagementPipeline(test_cutoff_date='2025-04-01')
    success = pipeline.run_full_pipeline()

    if success:
        print("\n" + "="*60)
        print("RISK MANAGEMENT SYSTEM READY")
        print("="*60)
        print("The system can now generate daily risk scores for traders.")
        print("Key outputs:")
        print("- Volatility-normalized risk predictions")
        print("- Three-tier risk signals (High/Neutral/Low)")
        print("- Actionable trading recommendations")
    else:
        print("Pipeline failed. Check error messages above.")

if __name__ == "__main__":
    main()
