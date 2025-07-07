import pandas as pd
import numpy as np
import pickle
from typing import Dict, Tuple, Optional, List
from pathlib import Path
import logging
import yaml
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.optimize import minimize_scalar, differential_evolution
import warnings
warnings.filterwarnings('ignore')

# Updated to use preprocessed data
# from src.data_processing import create_trader_day_panel
# from src.feature_engineering import build_features

logger = logging.getLogger(__name__)


class ThresholdOptimizer:
    """
    Optimizes VaR and loss probability thresholds for causal impact analysis.

    For each trader, finds optimal thresholds such that when:
    - VaR < var_threshold OR loss_prob > loss_prob_threshold
    Then PnL is reduced by 50% to simulate risk management intervention.
    """

    def __init__(self, models_dir: str = 'models/trader_specific_80pct', data_dir: str = 'data/processed/trader_splits'):
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.results_dir = Path('results/threshold_optimization')
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.sequence_length = 7  # From training

        logger.info(f"ThresholdOptimizer initialized with models from {self.models_dir}")
        logger.info(f"Using preprocessed data from {self.data_dir}")

    def load_trader_model(self, trader_id) -> Optional[Dict]:
        """Load trained models for a trader"""
        model_path = self.models_dir / f"{trader_id}_tuned_validated.pkl"
        if not model_path.exists():
            logger.warning(f"No model found for trader {trader_id}")
            return None

        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        return model_data

    def load_trader_data(self, trader_id: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """Load preprocessed training and test data for a trader"""
        trader_dir = self.data_dir / str(trader_id)

        if not trader_dir.exists():
            logger.warning(f"No data directory found for trader {trader_id}")
            return None, None, None

        # Load training and test data
        train_path = trader_dir / 'train_data.parquet'
        test_path = trader_dir / 'test_data.parquet'
        metadata_path = trader_dir / 'metadata.json'

        if not all(p.exists() for p in [train_path, test_path, metadata_path]):
            logger.warning(f"Missing data files for trader {trader_id}")
            return None, None, None

        train_data = pd.read_parquet(train_path)
        test_data = pd.read_parquet(test_path)

        with open(metadata_path, 'r') as f:
            import json
            metadata = json.load(f)

        # Convert date columns to datetime
        train_data['date'] = pd.to_datetime(train_data['date'])
        test_data['date'] = pd.to_datetime(test_data['date'])

        return train_data, test_data, metadata

    def prepare_trader_features(self, data: pd.DataFrame, trader_id: str, feature_names: List[str]) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Prepare features from preprocessed data (same as training logic)"""
        trader_data = data.sort_values('date').copy()

        # Get all feature columns (exclude metadata and targets)
        exclude_cols = ['trader_id', 'date', 'target_pnl', 'target_large_loss']
        available_feature_cols = [col for col in trader_data.columns if col not in exclude_cols]

        # Remove features that are all null
        valid_feature_cols = []
        for col in available_feature_cols:
            if not trader_data[col].isnull().all():
                valid_feature_cols.append(col)

        # Prepare data with sequence_length lookback
        features_list = []
        targets_cls = []
        targets_pnl = []
        feature_dates = []

        for i in range(self.sequence_length, len(trader_data)):
            # Get current row features
            current_features = trader_data.iloc[i][valid_feature_cols]
            target_cls = trader_data.iloc[i]['target_large_loss']
            target_pnl = trader_data.iloc[i]['target_pnl'] if 'target_pnl' in trader_data.columns else trader_data.iloc[i]['daily_pnl']
            current_date = trader_data.iloc[i]['date']

            # Check for valid data
            features_valid = not pd.isna(current_features.values).any()
            targets_valid = not pd.isna(target_cls) and not pd.isna(target_pnl)

            if features_valid and targets_valid:
                features_list.append(current_features.values)
                targets_cls.append(target_cls)
                targets_pnl.append(target_pnl)
                feature_dates.append(current_date)

        if len(features_list) == 0:
            return None, None, None

        # Convert to arrays/DataFrame
        features_df = pd.DataFrame(features_list, columns=valid_feature_cols)
        features_df['date'] = feature_dates
        targets_cls = np.array(targets_cls, dtype=np.float32)
        targets_pnl = np.array(targets_pnl, dtype=np.float32)

        # Clean data
        features_df[valid_feature_cols] = features_df[valid_feature_cols].replace([np.inf, -np.inf], np.nan)
        features_df[valid_feature_cols] = features_df[valid_feature_cols].fillna(
            features_df[valid_feature_cols].median()
        )
        features_df[valid_feature_cols] = features_df[valid_feature_cols].clip(-1e6, 1e6)

        # Align features with model's expected feature names
        common_features = [f for f in feature_names if f in valid_feature_cols]
        if len(common_features) != len(feature_names):
            logger.warning(f"Feature mismatch for trader {trader_id}. Expected: {len(feature_names)}, Available: {len(common_features)}")

        # Use only common features
        features_df = features_df[['date'] + common_features]

        return features_df, targets_cls, targets_pnl

    def generate_predictions(self, model_data: Dict, data: pd.DataFrame,
                           trader_id: str, date_range: Tuple[str, str]) -> pd.DataFrame:
        """Generate predictions for a trader in a specific date range using preprocessed data"""
        start_date, end_date = date_range

        # Prepare features from the data
        combined_features, targets_cls, targets_pnl = self.prepare_trader_features(
            data, trader_id, model_data['feature_names']
        )

        if combined_features is None:
            return pd.DataFrame()

        # Filter to date range
        date_mask = (combined_features['date'] >= start_date) & (combined_features['date'] <= end_date)
        combined_features_filtered = combined_features[date_mask]
        targets_cls_filtered = targets_cls[date_mask]
        targets_pnl_filtered = targets_pnl[date_mask]

        if len(combined_features_filtered) == 0:
            return pd.DataFrame()

        # Generate predictions
        feature_cols = [col for col in combined_features_filtered.columns if col != 'date']
        X = combined_features_filtered[feature_cols]

        # Classification predictions (loss probability)
        cls_model = model_data['classification_model']
        loss_prob = cls_model.predict_proba(X)[:, 1]

        # VaR predictions
        var_model = model_data['var_model']
        var_pred = var_model.predict(X)

        # Create results DataFrame
        results = pd.DataFrame({
            'trader_id': trader_id,
            'date': combined_features_filtered['date'].values,
            'loss_prob': loss_prob,
            'var_pred': var_pred,
            'actual_loss': targets_cls_filtered,
            'actual_pnl': targets_pnl_filtered
        })

        return results

    def calculate_causal_impact(self, predictions: pd.DataFrame,
                              var_threshold: float, loss_prob_threshold: float) -> Dict:
        """Calculate causal impact of applying thresholds"""
        if len(predictions) == 0:
            return {'total_pnl': 0, 'adjusted_pnl': 0, 'impact': 0, 'interventions': 0}

        # Identify intervention days
        intervention_mask = (
            (predictions['var_pred'] < var_threshold) |
            (predictions['loss_prob'] > loss_prob_threshold)
        )

        # Calculate original and adjusted PnL
        original_pnl = predictions['actual_pnl'].sum()
        adjusted_pnl = predictions['actual_pnl'].copy()

        # Apply 50% reduction on intervention days
        adjusted_pnl[intervention_mask] *= 0.5
        total_adjusted_pnl = adjusted_pnl.sum()

        # Calculate impact (improvement in PnL)
        impact = total_adjusted_pnl - original_pnl
        n_interventions = intervention_mask.sum()

        return {
            'total_pnl': original_pnl,
            'adjusted_pnl': total_adjusted_pnl,
            'impact': impact,
            'interventions': n_interventions,
            'intervention_rate': n_interventions / len(predictions)
        }

    def optimize_trader_thresholds(self, trader_id: str, data: pd.DataFrame,
                                 validation_range: Tuple[str, str]) -> Dict:
        """Optimize thresholds for a single trader using validation data"""
        logger.info(f"Optimizing thresholds for trader {trader_id}")

        # Load model
        model_data = self.load_trader_model(trader_id)
        if model_data is None:
            return None

        # Generate predictions on validation set
        predictions = self.generate_predictions(model_data, data, trader_id, validation_range)
        if len(predictions) == 0:
            logger.warning(f"No predictions generated for trader {trader_id}")
            return None

        logger.info(f"Generated {len(predictions)} predictions for trader {trader_id}")

        # Define search ranges based on data distribution
        # Expand ranges to ensure we can find solutions with ≤30% intervention rate
        var_percentiles = np.percentile(predictions['var_pred'], [1, 5, 10, 20, 30, 50, 70, 90])
        loss_prob_percentiles = np.percentile(predictions['loss_prob'], [30, 50, 70, 80, 90, 95, 99])

        # Objective function to maximize PnL improvement with intervention rate constraint
        def objective(params):
            var_thresh, loss_prob_thresh = params
            impact_result = self.calculate_causal_impact(predictions, var_thresh, loss_prob_thresh)

            # We want to maximize impact (PnL improvement) while keeping intervention rate ≤ 25% (with buffer)
            impact = impact_result['impact']
            intervention_rate = impact_result['intervention_rate']

            # Hard constraint: intervention rate must be ≤ 25% (conservative buffer for distribution shift)
            if intervention_rate > 0.25:
                # Return a large penalty to make this solution infeasible
                return 1e10

            return -impact  # Minimize negative impact (maximize positive impact)

        # Search bounds - use wider ranges to ensure feasible solutions
        bounds = [
            (var_percentiles[0], var_percentiles[-1]),  # VaR threshold range
            (loss_prob_percentiles[0], loss_prob_percentiles[-1])  # Loss prob threshold range
        ]

        # Optimize using differential evolution with increased iterations for constraint handling
        result = differential_evolution(
            objective,
            bounds,
            seed=42,
            maxiter=200,  # Increased iterations to handle constraint
            popsize=20,   # Increased population size
            atol=1e-6,
            tol=1e-6
        )

        if result.success:
            optimal_var_thresh, optimal_loss_prob_thresh = result.x

            # Calculate final metrics with optimal thresholds
            optimal_impact = self.calculate_causal_impact(
                predictions, optimal_var_thresh, optimal_loss_prob_thresh
            )

            # Store results
            optimization_result = {
                'trader_id': trader_id,
                'optimal_var_threshold': optimal_var_thresh,
                'optimal_loss_prob_threshold': optimal_loss_prob_thresh,
                'validation_impact': optimal_impact,
                'validation_samples': len(predictions),
                'var_threshold_percentile': np.searchsorted(np.sort(predictions['var_pred']), optimal_var_thresh) / len(predictions) * 100,
                'loss_prob_threshold_percentile': np.searchsorted(np.sort(predictions['loss_prob']), optimal_loss_prob_thresh) / len(predictions) * 100,
                'optimization_success': True
            }

            logger.info(f"Trader {trader_id} optimal thresholds: VaR={optimal_var_thresh:.2f}, "
                       f"Loss_Prob={optimal_loss_prob_thresh:.4f}, Impact={optimal_impact['impact']:.2f}, "
                       f"Validation_Rate={optimal_impact['intervention_rate']*100:.1f}%")

            return optimization_result
        else:
            logger.warning(f"Optimization failed for trader {trader_id}")
            return {
                'trader_id': trader_id,
                'optimization_success': False,
                'error': result.message
            }

    def evaluate_on_test_set(self, trader_id: str, data: pd.DataFrame,
                           test_range: Tuple[str, str], thresholds: Dict) -> Dict:
        """Evaluate optimized thresholds on unseen test data"""
        logger.info(f"Evaluating trader {trader_id} on test set")

        # Load model
        model_data = self.load_trader_model(trader_id)
        if model_data is None:
            return None

        # Generate predictions on test set
        predictions = self.generate_predictions(model_data, data, trader_id, test_range)
        if len(predictions) == 0:
            return None

        # Apply optimized thresholds
        test_impact = self.calculate_causal_impact(
            predictions,
            thresholds['optimal_var_threshold'],
            thresholds['optimal_loss_prob_threshold']
        )

        # Calculate baseline (no intervention)
        baseline_impact = self.calculate_causal_impact(
            predictions,
            float('-inf'),  # Never trigger VaR threshold
            1.0  # Never trigger loss prob threshold
        )

        evaluation_result = {
            'trader_id': trader_id,
            'test_samples': len(predictions),
            'test_impact': test_impact,
            'baseline_pnl': baseline_impact['total_pnl'],
            'improvement': test_impact['impact'],
            'improvement_pct': (test_impact['impact'] / abs(baseline_impact['total_pnl']) * 100) if baseline_impact['total_pnl'] != 0 else 0
        }

        logger.info(f"Trader {trader_id} test evaluation: "
                   f"Improvement={evaluation_result['improvement']:.2f} "
                   f"({evaluation_result['improvement_pct']:.2f}%)")

        return evaluation_result

    def get_available_traders(self) -> List[str]:
        """Get list of available trader IDs from saved models"""
        available_traders = []
        for model_file in self.models_dir.glob("*_tuned_validated.pkl"):
            trader_id = model_file.stem.split('_')[0]
            if trader_id != 'training':
                available_traders.append(trader_id)
        return sorted(available_traders)

    def run_threshold_optimization(self) -> Dict:
        """Run threshold optimization for all traders using preprocessed data"""
        logger.info("Starting threshold optimization for all traders")

        # Get available traders from saved models
        available_traders = self.get_available_traders()
        logger.info(f"Found {len(available_traders)} trained models: {available_traders}")

        results = {
            'optimization_results': {},
            'test_evaluations': {},
            'summary': {}
        }

        for trader_id in available_traders:
            try:
                # Load preprocessed data for this trader
                train_data, test_data, metadata = self.load_trader_data(trader_id)
                if train_data is None:
                    logger.warning(f"Could not load data for trader {trader_id}")
                    continue

                # Load model
                model_data = self.load_trader_model(trader_id)
                if model_data is None:
                    continue

                # For validation, we'll use a portion of training data (last 25% for more robust validation)
                # This simulates the validation split used during training
                train_dates = sorted(train_data['date'].unique())
                validation_split_idx = int(len(train_dates) * 0.75)  # Use last 25% for validation
                validation_start = train_dates[validation_split_idx]
                validation_end = train_dates[-1]

                validation_range = (
                    validation_start.strftime('%Y-%m-%d'),
                    validation_end.strftime('%Y-%m-%d')
                )

                # Test range from test data
                test_dates = sorted(test_data['date'].unique())
                test_range = (
                    test_dates[0].strftime('%Y-%m-%d'),
                    test_dates[-1].strftime('%Y-%m-%d')
                )

                logger.info(f"Trader {trader_id} - Validation: {validation_range}, Test: {test_range}")

                # Optimize thresholds on validation portion of training data
                optimization_result = self.optimize_trader_thresholds(
                    trader_id, train_data, validation_range
                )

                if optimization_result and optimization_result.get('optimization_success'):
                    results['optimization_results'][trader_id] = optimization_result

                    # Evaluate on test set
                    test_evaluation = self.evaluate_on_test_set(
                        trader_id, test_data, test_range, optimization_result
                    )

                    if test_evaluation:
                        results['test_evaluations'][trader_id] = test_evaluation

            except Exception as e:
                logger.error(f"Error processing trader {trader_id}: {str(e)}")
                continue

        # Calculate summary statistics
        if results['test_evaluations']:
            test_improvements = [r['improvement'] for r in results['test_evaluations'].values()]
            test_improvements_pct = [r['improvement_pct'] for r in results['test_evaluations'].values()]

            results['summary'] = {
                'total_traders_optimized': len(results['optimization_results']),
                'total_traders_evaluated': len(results['test_evaluations']),
                'mean_improvement': np.mean(test_improvements),
                'median_improvement': np.median(test_improvements),
                'mean_improvement_pct': np.mean(test_improvements_pct),
                'median_improvement_pct': np.median(test_improvements_pct),
                'positive_improvements': sum(1 for imp in test_improvements if imp > 0),
                'total_improvement': sum(test_improvements)
            }

        # Save results
        results_path = self.results_dir / 'threshold_optimization_results.pkl'
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)

        logger.info(f"Threshold optimization completed. Results saved to {results_path}")
        logger.info(f"Summary: {results['summary']}")

        return results


def run_threshold_optimization(data_dir: str = 'data/processed/trader_splits', models_dir: str = 'models/trader_specific_80pct'):
    """Main function to run threshold optimization using preprocessed data"""
    logging.basicConfig(level=logging.INFO)

    optimizer = ThresholdOptimizer(models_dir=models_dir, data_dir=data_dir)
    results = optimizer.run_threshold_optimization()

    return results


if __name__ == "__main__":
    results = run_threshold_optimization()

    if results['summary']:
        print(f"\n=== Threshold Optimization Summary ===")
        print(f"Traders optimized: {results['summary']['total_traders_optimized']}")
        print(f"Traders evaluated: {results['summary']['total_traders_evaluated']}")
        print(f"Mean improvement: ${results['summary']['mean_improvement']:.2f}")
        print(f"Mean improvement %: {results['summary']['mean_improvement_pct']:.2f}%")
        print(f"Positive improvements: {results['summary']['positive_improvements']}")
        print(f"Total improvement: ${results['summary']['total_improvement']:.2f}")
