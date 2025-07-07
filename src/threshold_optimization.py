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

from src.data_processing import create_trader_day_panel
from src.feature_engineering import build_features

logger = logging.getLogger(__name__)


class ThresholdOptimizer:
    """
    Optimizes VaR and loss probability thresholds for causal impact analysis.

    For each trader, finds optimal thresholds such that when:
    - VaR < var_threshold OR loss_prob > loss_prob_threshold
    Then PnL is reduced by 50% to simulate risk management intervention.
    """

    def __init__(self, models_dir: str = 'models/trader_specific_80pct'):
        self.models_dir = Path(models_dir)
        self.results_dir = Path('results/threshold_optimization')
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Load configuration
        with open('configs/main_config.yaml', 'r') as f:
            self.config = yaml.safe_load(f)

        self.sequence_length = 7  # From training

        logger.info(f"ThresholdOptimizer initialized with models from {self.models_dir}")

    def load_trader_model(self, trader_id) -> Optional[Dict]:
        """Load trained models for a trader"""
        model_path = self.models_dir / f"{trader_id}_tuned_validated.pkl"
        if not model_path.exists():
            logger.warning(f"No model found for trader {trader_id}")
            return None

        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        return model_data

    def prepare_trader_features(self, df: pd.DataFrame, trader_id,
                              feature_names: List[str]) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Prepare features for a trader using the same logic as training"""
        trader_data = df[df['trader_id'] == trader_id].sort_values('date').copy()

        # Raw feature columns from training
        raw_feature_cols = [
            'daily_pnl', 'daily_gross', 'daily_fees', 'daily_volume',
            'n_trades', 'gross_profit', 'gross_loss'
        ]

        # Get engineered features (exclude metadata and targets)
        exclude_cols = ['trader_id', 'date', 'target_pnl', 'target_large_loss']
        engineered_cols = [col for col in trader_data.columns
                          if col not in exclude_cols + raw_feature_cols]

        # Filter out features that are all null for this trader
        valid_engineered_cols = []
        for col in engineered_cols:
            if not trader_data[col].isnull().all():
                valid_engineered_cols.append(col)

        # Prepare sequential data and aligned engineered features
        combined_features_list = []
        aligned_targets = []
        aligned_pnl = []
        feature_dates = []

        for i in range(self.sequence_length, len(trader_data)):
            # Raw sequence: last sequence_length days (flattened)
            sequence_data = trader_data.iloc[i-self.sequence_length:i][raw_feature_cols].values
            flattened_sequence = sequence_data.flatten()

            # Current day's engineered features
            current_engineered = trader_data.iloc[i][valid_engineered_cols]

            # Target and PnL
            target = trader_data.iloc[i]['target_large_loss']
            pnl = trader_data.iloc[i]['daily_pnl']
            current_date = trader_data.iloc[i]['date']

            # Only include if we have valid data
            sequence_valid = not pd.isna(sequence_data).any()
            engineered_valid = not pd.isna(current_engineered.values).any()
            target_valid = not pd.isna(target)
            pnl_valid = not pd.isna(pnl)

            if sequence_valid and engineered_valid and target_valid and pnl_valid:
                # Combine engineered features with flattened sequence
                combined_row = np.concatenate([current_engineered.values, flattened_sequence])
                combined_features_list.append(combined_row)
                aligned_targets.append(target)
                aligned_pnl.append(pnl)
                feature_dates.append(current_date)

        if len(combined_features_list) == 0:
            return None, None, None

        # Create feature names (same order as training)
        engineered_feature_names = valid_engineered_cols
        sequence_feature_names = []
        for day in range(self.sequence_length):
            for feat in raw_feature_cols:
                sequence_feature_names.append(f"{feat}_lag_{day+1}")

        all_feature_names = engineered_feature_names + sequence_feature_names

        # Convert to DataFrame
        combined_features = pd.DataFrame(combined_features_list, columns=all_feature_names)
        combined_features['date'] = feature_dates

        # Clean data (same as training)
        feature_cols = [col for col in combined_features.columns if col != 'date']
        combined_features[feature_cols] = combined_features[feature_cols].replace([np.inf, -np.inf], np.nan)
        combined_features[feature_cols] = combined_features[feature_cols].fillna(
            combined_features[feature_cols].median()
        )
        combined_features[feature_cols] = combined_features[feature_cols].clip(-1e6, 1e6)

        # Reorder columns to match training
        try:
            combined_features = combined_features[['date'] + feature_names]
        except KeyError as e:
            logger.warning(f"Feature mismatch for trader {trader_id}: {e}")
            # Use available features
            available_features = [f for f in feature_names if f in combined_features.columns]
            combined_features = combined_features[['date'] + available_features]

        targets = np.array(aligned_targets, dtype=np.float32)
        pnl_values = np.array(aligned_pnl, dtype=np.float32)

        return combined_features, targets, pnl_values

    def generate_predictions(self, model_data: Dict, df: pd.DataFrame,
                           trader_id, date_range: Tuple[str, str]) -> pd.DataFrame:
        """Generate predictions for a trader in a specific date range"""
        start_date, end_date = date_range

        # Filter data to date range
        trader_data = df[
            (df['trader_id'] == trader_id) &
            (df['date'] >= start_date) &
            (df['date'] <= end_date)
        ].copy()

        if len(trader_data) == 0:
            return pd.DataFrame()

        # Prepare features
        combined_features, targets, pnl_values = self.prepare_trader_features(
            df, trader_id, model_data['feature_names']
        )

        if combined_features is None:
            return pd.DataFrame()

        # Filter to date range
        date_mask = (combined_features['date'] >= start_date) & (combined_features['date'] <= end_date)
        combined_features_filtered = combined_features[date_mask]
        targets_filtered = targets[date_mask]
        pnl_filtered = pnl_values[date_mask]

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
            'actual_loss': targets_filtered,
            'actual_pnl': pnl_filtered
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

    def optimize_trader_thresholds(self, trader_id, df: pd.DataFrame,
                                 validation_range: Tuple[str, str]) -> Dict:
        """Optimize thresholds for a single trader using validation data"""
        logger.info(f"Optimizing thresholds for trader {trader_id}")

        # Load model
        model_data = self.load_trader_model(trader_id)
        if model_data is None:
            return None

        # Generate predictions on validation set
        predictions = self.generate_predictions(model_data, df, trader_id, validation_range)
        if len(predictions) == 0:
            logger.warning(f"No predictions generated for trader {trader_id}")
            return None

        logger.info(f"Generated {len(predictions)} predictions for trader {trader_id}")

        # Define search ranges based on data distribution
        var_percentiles = np.percentile(predictions['var_pred'], [5, 10, 20, 30, 50])
        loss_prob_percentiles = np.percentile(predictions['loss_prob'], [50, 70, 80, 90, 95])

        # Objective function to maximize PnL improvement
        def objective(params):
            var_thresh, loss_prob_thresh = params
            impact_result = self.calculate_causal_impact(predictions, var_thresh, loss_prob_thresh)

            # We want to maximize impact (PnL improvement) while not intervening too frequently
            impact = impact_result['impact']
            intervention_rate = impact_result['intervention_rate']

            # Penalize very high intervention rates (>50%)
            if intervention_rate > 0.5:
                penalty = (intervention_rate - 0.5) * abs(impact_result['total_pnl']) * 0.1
                impact -= penalty

            return -impact  # Minimize negative impact (maximize positive impact)

        # Search bounds
        bounds = [
            (var_percentiles[0], var_percentiles[-1]),  # VaR threshold range
            (loss_prob_percentiles[0], loss_prob_percentiles[-1])  # Loss prob threshold range
        ]

        # Optimize using differential evolution
        result = differential_evolution(
            objective,
            bounds,
            seed=42,
            maxiter=100,
            popsize=15
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
                       f"Loss_Prob={optimal_loss_prob_thresh:.4f}, Impact={optimal_impact['impact']:.2f}")

            return optimization_result
        else:
            logger.warning(f"Optimization failed for trader {trader_id}")
            return {
                'trader_id': trader_id,
                'optimization_success': False,
                'error': result.message
            }

    def evaluate_on_test_set(self, trader_id, df: pd.DataFrame,
                           test_range: Tuple[str, str], thresholds: Dict) -> Dict:
        """Evaluate optimized thresholds on unseen test data"""
        logger.info(f"Evaluating trader {trader_id} on test set")

        # Load model
        model_data = self.load_trader_model(trader_id)
        if model_data is None:
            return None

        # Generate predictions on test set
        predictions = self.generate_predictions(model_data, df, trader_id, test_range)
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

    def run_threshold_optimization(self) -> Dict:
        """Run threshold optimization for all traders"""
        logger.info("Starting threshold optimization for all traders")

        # Load data
        df = create_trader_day_panel(self.config)
        df = df.rename(columns={'account_id': 'trader_id', 'trade_date': 'date'})

        # Apply feature engineering
        df_for_features = df.rename(columns={'trader_id': 'account_id', 'date': 'trade_date'})
        df_for_features = build_features(df_for_features, self.config)
        df = df_for_features.rename(columns={'account_id': 'trader_id', 'trade_date': 'date'})

        # Get available traders from saved models
        available_traders = []
        for model_file in self.models_dir.glob("*_tuned_validated.pkl"):
            trader_id = model_file.stem.split('_')[0]
            if trader_id != 'training':
                available_traders.append(int(trader_id))  # Convert to int to match data

        logger.info(f"Found {len(available_traders)} trained models: {available_traders}")

        results = {
            'optimization_results': {},
            'test_evaluations': {},
            'summary': {}
        }

        for trader_id in available_traders:
            try:
                # Load model to get boundaries
                model_data = self.load_trader_model(trader_id)
                if model_data is None:
                    continue

                boundaries = model_data['boundaries']

                # Define validation and test ranges from boundaries
                validation_range = (
                    boundaries['validation_start'].strftime('%Y-%m-%d'),
                    boundaries['validation_end'].strftime('%Y-%m-%d')
                )
                test_range = (
                    boundaries['test_start'].strftime('%Y-%m-%d'),
                    boundaries['test_end'].strftime('%Y-%m-%d')
                )

                logger.info(f"Trader {trader_id} - Validation: {validation_range}, Test: {test_range}")

                # Optimize thresholds on validation set
                optimization_result = self.optimize_trader_thresholds(
                    trader_id, df, validation_range
                )

                if optimization_result and optimization_result.get('optimization_success'):
                    results['optimization_results'][trader_id] = optimization_result

                    # Evaluate on test set
                    test_evaluation = self.evaluate_on_test_set(
                        trader_id, df, test_range, optimization_result
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


def run_threshold_optimization():
    """Main function to run threshold optimization"""
    logging.basicConfig(level=logging.INFO)

    optimizer = ThresholdOptimizer()
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
