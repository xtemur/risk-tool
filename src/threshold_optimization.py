import pandas as pd
import numpy as np
import pickle
import json
from typing import Dict, Tuple, Optional, List
from pathlib import Path
import logging
import yaml
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.optimize import minimize_scalar
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

    def __init__(self, models_dir: str = 'models/trader_specific', data_dir: str = 'data/processed/trader_splits', model_suffix: str = ''):
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.model_suffix = model_suffix
        self.results_dir = Path('results/threshold_optimization')
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.sequence_length = 7  # From training

        logger.info(f"ThresholdOptimizer initialized with models from {self.models_dir}")
        logger.info(f"Using preprocessed data from {self.data_dir}")

    def load_trader_model(self, trader_id) -> Optional[Dict]:
        """Load trained models for a trader"""
        model_path = self.models_dir / f"{trader_id}_tuned_validated{self.model_suffix}.pkl"
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

    def train_linear_regression_for_trader(self, trader_id: str, predictions: pd.DataFrame) -> Dict:
        """
        Train linear regression to find optimal thresholds for a trader with capped intervention rate.

        Uses linear regression to create a risk score, then finds thresholds that maximize
        PnL improvement while keeping intervention rate under control.
        """
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score

        # Prepare features: VaR and loss probability
        var_pred = predictions['var_pred'].values
        loss_prob = predictions['loss_prob'].values

        # Create target: binary indicator of large loss
        # We'll use actual PnL as proxy for risk outcome
        actual_pnl = predictions['actual_pnl'].values
        target = (actual_pnl < 0).astype(int)  # 1 if loss, 0 if profit

        # Normalize features
        var_min = var_pred.min()
        var_max = var_pred.max()

        # Normalize VaR to [0, 1] (higher = higher risk)
        if var_max != var_min:
            normalized_var = (var_max - var_pred) / (var_max - var_min)
        else:
            normalized_var = np.full_like(var_pred, 0.5)

        # Loss probability is already normalized [0, 1]
        normalized_loss_prob = loss_prob

        # Prepare feature matrix
        X = np.column_stack([normalized_var, normalized_loss_prob])
        y = target

        # Fit linear regression
        model = LinearRegression()
        model.fit(X, y)

        # Get coefficients
        alpha = model.coef_[0]  # VaR coefficient
        beta = model.coef_[1]   # Loss probability coefficient
        bias = model.intercept_

        # Calculate risk scores
        risk_scores = model.predict(X)

        # Maximum allowed intervention rate
        max_intervention_rate = 0.25

        # Sort indices by risk score (highest risk first)
        sorted_indices = np.argsort(-risk_scores)

        # Find the best thresholds by testing different intervention rates
        best_impact = float('-inf')
        best_var_threshold = None
        best_loss_prob_threshold = None
        best_intervention_rate = None

        # Test intervention rates from 5% to max_intervention_rate in steps
        for test_rate in np.arange(0.05, max_intervention_rate + 0.01, 0.02):
            # Get top risk indices based on test rate
            n_interventions = int(len(risk_scores) * test_rate)
            if n_interventions == 0:
                continue

            top_risk_indices = sorted_indices[:n_interventions]

            # Find threshold boundaries that capture these high-risk cases
            # We want the most restrictive thresholds that still capture all high-risk cases
            high_risk_var = var_pred[top_risk_indices]
            high_risk_loss_prob = loss_prob[top_risk_indices]

            # Set thresholds to capture these high-risk cases
            # For VaR: use the maximum (least negative) value among high-risk cases
            # For loss_prob: use the minimum value among high-risk cases
            test_var_threshold = np.max(high_risk_var)
            test_loss_prob_threshold = np.min(high_risk_loss_prob)

            # Calculate intervention mask with these thresholds
            intervention_mask = ((var_pred <= test_var_threshold) | (loss_prob >= test_loss_prob_threshold))
            actual_rate = intervention_mask.mean()

            # Skip if actual rate exceeds maximum
            if actual_rate > max_intervention_rate:
                continue

            # Calculate impact with these thresholds
            impact_result = self.calculate_causal_impact(
                predictions, test_var_threshold, test_loss_prob_threshold
            )

            # Update best if this is better
            if impact_result['impact'] > best_impact:
                best_impact = impact_result['impact']
                best_var_threshold = test_var_threshold
                best_loss_prob_threshold = test_loss_prob_threshold
                best_intervention_rate = actual_rate

        # If no valid thresholds found, use conservative defaults
        if best_var_threshold is None:
            # Use percentiles that give approximately 15% intervention rate
            best_var_threshold = np.percentile(var_pred, 15)
            best_loss_prob_threshold = np.percentile(loss_prob, 85)
            intervention_mask = ((var_pred <= best_var_threshold) | (loss_prob >= best_loss_prob_threshold))
            best_intervention_rate = intervention_mask.mean()

        # Calculate performance metrics
        try:
            auc_score = roc_auc_score(y, risk_scores)
        except:
            auc_score = 0.5

        r2 = r2_score(y, risk_scores)
        mse = mean_squared_error(y, risk_scores)

        return {
            'var_threshold': best_var_threshold,
            'loss_prob_threshold': best_loss_prob_threshold,
            'metrics': {
                'alpha': alpha,
                'beta': beta,
                'bias': bias,
                'auc': auc_score,
                'r2': r2,
                'mse': mse,
                'risk_scores_mean': risk_scores.mean(),
                'risk_scores_std': risk_scores.std(),
                'actual_intervention_rate': best_intervention_rate,
                'max_intervention_rate': max_intervention_rate,
                'best_impact': best_impact,
                'var_range': (var_min, var_max)
            }
        }

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
        """Optimize thresholds for a single trader using linear regression"""
        logger.info(f"Optimizing thresholds for trader {trader_id} using linear regression")

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

        # Use linear regression to find optimal thresholds
        linear_result = self.train_linear_regression_for_trader(trader_id, predictions)
        if linear_result is None:
            logger.warning(f"Linear regression failed for trader {trader_id}")
            return {
                'trader_id': trader_id,
                'optimization_success': False,
                'error': 'Linear regression failed'
            }

        # Calculate final metrics with linear regression thresholds
        optimal_impact = self.calculate_causal_impact(
            predictions, linear_result['var_threshold'], linear_result['loss_prob_threshold']
        )

        # Store results
        optimization_result = {
            'trader_id': trader_id,
            'optimal_var_threshold': linear_result['var_threshold'],
            'optimal_loss_prob_threshold': linear_result['loss_prob_threshold'],
            'validation_impact': optimal_impact,
            'validation_samples': len(predictions),
            'var_threshold_percentile': np.searchsorted(np.sort(predictions['var_pred']), linear_result['var_threshold']) / len(predictions) * 100,
            'loss_prob_threshold_percentile': np.searchsorted(np.sort(predictions['loss_prob']), linear_result['loss_prob_threshold']) / len(predictions) * 100,
            'optimization_success': True,
            'linear_regression_metrics': linear_result['metrics']
        }

        logger.info(f"Trader {trader_id} linear regression thresholds: VaR={linear_result['var_threshold']:.2f}, "
                   f"Loss_Prob={linear_result['loss_prob_threshold']:.4f}, Impact={optimal_impact['impact']:.2f}, "
                   f"Validation_Rate={optimal_impact['intervention_rate']*100:.1f}%")

        return optimization_result

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

        # Save thresholds as JSON to configs directory
        configs_dir = Path('configs/optimal_thresholds')
        configs_dir.mkdir(parents=True, exist_ok=True)

        # Extract thresholds for JSON format
        thresholds = []
        for trader_id, trader_results in results['optimization_results'].items():
            if 'optimal_var_threshold' in trader_results:
                thresholds.append({
                    "trader_id": trader_id,
                    "var_threshold": trader_results['optimal_var_threshold'],
                    "loss_prob_threshold": trader_results['optimal_loss_prob_threshold']
                })

        # Save thresholds in JSON format
        thresholds_data = {
            "thresholds": thresholds,
            "optimization_date": pd.Timestamp.now().isoformat(),
            "total_traders": len(thresholds)
        }

        thresholds_path = configs_dir / 'optimal_thresholds.json'
        with open(thresholds_path, 'w') as f:
            json.dump(thresholds_data, f, indent=2)

        logger.info(f"Threshold optimization completed. Results saved to {results_path}")
        logger.info(f"Thresholds saved to {thresholds_path}")
        logger.info(f"Summary: {results['summary']}")

        return results


def run_threshold_optimization(data_dir: str = 'data/processed/trader_splits', models_dir: str = 'models/trader_specific'):
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
