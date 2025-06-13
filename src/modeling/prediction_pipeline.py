"""
Prediction Pipeline

End-to-end pipeline that combines all modeling components for
comprehensive trader PnL prediction with proper validation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from pathlib import Path
from datetime import datetime
import json

from .config import ModelConfig
from .time_series_validator import TimeSeriesValidator
from .model_trainer import ModelTrainer
from .performance_evaluator import PerformanceEvaluator
from .causal_impact_analyzer import CausalImpactAnalyzer

logger = logging.getLogger(__name__)


class PredictionPipeline:
    """
    Complete end-to-end prediction pipeline
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize the pipeline

        Args:
            config: Model configuration object
        """
        self.config = config or ModelConfig()
        self.validator = TimeSeriesValidator(self.config)
        self.trainer = ModelTrainer(self.config)
        self.evaluator = PerformanceEvaluator(self.config)
        self.causal_analyzer = CausalImpactAnalyzer()

        # Results storage
        self.results = {}
        self.holdout_results = {}
        self.walk_forward_results = {}
        self.causal_impact_results = {}

    def run_full_pipeline(self, data: pd.DataFrame,
                         model_types: List[str] = None,
                         individual_traders: bool = True) -> Dict[str, Any]:
        """
        Run the complete pipeline: validation, training, and evaluation

        Args:
            data: Complete dataset with features and target
            model_types: List of model types to train (default: ['ridge'])
            individual_traders: Whether to train individual trader models

        Returns:
            Complete pipeline results
        """
        logger.info("Starting full prediction pipeline")

        if model_types is None:
            model_types = ['ridge']

        # Validate data
        if data.empty or 'target_next_pnl' not in data.columns:
            raise ValueError("Data must contain 'target_next_pnl' column")

        # 1. Create holdout split
        logger.info("Step 1: Creating holdout split")
        training_data, holdout_data = self.validator.create_holdout_split(data)

        if training_data.empty or holdout_data.empty:
            raise ValueError("Insufficient data for holdout split")

        # 2. Run walk-forward validation on training data
        logger.info("Step 2: Running walk-forward validation")
        wf_results = self.run_walk_forward_validation(
            training_data, model_types, individual_traders
        )

        # 3. Train final models on all training data
        logger.info("Step 3: Training final models")
        final_models = self.train_final_models(
            training_data, model_types, individual_traders
        )

        # 4. Evaluate on holdout data
        logger.info("Step 4: Evaluating on holdout data")
        holdout_results = self.evaluate_on_holdout(
            holdout_data, final_models, individual_traders
        )

        # 5. Calculate causal impact analysis
        logger.info("Step 5: Calculating causal impact analysis")
        causal_impact_results = self.calculate_causal_impact(
            holdout_data, final_models, individual_traders
        )

        # 6. Compile results
        pipeline_results = {
            'config': self.config.__dict__,
            'data_summary': {
                'total_samples': len(data),
                'training_samples': len(training_data),
                'holdout_samples': len(holdout_data),
                'traders': data['account_id'].nunique() if 'account_id' in data.columns else 1
            },
            'walk_forward_results': wf_results,
            'final_models': final_models,
            'holdout_results': holdout_results,
            'causal_impact_results': causal_impact_results,
            'validation_summary': self.validator.get_validation_summary(data)
        }

        self.results = pipeline_results

        logger.info("Pipeline completed successfully")
        return pipeline_results

    def run_walk_forward_validation(self, training_data: pd.DataFrame,
                                   model_types: List[str],
                                   individual_traders: bool = True) -> Dict[str, Any]:
        """
        Run walk-forward validation

        Args:
            training_data: Training dataset
            model_types: List of model types to validate
            individual_traders: Whether to validate individual trader models

        Returns:
            Walk-forward validation results
        """
        wf_results = {
            'model_types': model_types,
            'individual_traders': individual_traders,
            'results_by_fold': [],
            'aggregate_results': {}
        }

        if individual_traders and 'account_id' in training_data.columns:
            # Per-trader walk-forward validation
            traders = training_data['account_id'].unique()

            for trader in traders:
                logger.info(f"Walk-forward validation for trader {trader}")
                trader_data = training_data[training_data['account_id'] == trader]

                trader_wf_results = self._run_trader_walk_forward(
                    trader_data, trader, model_types
                )
                wf_results[f'trader_{trader}'] = trader_wf_results
        else:
            # Global walk-forward validation
            logger.info("Global walk-forward validation")
            global_wf_results = self._run_global_walk_forward(
                training_data, model_types
            )
            wf_results['global'] = global_wf_results

        return wf_results

    def _run_trader_walk_forward(self, trader_data: pd.DataFrame,
                                trader_id: str,
                                model_types: List[str]) -> Dict[str, Any]:
        """
        Run walk-forward validation for a single trader
        """
        trader_results = {'trader_id': trader_id, 'folds': []}

        # Get feature columns
        feature_cols = [col for col in trader_data.columns
                       if col not in ['account_id', 'date', 'target_next_pnl']]

        # Walk-forward splits
        fold_num = 0
        for train_fold, val_fold in self.validator.trader_walk_forward_split(
            trader_data, trader_id
        ):
            fold_num += 1
            logger.info(f"  Fold {fold_num}: {len(train_fold)} train, {len(val_fold)} val")

            fold_results = {'fold_num': fold_num, 'models': {}}

            # Prepare data
            X_train = train_fold[feature_cols]
            y_train = train_fold['target_next_pnl']
            X_val = val_fold[feature_cols]
            y_val = val_fold['target_next_pnl']

            # Train and evaluate each model type
            for model_type in model_types:
                try:
                    # Train model
                    model_key = f"{trader_id}_{model_type}_fold_{fold_num}"
                    train_result = self.trainer.train_model(
                        X_train, y_train,
                        model_type=model_type,
                        model_key=model_key,
                        optimize_hyperparams=True
                    )

                    # Make predictions
                    y_pred = self.trainer.predict(X_val, model_key)

                    # Evaluate
                    val_dates = pd.to_datetime(val_fold['date']) if 'date' in val_fold.columns else None
                    evaluation = self.evaluator.evaluate_model(
                        y_val.values, y_pred,
                        dates=val_dates,
                        model_name=f"{trader_id}_{model_type}"
                    )

                    fold_results['models'][model_type] = {
                        'train_metrics': train_result['metrics'],
                        'validation_evaluation': evaluation
                    }

                except Exception as e:
                    logger.error(f"Error in fold {fold_num} for {model_type}: {e}")
                    fold_results['models'][model_type] = {'error': str(e)}

            trader_results['folds'].append(fold_results)

        # Aggregate results across folds
        trader_results['aggregate'] = self._aggregate_fold_results(trader_results['folds'])

        return trader_results

    def _run_global_walk_forward(self, training_data: pd.DataFrame,
                                model_types: List[str]) -> Dict[str, Any]:
        """
        Run global walk-forward validation (all traders combined)
        """
        global_results = {'folds': []}

        # Get feature columns
        feature_cols = [col for col in training_data.columns
                       if col not in ['account_id', 'date', 'target_next_pnl']]

        # Walk-forward splits
        fold_num = 0
        for train_fold, val_fold in self.validator.walk_forward_split(training_data):
            fold_num += 1
            logger.info(f"  Global fold {fold_num}: {len(train_fold)} train, {len(val_fold)} val")

            fold_results = {'fold_num': fold_num, 'models': {}}

            # Prepare data
            X_train = train_fold[feature_cols]
            y_train = train_fold['target_next_pnl']
            X_val = val_fold[feature_cols]
            y_val = val_fold['target_next_pnl']

            # Train and evaluate each model type
            for model_type in model_types:
                try:
                    # Train model
                    model_key = f"global_{model_type}_fold_{fold_num}"
                    train_result = self.trainer.train_model(
                        X_train, y_train,
                        model_type=model_type,
                        model_key=model_key,
                        optimize_hyperparams=True
                    )

                    # Make predictions
                    y_pred = self.trainer.predict(X_val, model_key)

                    # Evaluate
                    val_dates = pd.to_datetime(val_fold['date']) if 'date' in val_fold.columns else None
                    evaluation = self.evaluator.evaluate_model(
                        y_val.values, y_pred,
                        dates=val_dates,
                        model_name=f"global_{model_type}"
                    )

                    fold_results['models'][model_type] = {
                        'train_metrics': train_result['metrics'],
                        'validation_evaluation': evaluation
                    }

                except Exception as e:
                    logger.error(f"Error in global fold {fold_num} for {model_type}: {e}")
                    fold_results['models'][model_type] = {'error': str(e)}

            global_results['folds'].append(fold_results)

        # Aggregate results
        global_results['aggregate'] = self._aggregate_fold_results(global_results['folds'])

        return global_results

    def _aggregate_fold_results(self, fold_results: List[Dict]) -> Dict[str, Any]:
        """
        Aggregate results across folds
        """
        if not fold_results:
            return {}

        # Get all model types
        model_types = set()
        for fold in fold_results:
            model_types.update(fold.get('models', {}).keys())

        aggregate = {}

        for model_type in model_types:
            model_results = []

            for fold in fold_results:
                if model_type in fold.get('models', {}):
                    fold_result = fold['models'][model_type]
                    if 'validation_evaluation' in fold_result:
                        model_results.append(fold_result['validation_evaluation'])

            if model_results:
                # Average key metrics across folds
                avg_metrics = {}

                # Statistical metrics
                stat_metrics = ['mae', 'rmse', 'r2', 'correlation']
                for metric in stat_metrics:
                    values = [r['statistical_metrics'][metric] for r in model_results
                             if metric in r['statistical_metrics']]
                    if values:
                        avg_metrics[f'avg_{metric}'] = np.mean(values)
                        avg_metrics[f'std_{metric}'] = np.std(values)

                # Financial metrics
                fin_metrics = ['hit_rate', 'actual_sharpe']
                for metric in fin_metrics:
                    values = [r['financial_metrics'][metric] for r in model_results
                             if metric in r['financial_metrics']]
                    if values:
                        avg_metrics[f'avg_{metric}'] = np.mean(values)
                        avg_metrics[f'std_{metric}'] = np.std(values)

                # Overall scores
                scores = [r['overall_score'] for r in model_results]
                if scores:
                    avg_metrics['avg_overall_score'] = np.mean(scores)
                    avg_metrics['std_overall_score'] = np.std(scores)

                aggregate[model_type] = {
                    'num_folds': len(model_results),
                    'average_metrics': avg_metrics
                }

        return aggregate

    def train_final_models(self, training_data: pd.DataFrame,
                          model_types: List[str],
                          individual_traders: bool = True) -> Dict[str, Any]:
        """
        Train final models on all training data
        """
        final_models = {}

        if individual_traders and 'account_id' in training_data.columns:
            # Train individual trader models
            traders = training_data['account_id'].unique()

            for trader in traders:
                logger.info(f"Training final model for trader {trader}")
                trader_data = training_data[training_data['account_id'] == trader]

                # Get features and target
                feature_cols = [col for col in trader_data.columns
                               if col not in ['account_id', 'date', 'target_next_pnl']]

                X = trader_data[feature_cols]
                y = trader_data['target_next_pnl']

                trader_models = {}
                for model_type in model_types:
                    try:
                        model_key = f"{trader}_final_{model_type}"
                        result = self.trainer.train_model(
                            X, y,
                            model_type=model_type,
                            model_key=model_key,
                            optimize_hyperparams=True
                        )
                        trader_models[model_type] = result

                    except Exception as e:
                        logger.error(f"Error training final {model_type} for trader {trader}: {e}")
                        trader_models[model_type] = {'error': str(e)}

                final_models[trader] = trader_models
        else:
            # Train global models
            logger.info("Training final global models")

            feature_cols = [col for col in training_data.columns
                           if col not in ['account_id', 'date', 'target_next_pnl']]

            X = training_data[feature_cols]
            y = training_data['target_next_pnl']

            global_models = {}
            for model_type in model_types:
                try:
                    model_key = f"global_final_{model_type}"
                    result = self.trainer.train_model(
                        X, y,
                        model_type=model_type,
                        model_key=model_key,
                        optimize_hyperparams=True
                    )
                    global_models[model_type] = result

                except Exception as e:
                    logger.error(f"Error training final global {model_type}: {e}")
                    global_models[model_type] = {'error': str(e)}

            final_models['global'] = global_models

        return final_models

    def evaluate_on_holdout(self, holdout_data: pd.DataFrame,
                           final_models: Dict[str, Any],
                           individual_traders: bool = True) -> Dict[str, Any]:
        """
        Evaluate final models on holdout data
        """
        holdout_results = {}

        if individual_traders and 'account_id' in holdout_data.columns:
            # Evaluate trader models
            traders = holdout_data['account_id'].unique()

            for trader in traders:
                if trader not in final_models:
                    logger.warning(f"No final model found for trader {trader}")
                    continue

                logger.info(f"Evaluating holdout for trader {trader}")
                trader_holdout = holdout_data[holdout_data['account_id'] == trader]

                if trader_holdout.empty:
                    continue

                # Get features and target
                feature_cols = [col for col in trader_holdout.columns
                               if col not in ['account_id', 'date', 'target_next_pnl']]

                X = trader_holdout[feature_cols]
                y = trader_holdout['target_next_pnl']
                dates = pd.to_datetime(trader_holdout['date']) if 'date' in trader_holdout.columns else None

                trader_results = {}
                for model_type, model_result in final_models[trader].items():
                    if 'error' in model_result:
                        trader_results[model_type] = model_result
                        continue

                    try:
                        # Make predictions
                        model_key = f"{trader}_final_{model_type}"
                        y_pred = self.trainer.predict(X, model_key)

                        # Evaluate
                        evaluation = self.evaluator.evaluate_model(
                            y.values, y_pred,
                            dates=dates,
                            model_name=f"{trader}_{model_type}_holdout"
                        )

                        trader_results[model_type] = evaluation

                    except Exception as e:
                        logger.error(f"Error evaluating holdout for {trader} {model_type}: {e}")
                        trader_results[model_type] = {'error': str(e)}

                holdout_results[trader] = trader_results
        else:
            # Evaluate global models
            logger.info("Evaluating global holdout")

            feature_cols = [col for col in holdout_data.columns
                           if col not in ['account_id', 'date', 'target_next_pnl']]

            X = holdout_data[feature_cols]
            y = holdout_data['target_next_pnl']
            dates = pd.to_datetime(holdout_data['date']) if 'date' in holdout_data.columns else None

            global_results = {}
            for model_type, model_result in final_models.get('global', {}).items():
                if 'error' in model_result:
                    global_results[model_type] = model_result
                    continue

                try:
                    # Make predictions
                    model_key = f"global_final_{model_type}"
                    y_pred = self.trainer.predict(X, model_key)

                    # Evaluate
                    evaluation = self.evaluator.evaluate_model(
                        y.values, y_pred,
                        dates=dates,
                        model_name=f"global_{model_type}_holdout"
                    )

                    global_results[model_type] = evaluation

                except Exception as e:
                        logger.error(f"Error evaluating global holdout {model_type}: {e}")
                        global_results[model_type] = {'error': str(e)}

            holdout_results['global'] = global_results

        return holdout_results

    def calculate_causal_impact(self, holdout_data: pd.DataFrame,
                               final_models: Dict[str, Any],
                               individual_traders: bool = True) -> Dict[str, Any]:
        """
        Calculate causal impact analysis for all models on holdout data

        Args:
            holdout_data: Holdout test data
            final_models: Trained final models
            individual_traders: Whether models are individual per trader

        Returns:
            Causal impact analysis results
        """
        causal_results = {}

        if individual_traders and 'account_id' in holdout_data.columns:
            # Per-trader causal impact analysis
            traders = holdout_data['account_id'].unique()
            trader_causal_results = {}

            for trader in traders:
                if trader not in final_models:
                    continue

                logger.info(f"Calculating causal impact for trader {trader}")
                trader_holdout = holdout_data[holdout_data['account_id'] == trader]

                if trader_holdout.empty:
                    continue

                # Get features and target
                feature_cols = [col for col in trader_holdout.columns
                               if col not in ['account_id', 'date', 'target_next_pnl']]

                X = trader_holdout[feature_cols]
                y_actual = trader_holdout['target_next_pnl'].values
                dates = pd.to_datetime(trader_holdout['date']) if 'date' in trader_holdout.columns else None

                trader_causal_models = {}

                # Calculate causal impact for each model type
                for model_type, model_result in final_models[trader].items():
                    if 'error' in model_result:
                        trader_causal_models[model_type] = {'error': model_result['error']}
                        continue

                    try:
                        # Get predictions
                        model_key = f"{trader}_final_{model_type}"
                        y_pred = self.trainer.predict(X, model_key)

                        # Calculate causal impact
                        causal_impact = self.causal_analyzer.calculate_trading_impact(
                            actual_pnl=y_actual,
                            predicted_pnl=y_pred,
                            dates=dates,
                            trader_id=trader
                        )

                        trader_causal_models[model_type] = causal_impact

                    except Exception as e:
                        logger.error(f"Error calculating causal impact for {trader} {model_type}: {e}")
                        trader_causal_models[model_type] = {'error': str(e)}

                trader_causal_results[trader] = trader_causal_models

            causal_results['individual_traders'] = trader_causal_results

            # Aggregate analysis across all traders
            logger.info("Calculating aggregate causal impact across traders")
            try:
                # Collect results for best model per trader
                trader_best_results = {}
                for trader, trader_models in trader_causal_results.items():
                    # Find best model for this trader (highest overall score from holdout results)
                    if trader in self.holdout_results:
                        best_model = None
                        best_score = -np.inf

                        for model_type, holdout_result in self.holdout_results[trader].items():
                            if 'error' not in holdout_result:
                                score = holdout_result.get('overall_score', 0)
                                if score > best_score:
                                    best_score = score
                                    best_model = model_type

                        if best_model and best_model in trader_models and 'error' not in trader_models[best_model]:
                            trader_best_results[trader] = trader_models[best_model]

                if trader_best_results:
                    aggregate_analysis = self.causal_analyzer.analyze_multiple_traders(trader_best_results)
                    causal_results['aggregate_analysis'] = aggregate_analysis

            except Exception as e:
                logger.error(f"Error in aggregate causal impact analysis: {e}")
                causal_results['aggregate_analysis'] = {'error': str(e)}

        else:
            # Global causal impact analysis
            logger.info("Calculating global causal impact")

            feature_cols = [col for col in holdout_data.columns
                           if col not in ['account_id', 'date', 'target_next_pnl']]

            X = holdout_data[feature_cols]
            y_actual = holdout_data['target_next_pnl'].values
            dates = pd.to_datetime(holdout_data['date']) if 'date' in holdout_data.columns else None

            global_causal_models = {}

            for model_type, model_result in final_models.get('global', {}).items():
                if 'error' in model_result:
                    global_causal_models[model_type] = {'error': model_result['error']}
                    continue

                try:
                    # Get predictions
                    model_key = f"global_final_{model_type}"
                    y_pred = self.trainer.predict(X, model_key)

                    # Calculate causal impact
                    causal_impact = self.causal_analyzer.calculate_trading_impact(
                        actual_pnl=y_actual,
                        predicted_pnl=y_pred,
                        dates=dates,
                        trader_id='global'
                    )

                    global_causal_models[model_type] = causal_impact

                except Exception as e:
                    logger.error(f"Error calculating global causal impact for {model_type}: {e}")
                    global_causal_models[model_type] = {'error': str(e)}

            causal_results['global'] = global_causal_models

        self.causal_impact_results = causal_results
        return causal_results

    def generate_causal_impact_reports(self, save_path: str = None) -> Dict[str, str]:
        """
        Generate comprehensive causal impact reports

        Args:
            save_path: Directory to save reports

        Returns:
            Dictionary mapping entity to report text
        """
        if not self.causal_impact_results:
            logger.warning("No causal impact results available. Run pipeline first.")
            return {}

        reports = {}

        if save_path:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)

        # Individual trader reports
        if 'individual_traders' in self.causal_impact_results:
            for trader, trader_models in self.causal_impact_results['individual_traders'].items():
                # Generate report for best model
                best_model = None
                best_score = -np.inf

                # Find best model from holdout results
                if hasattr(self, 'holdout_results') and trader in self.holdout_results:
                    for model_type, result in self.holdout_results[trader].items():
                        if 'error' not in result:
                            score = result.get('overall_score', 0)
                            if score > best_score:
                                best_score = score
                                best_model = model_type

                if best_model and best_model in trader_models and 'error' not in trader_models[best_model]:
                    causal_data = trader_models[best_model]
                    report = self.causal_analyzer.generate_impact_report(causal_data)
                    reports[f'trader_{trader}'] = report

                    if save_path:
                        report_file = save_path / f"causal_impact_trader_{trader}.txt"
                        with open(report_file, 'w') as f:
                            f.write(report)

        # Global report
        if 'global' in self.causal_impact_results:
            for model_type, causal_data in self.causal_impact_results['global'].items():
                if 'error' not in causal_data:
                    report = self.causal_analyzer.generate_impact_report(causal_data)
                    reports[f'global_{model_type}'] = report

                    if save_path:
                        report_file = save_path / f"causal_impact_global_{model_type}.txt"
                        with open(report_file, 'w') as f:
                            f.write(report)

        # Aggregate report
        if 'aggregate_analysis' in self.causal_impact_results:
            aggregate_data = self.causal_impact_results['aggregate_analysis']
            if 'error' not in aggregate_data:
                aggregate_report = self._generate_aggregate_report(aggregate_data)
                reports['aggregate'] = aggregate_report

                if save_path:
                    report_file = save_path / "causal_impact_aggregate.txt"
                    with open(report_file, 'w') as f:
                        f.write(aggregate_report)

        logger.info(f"Generated {len(reports)} causal impact reports")
        return reports

    def _generate_aggregate_report(self, aggregate_data: Dict[str, Any]) -> str:
        """
        Generate aggregate causal impact report across all traders
        """
        agg = aggregate_data['aggregate_metrics']

        report = f"""
AGGREGATE CAUSAL IMPACT ANALYSIS
================================

Portfolio Summary:
Total Traders: {agg['total_traders']}
Total Baseline PnL: ${agg['total_baseline_pnl']:,.2f}
Total Potential Improvement: ${agg['total_improvement']:,.2f}
Total Improvement %: {agg['total_improvement_pct']:.1f}%

Trader Success Metrics:
Traders Improved: {agg['traders_improved']} of {agg['total_traders']}
Traders Deteriorated: {agg['traders_deteriorated']} of {agg['total_traders']}
Success Rate: {agg['improvement_success_rate']:.1%}

Per-Trader Statistics:
Average Improvement: ${agg['avg_improvement_per_trader']:,.2f}
Median Improvement: ${agg['median_improvement_per_trader']:,.2f}
Average Improvement %: {agg['avg_improvement_pct']:.1f}%

Distribution Analysis:
25th Percentile: ${aggregate_data['distribution_stats']['improvement_percentiles']['25th']:,.2f}
50th Percentile: ${aggregate_data['distribution_stats']['improvement_percentiles']['50th']:,.2f}
75th Percentile: ${aggregate_data['distribution_stats']['improvement_percentiles']['75th']:,.2f}
90th Percentile: ${aggregate_data['distribution_stats']['improvement_percentiles']['90th']:,.2f}

Range: ${aggregate_data['distribution_stats']['improvement_range'][0]:,.2f} to ${aggregate_data['distribution_stats']['improvement_range'][1]:,.2f}
Standard Deviation: ${aggregate_data['distribution_stats']['improvement_std']:,.2f}

CONCLUSION:
{agg['improvement_success_rate']:.0%} of traders would benefit from following model predictions.
Portfolio-wide, the model shows potential for ${agg['total_improvement']:,.2f} ({agg['total_improvement_pct']:+.1f}%) improvement.
"""

        return report

    def save_results(self, save_path: str = None) -> str:
        """
        Save pipeline results

        Args:
            save_path: Directory to save results

        Returns:
            Path where results were saved
        """
        if save_path is None:
            save_path = self.config.RESULTS_PATH

        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save results as JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = save_path / f"pipeline_results_{timestamp}.json"

        # Convert numpy arrays to lists for JSON serialization
        serializable_results = self._make_json_serializable(self.results)

        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)

        # Save models
        if self.config.SAVE_MODELS:
            models_path = save_path / f"models_{timestamp}"
            self.trainer.save_models(str(models_path))

        logger.info(f"Pipeline results saved to {save_path}")
        return str(save_path)

    def _make_json_serializable(self, obj):
        """
        Make object JSON serializable
        """
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif pd.isna(obj):
            return None
        else:
            return obj

    def get_best_models(self) -> Dict[str, Any]:
        """
        Get the best performing models from the pipeline

        Returns:
            Dictionary of best models by trader/global
        """
        if 'holdout_results' not in self.results:
            return {}

        best_models = {}
        holdout_results = self.results['holdout_results']

        for entity, models in holdout_results.items():
            if not isinstance(models, dict):
                continue

            best_score = -np.inf
            best_model = None

            for model_type, result in models.items():
                if 'error' in result:
                    continue

                score = result.get('overall_score', 0)
                if score > best_score:
                    best_score = score
                    best_model = model_type

            if best_model:
                best_models[entity] = {
                    'model_type': best_model,
                    'score': best_score,
                    'result': models[best_model]
                }

        return best_models
