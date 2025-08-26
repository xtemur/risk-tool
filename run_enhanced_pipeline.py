#!/usr/bin/env python3
"""
Enhanced Risk Tool Pipeline with Fills-Based Features

This script runs the complete enhanced pipeline including:
1. Enhanced data processing with fills features
2. Enhanced feature engineering
3. Model retraining with new features
4. Causal impact evaluation with enhanced models
"""

import os
import sys
import yaml
import logging
from datetime import datetime
from pathlib import Path

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.enhanced_data_processing import create_enhanced_trader_day_panel, validate_enhanced_data_quality
from src.enhanced_feature_engineering import build_enhanced_features
from src.enhanced_trader_training import TraderSpecificModelTrainer
from src.enhanced_causal_impact_evaluation import EnhancedCausalImpactEvaluator

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def run_enhanced_data_processing(config):
    """Run enhanced data processing with fills features."""
    logger.info("="*50)
    logger.info("STEP 1: Enhanced Data Processing")
    logger.info("="*50)

    # Create enhanced panel data
    logger.info("Creating enhanced trader-day panel with fills features...")
    enhanced_panel = create_enhanced_trader_day_panel(config)

    # Validate enhanced data quality
    logger.info("Validating enhanced data quality...")
    validation_results = validate_enhanced_data_quality(enhanced_panel, config)

    # Save enhanced panel data
    enhanced_panel_path = "data/processed/enhanced_panel.parquet"
    os.makedirs(os.path.dirname(enhanced_panel_path), exist_ok=True)
    enhanced_panel.to_parquet(enhanced_panel_path, index=False)
    logger.info(f"Enhanced panel saved to: {enhanced_panel_path}")

    # Save validation results
    validation_path = "data/processed/enhanced_panel_validation.json"
    import json
    with open(validation_path, 'w') as f:
        json.dump(validation_results, f, indent=2, default=str)
    logger.info(f"Validation results saved to: {validation_path}")

    logger.info(f"Enhanced panel shape: {enhanced_panel.shape}")
    logger.info(f"Data quality issues: {validation_results['n_issues']}")
    logger.info(f"Data quality warnings: {validation_results['n_warnings']}")

    return enhanced_panel, validation_results


def run_enhanced_feature_engineering(enhanced_panel, config):
    """Run enhanced feature engineering."""
    logger.info("="*50)
    logger.info("STEP 2: Enhanced Feature Engineering")
    logger.info("="*50)

    # Build enhanced features
    logger.info("Building comprehensive feature set with fills data...")
    enhanced_features_df = build_enhanced_features(enhanced_panel, config)

    # Update config path for enhanced features
    enhanced_features_path = "data/processed/enhanced_features.parquet"
    config['paths']['processed_features'] = enhanced_features_path

    logger.info(f"Enhanced features shape: {enhanced_features_df.shape}")
    logger.info(f"Total features: {len(enhanced_features_df.columns)}")

    # Feature summary by category
    traditional_features = len([col for col in enhanced_features_df.columns if any(col.startswith(prefix) for prefix in
        ['ewm_', 'rolling_', 'avg_', 'win_rate', 'cumulative']) and not 'fills' in col])
    fills_features = len([col for col in enhanced_features_df.columns if 'fills' in col or 'order' in col])
    execution_features = len([col for col in enhanced_features_df.columns if any(keyword in col for keyword in
        ['impact', 'liquidity', 'aggressiveness', 'efficiency'])])
    cross_features = len([col for col in enhanced_features_df.columns if any(suffix in col for suffix in
        ['_ratio', '_trend', '_consistency', '_stress'])])

    logger.info(f"Feature breakdown:")
    logger.info(f"  Traditional features: {traditional_features}")
    logger.info(f"  Fills-based features: {fills_features}")
    logger.info(f"  Execution quality features: {execution_features}")
    logger.info(f"  Cross-feature interactions: {cross_features}")

    return enhanced_features_df


def run_enhanced_model_training(config):
    """Run model training with enhanced features."""
    logger.info("="*50)
    logger.info("STEP 3: Enhanced Model Training")
    logger.info("="*50)

    # Initialize trainer with enhanced features
    trainer = TraderSpecificModelTrainer(config_path='configs/main_config.yaml')

    # Train models for all active traders
    logger.info("Training enhanced models for all active traders...")
    results = trainer.train_all_traders()

    logger.info(f"Model training completed for {len(results)} traders")

    # Print summary of training results
    successful_models = [r for r in results.values() if r and r.get('model_trained', False)]
    logger.info(f"Successfully trained: {len(successful_models)} models")

    # Model performance summary
    if successful_models:
        var_scores = [r['validation_results']['var_model']['score'] for r in successful_models if 'validation_results' in r]
        loss_scores = [r['validation_results']['loss_model']['score'] for r in successful_models if 'validation_results' in r]

        if var_scores:
            logger.info(f"VaR model scores - Mean: {sum(var_scores)/len(var_scores):.4f}, Range: {min(var_scores):.4f}-{max(var_scores):.4f}")
        if loss_scores:
            logger.info(f"Loss model scores - Mean: {sum(loss_scores)/len(loss_scores):.4f}, Range: {min(loss_scores):.4f}-{max(loss_scores):.4f}")

    return results


def run_enhanced_causal_impact_evaluation():
    """Run causal impact evaluation with enhanced models."""
    logger.info("="*50)
    logger.info("STEP 4: Enhanced Causal Impact Evaluation")
    logger.info("="*50)

    # Initialize evaluator
    evaluator = EnhancedCausalImpactEvaluator()

    # Get active traders from config
    with open('configs/main_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    active_traders = config['active_traders']

    # Run evaluation for all traders
    evaluation_results = {}

    for trader_id in active_traders:
        logger.info(f"Evaluating enhanced model for trader {trader_id}...")
        try:
            # Load trader data and model
            test_data, model_data = evaluator.load_trader_data(str(trader_id))

            if test_data is not None and model_data is not None:
                # Generate predictions
                predictions_df = evaluator.generate_predictions(test_data, model_data)

                # Run causal impact analysis (if applicable)
                causal_results = evaluator.run_causal_impact_analysis(predictions_df, trader_id)

                evaluation_results[trader_id] = {
                    'test_data_shape': test_data.shape,
                    'predictions_shape': predictions_df.shape,
                    'mean_loss_probability': predictions_df['loss_probability'].mean(),
                    'mean_var_prediction': predictions_df['var_prediction'].mean(),
                    'causal_results': causal_results
                }

                logger.info(f"Trader {trader_id}: Mean loss prob = {predictions_df['loss_probability'].mean():.4f}")

            else:
                logger.warning(f"Could not load data for trader {trader_id}")
                evaluation_results[trader_id] = {'error': 'Could not load data'}

        except Exception as e:
            logger.error(f"Error evaluating trader {trader_id}: {str(e)}")
            evaluation_results[trader_id] = {'error': str(e)}

    # Save evaluation results
    results_path = f"reports/enhanced_evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    import json
    with open(results_path, 'w') as f:
        json.dump(evaluation_results, f, indent=2, default=str)

    logger.info(f"Enhanced evaluation results saved to: {results_path}")

    # Summary statistics
    successful_evaluations = [r for r in evaluation_results.values() if 'error' not in r]
    logger.info(f"Successfully evaluated: {len(successful_evaluations)}/{len(active_traders)} traders")

    if successful_evaluations:
        avg_loss_prob = sum(r['mean_loss_probability'] for r in successful_evaluations) / len(successful_evaluations)
        avg_var_pred = sum(r['mean_var_prediction'] for r in successful_evaluations) / len(successful_evaluations)
        logger.info(f"Average loss probability: {avg_loss_prob:.4f}")
        logger.info(f"Average VaR prediction: {avg_var_pred:.2f}")

    return evaluation_results


def main():
    """Run the complete enhanced pipeline."""
    start_time = datetime.now()

    logger.info("="*60)
    logger.info("ENHANCED RISK TOOL PIPELINE WITH FILLS FEATURES")
    logger.info("="*60)
    logger.info(f"Pipeline started at: {start_time}")

    try:
        # Load configuration
        with open('configs/main_config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        logger.info(f"Active traders: {config['active_traders']}")
        logger.info(f"Database path: {config['paths']['raw_data']}")

        # Step 1: Enhanced Data Processing
        enhanced_panel, validation_results = run_enhanced_data_processing(config)

        # Step 2: Enhanced Feature Engineering
        enhanced_features_df = run_enhanced_feature_engineering(enhanced_panel, config)

        # Step 3: Enhanced Model Training
        training_results = run_enhanced_model_training(config)

        # Step 4: Enhanced Causal Impact Evaluation
        evaluation_results = run_enhanced_causal_impact_evaluation()

        # Pipeline completion
        end_time = datetime.now()
        duration = end_time - start_time

        logger.info("="*60)
        logger.info("ENHANCED PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        logger.info(f"Total duration: {duration}")
        logger.info(f"Enhanced features shape: {enhanced_features_df.shape}")
        logger.info(f"Models trained: {len([r for r in training_results.values() if r and r.get('model_trained', False)])}")
        logger.info(f"Traders evaluated: {len([r for r in evaluation_results.values() if 'error' not in r])}")

        # Create summary report
        summary_report = {
            'pipeline_info': {
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': duration.total_seconds(),
                'enhanced_features_enabled': True
            },
            'data_processing': {
                'enhanced_panel_shape': enhanced_panel.shape,
                'validation_issues': validation_results['n_issues'],
                'validation_warnings': validation_results['n_warnings'],
                'fills_data_coverage': validation_results['data_summary'].get('fills_data_coverage', 0)
            },
            'feature_engineering': {
                'enhanced_features_shape': enhanced_features_df.shape,
                'traditional_features_count': len([col for col in enhanced_features_df.columns if any(col.startswith(prefix) for prefix in
                    ['ewm_', 'rolling_', 'avg_', 'win_rate', 'cumulative']) and not 'fills' in col]),
                'fills_features_count': len([col for col in enhanced_features_df.columns if 'fills' in col or 'order' in col]),
                'execution_features_count': len([col for col in enhanced_features_df.columns if any(keyword in col for keyword in
                    ['impact', 'liquidity', 'aggressiveness', 'efficiency'])])
            },
            'model_training': {
                'total_traders': len(config['active_traders']),
                'successful_models': len([r for r in training_results.values() if r and r.get('model_trained', False)])
            },
            'evaluation': {
                'total_evaluations': len(evaluation_results),
                'successful_evaluations': len([r for r in evaluation_results.values() if 'error' not in r])
            }
        }

        # Save summary report
        summary_path = f"reports/enhanced_pipeline_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary_report, f, indent=2, default=str)

        logger.info(f"Pipeline summary saved to: {summary_path}")

        return True

    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
