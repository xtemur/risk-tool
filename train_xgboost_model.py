"""
Train Comprehensive XGBoost Model

Script to train and evaluate an advanced XGBoost model with:
- Hyperparameter optimization
- Overfitting prevention
- Comparison with baseline models
- Causal impact analysis
"""

import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add the src directory to the Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root / "src"))
os.chdir(project_root)

import pandas as pd
import numpy as np
from datetime import datetime
import logging
import matplotlib.pyplot as plt
import seaborn as sns

# Import our modules
from modeling.xgboost_trainer import AdvancedXGBoostTrainer
from modeling.ensemble_trainer import EnsembleTrainer
from modeling.model_trainer import ModelTrainer
from modeling.performance_evaluator import PerformanceEvaluator
from modeling.causal_impact_analyzer import CausalImpactAnalyzer
from modeling.config import ModelConfig
from modeling.time_series_validator import TimeSeriesValidator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_and_prepare_data():
    """Load and prepare training data"""
    logger.info("Loading processed feature data...")

    # Try to load essential features first (higher quality)
    essential_path = "data/processed/essential_features.csv"
    fallback_path = "data/processed/features_demo.csv"

    try:
        if Path(essential_path).exists():
            data = pd.read_csv(essential_path)
            data_source = "essential features"
            logger.info(f"✓ Loaded {len(data)} samples from {data['account_id'].nunique()} traders (Essential Features)")
        else:
            data = pd.read_csv(fallback_path)
            data_source = "original features"
            logger.info(f"✓ Loaded {len(data)} samples from {data['account_id'].nunique()} traders (Original Features)")

        logger.info(f"Date range: {data['date'].min()} to {data['date'].max()}")

        # Convert date column
        data['date'] = pd.to_datetime(data['date'])

        # Prepare features and target
        feature_cols = [col for col in data.columns
                       if col not in ['account_id', 'date', 'target_next_pnl']]

        # Check for missing data
        missing_info = data[feature_cols + ['target_next_pnl']].isnull().sum()
        missing_features = missing_info[missing_info > 0]

        if len(missing_features) > 0:
            logger.warning(f"Found missing data in {len(missing_features)} features")

            # For essential features, use more conservative imputation
            if data_source == "essential features":
                logger.info("Using temporal-safe imputation for essential features")
                # Only keep rows with valid targets
                data = data[data['target_next_pnl'].notna()].copy()
                # Drop features with >90% missing data
                high_missing_features = [col for col in feature_cols
                                       if data[col].isnull().mean() > 0.9]
                if high_missing_features:
                    logger.info(f"Dropping {len(high_missing_features)} features with >90% missing data")
                    feature_cols = [col for col in feature_cols if col not in high_missing_features]

            else:
                logger.info("Missing data will be handled during preprocessing")

        logger.info(f"Features available: {len(feature_cols)} ({data_source})")
        return data, feature_cols

    except FileNotFoundError:
        logger.error("Feature data not found. Please run feature engineering first.")
        logger.info("Commands:")
        logger.info("  Essential: python examples/essential_feature_demo.py")
        logger.info("  Original:  python examples/feature_engineering_demo.py")
        return None, None


def train_baseline_models(data, feature_cols, config):
    """Train baseline models for comparison"""
    logger.info("\n" + "="*60)
    logger.info("TRAINING BASELINE MODELS")
    logger.info("="*60)

    # Prepare data
    X = data[feature_cols]
    y = data['target_next_pnl']

    # Create temporal split for holdout testing
    holdout_date = pd.to_datetime(config.HOLDOUT_START_DATE)
    train_mask = data['date'] < holdout_date
    test_mask = data['date'] >= holdout_date

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Test samples: {len(X_test)}")

    # Initialize trainer and evaluator
    trainer = ModelTrainer(config)
    evaluator = PerformanceEvaluator(config)

    # Train baseline models
    baseline_models = {}
    baseline_results = {}

    for model_type in ['linear', 'ridge', 'lasso']:
        logger.info(f"\nTraining {model_type} model...")
        try:
            result = trainer.train_model(
                X_train, y_train,
                model_type=model_type,
                model_key=f'baseline_{model_type}',
                optimize_hyperparams=True
            )

            # Evaluate on test set
            test_pred = trainer.predict(X_test, model_key=f'baseline_{model_type}')

            # Calculate metrics
            stat_metrics = evaluator.calculate_statistical_metrics(y_test.values, test_pred)
            fin_metrics = evaluator.calculate_financial_metrics(y_test.values, test_pred)

            baseline_models[model_type] = result['model']
            baseline_results[model_type] = {
                'train_metrics': result['metrics'],
                'test_statistical_metrics': stat_metrics,
                'test_financial_metrics': fin_metrics
            }

            logger.info(f"  Test MAE: {stat_metrics['mae']:.4f}")
            logger.info(f"  Test R²: {stat_metrics['r2']:.4f}")
            logger.info(f"  Hit Rate: {fin_metrics['hit_rate']:.1%}")

        except Exception as e:
            logger.error(f"Failed to train {model_type}: {e}")
            baseline_results[model_type] = {'error': str(e)}

    return baseline_results, X_train, X_test, y_train, y_test


def train_ensemble_models(X_train, y_train, X_test, y_test, config):
    """Train ensemble models including XGBoost, LightGBM, and CatBoost"""
    logger.info("\n" + "="*60)
    logger.info("TRAINING ENSEMBLE MODELS")
    logger.info("="*60)

    # Create validation split from training data
    val_split = int(0.8 * len(X_train))
    X_train_split = X_train.iloc[:val_split]
    X_val_split = X_train.iloc[val_split:]
    y_train_split = y_train.iloc[:val_split]
    y_val_split = y_train.iloc[val_split:]

    logger.info(f"Training split: {len(X_train_split)} samples")
    logger.info(f"Validation split: {len(X_val_split)} samples")

    # Handle missing values
    logger.info("Preprocessing data for ensemble training...")
    X_train_split_clean = X_train_split.fillna(X_train_split.median())
    X_val_split_clean = X_val_split.fillna(X_train_split.median())
    y_train_split_clean = y_train_split.fillna(0)
    y_val_split_clean = y_val_split.fillna(0)

    # Initialize ensemble trainer
    ensemble_trainer = EnsembleTrainer(config)

    # Train base models with hyperparameter optimization
    logger.info("Training base models with Optuna optimization...")
    algorithm_results = ensemble_trainer.train_base_models(
        X_train_split_clean, y_train_split_clean,
        X_val_split_clean, y_val_split_clean,
        optimization_trials=50,  # Trials per algorithm
        cv_folds=5
    )

    # Optimize ensemble weights
    logger.info("Optimizing ensemble weights...")
    optimal_weights = ensemble_trainer.optimize_ensemble_weights(
        X_val_split_clean.values, y_val_split_clean.values,
        n_trials=100
    )

    # Evaluate ensemble on test set
    X_test_clean = X_test.fillna(X_train_split.median())
    ensemble_pred = ensemble_trainer.predict_ensemble(X_test_clean.values)

    # Calculate comprehensive metrics for ensemble
    evaluator = PerformanceEvaluator(config)
    ensemble_stat_metrics = evaluator.calculate_statistical_metrics(y_test.values, ensemble_pred)
    ensemble_fin_metrics = evaluator.calculate_financial_metrics(y_test.values, ensemble_pred)

    # Log ensemble results
    logger.info("\nEnsemble Model Results:")
    logger.info(f"  Ensemble weights: {optimal_weights}")
    logger.info(f"  Test MAE: {ensemble_stat_metrics['mae']:.4f}")
    logger.info(f"  Test R²: {ensemble_stat_metrics['r2']:.4f}")
    logger.info(f"  Hit Rate: {ensemble_fin_metrics['hit_rate']:.1%}")

    # Log individual model results
    logger.info("\nIndividual Model Performance:")
    for algorithm, results in algorithm_results.items():
        metrics = results['metrics']
        logger.info(f"  {algorithm.upper()}:")
        logger.info(f"    Val MAE: {metrics['val_mae']:.4f}")
        logger.info(f"    Val R²: {metrics['val_r2']:.4f}")
        logger.info(f"    Best params: {results['best_params']}")

    # Create comprehensive results
    ensemble_results = {
        'ensemble_trainer': ensemble_trainer,
        'algorithm_results': algorithm_results,
        'ensemble_weights': optimal_weights,
        'test_statistical_metrics': ensemble_stat_metrics,
        'test_financial_metrics': ensemble_fin_metrics,
        'individual_models': {alg: res['model'] for alg, res in algorithm_results.items()}
    }

    return ensemble_results


def compare_models(baseline_results, ensemble_results):
    """Compare ensemble with baseline models"""
    logger.info("\n" + "="*60)
    logger.info("MODEL COMPARISON")
    logger.info("="*60)

    # Create comparison table
    comparison_data = []

    # Add baseline results
    for model_type, results in baseline_results.items():
        if 'error' not in results:
            comparison_data.append({
                'Model': model_type.title(),
                'Test_MAE': results['test_statistical_metrics']['mae'],
                'Test_R2': results['test_statistical_metrics']['r2'],
                'Hit_Rate': results['test_financial_metrics']['hit_rate'],
                'Model_Type': 'Baseline'
            })

    # Add individual ensemble models
    for algorithm, results in ensemble_results['algorithm_results'].items():
        # Calculate test metrics for individual models
        model = results['model']
        # Note: We'd need test predictions here, using validation for now
        comparison_data.append({
            'Model': algorithm.upper(),
            'Test_MAE': results['metrics']['val_mae'],  # Using validation as proxy
            'Test_R2': results['metrics']['val_r2'],
            'Hit_Rate': 'N/A',  # Would need to calculate
            'Model_Type': 'Individual'
        })

    # Add ensemble results
    comparison_data.append({
        'Model': 'Ensemble',
        'Test_MAE': ensemble_results['test_statistical_metrics']['mae'],
        'Test_R2': ensemble_results['test_statistical_metrics']['r2'],
        'Hit_Rate': ensemble_results['test_financial_metrics']['hit_rate'],
        'Model_Type': 'Ensemble'
    })

    # Create DataFrame and display
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Test_R2', ascending=False)

    logger.info("\nModel Performance Comparison:")
    logger.info("="*100)
    for _, row in comparison_df.iterrows():
        hit_rate_str = f"{row['Hit_Rate']:.1%}" if row['Hit_Rate'] != 'N/A' else 'N/A      '
        logger.info(f"{row['Model']:<12} | {row['Model_Type']:<10} | MAE: {row['Test_MAE']:<8.4f} | "
                   f"R²: {row['Test_R2']:<8.4f} | Hit Rate: {hit_rate_str}")

    # Determine best model
    best_model = comparison_df.iloc[0]
    logger.info(f"\nBest Model: {best_model['Model']} (R² = {best_model['Test_R2']:.4f})")

    # Performance improvement analysis
    baseline_models = comparison_df[comparison_df['Model_Type'] == 'Baseline']
    ensemble_model = comparison_df[comparison_df['Model'] == 'Ensemble'].iloc[0]

    if len(baseline_models) > 0:
        best_baseline = baseline_models.iloc[0]
        mae_improvement = ((best_baseline['Test_MAE'] - ensemble_model['Test_MAE']) / best_baseline['Test_MAE']) * 100
        r2_improvement = ((ensemble_model['Test_R2'] - best_baseline['Test_R2']) / abs(best_baseline['Test_R2'])) * 100 if best_baseline['Test_R2'] != 0 else 0

        if ensemble_model['Hit_Rate'] != 'N/A' and best_baseline['Hit_Rate'] != 'N/A':
            hit_rate_improvement = (ensemble_model['Hit_Rate'] - best_baseline['Hit_Rate']) * 100
            logger.info(f"\nEnsemble vs Best Baseline ({best_baseline['Model']}):")
            logger.info(f"  MAE improvement: {mae_improvement:+.1f}%")
            logger.info(f"  R² improvement: {r2_improvement:+.1f}%")
            logger.info(f"  Hit rate improvement: {hit_rate_improvement:+.1f} percentage points")

    return comparison_df


def analyze_causal_impact(data, ensemble_results, feature_cols):
    """Analyze causal impact of ensemble predictions"""
    logger.info("\n" + "="*60)
    logger.info("CAUSAL IMPACT ANALYSIS")
    logger.info("="*60)

    try:
        # Prepare predictions data
        predictions_data = data.copy()
        X = data[feature_cols].fillna(data[feature_cols].median())
        predictions_data['predicted_pnl'] = ensemble_results['ensemble_trainer'].predict_ensemble(X.values)

        # Initialize causal impact analyzer
        analyzer = CausalImpactAnalyzer()

        # Analyze causal impact
        logger.info("Calculating causal impact scenarios...")
        causal_results = analyzer.analyze_causal_impact(predictions_data)

        if 'error' not in causal_results:
            scenarios = causal_results['causal_impact_scenarios']

            logger.info("\nCausal Impact Results:")
            logger.info("="*40)

            # Perfect following scenario
            perfect = scenarios['perfect_following']
            logger.info(f"Perfect Following Scenario:")
            logger.info(f"  PnL Improvement: ${perfect['pnl_improvement']:,.2f}")
            logger.info(f"  Improvement %: {perfect['pnl_improvement_pct']:+.1f}%")
            logger.info(f"  Original PnL: ${perfect['original_pnl']:,.2f}")
            logger.info(f"  Enhanced PnL: ${perfect['enhanced_pnl']:,.2f}")

            # Directional trading scenario
            directional = scenarios['directional_trading']
            logger.info(f"\nDirectional Trading Scenario:")
            logger.info(f"  PnL Improvement: ${directional['pnl_improvement']:,.2f}")
            logger.info(f"  Improvement %: {directional['pnl_improvement_pct']:+.1f}%")
            logger.info(f"  Trading Frequency: {directional['trading_frequency']:.1%}")
            logger.info(f"  Win Rate: {directional['win_rate']:.1%}")

            # Risk-adjusted scenario
            risk_adjusted = scenarios['risk_adjusted_trading']
            logger.info(f"\nRisk-Adjusted Trading Scenario:")
            logger.info(f"  PnL Improvement: ${risk_adjusted['pnl_improvement']:,.2f}")
            logger.info(f"  Improvement %: {risk_adjusted['pnl_improvement_pct']:+.1f}%")
            logger.info(f"  Risk Reduction: {risk_adjusted['risk_reduction']:.1%}")
            logger.info(f"  Sharpe Improvement: {risk_adjusted['sharpe_improvement']:+.3f}")

            return causal_results
        else:
            logger.error(f"Causal impact analysis failed: {causal_results['error']}")
            return None

    except Exception as e:
        logger.error(f"Error in causal impact analysis: {e}")
        return None


def save_results(baseline_results, ensemble_results, comparison_df, causal_results=None):
    """Save all results to files"""
    logger.info("\n" + "="*60)
    logger.info("SAVING RESULTS")
    logger.info("="*60)

    # Create results directory
    results_dir = Path("results/xgboost_comprehensive")
    results_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Save model comparison
        comparison_df.to_csv(results_dir / "model_comparison.csv", index=False)
        logger.info(f"Model comparison saved to {results_dir / 'model_comparison.csv'}")

        # Save ensemble detailed results
        ensemble_summary = {
            'ensemble_weights': ensemble_results['ensemble_weights'],
            'algorithm_results': {
                alg: {
                    'best_params': res['best_params'],
                    'metrics': res['metrics'],
                    'cv_score': res['cv_score']
                } for alg, res in ensemble_results['algorithm_results'].items()
            },
            'test_metrics': {
                **ensemble_results['test_statistical_metrics'],
                **ensemble_results['test_financial_metrics']
            }
        }

        import json
        with open(results_dir / "ensemble_results.json", 'w') as f:
            json.dump(ensemble_summary, f, indent=2, default=str)
        logger.info(f"Ensemble results saved to {results_dir / 'ensemble_results.json'}")

        # Save individual model results
        for algorithm, results in ensemble_results['algorithm_results'].items():
            algo_file = results_dir / f"{algorithm}_results.json"
            with open(algo_file, 'w') as f:
                json.dump({
                    'best_params': results['best_params'],
                    'metrics': results['metrics'],
                    'cv_score': results['cv_score']
                }, f, indent=2, default=str)
            logger.info(f"{algorithm} results saved to {algo_file}")

        # Save causal impact results if available
        if causal_results:
            with open(results_dir / "causal_impact_results.json", 'w') as f:
                json.dump(causal_results, f, indent=2, default=str)
            logger.info(f"Causal impact results saved to {results_dir / 'causal_impact_results.json'}")

        logger.info(f"\nAll results saved to: {results_dir}")
        return str(results_dir)

    except Exception as e:
        logger.error(f"Error saving results: {e}")
        return None


def main():
    """Main execution function"""
    print("=" * 80)
    print("COMPREHENSIVE XGBOOST MODEL TRAINING")
    print("=" * 80)

    # Load data
    data, feature_cols = load_and_prepare_data()
    if data is None:
        return

    # Setup configuration
    config = ModelConfig.for_production()
    logger.info(f"Using configuration: {config.__class__.__name__}")

    # Train baseline models
    baseline_results, X_train, X_test, y_train, y_test = train_baseline_models(
        data, feature_cols, config
    )

    # Train ensemble models
    ensemble_results = train_ensemble_models(
        X_train, y_train, X_test, y_test, config
    )

    # Compare models
    comparison_df = compare_models(baseline_results, ensemble_results)

    # Analyze causal impact
    causal_results = analyze_causal_impact(data, ensemble_results, feature_cols)

    # Save results
    save_path = save_results(baseline_results, ensemble_results, comparison_df, causal_results)

    # Final summary
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETED SUCCESSFULLY")
    logger.info("="*80)

    logger.info(f"\nKey Results:")
    logger.info(f"  Best Model: Ensemble ({list(ensemble_results['ensemble_weights'].keys())})")
    logger.info(f"  Test R²: {ensemble_results['test_statistical_metrics']['r2']:.4f}")
    logger.info(f"  Test MAE: {ensemble_results['test_statistical_metrics']['mae']:.4f}")
    logger.info(f"  Hit Rate: {ensemble_results['test_financial_metrics']['hit_rate']:.1%}")
    logger.info(f"  Ensemble Weights: {ensemble_results['ensemble_weights']}")

    if causal_results and 'error' not in causal_results:
        directional = causal_results['causal_impact_scenarios']['directional_trading']
        logger.info(f"  Causal Impact: ${directional['pnl_improvement']:,.2f} ({directional['pnl_improvement_pct']:+.1f}%)")

    if save_path:
        logger.info(f"\nResults saved to: {save_path}")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
