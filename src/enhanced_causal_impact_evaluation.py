#!/usr/bin/env python3
"""
Enhanced Causal Impact Evaluation for Enhanced Risk Models

This module provides evaluation capabilities for the enhanced models that include fills-based features.
It extends the original causal impact evaluation with enhanced model compatibility and feature analysis.
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, Tuple, Optional, List

logger = logging.getLogger(__name__)


class EnhancedCausalImpactEvaluator:
    """
    Enhanced evaluator for models with fills-based features and improved risk prediction.
    """

    def __init__(self):
        self.model_dir = Path('models/trader_specific')
        self.features_path = 'data/processed/enhanced_features.parquet'

    def load_enhanced_features(self) -> pd.DataFrame:
        """Load enhanced features dataset."""
        try:
            if Path(self.features_path).exists():
                df = pd.read_parquet(self.features_path)
                logger.info(f"Loaded enhanced features: {df.shape}")
                return df
            else:
                logger.error(f"Enhanced features not found at {self.features_path}")
                return None
        except Exception as e:
            logger.error(f"Error loading enhanced features: {e}")
            return None

    def load_enhanced_trader_model(self, trader_id: str) -> Tuple[Optional[object], Optional[object], Optional[Dict]]:
        """Load enhanced models and metadata for a trader."""
        trader_dir = self.model_dir / str(trader_id)

        if not trader_dir.exists():
            logger.warning(f"No model directory found for trader {trader_id}")
            return None, None, None

        # Try to load enhanced models first
        enhanced_loss_model_path = trader_dir / 'enhanced_loss_model.pkl'
        enhanced_var_model_path = trader_dir / 'enhanced_var_model.pkl'
        enhanced_metadata_path = trader_dir / 'enhanced_model_metadata.json'

        loss_model = None
        var_model = None
        metadata = None

        # Load enhanced models if available
        if enhanced_loss_model_path.exists() and enhanced_var_model_path.exists():
            try:
                with open(enhanced_loss_model_path, 'rb') as f:
                    loss_model = pickle.load(f)

                with open(enhanced_var_model_path, 'rb') as f:
                    var_model = pickle.load(f)

                if enhanced_metadata_path.exists():
                    with open(enhanced_metadata_path, 'r') as f:
                        metadata = json.load(f)

                logger.info(f"Loaded enhanced models for trader {trader_id}")
                return loss_model, var_model, metadata

            except Exception as e:
                logger.warning(f"Error loading enhanced models for trader {trader_id}: {e}")

        # Fallback to original models if enhanced models not available
        fallback_paths = [
            trader_dir / 'loss_model.pkl',
            trader_dir / 'var_model.pkl',
            trader_dir / 'model_metadata.json'
        ]

        if all(p.exists() for p in fallback_paths[:2]):
            try:
                with open(fallback_paths[0], 'rb') as f:
                    loss_model = pickle.load(f)

                with open(fallback_paths[1], 'rb') as f:
                    var_model = pickle.load(f)

                if fallback_paths[2].exists():
                    with open(fallback_paths[2], 'r') as f:
                        metadata = json.load(f)

                logger.info(f"Loaded fallback models for trader {trader_id}")
                return loss_model, var_model, metadata

            except Exception as e:
                logger.error(f"Error loading fallback models for trader {trader_id}: {e}")

        return None, None, None

    def prepare_trader_data(self, df: pd.DataFrame, trader_id: str) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """Prepare trader data for enhanced model evaluation."""
        trader_data = df[df['account_id'] == int(trader_id)].copy()

        if len(trader_data) == 0:
            logger.warning(f"No data found for trader {trader_id}")
            return None, None, None

        # Sort by date
        trader_data = trader_data.sort_values('trade_date').reset_index(drop=True)

        # Remove rows with NaN targets
        trader_data = trader_data.dropna(subset=['target_pnl', 'target_large_loss'])

        if len(trader_data) < 100:
            logger.warning(f"Insufficient data for trader {trader_id}: {len(trader_data)} rows")
            return None, None, None

        # Split into train/test (same as training)
        split_idx = int(len(trader_data) * 0.8)
        train_data = trader_data.iloc[:split_idx].copy()
        test_data = trader_data.iloc[split_idx:].copy()

        # Get feature columns (exclude metadata and targets)
        feature_cols = [col for col in trader_data.columns
                       if col not in ['account_id', 'trade_date', 'target_pnl', 'target_large_loss',
                                      'target_high_impact', 'target_high_cost']]

        logger.info(f"Prepared trader {trader_id}: Train={len(train_data)}, Test={len(test_data)}, Features={len(feature_cols)}")

        return train_data, test_data, feature_cols

    def generate_enhanced_predictions(self, test_data: pd.DataFrame, loss_model: object,
                                    var_model: object, feature_cols: List[str],
                                    metadata: Dict) -> pd.DataFrame:
        """Generate predictions using enhanced models."""

        # Get selected features from metadata if available
        selected_features = feature_cols
        if metadata and 'selected_features' in metadata:
            selected_features = metadata['selected_features']
            logger.info(f"Using {len(selected_features)} selected features from model metadata")

        # Prepare features
        X_test = test_data[selected_features].copy()

        # Handle missing values and infinite values (same as training)
        for col in selected_features:
            if col in X_test.columns:
                # Skip non-numeric columns
                if X_test[col].dtype not in ['int64', 'float64', 'bool']:
                    continue

                # Convert boolean to numeric
                if X_test[col].dtype == 'bool':
                    X_test[col] = X_test[col].astype(int)

                # Replace inf and -inf with NaN first
                X_test[col] = X_test[col].replace([np.inf, -np.inf], np.nan)

                # Fill NaN values with median or 0
                if X_test[col].median() is not None and not pd.isna(X_test[col].median()):
                    X_test[col] = X_test[col].fillna(X_test[col].median())
                else:
                    X_test[col] = X_test[col].fillna(0)

                # Clip extreme values
                if X_test[col].dtype in ['int64', 'float64'] and X_test[col].std() > 0:
                    try:
                        lower_bound = X_test[col].quantile(0.01)
                        upper_bound = X_test[col].quantile(0.99)
                        if not pd.isna(lower_bound) and not pd.isna(upper_bound):
                            X_test[col] = X_test[col].clip(lower_bound, upper_bound)
                    except:
                        X_test[col] = X_test[col].clip(-1e6, 1e6)

        # Generate predictions
        try:
            # Classification predictions (high risk probability - position size < 0.7)
            risk_probs = loss_model.predict_proba(X_test)[:, 1]

            # Regression predictions (optimal position size 0.0 to 1.5)
            position_size_predictions = var_model.predict(X_test)

            # Clip position size predictions to valid range
            position_size_predictions = np.clip(position_size_predictions, 0.0, 1.5)

            # Create results dataframe
            results_df = pd.DataFrame({
                'date': test_data['trade_date'].values,
                'actual_pnl': test_data['target_pnl'].values,
                'risk_probability': risk_probs,
                'predicted_position_size': position_size_predictions
            })

            # Add legacy columns for backward compatibility
            if 'target_large_loss' in test_data.columns:
                results_df['actual_large_loss'] = test_data['target_large_loss'].values
                results_df['loss_probability'] = risk_probs  # Alias for risk_probability

            # Add position sizing specific columns
            if 'target_position_size' in test_data.columns:
                results_df['actual_position_size'] = test_data['target_position_size'].values

                # Calculate position sizing accuracy metrics
                results_df['position_size_error'] = np.abs(
                    results_df['predicted_position_size'] - results_df['actual_position_size']
                )

                # Categorize predictions
                results_df['predicted_category'] = pd.cut(
                    results_df['predicted_position_size'],
                    bins=[0, 0.5, 0.8, 1.2, 1.5],
                    labels=['Reduce', 'Conservative', 'Normal', 'Aggressive']
                )

                results_df['actual_category'] = pd.cut(
                    results_df['actual_position_size'],
                    bins=[0, 0.5, 0.8, 1.2, 1.5],
                    labels=['Reduce', 'Conservative', 'Normal', 'Aggressive']
                )

            # Backward compatibility - add var_prediction as alias for position_size
            results_df['var_prediction'] = position_size_predictions

            logger.info(f"Generated {len(results_df)} position sizing predictions")
            logger.info(f"  Position size range: {position_size_predictions.min():.3f} to {position_size_predictions.max():.3f}")
            logger.info(f"  Mean position size: {position_size_predictions.mean():.3f}")

            return results_df

        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
            return None

    def evaluate_enhanced_trader(self, trader_id: str) -> Dict:
        """Comprehensive evaluation of enhanced model for a trader."""
        logger.info(f"Evaluating enhanced model for trader {trader_id}...")

        try:
            # Load enhanced features
            df = self.load_enhanced_features()
            if df is None:
                return {'error': 'Could not load enhanced features'}

            # Load models
            loss_model, var_model, metadata = self.load_enhanced_trader_model(trader_id)
            if loss_model is None or var_model is None:
                return {'error': 'Could not load models'}

            # Prepare data
            train_data, test_data, feature_cols = self.prepare_trader_data(df, trader_id)
            if test_data is None:
                return {'error': 'Could not prepare trader data'}

            # Generate predictions
            predictions_df = self.generate_enhanced_predictions(
                test_data, loss_model, var_model, feature_cols, metadata
            )

            if predictions_df is None:
                return {'error': 'Could not generate predictions'}

            # Calculate evaluation metrics
            evaluation_metrics = self.calculate_evaluation_metrics(predictions_df)

            # Feature importance analysis
            feature_analysis = self.analyze_feature_importance(metadata)

            # Enhanced model insights
            model_insights = self.extract_model_insights(metadata, predictions_df)

            # Combine results
            results = {
                'trader_id': trader_id,
                'evaluation_timestamp': datetime.now().isoformat(),
                'enhanced_model': metadata.get('enhanced_features', False),
                'test_period': [
                    predictions_df['date'].min().strftime('%Y-%m-%d'),
                    predictions_df['date'].max().strftime('%Y-%m-%d')
                ],
                'predictions_count': len(predictions_df),
                'evaluation_metrics': evaluation_metrics,
                'feature_analysis': feature_analysis,
                'model_insights': model_insights,
                'predictions_sample': predictions_df.head(10).to_dict('records')
            }

            logger.info(f"Enhanced evaluation complete for trader {trader_id}")
            return results

        except Exception as e:
            logger.error(f"Error evaluating trader {trader_id}: {str(e)}")
            return {'error': str(e)}

    def calculate_evaluation_metrics(self, predictions_df: pd.DataFrame) -> Dict:
        """Calculate comprehensive evaluation metrics for position sizing predictions."""
        metrics = {}

        try:
            from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

            # ENHANCED: Position Sizing Metrics (Primary)
            if 'actual_position_size' in predictions_df.columns and 'predicted_position_size' in predictions_df.columns:
                actual_pos = predictions_df['actual_position_size'].dropna()
                predicted_pos = predictions_df['predicted_position_size'].loc[actual_pos.index]

                if len(actual_pos) > 0:
                    position_mae = mean_absolute_error(actual_pos, predicted_pos)
                    position_rmse = np.sqrt(mean_squared_error(actual_pos, predicted_pos))
                    position_r2 = r2_score(actual_pos, predicted_pos)

                    # Position-specific metrics
                    metrics['position_sizing'] = {
                        'mae': float(position_mae),
                        'rmse': float(position_rmse),
                        'r2': float(position_r2),
                        'mean_error': float(np.mean(predicted_pos - actual_pos)),
                        'std_error': float(np.std(predicted_pos - actual_pos)),
                        'mean_abs_error_pct': float(position_mae / actual_pos.mean() * 100) if actual_pos.mean() > 0 else 0,
                    }

                    # Directional accuracy (getting the risk level right)
                    reduce_actual = (actual_pos < 0.7)
                    reduce_predicted = (predicted_pos < 0.7)
                    aggressive_actual = (actual_pos > 1.2)
                    aggressive_predicted = (predicted_pos > 1.2)

                    metrics['position_sizing']['directional_accuracy'] = {
                        'reduce_precision': float(precision_score(reduce_actual, reduce_predicted, zero_division=0)),
                        'reduce_recall': float(recall_score(reduce_actual, reduce_predicted, zero_division=0)),
                        'aggressive_precision': float(precision_score(aggressive_actual, aggressive_predicted, zero_division=0)),
                        'aggressive_recall': float(recall_score(aggressive_actual, aggressive_predicted, zero_division=0)),
                    }

                    # Economic value metrics
                    if 'actual_pnl' in predictions_df.columns:
                        # Simulated performance with predicted position sizing
                        simulated_pnl = predictions_df['actual_pnl'] * predicted_pos
                        optimal_pnl = predictions_df['actual_pnl'] * actual_pos
                        baseline_pnl = predictions_df['actual_pnl']  # 100% position size

                        metrics['position_sizing']['economic_value'] = {
                            'simulated_total_pnl': float(simulated_pnl.sum()),
                            'optimal_total_pnl': float(optimal_pnl.sum()),
                            'baseline_total_pnl': float(baseline_pnl.sum()),
                            'value_capture_ratio': float((simulated_pnl.sum() / optimal_pnl.sum()) if optimal_pnl.sum() != 0 else 0),
                            'vs_baseline_improvement': float((simulated_pnl.sum() - baseline_pnl.sum()) / abs(baseline_pnl.sum()) * 100) if baseline_pnl.sum() != 0 else 0
                        }

            # Risk Classification Metrics (Secondary)
            if 'risk_probability' in predictions_df.columns:
                # Use risk probability for classification if available
                risk_prob_col = 'risk_probability'
            elif 'loss_probability' in predictions_df.columns:
                # Fallback to legacy column
                risk_prob_col = 'loss_probability'
            else:
                risk_prob_col = None

            if risk_prob_col and 'actual_large_loss' in predictions_df.columns:
                actual_risk = predictions_df['actual_large_loss'].dropna()
                risk_probs = predictions_df[risk_prob_col].loc[actual_risk.index]

                if len(np.unique(actual_risk)) > 1 and len(risk_probs) > 0:
                    metrics['risk_classification'] = {
                        'auc': float(roc_auc_score(actual_risk, risk_probs)),
                        'accuracy': float(accuracy_score(actual_risk, risk_probs > 0.5)),
                        'precision': float(precision_score(actual_risk, risk_probs > 0.5, zero_division=0)),
                        'recall': float(recall_score(actual_risk, risk_probs > 0.5, zero_division=0))
                    }
                else:
                    metrics['risk_classification'] = {'note': 'Insufficient class diversity for risk classification'}

            # Legacy regression metrics (for backward compatibility)
            if 'actual_pnl' in predictions_df.columns and 'var_prediction' in predictions_df.columns:
                actual_pnl = predictions_df['actual_pnl'].dropna()
                var_pred = predictions_df['var_prediction'].loc[actual_pnl.index]

                metrics['regression'] = {
                    'rmse': float(np.sqrt(mean_squared_error(actual_pnl, var_pred))),
                    'mae': float(mean_absolute_error(actual_pnl, var_pred)),
                    'r2': float(r2_score(actual_pnl, var_pred))
                }

            # Legacy classification metrics (for backward compatibility)
            if 'actual_large_loss' in predictions_df.columns and 'loss_probability' in predictions_df.columns:
                actual_loss = predictions_df['actual_large_loss'].dropna()
                loss_probs = predictions_df['loss_probability'].loc[actual_loss.index]

                if len(np.unique(actual_loss)) > 1:
                    metrics['classification'] = {
                        'auc': float(roc_auc_score(actual_loss, loss_probs)),
                        'accuracy': float(accuracy_score(actual_loss, loss_probs > 0.5)),
                        'precision': float(precision_score(actual_loss, loss_probs > 0.5, zero_division=0)),
                        'recall': float(recall_score(actual_loss, loss_probs > 0.5, zero_division=0))
                    }
                else:
                    metrics['classification'] = {'note': 'Insufficient class diversity for classification metrics'}

            # Risk metrics
            actual_losses = predictions_df[predictions_df['actual_large_loss'] == 1]
            if len(actual_losses) > 0:
                metrics['risk_metrics'] = {
                    'large_loss_rate': float(predictions_df['actual_large_loss'].mean()),
                    'avg_loss_probability_on_loss_days': float(actual_losses['loss_probability'].mean()),
                    'avg_loss_probability_on_normal_days': float(predictions_df[predictions_df['actual_large_loss'] == 0]['loss_probability'].mean()),
                    'var_accuracy_on_loss_days': float(actual_losses['var_prediction'].mean()),
                }

            # Prediction distribution
            metrics['prediction_distribution'] = {
                'loss_prob_mean': float(predictions_df['loss_probability'].mean()),
                'loss_prob_std': float(predictions_df['loss_probability'].std()),
                'var_pred_mean': float(predictions_df['var_prediction'].mean()),
                'var_pred_std': float(predictions_df['var_prediction'].std())
            }

        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            metrics['error'] = str(e)

        return metrics

    def analyze_feature_importance(self, metadata: Dict) -> Dict:
        """Analyze feature importance from model metadata."""
        feature_analysis = {}

        try:
            if metadata and 'models' in metadata:
                # Loss model feature importance
                if 'loss_model' in metadata['models']:
                    loss_importance = metadata['models']['loss_model'].get('importance_by_category', {})
                    feature_analysis['loss_model_importance'] = dict(
                        sorted(loss_importance.items(), key=lambda x: x[1], reverse=True)
                    )

                # VaR model feature importance
                if 'var_model' in metadata['models']:
                    var_importance = metadata['models']['var_model'].get('importance_by_category', {})
                    feature_analysis['var_model_importance'] = dict(
                        sorted(var_importance.items(), key=lambda x: x[1], reverse=True)
                    )

                # Feature categories
                if 'feature_categories' in metadata:
                    feature_analysis['feature_categories'] = metadata['feature_categories']

                    # Count features by category
                    category_counts = {cat: len(features) for cat, features in metadata['feature_categories'].items()}
                    feature_analysis['features_by_category'] = category_counts

        except Exception as e:
            logger.error(f"Error analyzing feature importance: {e}")
            feature_analysis['error'] = str(e)

        return feature_analysis

    def extract_model_insights(self, metadata: Dict, predictions_df: pd.DataFrame) -> Dict:
        """Extract insights from enhanced model performance."""
        insights = {}

        try:
            # Model type and configuration
            insights['model_type'] = 'enhanced' if metadata.get('enhanced_features', False) else 'traditional'
            insights['feature_count'] = metadata.get('feature_count', 0)
            insights['training_timestamp'] = metadata.get('training_timestamp', 'unknown')

            # Performance insights
            if 'models' in metadata:
                loss_model_info = metadata['models'].get('loss_model', {})
                var_model_info = metadata['models'].get('var_model', {})

                insights['training_performance'] = {
                    'loss_model_auc': loss_model_info.get('test_auc', 0),
                    'var_model_rmse': var_model_info.get('test_rmse', 0)
                }

            # Prediction characteristics
            high_risk_days = predictions_df[predictions_df['loss_probability'] > 0.7]
            insights['prediction_insights'] = {
                'high_risk_days_count': len(high_risk_days),
                'high_risk_days_rate': len(high_risk_days) / len(predictions_df),
                'avg_var_on_high_risk_days': float(high_risk_days['var_prediction'].mean()) if len(high_risk_days) > 0 else 0,
                'actual_losses_on_high_risk_days': int(high_risk_days['actual_large_loss'].sum()) if len(high_risk_days) > 0 else 0
            }

            # Enhanced features impact (if available)
            if metadata.get('enhanced_features', False) and 'feature_analysis' in metadata:
                fills_importance = 0
                execution_importance = 0

                if 'models' in metadata and 'loss_model' in metadata['models']:
                    importance_by_cat = metadata['models']['loss_model'].get('importance_by_category', {})
                    fills_importance = importance_by_cat.get('fills_based', 0)
                    execution_importance = importance_by_cat.get('execution_quality', 0)

                insights['enhanced_features_impact'] = {
                    'fills_based_importance': fills_importance,
                    'execution_quality_importance': execution_importance,
                    'enhanced_features_total': fills_importance + execution_importance
                }

        except Exception as e:
            logger.error(f"Error extracting insights: {e}")
            insights['error'] = str(e)

        return insights

    def evaluate_all_enhanced_traders(self, trader_ids: Optional[List[str]] = None) -> Dict:
        """Evaluate enhanced models for all or specified traders."""
        if trader_ids is None:
            # Get all traders with enhanced models
            trader_ids = []
            for trader_dir in self.model_dir.iterdir():
                if trader_dir.is_dir() and (trader_dir / 'enhanced_loss_model.pkl').exists():
                    trader_ids.append(trader_dir.name)

            if not trader_ids:
                # Fallback to all traders with any models
                trader_ids = [d.name for d in self.model_dir.iterdir() if d.is_dir()]

        logger.info(f"Evaluating {len(trader_ids)} traders: {trader_ids}")

        results = {}
        successful_evaluations = 0

        for trader_id in trader_ids:
            try:
                result = self.evaluate_enhanced_trader(trader_id)
                results[trader_id] = result

                if 'error' not in result:
                    successful_evaluations += 1
                    logger.info(f"✅ Trader {trader_id}: AUC={result['evaluation_metrics'].get('classification', {}).get('auc', 'N/A'):.4f}, RMSE={result['evaluation_metrics'].get('regression', {}).get('rmse', 'N/A'):.1f}")
                else:
                    logger.warning(f"❌ Trader {trader_id}: {result['error']}")

            except Exception as e:
                logger.error(f"Error evaluating trader {trader_id}: {e}")
                results[trader_id] = {'error': str(e)}

        # Summary statistics
        summary = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'total_traders': len(trader_ids),
            'successful_evaluations': successful_evaluations,
            'failed_evaluations': len(trader_ids) - successful_evaluations,
            'individual_results': results
        }

        # Aggregate metrics for successful evaluations
        if successful_evaluations > 0:
            successful_results = [r for r in results.values() if 'error' not in r]

            # Classification metrics
            aucs = [r['evaluation_metrics']['classification']['auc']
                   for r in successful_results
                   if 'classification' in r['evaluation_metrics']
                   and 'auc' in r['evaluation_metrics']['classification']]

            # Regression metrics
            rmses = [r['evaluation_metrics']['regression']['rmse']
                    for r in successful_results
                    if 'regression' in r['evaluation_metrics']]

            if aucs:
                summary['aggregate_metrics'] = {
                    'mean_auc': sum(aucs) / len(aucs),
                    'mean_rmse': sum(rmses) / len(rmses) if rmses else 0,
                    'auc_range': [min(aucs), max(aucs)],
                    'rmse_range': [min(rmses), max(rmses)] if rmses else [0, 0]
                }

            # Enhanced features analysis
            enhanced_traders = [r for r in successful_results if r.get('enhanced_model', False)]
            if enhanced_traders:
                summary['enhanced_features_analysis'] = {
                    'enhanced_traders_count': len(enhanced_traders),
                    'average_feature_count': sum(r['feature_analysis'].get('features_by_category', {}).get('fills_based', 0)
                                                for r in enhanced_traders) / len(enhanced_traders),
                    'fills_importance_avg': sum(r['model_insights'].get('enhanced_features_impact', {}).get('fills_based_importance', 0)
                                              for r in enhanced_traders) / len(enhanced_traders)
                }

        logger.info(f"Enhanced evaluation complete: {successful_evaluations}/{len(trader_ids)} successful")
        return summary


if __name__ == "__main__":
    # Test enhanced evaluation
    evaluator = EnhancedCausalImpactEvaluator()

    # Test single trader evaluation
    test_result = evaluator.evaluate_enhanced_trader('3950')
    print(f"Test result: {json.dumps(test_result, indent=2, default=str)}")
