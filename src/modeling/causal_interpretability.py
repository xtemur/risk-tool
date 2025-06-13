"""
Causal Interpretability Module

Integration of SHAP, LIME, and other interpretability methods with proper causal warnings.
Provides feature importance and model explanations while clearly distinguishing between
predictive relationships and causal relationships.

Key features:
- SHAP integration with causal interpretation warnings
- LIME explanations with correlation vs causation disclaimers
- Causal feature importance ranking
- Counterfactual explanations for individual predictions
- Model interpretability reports with causal context
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
import logging
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

logger = logging.getLogger(__name__)


class CausalInterpretabilityAnalyzer:
    """
    Model interpretability analyzer with causal inference context

    Provides SHAP and LIME explanations while clearly warning about
    the difference between predictive importance and causal effects.
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize causal interpretability analyzer

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.shap_explainer = None
        self.lime_explainer = None

        # Check availability of interpretation libraries
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available. Install with: pip install shap")
        if not LIME_AVAILABLE:
            logger.warning("LIME not available. Install with: pip install lime")

    def analyze_model_interpretability(self,
                                     model: Any,
                                     X_train: pd.DataFrame,
                                     X_explain: pd.DataFrame,
                                     feature_names: Optional[List[str]] = None,
                                     causal_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Comprehensive model interpretability analysis with causal warnings

        Args:
            model: Trained ML model
            X_train: Training data for explainer fitting
            X_explain: Data to explain (subset of observations)
            feature_names: Optional feature names
            causal_context: Optional causal inference context

        Returns:
            Comprehensive interpretability analysis with causal warnings
        """
        logger.info("Starting model interpretability analysis with causal context")

        results = {
            'causal_warnings': self._generate_causal_warnings(),
            'interpretability_summary': {
                'n_features': X_train.shape[1],
                'n_explained_instances': X_explain.shape[0],
                'methods_available': {
                    'shap': SHAP_AVAILABLE,
                    'lime': LIME_AVAILABLE
                }
            }
        }

        # 1. SHAP Analysis
        if SHAP_AVAILABLE:
            try:
                logger.info("Running SHAP analysis...")
                shap_results = self._run_shap_analysis(model, X_train, X_explain, feature_names)
                results['shap_analysis'] = shap_results
            except Exception as e:
                logger.error(f"SHAP analysis failed: {e}")
                results['shap_analysis'] = {'error': str(e)}

        # 2. LIME Analysis
        if LIME_AVAILABLE:
            try:
                logger.info("Running LIME analysis...")
                lime_results = self._run_lime_analysis(model, X_train, X_explain, feature_names)
                results['lime_analysis'] = lime_results
            except Exception as e:
                logger.error(f"LIME analysis failed: {e}")
                results['lime_analysis'] = {'error': str(e)}

        # 3. Causal Feature Ranking
        try:
            logger.info("Computing causal feature ranking...")
            causal_ranking = self._compute_causal_feature_ranking(
                results, causal_context
            )
            results['causal_feature_ranking'] = causal_ranking
        except Exception as e:
            logger.error(f"Causal feature ranking failed: {e}")
            results['causal_feature_ranking'] = {'error': str(e)}

        # 4. Counterfactual Explanations
        try:
            logger.info("Generating counterfactual explanations...")
            counterfactuals = self._generate_counterfactual_explanations(
                model, X_explain, feature_names
            )
            results['counterfactual_explanations'] = counterfactuals
        except Exception as e:
            logger.error(f"Counterfactual explanations failed: {e}")
            results['counterfactual_explanations'] = {'error': str(e)}

        # 5. Interpretation Summary
        try:
            interpretation_summary = self._generate_interpretation_summary(results)
            results['interpretation_summary'] = interpretation_summary
        except Exception as e:
            logger.error(f"Interpretation summary failed: {e}")
            results['interpretation_summary'] = {'error': str(e)}

        logger.info("Model interpretability analysis completed")
        return results

    def _generate_causal_warnings(self) -> Dict[str, Any]:
        """Generate comprehensive causal interpretation warnings"""

        return {
            'primary_warning': (
                "⚠️  IMPORTANT: Model interpretability methods (SHAP, LIME) show predictive "
                "importance, NOT causal effects. Do not use these results to make causal claims "
                "without proper causal inference methods."
            ),
            'detailed_warnings': {
                'correlation_vs_causation': (
                    "Feature importance indicates correlation with the outcome, not causation. "
                    "High SHAP values do not mean changing that feature will cause the outcome to change."
                ),
                'confounding_bias': (
                    "Feature importance may be biased by confounding variables. Features may appear "
                    "important because they correlate with unmeasured causal factors."
                ),
                'spurious_correlations': (
                    "Some features may show high importance due to spurious correlations or "
                    "data leakage rather than meaningful relationships."
                ),
                'policy_implications': (
                    "Do not use feature importance to guide policy or intervention decisions. "
                    "Use proper causal inference methods (DML, IV, experiments) for causal claims."
                )
            },
            'safe_uses': [
                "Understanding model behavior and debugging",
                "Identifying which features the model relies on for predictions",
                "Detecting potential data quality issues or biases",
                "Model validation and consistency checking"
            ],
            'unsafe_uses': [
                "Making causal claims about feature effects",
                "Guiding business interventions or policy decisions",
                "Inferring what will happen if features are changed",
                "Attributing causality to model predictions"
            ]
        }

    def _run_shap_analysis(self,
                          model: Any,
                          X_train: pd.DataFrame,
                          X_explain: pd.DataFrame,
                          feature_names: Optional[List[str]]) -> Dict[str, Any]:
        """Run SHAP analysis with causal warnings"""

        if not SHAP_AVAILABLE:
            return {'error': 'SHAP not available'}

        # Initialize SHAP explainer
        try:
            # Try TreeExplainer first (for tree-based models)
            if hasattr(model, 'feature_importances_'):
                self.shap_explainer = shap.TreeExplainer(model)
            else:
                # Fall back to KernelExplainer (model-agnostic but slower)
                background_sample = shap.sample(X_train, min(100, len(X_train)), random_state=self.random_state)
                self.shap_explainer = shap.KernelExplainer(model.predict, background_sample)

        except Exception as e:
            logger.warning(f"Could not create SHAP explainer: {e}")
            return {'error': f'SHAP explainer creation failed: {e}'}

        # Calculate SHAP values
        try:
            shap_values = self.shap_explainer.shap_values(X_explain)

            # Handle multi-output models
            if isinstance(shap_values, list):
                shap_values = shap_values[0]  # Use first output for simplicity

        except Exception as e:
            logger.error(f"SHAP value calculation failed: {e}")
            return {'error': f'SHAP calculation failed: {e}'}

        # Process SHAP results
        feature_names_used = feature_names or X_explain.columns.tolist()

        # Global feature importance (mean absolute SHAP values)
        global_importance = np.mean(np.abs(shap_values), axis=0)

        feature_importance_df = pd.DataFrame({
            'feature': feature_names_used,
            'importance': global_importance
        }).sort_values('importance', ascending=False)

        # Instance-level explanations (for first few instances)
        n_explain = min(5, len(X_explain))
        instance_explanations = []

        for i in range(n_explain):
            instance_shap = shap_values[i]

            explanation = {
                'instance_id': i,
                'prediction': float(model.predict(X_explain.iloc[[i]])[0]) if hasattr(model, 'predict') else None,
                'feature_contributions': [
                    {
                        'feature': feature_names_used[j],
                        'value': float(X_explain.iloc[i, j]),
                        'shap_value': float(instance_shap[j])
                    }
                    for j in range(len(feature_names_used))
                ],
                'top_positive_contributors': [],
                'top_negative_contributors': []
            }

            # Sort contributions
            contributions = explanation['feature_contributions']
            contributions.sort(key=lambda x: x['shap_value'], reverse=True)

            explanation['top_positive_contributors'] = [
                c for c in contributions if c['shap_value'] > 0
            ][:3]

            explanation['top_negative_contributors'] = [
                c for c in contributions if c['shap_value'] < 0
            ][-3:]

            instance_explanations.append(explanation)

        return {
            'global_feature_importance': {
                'ranking': feature_importance_df.to_dict('records'),
                'top_features': feature_importance_df.head(10).to_dict('records')
            },
            'instance_explanations': instance_explanations,
            'shap_statistics': {
                'mean_abs_shap': float(np.mean(np.abs(shap_values))),
                'max_abs_shap': float(np.max(np.abs(shap_values))),
                'feature_interaction_strength': float(np.std(shap_values, axis=0).mean())
            },
            'causal_disclaimer': (
                "⚠️ SHAP values show predictive importance, not causal effects. "
                "Do not interpret high SHAP values as evidence of causation."
            )
        }

    def _run_lime_analysis(self,
                          model: Any,
                          X_train: pd.DataFrame,
                          X_explain: pd.DataFrame,
                          feature_names: Optional[List[str]]) -> Dict[str, Any]:
        """Run LIME analysis with causal warnings"""

        if not LIME_AVAILABLE:
            return {'error': 'LIME not available'}

        # Initialize LIME explainer
        try:
            self.lime_explainer = LimeTabularExplainer(
                X_train.values,
                feature_names=feature_names or X_train.columns.tolist(),
                mode='regression',  # Assume regression for trading models
                random_state=self.random_state
            )
        except Exception as e:
            return {'error': f'LIME explainer creation failed: {e}'}

        # Generate LIME explanations for sample instances
        n_explain = min(3, len(X_explain))  # LIME is slower, so fewer instances
        lime_explanations = []

        for i in range(n_explain):
            try:
                instance = X_explain.iloc[i].values

                # Generate explanation
                explanation = self.lime_explainer.explain_instance(
                    instance,
                    model.predict,
                    num_features=min(10, len(instance))
                )

                # Extract explanation details
                feature_importance = explanation.as_list()

                lime_explanations.append({
                    'instance_id': i,
                    'prediction': float(model.predict(X_explain.iloc[[i]])[0]) if hasattr(model, 'predict') else None,
                    'local_importance': [
                        {
                            'feature': feat,
                            'importance': float(imp)
                        }
                        for feat, imp in feature_importance
                    ],
                    'explanation_score': float(explanation.score) if hasattr(explanation, 'score') else None
                })

            except Exception as e:
                logger.warning(f"LIME explanation for instance {i} failed: {e}")

        return {
            'local_explanations': lime_explanations,
            'lime_summary': {
                'n_explained_instances': len(lime_explanations),
                'avg_explanation_score': np.mean([
                    exp['explanation_score'] for exp in lime_explanations
                    if exp['explanation_score'] is not None
                ]) if lime_explanations else None
            },
            'causal_disclaimer': (
                "⚠️ LIME explanations show local predictive importance, not causal effects. "
                "Local importance may vary significantly across instances."
            )
        }

    def _compute_causal_feature_ranking(self,
                                      interpretability_results: Dict[str, Any],
                                      causal_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute causal feature ranking by combining interpretability with causal evidence
        """

        ranking_results = {
            'methodology': (
                "Causal feature ranking combines predictive importance with causal evidence. "
                "Features are ranked by both their predictive power and their potential "
                "for causal interpretation based on domain knowledge and causal analysis."
            ),
            'ranking_criteria': {
                'predictive_importance': "Based on SHAP/LIME feature importance",
                'causal_plausibility': "Based on domain knowledge and causal theory",
                'temporal_precedence': "Whether feature precedes outcome in time",
                'manipulation_feasibility': "Whether feature can be realistically changed"
            }
        }

        # Extract predictive importance from SHAP/LIME
        predictive_ranking = []

        if 'shap_analysis' in interpretability_results and 'error' not in interpretability_results['shap_analysis']:
            shap_features = interpretability_results['shap_analysis']['global_feature_importance']['ranking']
            predictive_ranking = [
                {
                    'feature': feat['feature'],
                    'predictive_importance': feat['importance'],
                    'method': 'shap'
                }
                for feat in shap_features
            ]

        # Add causal context if available
        if causal_context and 'causal_features' in causal_context:
            causal_features = causal_context['causal_features']

            for feature_info in predictive_ranking:
                feature_name = feature_info['feature']

                # Look up causal information
                causal_info = causal_features.get(feature_name, {})

                feature_info.update({
                    'causal_plausibility': causal_info.get('causal_plausibility', 'unknown'),
                    'temporal_precedence': causal_info.get('temporal_precedence', 'unknown'),
                    'manipulation_feasibility': causal_info.get('manipulation_feasibility', 'unknown'),
                    'domain_knowledge': causal_info.get('domain_knowledge', 'none')
                })

                # Calculate combined causal score
                causal_score = 0
                if causal_info.get('causal_plausibility') == 'high':
                    causal_score += 0.4
                elif causal_info.get('causal_plausibility') == 'medium':
                    causal_score += 0.2

                if causal_info.get('temporal_precedence') == 'yes':
                    causal_score += 0.3

                if causal_info.get('manipulation_feasibility') == 'high':
                    causal_score += 0.3

                feature_info['causal_score'] = causal_score
                feature_info['combined_score'] = (
                    0.6 * feature_info['predictive_importance'] + 0.4 * causal_score
                )

        # Sort by combined score if available, otherwise by predictive importance
        if predictive_ranking and 'combined_score' in predictive_ranking[0]:
            predictive_ranking.sort(key=lambda x: x['combined_score'], reverse=True)
            ranking_results['ranking_method'] = 'combined_predictive_and_causal'
        else:
            ranking_results['ranking_method'] = 'predictive_only'

        ranking_results['feature_ranking'] = predictive_ranking

        # Generate actionable insights
        actionable_features = []
        descriptive_features = []

        for feature in predictive_ranking[:10]:  # Top 10 features
            if feature.get('manipulation_feasibility') == 'high':
                actionable_features.append(feature)
            else:
                descriptive_features.append(feature)

        ranking_results['actionable_insights'] = {
            'actionable_features': actionable_features,
            'descriptive_features': descriptive_features,
            'causal_interpretation_warning': (
                "Only features with high causal plausibility and manipulation feasibility "
                "should be considered for interventions. High predictive importance alone "
                "does not justify causal interventions."
            )
        }

        return ranking_results

    def _generate_counterfactual_explanations(self,
                                            model: Any,
                                            X_explain: pd.DataFrame,
                                            feature_names: Optional[List[str]]) -> Dict[str, Any]:
        """
        Generate counterfactual explanations for individual predictions
        """

        counterfactuals = {
            'methodology': (
                "Counterfactual explanations show what feature changes would lead to "
                "different predictions. These are 'what-if' scenarios for model behavior."
            ),
            'explanations': []
        }

        # Generate counterfactuals for a few instances
        n_explain = min(3, len(X_explain))

        for i in range(n_explain):
            try:
                instance = X_explain.iloc[i].copy()
                original_prediction = model.predict(instance.values.reshape(1, -1))[0]

                counterfactual_scenarios = []

                # Generate simple counterfactuals by perturbing each feature
                for j, feature in enumerate(instance.index):
                    original_value = instance.iloc[j]

                    # Try different perturbations
                    perturbations = []

                    if pd.api.types.is_numeric_dtype(type(original_value)):
                        # For numeric features, try ±10%, ±25%
                        for pct in [-0.25, -0.1, 0.1, 0.25]:
                            new_value = original_value * (1 + pct)
                            perturbations.append(new_value)
                    else:
                        # For categorical features, would need more sophisticated approach
                        continue

                    for new_value in perturbations:
                        counterfactual_instance = instance.copy()
                        counterfactual_instance.iloc[j] = new_value

                        new_prediction = model.predict(counterfactual_instance.values.reshape(1, -1))[0]

                        if abs(new_prediction - original_prediction) > abs(original_prediction * 0.05):  # 5% change threshold
                            counterfactual_scenarios.append({
                                'feature_changed': feature,
                                'original_value': float(original_value),
                                'new_value': float(new_value),
                                'change_magnitude': float(abs(new_value - original_value)),
                                'original_prediction': float(original_prediction),
                                'new_prediction': float(new_prediction),
                                'prediction_change': float(new_prediction - original_prediction),
                                'relative_change': float((new_value - original_value) / original_value) if original_value != 0 else 0
                            })

                # Sort by prediction change magnitude
                counterfactual_scenarios.sort(key=lambda x: abs(x['prediction_change']), reverse=True)

                counterfactuals['explanations'].append({
                    'instance_id': i,
                    'original_prediction': float(original_prediction),
                    'counterfactual_scenarios': counterfactual_scenarios[:5],  # Top 5 scenarios
                    'most_sensitive_feature': counterfactual_scenarios[0]['feature_changed'] if counterfactual_scenarios else None
                })

            except Exception as e:
                logger.warning(f"Counterfactual generation for instance {i} failed: {e}")

        counterfactuals['causal_disclaimer'] = (
            "⚠️ Counterfactual explanations show model behavior under hypothetical changes, "
            "not real-world causal effects. These scenarios may not be achievable in practice "
            "due to feature dependencies and causal constraints."
        )

        return counterfactuals

    def _generate_interpretation_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive interpretation summary"""

        summary = {
            'key_findings': [],
            'most_important_features': [],
            'model_behavior_insights': [],
            'causal_interpretation_guidance': [],
            'limitations': []
        }

        # Extract key findings from SHAP
        if 'shap_analysis' in results and 'error' not in results['shap_analysis']:
            shap_top_features = results['shap_analysis']['global_feature_importance']['top_features'][:3]
            summary['most_important_features'].extend([
                f"{feat['feature']} (SHAP importance: {feat['importance']:.3f})"
                for feat in shap_top_features
            ])

            summary['key_findings'].append(
                f"SHAP analysis identifies {len(shap_top_features)} key predictive features"
            )

        # Extract insights from counterfactuals
        if 'counterfactual_explanations' in results and 'error' not in results['counterfactual_explanations']:
            cf_explanations = results['counterfactual_explanations']['explanations']

            if cf_explanations:
                sensitive_features = [exp['most_sensitive_feature'] for exp in cf_explanations if exp['most_sensitive_feature']]

                if sensitive_features:
                    from collections import Counter
                    most_common_sensitive = Counter(sensitive_features).most_common(1)[0][0]
                    summary['model_behavior_insights'].append(
                        f"Model predictions are most sensitive to changes in '{most_common_sensitive}'"
                    )

        # Causal interpretation guidance
        summary['causal_interpretation_guidance'] = [
            "Use interpretability results to understand model behavior, not to make causal claims",
            "High feature importance indicates correlation with the outcome, not causation",
            "For causal inference, use dedicated methods like DML, IV, or experiments",
            "Be cautious of confounding when interpreting feature importance",
            "Consider temporal relationships and manipulation feasibility for actionable insights"
        ]

        # Limitations
        summary['limitations'] = [
            "Interpretability methods show correlations, not causal relationships",
            "Feature importance may be biased by unmeasured confounders",
            "Local explanations (LIME) may not generalize across instances",
            "Counterfactual scenarios may not be realistic or achievable",
            "Model explanations are only as good as the underlying model quality"
        ]

        return summary
