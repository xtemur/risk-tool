"""
Advanced Causal Impact Analyzer

State-of-the-art causal inference implementation that combines multiple methods:
- Double Machine Learning (DML) for robust causal effect estimation
- Synthetic Control Method for counterfactual analysis
- Enhanced statistical validation and robustness testing
- Model interpretability with causal caveats

This analyzer provides comprehensive causal analysis to prove model utility
using modern econometric and machine learning techniques.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
import logging
from datetime import datetime
import warnings

# Import our custom modules
from .double_ml_estimator import DoubleMachineLearningEstimator, TradingDMLAnalyzer
from .synthetic_control_analyzer import SyntheticControlAnalyzer
from .causal_impact_analyzer import CausalImpactAnalyzer

logger = logging.getLogger(__name__)


class AdvancedCausalAnalyzer:
    """
    Advanced causal impact analyzer combining state-of-the-art methods

    Integrates multiple causal inference approaches:
    1. Double Machine Learning for robust treatment effect estimation
    2. Synthetic Control Method for counterfactual analysis
    3. Traditional causal impact scenarios
    4. Comprehensive validation and robustness testing
    """

    def __init__(self,
                 random_state: int = 42,
                 enable_dml: bool = True,
                 enable_synthetic_control: bool = True,
                 enable_traditional_analysis: bool = True):
        """
        Initialize advanced causal analyzer

        Args:
            random_state: Random seed for reproducibility
            enable_dml: Whether to run Double ML analysis
            enable_synthetic_control: Whether to run Synthetic Control analysis
            enable_traditional_analysis: Whether to run traditional causal impact analysis
        """
        self.random_state = random_state
        self.enable_dml = enable_dml
        self.enable_synthetic_control = enable_synthetic_control
        self.enable_traditional_analysis = enable_traditional_analysis

        # Initialize component analyzers
        if self.enable_dml:
            self.dml_analyzer = TradingDMLAnalyzer()

        if self.enable_synthetic_control:
            self.synthetic_control_analyzer = SyntheticControlAnalyzer(random_state=random_state)

        if self.enable_traditional_analysis:
            self.traditional_analyzer = CausalImpactAnalyzer()

        # Results storage
        self.results_ = {}
        self.is_fitted_ = False

    def analyze_comprehensive_impact(self,
                                   actual_pnl: Union[pd.Series, pd.DataFrame],
                                   predicted_pnl: Union[pd.Series, pd.DataFrame],
                                   features: pd.DataFrame,
                                   trader_ids: Optional[pd.Series] = None,
                                   dates: Optional[pd.DatetimeIndex] = None,
                                   model_confidence: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Comprehensive causal impact analysis using multiple state-of-the-art methods

        Args:
            actual_pnl: Actual trading PnL (can be Series for single trader or DataFrame for panel)
            predicted_pnl: Model predicted PnL (same structure as actual_pnl)
            features: Features/covariates for causal analysis
            trader_ids: Optional trader identifiers for panel data
            dates: Optional date index
            model_confidence: Optional model confidence scores

        Returns:
            Comprehensive causal analysis results
        """
        logger.info("Starting comprehensive causal impact analysis with state-of-the-art methods")

        # Prepare data for analysis
        analysis_data = self._prepare_analysis_data(
            actual_pnl, predicted_pnl, features, trader_ids, dates, model_confidence
        )

        results = {
            'analysis_metadata': {
                'analysis_date': datetime.now().isoformat(),
                'n_observations': len(analysis_data['actual_pnl']),
                'n_traders': analysis_data.get('n_traders', 1),
                'date_range': {
                    'start': analysis_data['dates'].min().isoformat() if analysis_data['dates'] is not None else None,
                    'end': analysis_data['dates'].max().isoformat() if analysis_data['dates'] is not None else None
                },
                'methods_enabled': {
                    'double_ml': self.enable_dml,
                    'synthetic_control': self.enable_synthetic_control,
                    'traditional_scenarios': self.enable_traditional_analysis
                }
            }
        }

        # 1. Double Machine Learning Analysis
        if self.enable_dml:
            try:
                logger.info("Running Double Machine Learning analysis...")
                dml_results = self._run_dml_analysis(analysis_data)
                results['double_ml_analysis'] = dml_results
                logger.info(f"DML analysis completed. ATE: {dml_results.get('dml_causal_effect', {}).get('ate_estimate', 'N/A')}")
            except Exception as e:
                logger.error(f"DML analysis failed: {e}")
                results['double_ml_analysis'] = {'error': str(e)}

        # 2. Synthetic Control Analysis (if panel data available)
        if self.enable_synthetic_control and analysis_data.get('is_panel_data', False):
            try:
                logger.info("Running Synthetic Control analysis...")
                sc_results = self._run_synthetic_control_analysis(analysis_data)
                results['synthetic_control_analysis'] = sc_results
                logger.info("Synthetic Control analysis completed")
            except Exception as e:
                logger.error(f"Synthetic Control analysis failed: {e}")
                results['synthetic_control_analysis'] = {'error': str(e)}

        # 3. Traditional Causal Impact Scenarios
        if self.enable_traditional_analysis:
            try:
                logger.info("Running traditional causal impact analysis...")
                traditional_results = self._run_traditional_analysis(analysis_data)
                results['traditional_causal_analysis'] = traditional_results
                logger.info("Traditional analysis completed")
            except Exception as e:
                logger.error(f"Traditional analysis failed: {e}")
                results['traditional_causal_analysis'] = {'error': str(e)}

        # 4. Cross-Method Validation and Synthesis
        try:
            logger.info("Synthesizing results across methods...")
            synthesis = self._synthesize_results(results)
            results['synthesis_and_validation'] = synthesis
        except Exception as e:
            logger.error(f"Results synthesis failed: {e}")
            results['synthesis_and_validation'] = {'error': str(e)}

        # 5. Business Impact Assessment
        try:
            business_impact = self._assess_business_impact(results, analysis_data)
            results['business_impact_assessment'] = business_impact
        except Exception as e:
            logger.error(f"Business impact assessment failed: {e}")
            results['business_impact_assessment'] = {'error': str(e)}

        # 6. Recommendations and Interpretation
        try:
            recommendations = self._generate_recommendations(results)
            results['recommendations_and_interpretation'] = recommendations
        except Exception as e:
            logger.error(f"Recommendations generation failed: {e}")
            results['recommendations_and_interpretation'] = {'error': str(e)}

        self.results_ = results
        self.is_fitted_ = True

        logger.info("Comprehensive causal impact analysis completed")
        return results

    def _prepare_analysis_data(self,
                             actual_pnl: Union[pd.Series, pd.DataFrame],
                             predicted_pnl: Union[pd.Series, pd.DataFrame],
                             features: pd.DataFrame,
                             trader_ids: Optional[pd.Series],
                             dates: Optional[pd.DatetimeIndex],
                             model_confidence: Optional[pd.Series]) -> Dict[str, Any]:
        """Prepare and validate data for analysis"""

        # Determine if we have panel data
        is_panel_data = isinstance(actual_pnl, pd.DataFrame) and len(actual_pnl.columns) > 1

        # Convert to consistent format
        if isinstance(actual_pnl, pd.Series):
            actual_pnl_df = actual_pnl.to_frame('trader_1')
            predicted_pnl_df = predicted_pnl.to_frame('trader_1')
        else:
            actual_pnl_df = actual_pnl
            predicted_pnl_df = predicted_pnl

        # Handle dates
        if dates is None:
            dates = actual_pnl_df.index

        # Handle trader IDs
        if trader_ids is None and is_panel_data:
            trader_ids = actual_pnl_df.columns

        return {
            'actual_pnl': actual_pnl if isinstance(actual_pnl, pd.Series) else actual_pnl.stack(),
            'predicted_pnl': predicted_pnl if isinstance(predicted_pnl, pd.Series) else predicted_pnl.stack(),
            'actual_pnl_df': actual_pnl_df,
            'predicted_pnl_df': predicted_pnl_df,
            'features': features,
            'trader_ids': trader_ids,
            'dates': dates,
            'model_confidence': model_confidence,
            'is_panel_data': is_panel_data,
            'n_traders': len(actual_pnl_df.columns) if is_panel_data else 1
        }

    def _run_dml_analysis(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run Double Machine Learning analysis"""

        # For DML, we need to create a treatment variable
        # Treatment = model recommends positive return
        treatment = (analysis_data['predicted_pnl'] > 0).astype(int)

        # Prepare features for DML
        features_dml = analysis_data['features'].copy()

        # Add additional controls
        if analysis_data['model_confidence'] is not None:
            features_dml['model_confidence'] = analysis_data['model_confidence']

        # Add trader fixed effects if panel data
        if analysis_data['is_panel_data'] and analysis_data['trader_ids'] is not None:
            # Create trader dummies
            trader_dummies = pd.get_dummies(
                analysis_data['actual_pnl'].index.get_level_values(1) if hasattr(analysis_data['actual_pnl'].index, 'levels')
                else pd.Series(['trader_1'] * len(analysis_data['actual_pnl']), index=analysis_data['actual_pnl'].index),
                prefix='trader'
            )
            features_dml = pd.concat([features_dml, trader_dummies], axis=1)

        # Ensure indices match
        common_index = features_dml.index.intersection(analysis_data['actual_pnl'].index)

        features_dml = features_dml.loc[common_index]
        actual_pnl_aligned = analysis_data['actual_pnl'].loc[common_index]
        treatment_aligned = treatment.loc[common_index]

        # Run DML analysis
        dml_results = self.dml_analyzer.analyze_model_impact(
            actual_pnl=actual_pnl_aligned,
            predicted_pnl=analysis_data['predicted_pnl'].loc[common_index],
            features=features_dml,
            model_confidence=analysis_data['model_confidence'].loc[common_index] if analysis_data['model_confidence'] is not None else None
        )

        return dml_results

    def _run_synthetic_control_analysis(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run Synthetic Control analysis"""

        if not analysis_data['is_panel_data']:
            return {'error': 'Synthetic Control requires panel data with multiple traders'}

        # Create treatment indicators (model usage)
        # For simplicity, assume treatment = model recommends positive return
        treatment_indicators = (analysis_data['predicted_pnl_df'] > 0).astype(int)

        # Run synthetic control analysis
        sc_results = self.synthetic_control_analyzer.analyze_trading_impact(
            trader_pnl=analysis_data['actual_pnl_df'],
            model_usage=treatment_indicators,
            trader_features=None  # Could add trader-specific features here
        )

        return sc_results

    def _run_traditional_analysis(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run traditional causal impact analysis"""

        # Run for primary trader or aggregate if panel data
        if analysis_data['is_panel_data']:
            # Use first trader or aggregate
            actual_pnl = analysis_data['actual_pnl_df'].iloc[:, 0]
            predicted_pnl = analysis_data['predicted_pnl_df'].iloc[:, 0]
        else:
            actual_pnl = analysis_data['actual_pnl']
            predicted_pnl = analysis_data['predicted_pnl']

        traditional_results = self.traditional_analyzer.calculate_trading_impact(
            actual_pnl=actual_pnl.values,
            predicted_pnl=predicted_pnl.values,
            dates=analysis_data['dates']
        )

        return traditional_results

    def _synthesize_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize results across different methods"""

        synthesis = {
            'method_comparison': {},
            'consistency_check': {},
            'robustness_assessment': {},
            'confidence_rating': 'medium'
        }

        # Extract causal effect estimates from each method
        effect_estimates = {}

        # DML estimate
        if 'double_ml_analysis' in results and 'error' not in results['double_ml_analysis']:
            dml_effect = results['double_ml_analysis']['dml_causal_effect']['ate_estimate']
            effect_estimates['double_ml'] = dml_effect

        # Synthetic Control estimate
        if 'synthetic_control_analysis' in results and 'error' not in results['synthetic_control_analysis']:
            if 'aggregate_effects' in results['synthetic_control_analysis']:
                sc_effect = results['synthetic_control_analysis']['aggregate_effects']['average_treatment_effect']['difference']
                effect_estimates['synthetic_control'] = sc_effect

        # Traditional estimate
        if 'traditional_causal_analysis' in results and 'error' not in results['traditional_causal_analysis']:
            trad_effect = results['traditional_causal_analysis']['causal_impact_scenarios']['perfect_following']['pnl_improvement']
            effect_estimates['traditional'] = trad_effect

        synthesis['method_comparison'] = {
            'effect_estimates': effect_estimates,
            'estimate_range': [min(effect_estimates.values()), max(effect_estimates.values())] if effect_estimates else [0, 0],
            'estimate_std': float(np.std(list(effect_estimates.values()))) if len(effect_estimates) > 1 else 0,
            'methods_agree': len(set(np.sign(list(effect_estimates.values())))) <= 1 if effect_estimates else False
        }

        # Consistency check
        if len(effect_estimates) >= 2:
            estimates_array = np.array(list(effect_estimates.values()))
            cv = np.std(estimates_array) / (np.abs(np.mean(estimates_array)) + 1e-6)

            synthesis['consistency_check'] = {
                'coefficient_of_variation': float(cv),
                'is_consistent': cv < 0.5,  # Estimates within 50% of each other
                'consensus_estimate': float(np.mean(estimates_array)),
                'consensus_direction': 'positive' if np.mean(estimates_array) > 0 else 'negative'
            }

        # Robustness assessment
        significance_count = 0
        total_tests = 0

        # Check DML significance
        if 'double_ml_analysis' in results and 'error' not in results['double_ml_analysis']:
            if results['double_ml_analysis']['dml_causal_effect']['is_significant']:
                significance_count += 1
            total_tests += 1

        # Check traditional significance
        if 'traditional_causal_analysis' in results and 'error' not in results['traditional_causal_analysis']:
            if results['traditional_causal_analysis']['statistical_significance']['is_significant']:
                significance_count += 1
            total_tests += 1

        synthesis['robustness_assessment'] = {
            'significant_methods': significance_count,
            'total_methods': total_tests,
            'robustness_score': significance_count / max(total_tests, 1)
        }

        # Overall confidence rating
        if synthesis['robustness_assessment']['robustness_score'] >= 0.75:
            synthesis['confidence_rating'] = 'high'
        elif synthesis['robustness_assessment']['robustness_score'] >= 0.5:
            synthesis['confidence_rating'] = 'medium'
        else:
            synthesis['confidence_rating'] = 'low'

        return synthesis

    def _assess_business_impact(self, results: Dict[str, Any], analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess business impact and practical significance"""

        # Get consensus estimate
        consensus_effect = 0
        if 'synthesis_and_validation' in results and 'consistency_check' in results['synthesis_and_validation']:
            consensus_effect = results['synthesis_and_validation']['consistency_check'].get('consensus_estimate', 0)

        # Calculate business metrics
        total_trading_days = len(analysis_data['actual_pnl'])
        total_causal_impact = consensus_effect * total_trading_days

        # Calculate baseline performance
        baseline_total_pnl = analysis_data['actual_pnl'].sum()
        baseline_volatility = analysis_data['actual_pnl'].std()

        # Relative impact
        relative_improvement = (total_causal_impact / abs(baseline_total_pnl) * 100) if baseline_total_pnl != 0 else 0

        # Risk-adjusted metrics
        if baseline_volatility > 0:
            effect_size = consensus_effect / baseline_volatility
            information_ratio = consensus_effect / baseline_volatility * np.sqrt(252)  # Annualized
        else:
            effect_size = 0
            information_ratio = 0

        return {
            'consensus_daily_effect': float(consensus_effect),
            'total_causal_impact': float(total_causal_impact),
            'relative_improvement_pct': float(relative_improvement),
            'baseline_metrics': {
                'total_pnl': float(baseline_total_pnl),
                'daily_volatility': float(baseline_volatility),
                'trading_days': int(total_trading_days)
            },
            'risk_adjusted_metrics': {
                'effect_size': float(effect_size),
                'information_ratio': float(information_ratio),
                'practical_significance': 'high' if abs(effect_size) > 0.5 else 'medium' if abs(effect_size) > 0.2 else 'low'
            },
            'business_value_assessment': {
                'economic_significance': abs(total_causal_impact) > abs(baseline_total_pnl) * 0.05,  # 5% improvement threshold
                'estimated_annual_impact': float(consensus_effect * 252),  # Assume 252 trading days
                'roi_category': 'high' if abs(relative_improvement) > 10 else 'medium' if abs(relative_improvement) > 5 else 'low'
            }
        }

    def _generate_recommendations(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate actionable recommendations based on analysis"""

        recommendations = {
            'primary_recommendation': '',
            'confidence_level': '',
            'key_findings': [],
            'implementation_guidance': [],
            'monitoring_recommendations': [],
            'limitations_and_caveats': []
        }

        # Extract key metrics
        confidence_rating = results.get('synthesis_and_validation', {}).get('confidence_rating', 'medium')
        consensus_effect = results.get('business_impact_assessment', {}).get('consensus_daily_effect', 0)
        practical_significance = results.get('business_impact_assessment', {}).get('risk_adjusted_metrics', {}).get('practical_significance', 'medium')

        # Primary recommendation
        if consensus_effect > 0 and confidence_rating in ['high', 'medium']:
            recommendations['primary_recommendation'] = f"RECOMMENDED: Model deployment shows positive causal impact of ${consensus_effect:.2f} per day with {confidence_rating} confidence"
        elif consensus_effect < 0:
            recommendations['primary_recommendation'] = f"CAUTION: Model shows negative causal impact of ${consensus_effect:.2f} per day - review model implementation"
        else:
            recommendations['primary_recommendation'] = "NEUTRAL: No significant causal impact detected - model may need improvement"

        recommendations['confidence_level'] = confidence_rating

        # Key findings
        findings = []

        if 'double_ml_analysis' in results and 'error' not in results['double_ml_analysis']:
            dml_effect = results['double_ml_analysis']['dml_causal_effect']['ate_estimate']
            dml_sig = results['double_ml_analysis']['dml_causal_effect']['is_significant']
            findings.append(f"Double ML method estimates ${dml_effect:.2f} daily effect ({'significant' if dml_sig else 'not significant'})")

        if 'synthesis_and_validation' in results:
            methods_agree = results['synthesis_and_validation']['method_comparison'].get('methods_agree', False)
            findings.append(f"Multiple methods {'agree' if methods_agree else 'show mixed results'} on effect direction")

        recommendations['key_findings'] = findings

        # Implementation guidance
        guidance = []

        if consensus_effect > 0:
            guidance.append("Implement model recommendations in live trading environment")
            guidance.append("Start with reduced position sizes to validate results")
            guidance.append("Monitor actual vs predicted performance closely")
        else:
            guidance.append("Do not implement current model in live trading")
            guidance.append("Investigate model calibration and feature engineering")
            guidance.append("Consider alternative modeling approaches")

        recommendations['implementation_guidance'] = guidance

        # Monitoring recommendations
        monitoring = [
            "Track daily model recommendations vs actual outcomes",
            "Monitor hit rate and performance degradation over time",
            "Conduct monthly causal impact reassessment",
            "Set up alerts for significant performance deviations"
        ]

        if practical_significance == 'high':
            monitoring.append("Implement real-time model performance dashboard")

        recommendations['monitoring_recommendations'] = monitoring

        # Limitations and caveats
        limitations = [
            "Causal estimates assume no unmeasured confounders",
            "Results may not generalize to different market conditions",
            "Model performance may decay over time",
            "External factors (regulation, market structure) may affect results"
        ]

        if confidence_rating == 'low':
            limitations.append("Low confidence rating suggests results should be interpreted cautiously")

        recommendations['limitations_and_caveats'] = limitations

        return recommendations

    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive causal impact report"""

        if not self.is_fitted_:
            raise ValueError("Analyzer must be fitted before generating report")

        results = self.results_

        report = f"""
ADVANCED CAUSAL IMPACT ANALYSIS REPORT
=====================================

Analysis Date: {results['analysis_metadata']['analysis_date']}
Observations: {results['analysis_metadata']['n_observations']}
Traders: {results['analysis_metadata']['n_traders']}
Methods Used: {', '.join([k for k, v in results['analysis_metadata']['methods_enabled'].items() if v])}

EXECUTIVE SUMMARY
-----------------
{results.get('recommendations_and_interpretation', {}).get('primary_recommendation', 'No recommendation available')}

Confidence Level: {results.get('recommendations_and_interpretation', {}).get('confidence_level', 'Unknown')}

CAUSAL EFFECT ESTIMATES
-----------------------
"""

        # Add method-specific results
        if 'double_ml_analysis' in results and 'error' not in results['double_ml_analysis']:
            dml = results['double_ml_analysis']['dml_causal_effect']
            report += f"""
Double Machine Learning:
  Daily Effect: ${dml['ate_estimate']:.2f} ± ${dml['ate_standard_error']:.2f}
  95% CI: [${dml['ate_confidence_interval'][0]:.2f}, ${dml['ate_confidence_interval'][1]:.2f}]
  P-value: {dml['ate_p_value']:.4f}
  Significant: {'Yes' if dml['is_significant'] else 'No'}
"""

        if 'synthetic_control_analysis' in results and 'error' not in results['synthetic_control_analysis']:
            sc = results['synthetic_control_analysis']['aggregate_effects']['average_treatment_effect']
            report += f"""
Synthetic Control:
  Daily Effect: ${sc['difference']:.2f}
  Cumulative Effect: ${sc['cumulative_effect']:.2f}
  Relative Improvement: {sc['relative_improvement']:.1f}%
"""

        if 'traditional_causal_analysis' in results and 'error' not in results['traditional_causal_analysis']:
            trad = results['traditional_causal_analysis']['causal_impact_scenarios']['perfect_following']
            report += f"""
Traditional Analysis:
  Perfect Following Impact: ${trad['pnl_improvement']:.2f}
  Improvement %: {trad['pnl_improvement_pct']:.1f}%
"""

        # Business impact
        if 'business_impact_assessment' in results:
            biz = results['business_impact_assessment']
            report += f"""
BUSINESS IMPACT ASSESSMENT
--------------------------
Consensus Daily Effect: ${biz['consensus_daily_effect']:.2f}
Total Causal Impact: ${biz['total_causal_impact']:.2f}
Relative Improvement: {biz['relative_improvement_pct']:.1f}%
Effect Size: {biz['risk_adjusted_metrics']['effect_size']:.2f}
Practical Significance: {biz['risk_adjusted_metrics']['practical_significance']}
Estimated Annual Impact: ${biz['business_value_assessment']['estimated_annual_impact']:.2f}
"""

        # Recommendations
        if 'recommendations_and_interpretation' in results:
            rec = results['recommendations_and_interpretation']
            report += f"""
RECOMMENDATIONS
---------------
"""
            for finding in rec.get('key_findings', []):
                report += f"• {finding}\n"

            report += f"""
Implementation Guidance:
"""
            for guidance in rec.get('implementation_guidance', []):
                report += f"• {guidance}\n"

            report += f"""
LIMITATIONS AND CAVEATS
-----------------------
"""
            for limitation in rec.get('limitations_and_caveats', []):
                report += f"• {limitation}\n"

        return report
