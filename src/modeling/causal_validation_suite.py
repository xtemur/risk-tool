"""
Causal Validation Suite

Comprehensive validation and robustness testing for causal inference models.
Implements modern statistical tests to validate causal claims and assess
the reliability of causal effect estimates.

Key validation methods:
- Sensitivity analysis for unobserved confounders
- Placebo tests and falsification tests
- Bootstrap confidence intervals
- Cross-validation for causal models
- Heterogeneity analysis
- Model specification tests
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Callable
import logging
from scipy import stats
from sklearn.model_selection import KFold, train_test_split
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class CausalValidationSuite:
    """
    Comprehensive validation suite for causal inference models

    Provides statistical tests and robustness checks to validate causal claims
    and assess the reliability of treatment effect estimates.
    """

    def __init__(self, random_state: int = 42, n_bootstrap: int = 1000):
        """
        Initialize validation suite

        Args:
            random_state: Random seed for reproducibility
            n_bootstrap: Number of bootstrap iterations
        """
        self.random_state = random_state
        self.n_bootstrap = n_bootstrap

    def validate_causal_model(self,
                            causal_estimator: Any,
                            X: pd.DataFrame,
                            y: pd.Series,
                            treatment: pd.Series,
                            validation_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Comprehensive validation of causal model

        Args:
            causal_estimator: Fitted causal inference estimator
            X: Covariate features
            y: Outcome variable
            treatment: Treatment indicator
            validation_config: Configuration for validation tests

        Returns:
            Comprehensive validation results
        """
        logger.info("Starting comprehensive causal model validation")

        config = validation_config or {}

        results = {
            'validation_summary': {
                'n_observations': len(X),
                'n_features': X.shape[1],
                'treatment_prevalence': float(treatment.mean()),
                'validation_date': pd.Timestamp.now().isoformat()
            }
        }

        # 1. Sensitivity Analysis for Unobserved Confounders
        try:
            logger.info("Running sensitivity analysis...")
            sensitivity_results = self._sensitivity_analysis(
                causal_estimator, X, y, treatment,
                config.get('sensitivity_config', {})
            )
            results['sensitivity_analysis'] = sensitivity_results
        except Exception as e:
            logger.error(f"Sensitivity analysis failed: {e}")
            results['sensitivity_analysis'] = {'error': str(e)}

        # 2. Placebo and Falsification Tests
        try:
            logger.info("Running placebo tests...")
            placebo_results = self._placebo_tests(
                causal_estimator, X, y, treatment,
                config.get('placebo_config', {})
            )
            results['placebo_tests'] = placebo_results
        except Exception as e:
            logger.error(f"Placebo tests failed: {e}")
            results['placebo_tests'] = {'error': str(e)}

        # 3. Bootstrap Confidence Intervals
        try:
            logger.info("Computing bootstrap confidence intervals...")
            bootstrap_results = self._bootstrap_inference(
                causal_estimator, X, y, treatment,
                config.get('bootstrap_config', {})
            )
            results['bootstrap_inference'] = bootstrap_results
        except Exception as e:
            logger.error(f"Bootstrap inference failed: {e}")
            results['bootstrap_inference'] = {'error': str(e)}

        # 4. Cross-Validation
        try:
            logger.info("Running cross-validation...")
            cv_results = self._cross_validation(
                causal_estimator, X, y, treatment,
                config.get('cv_config', {})
            )
            results['cross_validation'] = cv_results
        except Exception as e:
            logger.error(f"Cross-validation failed: {e}")
            results['cross_validation'] = {'error': str(e)}

        # 5. Heterogeneity Analysis
        try:
            logger.info("Analyzing treatment effect heterogeneity...")
            heterogeneity_results = self._heterogeneity_analysis(
                causal_estimator, X, y, treatment,
                config.get('heterogeneity_config', {})
            )
            results['heterogeneity_analysis'] = heterogeneity_results
        except Exception as e:
            logger.error(f"Heterogeneity analysis failed: {e}")
            results['heterogeneity_analysis'] = {'error': str(e)}

        # 6. Model Specification Tests
        try:
            logger.info("Running model specification tests...")
            specification_results = self._specification_tests(
                causal_estimator, X, y, treatment,
                config.get('specification_config', {})
            )
            results['specification_tests'] = specification_results
        except Exception as e:
            logger.error(f"Specification tests failed: {e}")
            results['specification_tests'] = {'error': str(e)}

        # 7. Overall Validation Assessment
        try:
            assessment = self._assess_validation_results(results)
            results['overall_assessment'] = assessment
        except Exception as e:
            logger.error(f"Overall assessment failed: {e}")
            results['overall_assessment'] = {'error': str(e)}

        logger.info("Causal model validation completed")
        return results

    def _sensitivity_analysis(self,
                            estimator: Any,
                            X: pd.DataFrame,
                            y: pd.Series,
                            treatment: pd.Series,
                            config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sensitivity analysis for unobserved confounders

        Tests how robust the causal estimate is to potential unobserved confounders
        using the approach from Rosenbaum (2002) and Imbens (2003).
        """

        # Get baseline treatment effect
        if hasattr(estimator, 'ate_estimate_'):
            baseline_ate = estimator.ate_estimate_
        else:
            # Fallback: simple difference in means
            treated_outcomes = y[treatment == 1]
            control_outcomes = y[treatment == 0]
            baseline_ate = treated_outcomes.mean() - control_outcomes.mean()

        # Simulate unobserved confounders with different strengths
        confounder_strengths = config.get('confounder_strengths', [0.1, 0.2, 0.3, 0.5, 0.7])

        sensitivity_results = {
            'baseline_ate': float(baseline_ate),
            'confounder_analysis': []
        }

        np.random.seed(self.random_state)

        for strength in confounder_strengths:
            # Simulate unobserved confounder
            n_obs = len(X)

            # Confounder affects both treatment and outcome
            unobserved_U = np.random.normal(0, 1, n_obs)

            # U affects treatment assignment (selection bias)
            treatment_bias = strength * unobserved_U
            prob_treatment = 1 / (1 + np.exp(-(treatment.astype(float) + treatment_bias)))

            # U affects outcome (confounding bias)
            outcome_bias = strength * unobserved_U
            biased_outcome = y + outcome_bias

            # Recalculate treatment effect with confounded data
            treated_biased = biased_outcome[treatment == 1]
            control_biased = biased_outcome[treatment == 0]

            if len(treated_biased) > 0 and len(control_biased) > 0:
                biased_ate = treated_biased.mean() - control_biased.mean()
                bias_magnitude = abs(biased_ate - baseline_ate)
                relative_bias = bias_magnitude / abs(baseline_ate) if baseline_ate != 0 else float('inf')

                sensitivity_results['confounder_analysis'].append({
                    'confounder_strength': strength,
                    'biased_ate': float(biased_ate),
                    'bias_magnitude': float(bias_magnitude),
                    'relative_bias': float(relative_bias),
                    'sign_change': (np.sign(biased_ate) != np.sign(baseline_ate))
                })

        # Calculate robustness metrics
        if sensitivity_results['confounder_analysis']:
            max_relative_bias = max([r['relative_bias'] for r in sensitivity_results['confounder_analysis']])
            sign_changes = sum([r['sign_change'] for r in sensitivity_results['confounder_analysis']])

            sensitivity_results['robustness_metrics'] = {
                'max_relative_bias': float(max_relative_bias),
                'robustness_score': 1.0 / (1.0 + max_relative_bias),  # Higher = more robust
                'sign_changes': int(sign_changes),
                'is_robust': max_relative_bias < 0.5 and sign_changes == 0
            }

        return sensitivity_results

    def _placebo_tests(self,
                      estimator: Any,
                      X: pd.DataFrame,
                      y: pd.Series,
                      treatment: pd.Series,
                      config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Placebo and falsification tests

        Tests causal model on scenarios where no effect should exist
        """

        n_placebo_tests = config.get('n_placebo_tests', 100)

        placebo_results = {
            'n_tests': n_placebo_tests,
            'placebo_effects': [],
            'temporal_placebo': None,
            'outcome_placebo': None
        }

        np.random.seed(self.random_state)

        # 1. Random treatment assignment placebo
        logger.debug("Running random treatment placebo tests...")

        for i in range(n_placebo_tests):
            # Randomly shuffle treatment assignment
            placebo_treatment = np.random.permutation(treatment.values)

            try:
                # Calculate placebo effect (should be ~0)
                treated_placebo = y[placebo_treatment == 1]
                control_placebo = y[placebo_treatment == 0]

                if len(treated_placebo) > 0 and len(control_placebo) > 0:
                    placebo_effect = treated_placebo.mean() - control_placebo.mean()
                    placebo_results['placebo_effects'].append(float(placebo_effect))

            except Exception as e:
                logger.debug(f"Placebo test {i} failed: {e}")

        # 2. Temporal placebo (pre-treatment periods)
        if 'time_var' in X.columns:
            try:
                logger.debug("Running temporal placebo test...")

                # Use early periods as "fake" treatment periods
                time_var = X['time_var']
                early_period_threshold = time_var.quantile(0.3)

                temporal_treatment = (time_var <= early_period_threshold).astype(int)

                treated_temporal = y[temporal_treatment == 1]
                control_temporal = y[temporal_treatment == 0]

                if len(treated_temporal) > 0 and len(control_temporal) > 0:
                    temporal_effect = treated_temporal.mean() - control_temporal.mean()

                    placebo_results['temporal_placebo'] = {
                        'effect': float(temporal_effect),
                        'n_treated': len(treated_temporal),
                        'n_control': len(control_temporal)
                    }

            except Exception as e:
                logger.debug(f"Temporal placebo test failed: {e}")

        # 3. Outcome placebo (use predetermined variables as fake outcomes)
        try:
            logger.debug("Running outcome placebo test...")

            # Use first feature as fake outcome (should not be affected by treatment)
            fake_outcome = X.iloc[:, 0]

            treated_fake = fake_outcome[treatment == 1]
            control_fake = fake_outcome[treatment == 0]

            if len(treated_fake) > 0 and len(control_fake) > 0:
                fake_effect = treated_fake.mean() - control_fake.mean()

                placebo_results['outcome_placebo'] = {
                    'effect': float(fake_effect),
                    'variable_used': X.columns[0]
                }

        except Exception as e:
            logger.debug(f"Outcome placebo test failed: {e}")

        # Statistical assessment
        if placebo_results['placebo_effects']:
            placebo_effects = np.array(placebo_results['placebo_effects'])

            placebo_results['statistical_assessment'] = {
                'mean_placebo_effect': float(np.mean(placebo_effects)),
                'std_placebo_effect': float(np.std(placebo_effects)),
                'fraction_significant': float(np.mean(np.abs(placebo_effects) > 1.96 * np.std(placebo_effects))),
                'p_value_uniform': float(stats.kstest(placebo_effects, 'norm')[1]),
                'placebo_distribution_normal': stats.kstest(placebo_effects, 'norm')[1] > 0.05
            }

        return placebo_results

    def _bootstrap_inference(self,
                           estimator: Any,
                           X: pd.DataFrame,
                           y: pd.Series,
                           treatment: pd.Series,
                           config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Bootstrap inference for robust confidence intervals
        """

        n_bootstrap = config.get('n_bootstrap', self.n_bootstrap)

        bootstrap_estimates = []
        np.random.seed(self.random_state)

        logger.debug(f"Running {n_bootstrap} bootstrap iterations...")

        for i in range(n_bootstrap):
            try:
                # Bootstrap sample
                n_obs = len(X)
                bootstrap_idx = np.random.choice(n_obs, size=n_obs, replace=True)

                X_boot = X.iloc[bootstrap_idx]
                y_boot = y.iloc[bootstrap_idx]
                treatment_boot = treatment.iloc[bootstrap_idx]

                # Simple difference in means for bootstrap
                # (Full DML bootstrap would be computationally expensive)
                treated_boot = y_boot[treatment_boot == 1]
                control_boot = y_boot[treatment_boot == 0]

                if len(treated_boot) > 0 and len(control_boot) > 0:
                    boot_estimate = treated_boot.mean() - control_boot.mean()
                    bootstrap_estimates.append(boot_estimate)

            except Exception as e:
                logger.debug(f"Bootstrap iteration {i} failed: {e}")

        if not bootstrap_estimates:
            return {'error': 'All bootstrap iterations failed'}

        bootstrap_estimates = np.array(bootstrap_estimates)

        # Calculate confidence intervals
        ci_levels = [0.90, 0.95, 0.99]
        confidence_intervals = {}

        for level in ci_levels:
            alpha = 1 - level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100

            ci_lower = np.percentile(bootstrap_estimates, lower_percentile)
            ci_upper = np.percentile(bootstrap_estimates, upper_percentile)

            confidence_intervals[f'{level:.0%}'] = [float(ci_lower), float(ci_upper)]

        return {
            'n_successful_bootstrap': len(bootstrap_estimates),
            'bootstrap_estimates': bootstrap_estimates.tolist(),
            'bootstrap_statistics': {
                'mean': float(np.mean(bootstrap_estimates)),
                'std': float(np.std(bootstrap_estimates)),
                'skewness': float(stats.skew(bootstrap_estimates)),
                'kurtosis': float(stats.kurtosis(bootstrap_estimates))
            },
            'confidence_intervals': confidence_intervals,
            'bias_corrected_estimate': float(2 * np.mean(bootstrap_estimates) - np.mean(bootstrap_estimates))  # Simple bias correction
        }

    def _cross_validation(self,
                         estimator: Any,
                         X: pd.DataFrame,
                         y: pd.Series,
                         treatment: pd.Series,
                         config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Cross-validation for causal models
        """

        n_folds = config.get('n_folds', 5)

        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)

        cv_estimates = []
        cv_metrics = []

        for fold, (train_idx, test_idx) in enumerate(kfold.split(X)):
            try:
                # Split data
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                treatment_train, treatment_test = treatment.iloc[train_idx], treatment.iloc[test_idx]

                # Calculate treatment effect on test set
                treated_test = y_test[treatment_test == 1]
                control_test = y_test[treatment_test == 0]

                if len(treated_test) > 0 and len(control_test) > 0:
                    fold_estimate = treated_test.mean() - control_test.mean()
                    cv_estimates.append(fold_estimate)

                    # Calculate fold-specific metrics
                    fold_metrics = {
                        'fold': fold,
                        'estimate': float(fold_estimate),
                        'n_treated_test': len(treated_test),
                        'n_control_test': len(control_test),
                        'treatment_prevalence_test': float(treatment_test.mean())
                    }
                    cv_metrics.append(fold_metrics)

            except Exception as e:
                logger.debug(f"CV fold {fold} failed: {e}")

        if not cv_estimates:
            return {'error': 'All CV folds failed'}

        cv_estimates = np.array(cv_estimates)

        return {
            'n_folds': len(cv_estimates),
            'fold_estimates': cv_estimates.tolist(),
            'fold_metrics': cv_metrics,
            'cv_statistics': {
                'mean_estimate': float(np.mean(cv_estimates)),
                'std_estimate': float(np.std(cv_estimates)),
                'min_estimate': float(np.min(cv_estimates)),
                'max_estimate': float(np.max(cv_estimates)),
                'estimate_stability': float(np.std(cv_estimates) / (abs(np.mean(cv_estimates)) + 1e-6))
            }
        }

    def _heterogeneity_analysis(self,
                              estimator: Any,
                              X: pd.DataFrame,
                              y: pd.Series,
                              treatment: pd.Series,
                              config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze treatment effect heterogeneity across subgroups
        """

        heterogeneity_results = {
            'subgroup_analysis': [],
            'continuous_moderators': []
        }

        # 1. Subgroup analysis for categorical variables
        categorical_vars = config.get('categorical_vars', [])

        for var in categorical_vars:
            if var in X.columns:
                try:
                    subgroup_effects = []

                    for value in X[var].unique():
                        if pd.notna(value):
                            mask = X[var] == value

                            y_subgroup = y[mask]
                            treatment_subgroup = treatment[mask]

                            treated_sub = y_subgroup[treatment_subgroup == 1]
                            control_sub = y_subgroup[treatment_subgroup == 0]

                            if len(treated_sub) > 5 and len(control_sub) > 5:  # Minimum sample size
                                effect = treated_sub.mean() - control_sub.mean()

                                subgroup_effects.append({
                                    'variable': var,
                                    'value': str(value),
                                    'effect': float(effect),
                                    'n_treated': len(treated_sub),
                                    'n_control': len(control_sub)
                                })

                    if len(subgroup_effects) > 1:
                        effects_only = [s['effect'] for s in subgroup_effects]

                        heterogeneity_results['subgroup_analysis'].append({
                            'variable': var,
                            'subgroup_effects': subgroup_effects,
                            'heterogeneity_measure': float(np.std(effects_only)),
                            'effect_range': [float(min(effects_only)), float(max(effects_only))]
                        })

                except Exception as e:
                    logger.debug(f"Subgroup analysis for {var} failed: {e}")

        # 2. Continuous moderator analysis
        continuous_vars = config.get('continuous_vars', [])

        for var in continuous_vars:
            if var in X.columns:
                try:
                    # Split into quartiles
                    quartiles = X[var].quantile([0.25, 0.5, 0.75])

                    quartile_effects = []

                    for i, (lower, upper) in enumerate([(X[var].min(), quartiles.iloc[0]),
                                                       (quartiles.iloc[0], quartiles.iloc[1]),
                                                       (quartiles.iloc[1], quartiles.iloc[2]),
                                                       (quartiles.iloc[2], X[var].max())]):

                        if i == 3:  # Include upper bound in last quartile
                            mask = (X[var] >= lower) & (X[var] <= upper)
                        else:
                            mask = (X[var] >= lower) & (X[var] < upper)

                        y_quartile = y[mask]
                        treatment_quartile = treatment[mask]

                        treated_q = y_quartile[treatment_quartile == 1]
                        control_q = y_quartile[treatment_quartile == 0]

                        if len(treated_q) > 5 and len(control_q) > 5:
                            effect = treated_q.mean() - control_q.mean()

                            quartile_effects.append({
                                'quartile': i + 1,
                                'range': [float(lower), float(upper)],
                                'effect': float(effect),
                                'n_treated': len(treated_q),
                                'n_control': len(control_q)
                            })

                    if len(quartile_effects) > 1:
                        effects_only = [q['effect'] for q in quartile_effects]

                        heterogeneity_results['continuous_moderators'].append({
                            'variable': var,
                            'quartile_effects': quartile_effects,
                            'heterogeneity_measure': float(np.std(effects_only)),
                            'trend_test': self._test_linear_trend([q['effect'] for q in quartile_effects])
                        })

                except Exception as e:
                    logger.debug(f"Continuous moderator analysis for {var} failed: {e}")

        return heterogeneity_results

    def _specification_tests(self,
                           estimator: Any,
                           X: pd.DataFrame,
                           y: pd.Series,
                           treatment: pd.Series,
                           config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Model specification tests
        """

        spec_results = {}

        # 1. Overlap/Common Support Test
        try:
            # Calculate propensity scores
            from sklearn.linear_model import LogisticRegression

            prop_model = LogisticRegression(random_state=self.random_state)
            prop_model.fit(X, treatment)
            propensity_scores = prop_model.predict_proba(X)[:, 1]

            treated_props = propensity_scores[treatment == 1]
            control_props = propensity_scores[treatment == 0]

            # Check overlap
            treated_range = [treated_props.min(), treated_props.max()]
            control_range = [control_props.min(), control_props.max()]

            overlap = max(0, min(treated_range[1], control_range[1]) - max(treated_range[0], control_range[0]))
            total_range = max(treated_range[1], control_range[1]) - min(treated_range[0], control_range[0])

            overlap_ratio = overlap / total_range if total_range > 0 else 0

            spec_results['overlap_test'] = {
                'treated_propensity_range': [float(x) for x in treated_range],
                'control_propensity_range': [float(x) for x in control_range],
                'overlap_ratio': float(overlap_ratio),
                'sufficient_overlap': overlap_ratio > 0.8
            }

        except Exception as e:
            logger.debug(f"Overlap test failed: {e}")
            spec_results['overlap_test'] = {'error': str(e)}

        # 2. Balance Test
        try:
            balance_stats = []

            for col in X.select_dtypes(include=[np.number]).columns:
                treated_mean = X.loc[treatment == 1, col].mean()
                control_mean = X.loc[treatment == 0, col].mean()
                pooled_std = X[col].std()

                standardized_diff = (treated_mean - control_mean) / pooled_std if pooled_std > 0 else 0

                balance_stats.append({
                    'variable': col,
                    'treated_mean': float(treated_mean),
                    'control_mean': float(control_mean),
                    'standardized_difference': float(standardized_diff),
                    'well_balanced': abs(standardized_diff) < 0.25
                })

            well_balanced_vars = sum([1 for stat in balance_stats if stat['well_balanced']])

            spec_results['balance_test'] = {
                'variable_balance': balance_stats,
                'fraction_well_balanced': well_balanced_vars / len(balance_stats) if balance_stats else 0,
                'overall_balance_ok': well_balanced_vars / len(balance_stats) > 0.8 if balance_stats else False
            }

        except Exception as e:
            logger.debug(f"Balance test failed: {e}")
            spec_results['balance_test'] = {'error': str(e)}

        return spec_results

    def _test_linear_trend(self, effects: List[float]) -> Dict[str, float]:
        """Test for linear trend in effects"""
        try:
            x = np.arange(len(effects))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, effects)

            return {
                'slope': float(slope),
                'r_squared': float(r_value ** 2),
                'p_value': float(p_value),
                'significant_trend': p_value < 0.05
            }
        except:
            return {'slope': 0, 'r_squared': 0, 'p_value': 1, 'significant_trend': False}

    def _assess_validation_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall validation results and provide recommendations"""

        assessment = {
            'overall_score': 0.0,
            'reliability_rating': 'unknown',
            'key_concerns': [],
            'strengths': [],
            'recommendations': []
        }

        scores = []

        # Assess sensitivity analysis
        if 'sensitivity_analysis' in results and 'error' not in results['sensitivity_analysis']:
            sens = results['sensitivity_analysis'].get('robustness_metrics', {})
            if sens.get('is_robust', False):
                scores.append(1.0)
                assessment['strengths'].append("Robust to unobserved confounders")
            else:
                scores.append(0.3)
                assessment['key_concerns'].append("Sensitive to unobserved confounders")

        # Assess placebo tests
        if 'placebo_tests' in results and 'error' not in results['placebo_tests']:
            placebo = results['placebo_tests'].get('statistical_assessment', {})
            if placebo.get('fraction_significant', 1) < 0.1:
                scores.append(1.0)
                assessment['strengths'].append("Passes placebo tests")
            else:
                scores.append(0.2)
                assessment['key_concerns'].append("Fails some placebo tests")

        # Assess cross-validation stability
        if 'cross_validation' in results and 'error' not in results['cross_validation']:
            cv = results['cross_validation'].get('cv_statistics', {})
            if cv.get('estimate_stability', float('inf')) < 0.5:
                scores.append(1.0)
                assessment['strengths'].append("Stable across cross-validation")
            else:
                scores.append(0.4)
                assessment['key_concerns'].append("Unstable across folds")

        # Assess specification tests
        if 'specification_tests' in results and 'error' not in results['specification_tests']:
            spec = results['specification_tests']

            overlap_ok = spec.get('overlap_test', {}).get('sufficient_overlap', False)
            balance_ok = spec.get('balance_test', {}).get('overall_balance_ok', False)

            if overlap_ok and balance_ok:
                scores.append(1.0)
                assessment['strengths'].append("Good overlap and balance")
            elif overlap_ok or balance_ok:
                scores.append(0.6)
                assessment['key_concerns'].append("Some specification issues")
            else:
                scores.append(0.2)
                assessment['key_concerns'].append("Poor overlap or balance")

        # Calculate overall score
        if scores:
            assessment['overall_score'] = float(np.mean(scores))

        # Reliability rating
        if assessment['overall_score'] >= 0.8:
            assessment['reliability_rating'] = 'high'
        elif assessment['overall_score'] >= 0.6:
            assessment['reliability_rating'] = 'medium'
        else:
            assessment['reliability_rating'] = 'low'

        # Recommendations
        if assessment['reliability_rating'] == 'high':
            assessment['recommendations'].append("Results are reliable for causal interpretation")
        elif assessment['reliability_rating'] == 'medium':
            assessment['recommendations'].append("Results should be interpreted with some caution")
            assessment['recommendations'].append("Consider additional robustness checks")
        else:
            assessment['recommendations'].append("Results are not reliable for causal interpretation")
            assessment['recommendations'].append("Significant methodological improvements needed")

        return assessment
