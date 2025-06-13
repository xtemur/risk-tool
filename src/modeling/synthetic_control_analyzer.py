"""
Synthetic Control Method for Causal Inference

Implementation of synthetic control method to create counterfactual scenarios
for analyzing the causal impact of model recommendations on trading performance.

Based on Abadie & Gardeazabal (2003) and Abadie et al. (2010).
Creates synthetic "control" traders from weighted combinations of untreated traders
to estimate what would have happened without model intervention.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import logging
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class SyntheticControlAnalyzer:
    """
    Synthetic Control Method for trading causal inference

    Creates synthetic counterfactuals by finding optimal weighted combinations
    of control units (traders not following model recommendations) to match
    pre-treatment characteristics of treated units.
    """

    def __init__(self,
                 matching_period_length: int = 30,
                 optimization_method: str = 'SLSQP',
                 random_state: int = 42):
        """
        Initialize Synthetic Control analyzer

        Args:
            matching_period_length: Number of pre-treatment periods for matching
            optimization_method: Optimization method for weight calculation
            random_state: Random seed for reproducibility
        """
        self.matching_period_length = matching_period_length
        self.optimization_method = optimization_method
        self.random_state = random_state

        # Results storage
        self.synthetic_weights_ = {}
        self.synthetic_controls_ = {}
        self.causal_effects_ = {}
        self.is_fitted_ = False

    def fit_synthetic_control(self,
                            outcome_data: pd.DataFrame,
                            treatment_indicators: pd.DataFrame,
                            feature_data: Optional[pd.DataFrame] = None,
                            treatment_start_period: Optional[int] = None) -> 'SyntheticControlAnalyzer':
        """
        Fit synthetic control model

        Args:
            outcome_data: Panel data with outcomes (rows=time, cols=units)
            treatment_indicators: Binary treatment indicators (rows=time, cols=units)
            feature_data: Optional additional features for matching
            treatment_start_period: When treatment began (auto-detect if None)

        Returns:
            Fitted analyzer
        """
        logger.info(f"Fitting synthetic control with {outcome_data.shape[1]} units, {outcome_data.shape[0]} periods")

        # Validate inputs
        self._validate_inputs(outcome_data, treatment_indicators)

        # Detect treatment start if not provided
        if treatment_start_period is None:
            treatment_start_period = self._detect_treatment_start(treatment_indicators)

        self.treatment_start_period_ = treatment_start_period

        # Define pre/post treatment periods
        pre_treatment_end = max(0, treatment_start_period - 1)
        pre_treatment_start = max(0, pre_treatment_end - self.matching_period_length)

        # For each treated unit, find synthetic control
        treated_units = self._identify_treated_units(treatment_indicators, treatment_start_period)

        for unit in treated_units:
            logger.debug(f"Creating synthetic control for unit {unit}")

            # Get control units (never treated or treated at different times)
            control_units = self._identify_control_units(
                treatment_indicators, unit, treatment_start_period
            )

            if len(control_units) < 2:
                logger.warning(f"Insufficient control units for {unit}, skipping")
                continue

            # Extract pre-treatment data for matching
            treated_pre = outcome_data.loc[pre_treatment_start:pre_treatment_end, unit].values
            control_pre = outcome_data.loc[pre_treatment_start:pre_treatment_end, control_units].values

            # Add feature matching if provided
            if feature_data is not None:
                treated_features = feature_data.loc[pre_treatment_start:pre_treatment_end, unit].values
                control_features = feature_data.loc[pre_treatment_start:pre_treatment_end, control_units].values

                # Combine outcome and feature data for matching
                treated_pre = np.concatenate([treated_pre, treated_features])
                control_pre = np.vstack([control_pre, control_features.T])

            # Optimize weights to match pre-treatment characteristics
            weights = self._optimize_weights(treated_pre, control_pre)

            # Create synthetic control time series
            synthetic_control = (outcome_data[control_units] * weights).sum(axis=1)

            # Calculate causal effects
            causal_effect = outcome_data[unit] - synthetic_control

            # Store results
            self.synthetic_weights_[unit] = dict(zip(control_units, weights))
            self.synthetic_controls_[unit] = synthetic_control
            self.causal_effects_[unit] = causal_effect

        self.is_fitted_ = True
        logger.info(f"Synthetic control fitting completed for {len(self.causal_effects_)} treated units")

        return self

    def analyze_trading_impact(self,
                             trader_pnl: pd.DataFrame,
                             model_usage: pd.DataFrame,
                             trader_features: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Analyze causal impact of model usage on trading performance

        Args:
            trader_pnl: Daily PnL by trader (rows=dates, cols=trader_ids)
            model_usage: Binary indicators of model usage (same shape)
            trader_features: Optional trader characteristics for matching

        Returns:
            Comprehensive synthetic control analysis
        """
        logger.info("Analyzing trading impact using synthetic control method")

        # Fit synthetic control
        self.fit_synthetic_control(
            outcome_data=trader_pnl,
            treatment_indicators=model_usage,
            feature_data=trader_features
        )

        if not self.causal_effects_:
            logger.warning("No causal effects estimated, insufficient data")
            return {}

        # Calculate aggregate results
        results = {
            'individual_effects': self._analyze_individual_effects(),
            'aggregate_effects': self._analyze_aggregate_effects(),
            'robustness_tests': self._conduct_robustness_tests(trader_pnl, model_usage),
            'placebo_tests': self._conduct_placebo_tests(trader_pnl, model_usage),
            'interpretation': self._interpret_results()
        }

        return results

    def _validate_inputs(self, outcome_data: pd.DataFrame, treatment_indicators: pd.DataFrame):
        """Validate input data"""
        if outcome_data.shape != treatment_indicators.shape:
            raise ValueError("Outcome data and treatment indicators must have same shape")

        if outcome_data.isnull().sum().sum() > outcome_data.size * 0.1:
            logger.warning("High proportion of missing values in outcome data")

        if not np.all(np.isin(treatment_indicators.values, [0, 1, np.nan])):
            raise ValueError("Treatment indicators must be binary (0/1) or NaN")

    def _detect_treatment_start(self, treatment_indicators: pd.DataFrame) -> int:
        """Auto-detect when treatment started"""
        # Find first period with any treatment
        treatment_counts = treatment_indicators.sum(axis=1)
        first_treatment_idx = treatment_counts[treatment_counts > 0].index

        if len(first_treatment_idx) == 0:
            raise ValueError("No treatment periods found")

        return treatment_indicators.index.get_loc(first_treatment_idx[0])

    def _identify_treated_units(self, treatment_indicators: pd.DataFrame, treatment_start: int) -> List:
        """Identify units that received treatment"""
        post_treatment = treatment_indicators.iloc[treatment_start:]
        treated_units = post_treatment.columns[post_treatment.sum() > 0].tolist()
        return treated_units

    def _identify_control_units(self,
                              treatment_indicators: pd.DataFrame,
                              treated_unit: str,
                              treatment_start: int) -> List:
        """Identify potential control units for a given treated unit"""
        # Units that were never treated or treated at different times
        all_units = treatment_indicators.columns.tolist()
        all_units.remove(treated_unit)

        control_units = []
        for unit in all_units:
            unit_treatment = treatment_indicators[unit]

            # Check if unit was treated in the same period
            if unit_treatment.iloc[treatment_start:].sum() == 0:
                control_units.append(unit)

        return control_units

    def _optimize_weights(self, treated_pre: np.ndarray, control_pre: np.ndarray) -> np.ndarray:
        """Optimize weights to minimize pre-treatment differences"""
        n_controls = control_pre.shape[1] if control_pre.ndim > 1 else 1

        if n_controls == 1:
            return np.array([1.0])

        # Objective function: minimize squared differences
        def objective(weights):
            if control_pre.ndim == 1:
                synthetic = control_pre * weights[0]
            else:
                synthetic = control_pre @ weights
            return np.sum((treated_pre - synthetic) ** 2)

        # Constraints: weights sum to 1 and are non-negative
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        bounds = [(0, 1) for _ in range(n_controls)]

        # Initial guess: equal weights
        initial_weights = np.ones(n_controls) / n_controls

        # Optimize
        try:
            result = minimize(
                objective,
                initial_weights,
                method=self.optimization_method,
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )

            if result.success:
                return result.x
            else:
                logger.warning(f"Optimization failed: {result.message}")
                return initial_weights

        except Exception as e:
            logger.warning(f"Optimization error: {e}, using equal weights")
            return initial_weights

    def _analyze_individual_effects(self) -> Dict[str, Any]:
        """Analyze causal effects for individual units"""
        individual_results = {}

        for unit, causal_effect in self.causal_effects_.items():
            # Split into pre/post treatment
            pre_treatment = causal_effect.iloc[:self.treatment_start_period_]
            post_treatment = causal_effect.iloc[self.treatment_start_period_:]

            # Calculate metrics
            individual_results[unit] = {
                'pre_treatment_effect': {
                    'mean': float(pre_treatment.mean()),
                    'std': float(pre_treatment.std()),
                    'rmse': float(np.sqrt(np.mean(pre_treatment ** 2)))
                },
                'post_treatment_effect': {
                    'mean': float(post_treatment.mean()),
                    'std': float(post_treatment.std()),
                    'total': float(post_treatment.sum()),
                    'cumulative': post_treatment.cumsum().tolist()
                },
                'statistical_tests': {
                    'pre_post_difference': float(post_treatment.mean() - pre_treatment.mean()),
                    't_statistic': self._calculate_t_statistic(pre_treatment, post_treatment),
                    'effect_size': float(post_treatment.mean() / (pre_treatment.std() + 1e-6))
                }
            }

        return individual_results

    def _analyze_aggregate_effects(self) -> Dict[str, Any]:
        """Analyze aggregate causal effects across all treated units"""
        if not self.causal_effects_:
            return {}

        # Combine effects across units
        all_effects = pd.concat(self.causal_effects_.values(), axis=1)

        # Average effect across units
        avg_effect = all_effects.mean(axis=1)

        # Split into pre/post treatment
        pre_treatment = avg_effect.iloc[:self.treatment_start_period_]
        post_treatment = avg_effect.iloc[self.treatment_start_period_:]

        return {
            'average_treatment_effect': {
                'pre_treatment_mean': float(pre_treatment.mean()),
                'post_treatment_mean': float(post_treatment.mean()),
                'difference': float(post_treatment.mean() - pre_treatment.mean()),
                'cumulative_effect': float(post_treatment.sum()),
                'relative_improvement': float(post_treatment.mean() / abs(pre_treatment.mean()) * 100) if pre_treatment.mean() != 0 else 0
            },
            'heterogeneity': {
                'std_across_units': float(all_effects.iloc[self.treatment_start_period_:].mean().std()),
                'range': [
                    float(all_effects.iloc[self.treatment_start_period_:].mean().min()),
                    float(all_effects.iloc[self.treatment_start_period_:].mean().max())
                ],
                'units_with_positive_effect': int((all_effects.iloc[self.treatment_start_period_:].mean() > 0).sum()),
                'units_with_negative_effect': int((all_effects.iloc[self.treatment_start_period_:].mean() < 0).sum())
            },
            'time_varying_effects': {
                'daily_effects': post_treatment.tolist(),
                'cumulative_effects': post_treatment.cumsum().tolist(),
                'effect_stability': float(post_treatment.std() / (abs(post_treatment.mean()) + 1e-6))
            }
        }

    def _conduct_robustness_tests(self,
                                trader_pnl: pd.DataFrame,
                                model_usage: pd.DataFrame) -> Dict[str, Any]:
        """Conduct robustness tests for synthetic control estimates"""

        robustness_results = {}

        # Test 1: Alternative matching periods
        for period_length in [15, 45, 60]:
            if period_length != self.matching_period_length:
                try:
                    alt_analyzer = SyntheticControlAnalyzer(
                        matching_period_length=period_length,
                        random_state=self.random_state
                    )
                    alt_analyzer.fit_synthetic_control(trader_pnl, model_usage)

                    if alt_analyzer.causal_effects_:
                        alt_effects = pd.concat(alt_analyzer.causal_effects_.values(), axis=1)
                        alt_post_effect = alt_effects.iloc[self.treatment_start_period_:].mean().mean()

                        robustness_results[f'matching_period_{period_length}'] = {
                            'effect_estimate': float(alt_post_effect),
                            'correlation_with_main': float(self._calculate_correlation(alt_effects))
                        }
                except Exception as e:
                    logger.warning(f"Robustness test with period {period_length} failed: {e}")

        # Test 2: Leave-one-out analysis (exclude each control unit)
        loo_effects = []
        for unit in self.causal_effects_.keys():
            try:
                # Get original control units
                original_controls = list(self.synthetic_weights_[unit].keys())

                if len(original_controls) > 2:  # Need at least 2 controls
                    for exclude_control in original_controls:
                        remaining_controls = [c for c in original_controls if c != exclude_control]

                        # Recalculate with reduced control set
                        control_data = trader_pnl[remaining_controls]
                        treated_data = trader_pnl[unit]

                        # Simple equal weighting for LOO test
                        weights = np.ones(len(remaining_controls)) / len(remaining_controls)
                        synthetic = (control_data * weights).sum(axis=1)
                        effect = treated_data - synthetic

                        post_effect = effect.iloc[self.treatment_start_period_:].mean()
                        loo_effects.append(post_effect)
            except Exception as e:
                logger.debug(f"LOO test failed for unit {unit}: {e}")

        if loo_effects:
            robustness_results['leave_one_out'] = {
                'mean_effect': float(np.mean(loo_effects)),
                'std_effect': float(np.std(loo_effects)),
                'min_effect': float(np.min(loo_effects)),
                'max_effect': float(np.max(loo_effects))
            }

        return robustness_results

    def _conduct_placebo_tests(self,
                             trader_pnl: pd.DataFrame,
                             model_usage: pd.DataFrame) -> Dict[str, Any]:
        """Conduct placebo tests by randomly assigning treatment"""

        placebo_results = {}
        np.random.seed(self.random_state)

        n_placebo_tests = 100
        placebo_effects = []

        for i in range(n_placebo_tests):
            try:
                # Create random treatment assignment
                placebo_treatment = model_usage.copy()

                # Randomly shuffle treatment assignment across units
                for col in placebo_treatment.columns:
                    placebo_treatment[col] = np.random.permutation(placebo_treatment[col].values)

                # Fit synthetic control with placebo treatment
                placebo_analyzer = SyntheticControlAnalyzer(
                    matching_period_length=self.matching_period_length,
                    random_state=self.random_state + i
                )
                placebo_analyzer.fit_synthetic_control(trader_pnl, placebo_treatment)

                if placebo_analyzer.causal_effects_:
                    placebo_effect_data = pd.concat(placebo_analyzer.causal_effects_.values(), axis=1)
                    placebo_post_effect = placebo_effect_data.iloc[self.treatment_start_period_:].mean().mean()
                    placebo_effects.append(placebo_post_effect)

            except Exception as e:
                logger.debug(f"Placebo test {i} failed: {e}")

        if placebo_effects:
            # Compare actual effect to placebo distribution
            actual_effect = self._get_actual_average_effect()

            placebo_effects = np.array(placebo_effects)
            p_value = np.mean(np.abs(placebo_effects) >= np.abs(actual_effect))

            placebo_results = {
                'n_placebo_tests': len(placebo_effects),
                'placebo_effects_summary': {
                    'mean': float(np.mean(placebo_effects)),
                    'std': float(np.std(placebo_effects)),
                    'percentiles': {
                        '5th': float(np.percentile(placebo_effects, 5)),
                        '95th': float(np.percentile(placebo_effects, 95))
                    }
                },
                'actual_vs_placebo': {
                    'actual_effect': float(actual_effect),
                    'rank_among_placebos': int(np.sum(placebo_effects <= actual_effect)),
                    'p_value': float(p_value),
                    'is_significant': p_value < 0.05
                }
            }

        return placebo_results

    def _calculate_t_statistic(self, pre: pd.Series, post: pd.Series) -> float:
        """Calculate t-statistic for difference in means"""
        try:
            from scipy import stats
            t_stat, _ = stats.ttest_ind(post, pre, equal_var=False)
            return float(t_stat)
        except:
            return 0.0

    def _calculate_correlation(self, alt_effects: pd.DataFrame) -> float:
        """Calculate correlation with main results"""
        try:
            main_effects = pd.concat(self.causal_effects_.values(), axis=1)

            # Compare post-treatment periods
            main_post = main_effects.iloc[self.treatment_start_period_:].mean()
            alt_post = alt_effects.iloc[self.treatment_start_period_:].mean()

            return np.corrcoef(main_post, alt_post)[0, 1]
        except:
            return 0.0

    def _get_actual_average_effect(self) -> float:
        """Get the actual average treatment effect from main analysis"""
        if not self.causal_effects_:
            return 0.0

        all_effects = pd.concat(self.causal_effects_.values(), axis=1)
        post_effects = all_effects.iloc[self.treatment_start_period_:]
        return float(post_effects.mean().mean())

    def _interpret_results(self) -> Dict[str, str]:
        """Generate interpretation of synthetic control results"""
        if not self.causal_effects_:
            return {'error': 'No results to interpret'}

        avg_effect = self._get_actual_average_effect()

        # Determine effect size and direction
        direction = "positive" if avg_effect > 0 else "negative"
        magnitude = "small"

        # Compare to typical volatility
        all_effects = pd.concat(self.causal_effects_.values(), axis=1)
        volatility = all_effects.std().mean()

        if abs(avg_effect) > 0.2 * volatility:
            magnitude = "moderate"
        if abs(avg_effect) > 0.5 * volatility:
            magnitude = "large"

        return {
            'main_finding': f"Synthetic control analysis reveals a {magnitude} {direction} causal effect of ${avg_effect:.2f} per day from following model recommendations",
            'method_explanation': "Synthetic control creates counterfactual outcomes by optimally weighting similar untreated traders to match pre-treatment characteristics",
            'causal_interpretation': f"The estimated effect represents what traders would have gained/lost if they had followed model recommendations compared to their synthetic counterfactual",
            'reliability': f"Results based on {len(self.causal_effects_)} treated traders with synthetic controls created from weighted combinations of similar untreated traders"
        }
