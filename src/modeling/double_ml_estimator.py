"""
Double Machine Learning (DML) Estimator

Implementation of Double Machine Learning framework for robust causal effect estimation.
Based on Chernozhukov et al. (2018) "Double/Debiased Machine Learning for Treatment and Structural Parameters"

Key features:
- Cross-fitting to reduce overfitting bias
- Orthogonal moment conditions to reduce regularization bias
- Support for heterogeneous treatment effects
- Robust standard errors and confidence intervals
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union
import logging
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import mean_squared_error, log_loss
import warnings

logger = logging.getLogger(__name__)


class DoubleMachineLearningEstimator:
    """
    Double Machine Learning estimator for causal inference

    Estimates average treatment effect (ATE) and conditional average treatment effect (CATE)
    using cross-fitting and orthogonal moments to reduce bias from regularization and overfitting.
    """

    def __init__(self,
                 outcome_learner: Optional[BaseEstimator] = None,
                 treatment_learner: Optional[BaseEstimator] = None,
                 n_folds: int = 5,
                 random_state: int = 42):
        """
        Initialize DML estimator

        Args:
            outcome_learner: ML model for outcome regression (Y ~ X)
            treatment_learner: ML model for treatment propensity (T ~ X)
            n_folds: Number of cross-validation folds
            random_state: Random seed for reproducibility
        """
        self.outcome_learner = outcome_learner or RandomForestRegressor(
            n_estimators=100, random_state=random_state, n_jobs=-1
        )
        self.treatment_learner = treatment_learner or RandomForestClassifier(
            n_estimators=100, random_state=random_state, n_jobs=-1
        )
        self.n_folds = n_folds
        self.random_state = random_state

        # Results storage
        self.ate_estimate_ = None
        self.ate_se_ = None
        self.ate_ci_ = None
        self.cate_estimates_ = None
        self.fold_results_ = []
        self.is_fitted_ = False

    def fit(self,
            X: pd.DataFrame,
            y: pd.Series,
            treatment: pd.Series,
            sample_weights: Optional[pd.Series] = None) -> 'DoubleMachineLearningEstimator':
        """
        Fit DML estimator using cross-fitting

        Args:
            X: Covariate features
            y: Outcome variable
            treatment: Binary treatment indicator (0/1)
            sample_weights: Optional sample weights

        Returns:
            Fitted estimator
        """
        logger.info(f"Fitting DML estimator with {len(X)} samples, {X.shape[1]} features")

        # Validate inputs
        X, y, treatment = self._validate_inputs(X, y, treatment)

        # Initialize cross-validation
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        # Storage for cross-fitting results
        residuals_y = np.zeros(len(y))
        residuals_t = np.zeros(len(treatment))
        treatment_effects = np.zeros(len(y))

        self.fold_results_ = []

        # Cross-fitting procedure
        for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X)):
            logger.debug(f"Processing fold {fold_idx + 1}/{self.n_folds}")

            # Split data
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            t_train, t_test = treatment.iloc[train_idx], treatment.iloc[test_idx]

            # Fit outcome model: E[Y|X]
            outcome_model = clone(self.outcome_learner)
            outcome_model.fit(X_train, y_train)
            y_pred = outcome_model.predict(X_test)

            # Fit treatment model: E[T|X] (propensity score)
            treatment_model = clone(self.treatment_learner)
            treatment_model.fit(X_train, t_train)

            if hasattr(treatment_model, 'predict_proba'):
                t_pred = treatment_model.predict_proba(X_test)[:, 1]
            else:
                t_pred = treatment_model.predict(X_test)

            # Calculate residuals (orthogonal components)
            residuals_y[test_idx] = y_test - y_pred
            residuals_t[test_idx] = t_test - t_pred

            # Calculate individual treatment effects (for CATE)
            # This is a simplified CATE estimate - could be enhanced with more sophisticated methods
            individual_effects = np.where(
                np.abs(residuals_t[test_idx]) > 1e-6,
                residuals_y[test_idx] / residuals_t[test_idx],
                0
            )
            treatment_effects[test_idx] = individual_effects

            # Store fold metrics
            fold_result = {
                'fold': fold_idx,
                'outcome_mse': mean_squared_error(y_test, y_pred),
                'treatment_score': self._calculate_treatment_score(t_test, t_pred),
                'n_treated': np.sum(t_test),
                'n_control': np.sum(1 - t_test)
            }
            self.fold_results_.append(fold_result)

        # Calculate Average Treatment Effect (ATE) using orthogonal moments
        # ATE = E[ψ(W, θ)] where ψ is the orthogonal moment function
        valid_residuals = np.abs(residuals_t) > 1e-6  # Avoid division by zero

        if np.sum(valid_residuals) > 0:
            # Orthogonal moment: ψ(W,θ) = (Y - m(X))(T - e(X)) / Var(T|X) - θ(T - e(X))
            # Simplified version: weighted average of residual ratios
            weights = np.abs(residuals_t[valid_residuals])
            weights = weights / np.sum(weights)  # Normalize weights

            self.ate_estimate_ = np.average(
                residuals_y[valid_residuals] / residuals_t[valid_residuals],
                weights=weights
            )
        else:
            logger.warning("No valid residuals for ATE calculation, using fallback method")
            # Fallback: simple difference in means
            treated_idx = treatment == 1
            control_idx = treatment == 0

            if np.sum(treated_idx) > 0 and np.sum(control_idx) > 0:
                self.ate_estimate_ = np.mean(y[treated_idx]) - np.mean(y[control_idx])
            else:
                self.ate_estimate_ = 0.0

        # Calculate standard error using influence function
        self.ate_se_ = self._calculate_standard_error(residuals_y, residuals_t, treatment)

        # Calculate confidence interval
        self.ate_ci_ = self._calculate_confidence_interval(self.ate_estimate_, self.ate_se_)

        # Store CATE estimates
        self.cate_estimates_ = treatment_effects

        self.is_fitted_ = True

        logger.info(f"DML fitting completed. ATE: {self.ate_estimate_:.4f} ± {self.ate_se_:.4f}")

        return self

    def predict_treatment_effect(self, X: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Predict conditional average treatment effects (CATE)

        Args:
            X: Features for prediction (if None, returns training CATE estimates)

        Returns:
            Array of treatment effect predictions
        """
        if not self.is_fitted_:
            raise ValueError("Estimator must be fitted before prediction")

        if X is None:
            return self.cate_estimates_

        # For out-of-sample prediction, we would need a more sophisticated CATE model
        # This is a simplified implementation
        logger.warning("Out-of-sample CATE prediction not fully implemented")
        return np.full(len(X), self.ate_estimate_)

    def get_results(self) -> Dict[str, Any]:
        """
        Get comprehensive DML results

        Returns:
            Dictionary with all estimation results
        """
        if not self.is_fitted_:
            raise ValueError("Estimator must be fitted first")

        # Calculate additional diagnostics
        fold_metrics = pd.DataFrame(self.fold_results_)

        return {
            'ate_estimate': float(self.ate_estimate_),
            'ate_standard_error': float(self.ate_se_),
            'ate_confidence_interval': [float(x) for x in self.ate_ci_],
            'ate_t_statistic': float(self.ate_estimate_ / self.ate_se_) if self.ate_se_ > 0 else 0,
            'ate_p_value': self._calculate_p_value(self.ate_estimate_, self.ate_se_),
            'is_significant': abs(self.ate_estimate_ / self.ate_se_) > 1.96 if self.ate_se_ > 0 else False,
            'cate_summary': {
                'mean': float(np.mean(self.cate_estimates_)),
                'std': float(np.std(self.cate_estimates_)),
                'min': float(np.min(self.cate_estimates_)),
                'max': float(np.max(self.cate_estimates_)),
                'heterogeneity_measure': float(np.std(self.cate_estimates_) / (abs(np.mean(self.cate_estimates_)) + 1e-6))
            },
            'cross_validation_metrics': {
                'mean_outcome_mse': float(fold_metrics['outcome_mse'].mean()),
                'mean_treatment_score': float(fold_metrics['treatment_score'].mean()),
                'fold_results': self.fold_results_
            },
            'model_info': {
                'outcome_learner': str(type(self.outcome_learner).__name__),
                'treatment_learner': str(type(self.treatment_learner).__name__),
                'n_folds': self.n_folds,
                'n_observations': len(self.cate_estimates_)
            }
        }

    def _validate_inputs(self, X: pd.DataFrame, y: pd.Series, treatment: pd.Series) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Validate and clean inputs"""
        if len(X) != len(y) or len(X) != len(treatment):
            raise ValueError("X, y, and treatment must have same length")

        if not np.all(np.isin(treatment.unique(), [0, 1])):
            raise ValueError("Treatment must be binary (0/1)")

        # Remove rows with missing values
        valid_idx = ~(X.isnull().any(axis=1) | y.isnull() | treatment.isnull())

        if not valid_idx.all():
            logger.warning(f"Removing {(~valid_idx).sum()} rows with missing values")
            X = X[valid_idx]
            y = y[valid_idx]
            treatment = treatment[valid_idx]

        return X, y, treatment

    def _calculate_treatment_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate treatment model performance score"""
        try:
            if hasattr(self.treatment_learner, 'predict_proba'):
                # For probabilistic classifiers, use log-loss
                return -log_loss(y_true, y_pred)
            else:
                # For deterministic classifiers, use accuracy
                return np.mean(y_true == (y_pred > 0.5))
        except:
            return 0.0

    def _calculate_standard_error(self, residuals_y: np.ndarray, residuals_t: np.ndarray, treatment: np.ndarray) -> float:
        """Calculate robust standard error using influence function"""
        try:
            # Simplified robust standard error calculation
            # In practice, this should use the full influence function theory
            valid_mask = np.abs(residuals_t) > 1e-6

            if np.sum(valid_mask) < 2:
                return float('inf')

            # Calculate influence function components
            influence_scores = np.where(
                valid_mask,
                residuals_y / residuals_t - self.ate_estimate_,
                0
            )

            # Robust variance estimate
            variance = np.var(influence_scores[valid_mask]) / np.sum(valid_mask)

            return np.sqrt(variance)

        except Exception as e:
            logger.warning(f"Error calculating standard error: {e}")
            return float('inf')

    def _calculate_confidence_interval(self, estimate: float, se: float, alpha: float = 0.05) -> Tuple[float, float]:
        """Calculate confidence interval"""
        if se == float('inf') or se == 0:
            return (estimate, estimate)

        z_score = 1.96  # 95% confidence interval
        margin = z_score * se

        return (estimate - margin, estimate + margin)

    def _calculate_p_value(self, estimate: float, se: float) -> float:
        """Calculate two-tailed p-value"""
        if se == 0 or se == float('inf'):
            return 1.0

        from scipy import stats
        t_stat = estimate / se
        return 2 * (1 - stats.norm.cdf(abs(t_stat)))


class TradingDMLAnalyzer:
    """
    Specialized DML analyzer for trading model causal impact

    Applies DML to estimate causal impact of following model recommendations
    on trading performance.
    """

    def __init__(self, dml_estimator: Optional[DoubleMachineLearningEstimator] = None):
        """
        Initialize trading DML analyzer

        Args:
            dml_estimator: Pre-configured DML estimator (optional)
        """
        self.dml_estimator = dml_estimator or DoubleMachineLearningEstimator()
        self.results_ = None

    def analyze_model_impact(self,
                           actual_pnl: pd.Series,
                           predicted_pnl: pd.Series,
                           features: pd.DataFrame,
                           model_confidence: Optional[pd.Series] = None,
                           trader_id: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Analyze causal impact of model recommendations using DML

        Args:
            actual_pnl: Actual trading PnL
            predicted_pnl: Model predicted PnL
            features: Trading features (market conditions, trader characteristics, etc.)
            model_confidence: Optional model confidence scores
            trader_id: Optional trader identifiers for heterogeneity analysis

        Returns:
            Comprehensive DML causal analysis results
        """
        logger.info("Starting DML analysis of model impact on trading performance")

        # Create treatment variable: whether model suggested positive return
        treatment = (predicted_pnl > 0).astype(int)

        # Outcome variable
        outcome = actual_pnl

        # Covariates (control for confounders)
        X = features.copy()

        # Add additional controls
        if model_confidence is not None:
            X['model_confidence'] = model_confidence

        if trader_id is not None:
            # Add trader fixed effects (dummy variables)
            trader_dummies = pd.get_dummies(trader_id, prefix='trader')
            X = pd.concat([X, trader_dummies], axis=1)

        # Add lagged performance controls
        X['lagged_pnl'] = actual_pnl.shift(1).fillna(0)
        X['lagged_volatility'] = actual_pnl.rolling(5).std().fillna(actual_pnl.std())

        # Fit DML estimator
        self.dml_estimator.fit(X, outcome, treatment)

        # Get results
        dml_results = self.dml_estimator.get_results()

        # Calculate additional trading-specific metrics
        trading_metrics = self._calculate_trading_metrics(
            actual_pnl, predicted_pnl, treatment, dml_results['ate_estimate']
        )

        # Combine results
        self.results_ = {
            'dml_causal_effect': dml_results,
            'trading_metrics': trading_metrics,
            'interpretation': self._interpret_results(dml_results, trading_metrics)
        }

        logger.info(f"DML analysis completed. Causal effect: {dml_results['ate_estimate']:.4f}")

        return self.results_

    def _calculate_trading_metrics(self,
                                 actual_pnl: pd.Series,
                                 predicted_pnl: pd.Series,
                                 treatment: pd.Series,
                                 causal_effect: float) -> Dict[str, Any]:
        """Calculate trading-specific metrics"""

        # Basic performance metrics
        total_actual = actual_pnl.sum()
        total_predicted = predicted_pnl.sum()

        # Treatment group analysis
        treated_mask = treatment == 1
        control_mask = treatment == 0

        treated_pnl = actual_pnl[treated_mask].sum() if treated_mask.any() else 0
        control_pnl = actual_pnl[control_mask].sum() if control_mask.any() else 0

        # Calculate implied total causal impact
        n_treated = treated_mask.sum()
        total_causal_impact = causal_effect * n_treated

        return {
            'baseline_performance': {
                'total_pnl': float(total_actual),
                'mean_daily_pnl': float(actual_pnl.mean()),
                'volatility': float(actual_pnl.std()),
                'sharpe_ratio': float(actual_pnl.mean() / (actual_pnl.std() + 1e-6) * np.sqrt(252))
            },
            'treatment_analysis': {
                'n_positive_predictions': int(n_treated),
                'n_negative_predictions': int((treatment == 0).sum()),
                'treatment_frequency': float(treated_mask.mean()),
                'treated_total_pnl': float(treated_pnl),
                'control_total_pnl': float(control_pnl),
                'naive_difference': float(treated_pnl / max(n_treated, 1) - control_pnl / max((treatment == 0).sum(), 1))
            },
            'causal_impact': {
                'per_day_effect': float(causal_effect),
                'total_effect': float(total_causal_impact),
                'relative_improvement': float(total_causal_impact / abs(total_actual) * 100) if total_actual != 0 else 0,
                'counterfactual_total_pnl': float(total_actual + total_causal_impact)
            }
        }

    def _interpret_results(self, dml_results: Dict[str, Any], trading_metrics: Dict[str, Any]) -> Dict[str, str]:
        """Generate interpretation of results"""

        ate = dml_results['ate_estimate']
        is_sig = dml_results['is_significant']
        p_value = dml_results['ate_p_value']

        # Effect size interpretation
        baseline_vol = trading_metrics['baseline_performance']['volatility']
        effect_size_interpretation = "small"
        if abs(ate) > 0.2 * baseline_vol:
            effect_size_interpretation = "medium"
        if abs(ate) > 0.5 * baseline_vol:
            effect_size_interpretation = "large"

        # Direction
        direction = "positive" if ate > 0 else "negative"

        # Significance
        significance = "statistically significant" if is_sig else "not statistically significant"

        return {
            'main_finding': f"Following model recommendations has a {direction} causal effect of ${ate:.2f} per trading day",
            'statistical_significance': f"The effect is {significance} (p-value: {p_value:.4f})",
            'effect_size': f"The effect size is {effect_size_interpretation} relative to typical daily volatility",
            'business_impact': f"Total estimated causal impact: ${trading_metrics['causal_impact']['total_effect']:.2f} ({trading_metrics['causal_impact']['relative_improvement']:+.1f}%)",
            'confidence': f"95% confidence interval: [${dml_results['ate_confidence_interval'][0]:.2f}, ${dml_results['ate_confidence_interval'][1]:.2f}]"
        }
