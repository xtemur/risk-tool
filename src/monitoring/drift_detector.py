"""
Drift Detector
Detects data drift, concept drift, and model degradation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime
import logging
from scipy import stats
from scipy.spatial.distance import jensenshannon
import warnings

from src.core.constants import TradingConstants as TC

warnings.filterwarnings('ignore', category=RuntimeWarning)
logger = logging.getLogger(__name__)


@dataclass
class DriftResult:
    """Result of drift detection"""
    feature_name: str
    drift_score: float
    p_value: float
    is_drifted: bool
    drift_type: str  # 'distributional', 'statistical', 'magnitude'
    reference_stats: Dict[str, float]
    current_stats: Dict[str, float]
    threshold: float


class DriftDetector:
    """
    Comprehensive drift detection for trading systems
    Monitors data drift, concept drift, and performance degradation
    """

    def __init__(self,
                 confidence_level: float = 0.95,
                 psi_threshold: float = 0.2,
                 kl_threshold: float = 0.1,
                 js_threshold: float = 0.1,
                 ks_threshold: float = 0.05):
        """
        Initialize drift detector

        Args:
            confidence_level: Confidence level for statistical tests
            psi_threshold: Population Stability Index threshold
            kl_threshold: KL divergence threshold
            js_threshold: Jensen-Shannon divergence threshold
            ks_threshold: Kolmogorov-Smirnov test threshold
        """
        self.confidence_level = confidence_level
        self.psi_threshold = psi_threshold
        self.kl_threshold = kl_threshold
        self.js_threshold = js_threshold
        self.ks_threshold = ks_threshold

        # Reference distributions
        self.reference_data = None
        self.reference_stats = {}

    def set_reference(self, reference_data: pd.DataFrame):
        """
        Set reference data for drift detection

        Args:
            reference_data: Reference DataFrame (training data)
        """
        self.reference_data = reference_data
        self.reference_stats = self._calculate_statistics(reference_data)
        logger.info(f"Reference data set with {len(reference_data)} samples, {len(reference_data.columns)} features")

    def detect_drift(self,
                    current_data: pd.DataFrame,
                    features: Optional[List[str]] = None,
                    methods: List[str] = None) -> Dict[str, DriftResult]:
        """
        Detect drift in current data compared to reference

        Args:
            current_data: Current data to check for drift
            features: List of features to check (None for all)
            methods: List of methods to use (None for all)

        Returns:
            Dictionary mapping feature names to drift results
        """
        if self.reference_data is None:
            raise ValueError("Reference data not set. Call set_reference() first.")

        # Default to all numeric features
        if features is None:
            numeric_cols = current_data.select_dtypes(include=[np.number]).columns
            features = [col for col in numeric_cols if col in self.reference_data.columns]

        # Default methods
        if methods is None:
            methods = ['psi', 'ks', 'js', 'magnitude', 'mean_std']

        drift_results = {}

        for feature in features:
            if feature not in current_data.columns:
                continue

            # Get feature data
            ref_values = self.reference_data[feature].dropna()
            curr_values = current_data[feature].dropna()

            if len(ref_values) == 0 or len(curr_values) == 0:
                continue

            # Run drift detection methods
            drift_scores = {}

            if 'psi' in methods:
                drift_scores['psi'] = self._calculate_psi(ref_values, curr_values)

            if 'ks' in methods:
                drift_scores['ks'] = self._calculate_ks_test(ref_values, curr_values)

            if 'js' in methods:
                drift_scores['js'] = self._calculate_js_divergence(ref_values, curr_values)

            if 'magnitude' in methods:
                drift_scores['magnitude'] = self._calculate_magnitude_drift(ref_values, curr_values)

            if 'mean_std' in methods:
                drift_scores['mean_std'] = self._calculate_mean_std_drift(ref_values, curr_values)

            # Determine if drifted
            is_drifted, drift_type, score, p_value, threshold = self._evaluate_drift(drift_scores)

            # Create result
            drift_results[feature] = DriftResult(
                feature_name=feature,
                drift_score=score,
                p_value=p_value,
                is_drifted=is_drifted,
                drift_type=drift_type,
                reference_stats={
                    'mean': float(ref_values.mean()),
                    'std': float(ref_values.std()),
                    'min': float(ref_values.min()),
                    'max': float(ref_values.max()),
                    'q25': float(ref_values.quantile(0.25)),
                    'q75': float(ref_values.quantile(0.75))
                },
                current_stats={
                    'mean': float(curr_values.mean()),
                    'std': float(curr_values.std()),
                    'min': float(curr_values.min()),
                    'max': float(curr_values.max()),
                    'q25': float(curr_values.quantile(0.25)),
                    'q75': float(curr_values.quantile(0.75))
                },
                threshold=threshold
            )

        return drift_results

    def _calculate_psi(self, reference: pd.Series, current: pd.Series) -> Tuple[float, float]:
        """
        Calculate Population Stability Index (PSI)

        PSI = Σ (current% - reference%) * ln(current% / reference%)
        """
        # Create bins based on reference data
        n_bins = min(10, int(np.sqrt(len(reference))))
        _, bin_edges = np.histogram(reference, bins=n_bins)

        # Ensure edge cases are covered
        bin_edges[0] = min(reference.min(), current.min()) - 0.001
        bin_edges[-1] = max(reference.max(), current.max()) + 0.001

        # Calculate distributions
        ref_hist, _ = np.histogram(reference, bins=bin_edges)
        curr_hist, _ = np.histogram(current, bins=bin_edges)

        # Convert to proportions
        ref_prop = ref_hist / ref_hist.sum()
        curr_prop = curr_hist / curr_hist.sum()

        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        ref_prop = ref_prop + epsilon
        curr_prop = curr_prop + epsilon

        # Calculate PSI
        psi = np.sum((curr_prop - ref_prop) * np.log(curr_prop / ref_prop))

        # Convert to p-value approximation
        # PSI doesn't have a standard p-value, so we use a heuristic
        p_value = 1.0 if psi < self.psi_threshold else 0.0

        return psi, p_value

    def _calculate_ks_test(self, reference: pd.Series, current: pd.Series) -> Tuple[float, float]:
        """
        Kolmogorov-Smirnov test for distributional drift
        """
        ks_stat, p_value = stats.ks_2samp(reference, current)
        return ks_stat, p_value

    def _calculate_js_divergence(self, reference: pd.Series, current: pd.Series) -> Tuple[float, float]:
        """
        Jensen-Shannon divergence for distributional drift
        """
        # Create histograms
        n_bins = min(20, int(np.sqrt(len(reference))))

        # Common range
        min_val = min(reference.min(), current.min())
        max_val = max(reference.max(), current.max())
        bins = np.linspace(min_val, max_val, n_bins + 1)

        # Calculate distributions
        ref_hist, _ = np.histogram(reference, bins=bins, density=True)
        curr_hist, _ = np.histogram(current, bins=bins, density=True)

        # Normalize
        ref_hist = ref_hist / ref_hist.sum()
        curr_hist = curr_hist / curr_hist.sum()

        # Add small epsilon
        epsilon = 1e-10
        ref_hist = ref_hist + epsilon
        curr_hist = curr_hist + epsilon

        # Calculate JS divergence
        js_div = jensenshannon(ref_hist, curr_hist)

        # Convert to p-value approximation
        p_value = 1.0 if js_div < self.js_threshold else 0.0

        return js_div, p_value

    def _calculate_magnitude_drift(self, reference: pd.Series, current: pd.Series) -> Tuple[float, float]:
        """
        Calculate magnitude of change in key statistics
        """
        ref_mean = reference.mean()
        curr_mean = current.mean()
        ref_std = reference.std()
        curr_std = current.std()

        # Relative change in mean
        mean_change = abs(curr_mean - ref_mean) / (abs(ref_mean) + TC.MIN_VARIANCE)

        # Relative change in std
        std_change = abs(curr_std - ref_std) / (ref_std + TC.MIN_VARIANCE)

        # Combined magnitude score
        magnitude = np.sqrt(mean_change**2 + std_change**2)

        # Threshold for significant change (20% change)
        threshold = 0.2
        p_value = 1.0 if magnitude < threshold else 0.0

        return magnitude, p_value

    def _calculate_mean_std_drift(self, reference: pd.Series, current: pd.Series) -> Tuple[float, float]:
        """
        Statistical test for mean and variance drift
        """
        # Test for mean difference (Welch's t-test)
        t_stat, t_pvalue = stats.ttest_ind(reference, current, equal_var=False)

        # Test for variance difference (Levene's test)
        f_stat, f_pvalue = stats.levene(reference, current)

        # Combined p-value (Bonferroni correction)
        combined_pvalue = min(t_pvalue * 2, f_pvalue * 2, 1.0)

        # Combined statistic (normalized)
        combined_stat = np.sqrt(t_stat**2 + f_stat**2) / np.sqrt(2)

        return combined_stat, combined_pvalue

    def _evaluate_drift(self, drift_scores: Dict[str, Tuple[float, float]]) -> Tuple[bool, str, float, float, float]:
        """
        Evaluate drift based on multiple methods

        Returns:
            (is_drifted, drift_type, score, p_value, threshold)
        """
        # Check each method
        for method, (score, p_value) in drift_scores.items():
            threshold = self._get_threshold(method)

            if method in ['ks', 'mean_std']:
                # For statistical tests, use p-value
                if p_value < (1 - self.confidence_level):
                    return True, f'statistical_{method}', score, p_value, threshold
            else:
                # For other methods, use score threshold
                if method == 'psi' and score > self.psi_threshold:
                    return True, 'distributional_psi', score, p_value, self.psi_threshold
                elif method == 'js' and score > self.js_threshold:
                    return True, 'distributional_js', score, p_value, self.js_threshold
                elif method == 'magnitude' and score > 0.2:
                    return True, 'magnitude', score, p_value, 0.2

        # No drift detected
        # Return the maximum score
        if drift_scores:
            max_method = max(drift_scores.items(), key=lambda x: x[1][0])
            return False, 'none', max_method[1][0], max_method[1][1], self._get_threshold(max_method[0])

        return False, 'none', 0.0, 1.0, 0.0

    def _get_threshold(self, method: str) -> float:
        """Get threshold for a specific method"""
        thresholds = {
            'psi': self.psi_threshold,
            'ks': 1 - self.confidence_level,
            'js': self.js_threshold,
            'magnitude': 0.2,
            'mean_std': 1 - self.confidence_level
        }
        return thresholds.get(method, 0.1)

    def _calculate_statistics(self, data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculate reference statistics for all features"""
        stats_dict = {}

        numeric_cols = data.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            values = data[col].dropna()
            if len(values) > 0:
                stats_dict[col] = {
                    'mean': float(values.mean()),
                    'std': float(values.std()),
                    'min': float(values.min()),
                    'max': float(values.max()),
                    'q25': float(values.quantile(0.25)),
                    'q50': float(values.quantile(0.50)),
                    'q75': float(values.quantile(0.75)),
                    'skew': float(values.skew()),
                    'kurt': float(values.kurt())
                }

        return stats_dict

    def detect_concept_drift(self,
                           predictions_history: List[Dict[str, Any]],
                           window_size: int = 30,
                           method: str = 'page_hinkley') -> Dict[str, Any]:
        """
        Detect concept drift in model predictions

        Args:
            predictions_history: List of prediction results with timestamps
            window_size: Window size for drift detection
            method: Detection method ('page_hinkley', 'adwin', 'sliding_window')

        Returns:
            Drift detection results
        """
        if len(predictions_history) < window_size:
            return {'drift_detected': False, 'message': 'Insufficient data'}

        # Extract prediction errors
        errors = []
        for record in predictions_history:
            if 'error' in record:
                errors.append(record['error'])
            elif 'actual' in record and 'predicted' in record:
                errors.append(record['actual'] - record['predicted'])

        if not errors:
            return {'drift_detected': False, 'message': 'No error data available'}

        errors = np.array(errors)

        if method == 'page_hinkley':
            return self._page_hinkley_test(errors)
        elif method == 'sliding_window':
            return self._sliding_window_test(errors, window_size)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _page_hinkley_test(self, errors: np.ndarray,
                          delta: float = 0.005,
                          lambda_: float = 50) -> Dict[str, Any]:
        """
        Page-Hinkley test for concept drift
        """
        n = len(errors)
        mean_errors = np.cumsum(errors) / np.arange(1, n + 1)

        # Page-Hinkley statistic
        s = np.zeros(n)
        m = np.zeros(n)

        for i in range(1, n):
            s[i] = max(0, s[i-1] + errors[i] - mean_errors[i] - delta)
            m[i] = max(m[i-1], s[i])

        # Check for drift
        ph_values = m - s
        drift_points = np.where(ph_values > lambda_)[0]

        if len(drift_points) > 0:
            return {
                'drift_detected': True,
                'drift_point': int(drift_points[0]),
                'confidence': float(ph_values[drift_points[0]] / lambda_),
                'method': 'page_hinkley'
            }

        return {
            'drift_detected': False,
            'max_statistic': float(ph_values.max()),
            'threshold': lambda_,
            'method': 'page_hinkley'
        }

    def _sliding_window_test(self, errors: np.ndarray, window_size: int) -> Dict[str, Any]:
        """
        Sliding window test for concept drift
        """
        if len(errors) < 2 * window_size:
            return {'drift_detected': False, 'message': 'Insufficient data for sliding window'}

        # Compare recent window to previous window
        recent_window = errors[-window_size:]
        previous_window = errors[-2*window_size:-window_size]

        # Statistical test
        t_stat, p_value = stats.ttest_ind(recent_window, previous_window)

        # Check for significant difference
        drift_detected = p_value < (1 - self.confidence_level)

        return {
            'drift_detected': drift_detected,
            'p_value': float(p_value),
            't_statistic': float(t_stat),
            'recent_mean_error': float(recent_window.mean()),
            'previous_mean_error': float(previous_window.mean()),
            'method': 'sliding_window'
        }

    def create_drift_report(self, drift_results: Dict[str, DriftResult]) -> str:
        """Create formatted drift detection report"""

        report = []
        report.append("=" * 60)
        report.append("DRIFT DETECTION REPORT")
        report.append("=" * 60)
        report.append(f"Features Analyzed: {len(drift_results)}")

        # Count drifted features
        drifted_features = [r for r in drift_results.values() if r.is_drifted]
        report.append(f"Features with Drift: {len(drifted_features)}")
        report.append("")

        if drifted_features:
            report.append("DRIFTED FEATURES:")
            for result in sorted(drifted_features, key=lambda x: x.drift_score, reverse=True):
                report.append(f"\n{result.feature_name}:")
                report.append(f"  Drift Type: {result.drift_type}")
                report.append(f"  Drift Score: {result.drift_score:.4f}")
                report.append(f"  P-value: {result.p_value:.4f}")
                report.append(f"  Reference Mean: {result.reference_stats['mean']:.4f}")
                report.append(f"  Current Mean: {result.current_stats['mean']:.4f}")
                report.append(f"  Change: {((result.current_stats['mean'] - result.reference_stats['mean']) / (abs(result.reference_stats['mean']) + TC.MIN_VARIANCE) * 100):.1f}%")
        else:
            report.append("No significant drift detected in any features.")

        # Summary statistics
        report.append("\nSUMMARY STATISTICS:")
        report.append(f"  Drift Rate: {len(drifted_features) / len(drift_results) * 100:.1f}%")

        if drift_results:
            avg_score = np.mean([r.drift_score for r in drift_results.values()])
            report.append(f"  Average Drift Score: {avg_score:.4f}")

        report.append("=" * 60)

        return "\n".join(report)
