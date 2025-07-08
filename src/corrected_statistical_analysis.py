#!/usr/bin/env python3
"""
Corrected Statistical Analysis for Risk Management System
Proper handling of small sample sizes and accurate reporting
"""

import pandas as pd
import numpy as np
from scipy import stats
import pickle
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

class CorrectedStatisticalAnalyzer:
    """Corrected statistical analysis with proper small sample handling."""

    def __init__(self, base_path="/Users/temurbekkhujaev/Repos/risk-tool"):
        self.base_path = Path(base_path)
        self.results_path = self.base_path / "results" / "causal_impact_comparison"

    def load_scenario_results(self, reduction_pct):
        """Load results for a specific reduction percentage."""
        try:
            with open(self.results_path / f'reduction_{reduction_pct}pct/detailed_results.pkl', 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            print(f"Warning: Could not find detailed results for {reduction_pct}% reduction")
            return None

    def extract_trader_metrics(self, results):
        """Extract individual trader metrics from results."""
        metrics = []
        trader_ids = []

        individual_results = results.get('individual_results', {})

        for trader_id, trader_data in individual_results.items():
            if isinstance(trader_data, dict) and 'metrics' in trader_data:
                metrics_dict = trader_data['metrics']
                metrics.append({
                    'trader_id': trader_id,
                    'original_pnl': metrics_dict.get('total_actual_pnl', 0),
                    'adjusted_pnl': metrics_dict.get('total_adjusted_pnl', 0),
                    'net_benefit': metrics_dict.get('net_benefit', 0),
                    'intervention_rate': metrics_dict.get('intervention_rate', 0),
                    'improvement_pct': metrics_dict.get('overall_improvement', 0)
                })
                trader_ids.append(trader_id)

        return pd.DataFrame(metrics), trader_ids

    def corrected_statistical_tests(self, reduction_pct):
        """Perform corrected statistical tests for small sample size."""
        print(f"\n{'='*60}")
        print(f"CORRECTED STATISTICAL ANALYSIS: {reduction_pct}% RISK REDUCTION")
        print(f"{'='*60}")

        results = self.load_scenario_results(reduction_pct)
        if results is None:
            print("Could not load data for analysis")
            return None

        metrics_df, trader_ids = self.extract_trader_metrics(results)

        if len(metrics_df) == 0:
            print("No trader metrics found")
            return None

        # Basic descriptive statistics
        n_traders = len(metrics_df)
        positive_benefits = (metrics_df['net_benefit'] > 0).sum()
        success_rate = positive_benefits / n_traders

        print(f"Sample size: {n_traders} traders")
        print(f"Positive benefits: {positive_benefits} traders")
        print(f"Success rate: {success_rate:.1%} ({positive_benefits}/{n_traders})")
        print(f"Mean net benefit: ${metrics_df['net_benefit'].mean():,.2f}")
        print(f"Median net benefit: ${metrics_df['net_benefit'].median():,.2f}")
        print(f"Std deviation: ${metrics_df['net_benefit'].std():,.2f}")

        # Test if net benefits are significantly different from zero
        print(f"\n{'='*40}")
        print("STATISTICAL SIGNIFICANCE TESTING")
        print(f"{'='*40}")

        # 1. One-sample t-test (parametric)
        net_benefits = metrics_df['net_benefit'].values
        t_stat, t_p_value = stats.ttest_1samp(net_benefits, 0)

        print(f"\n1. ONE-SAMPLE T-TEST (H0: mean benefit = 0)")
        print(f"   t-statistic: {t_stat:.4f}")
        print(f"   p-value: {t_p_value:.6f}")

        # 2. Wilcoxon signed-rank test (non-parametric, better for small samples)
        try:
            w_stat, w_p_value = stats.wilcoxon(net_benefits, alternative='two-sided')
            print(f"\n2. WILCOXON SIGNED-RANK TEST (H0: median benefit = 0)")
            print(f"   W-statistic: {w_stat:.0f}")
            print(f"   p-value: {w_p_value:.6f}")
        except ValueError as e:
            print(f"\n2. WILCOXON TEST: Could not perform ({e})")
            w_p_value = None

        # 3. Bootstrap confidence interval for mean
        print(f"\n3. BOOTSTRAP CONFIDENCE INTERVALS")
        ci_lower, ci_upper = self.bootstrap_confidence_interval(net_benefits)
        print(f"   95% CI for mean benefit: [${ci_lower:,.2f}, ${ci_upper:,.2f}]")

        # 4. Effect size calculation
        cohens_d = np.mean(net_benefits) / np.std(net_benefits, ddof=1) if np.std(net_benefits) > 0 else 0
        print(f"   Cohen's d (effect size): {cohens_d:.4f}")

        # 5. Sign test for success rate
        print(f"\n4. SIGN TEST FOR SUCCESS RATE")
        # Test if success rate is significantly different from 50%
        from scipy.stats import binomtest
        sign_result = binomtest(positive_benefits, n_traders, p=0.5)
        sign_p_value = sign_result.pvalue
        print(f"   Successes: {positive_benefits}/{n_traders}")
        print(f"   p-value (vs 50% null): {sign_p_value:.6f}")

        # Interpretation with proper caveats
        print(f"\n{'='*40}")
        print("STATISTICAL INTERPRETATION")
        print(f"{'='*40}")

        # Significance interpretation with small sample caveats
        primary_p = w_p_value if w_p_value is not None else t_p_value

        if primary_p < 0.001:
            significance = "highly significant (p < 0.001)"
        elif primary_p < 0.01:
            significance = "very significant (p < 0.01)"
        elif primary_p < 0.05:
            significance = "significant (p < 0.05)"
        elif primary_p < 0.1:
            significance = "marginally significant (p < 0.1)"
        else:
            significance = "not significant (p ≥ 0.1)"

        print(f"Primary result: {significance}")

        # Effect size interpretation
        if abs(cohens_d) < 0.2:
            effect_size = "negligible"
        elif abs(cohens_d) < 0.5:
            effect_size = "small"
        elif abs(cohens_d) < 0.8:
            effect_size = "medium"
        else:
            effect_size = "large"

        print(f"Effect size: {effect_size}")

        # IMPORTANT CAVEATS
        print(f"\n⚠️  IMPORTANT STATISTICAL CAVEATS:")
        print(f"   • Small sample size (n={n_traders}) limits statistical power")
        print(f"   • Results may not be robust to outliers")
        print(f"   • Multiple testing correction not applied")
        print(f"   • Assumes independence between traders")

        if ci_lower > 0:
            print(f"   ✓ 95% confidence interval excludes zero")
        elif ci_upper < 0:
            print(f"   ✗ 95% confidence interval suggests negative effect")
        else:
            print(f"   ⚠ 95% confidence interval includes zero")

        return {
            'n_traders': n_traders,
            'positive_benefits': positive_benefits,
            'success_rate': success_rate,
            'mean_benefit': metrics_df['net_benefit'].mean(),
            'median_benefit': metrics_df['net_benefit'].median(),
            'std_benefit': metrics_df['net_benefit'].std(),
            't_statistic': t_stat,
            't_p_value': t_p_value,
            'w_p_value': w_p_value,
            'cohens_d': cohens_d,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'sign_p_value': sign_p_value
        }

    def bootstrap_confidence_interval(self, data, n_bootstrap=10000, confidence_level=0.95):
        """Calculate bootstrap confidence interval for mean."""
        if len(data) == 0:
            return None, None

        bootstrap_means = []
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))

        alpha = 1 - confidence_level
        lower_percentile = (alpha/2) * 100
        upper_percentile = (1 - alpha/2) * 100

        ci_lower = np.percentile(bootstrap_means, lower_percentile)
        ci_upper = np.percentile(bootstrap_means, upper_percentile)

        return ci_lower, ci_upper

    def analyze_all_scenarios(self):
        """Analyze all risk reduction scenarios with corrected statistics."""
        scenarios = [25, 50, 70, 90]
        all_results = {}

        print("="*80)
        print("CORRECTED STATISTICAL ANALYSIS FOR ALL SCENARIOS")
        print("="*80)

        for scenario in scenarios:
            result = self.corrected_statistical_tests(scenario)
            if result:
                all_results[scenario] = result

        # Summary comparison
        print(f"\n{'='*80}")
        print("SCENARIO COMPARISON SUMMARY")
        print(f"{'='*80}")

        print(f"{'Scenario':<10} {'Success Rate':<15} {'Mean Benefit':<15} {'p-value':<12} {'Effect Size'}")
        print("-" * 70)

        for scenario, result in all_results.items():
            primary_p = result['w_p_value'] if result['w_p_value'] is not None else result['t_p_value']
            print(f"{scenario}%{'':<7} {result['success_rate']:.1%} ({result['positive_benefits']}/{result['n_traders']}){'':<4} "
                  f"${result['mean_benefit']:>10,.0f}{'':<3} {primary_p:<12.4f} {result['cohens_d']:.3f}")

        return all_results

if __name__ == "__main__":
    analyzer = CorrectedStatisticalAnalyzer()
    results = analyzer.analyze_all_scenarios()
