#!/usr/bin/env python3
"""
Statistical Significance Analysis for Risk Management System
Performs rigorous statistical testing on the causal impact results
"""

import pandas as pd
import numpy as np
from scipy import stats
import pickle
import warnings
warnings.filterwarnings('ignore')

def load_detailed_results(reduction_pct):
    """Load detailed results for a specific reduction percentage."""
    try:
        with open(f'results/causal_impact_comparison/reduction_{reduction_pct}pct/detailed_results.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"Warning: Could not find detailed results for {reduction_pct}% reduction")
        return None

def perform_paired_t_test(before_values, after_values, scenario_name):
    """Perform paired t-test on before/after values."""
    # Remove any pairs where either value is NaN
    valid_pairs = ~(np.isnan(before_values) | np.isnan(after_values))
    before_clean = before_values[valid_pairs]
    after_clean = after_values[valid_pairs]

    if len(before_clean) < 3:
        return None, None, None, len(before_clean)

    # Perform paired t-test
    t_stat, p_value = stats.ttest_rel(after_clean, before_clean)

    # Calculate effect size (Cohen's d for paired samples)
    differences = after_clean - before_clean
    pooled_std = np.sqrt(((np.std(before_clean, ddof=1)**2 + np.std(after_clean, ddof=1)**2) / 2))
    cohens_d = np.mean(differences) / pooled_std if pooled_std > 0 else 0

    return t_stat, p_value, cohens_d, len(before_clean)

def perform_wilcoxon_test(before_values, after_values):
    """Perform Wilcoxon signed-rank test (non-parametric alternative)."""
    valid_pairs = ~(np.isnan(before_values) | np.isnan(after_values))
    before_clean = before_values[valid_pairs]
    after_clean = after_values[valid_pairs]

    if len(before_clean) < 3:
        return None, None

    try:
        statistic, p_value = stats.wilcoxon(after_clean, before_clean, alternative='two-sided')
        return statistic, p_value
    except:
        return None, None

def bootstrap_confidence_interval(data, n_bootstrap=1000, confidence_level=0.95):
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

def analyze_scenario_significance(reduction_pct):
    """Analyze statistical significance for a specific reduction scenario."""
    print(f"\n{'='*60}")
    print(f"STATISTICAL ANALYSIS: {reduction_pct}% RISK REDUCTION")
    print(f"{'='*60}")

    # Load detailed results
    results = load_detailed_results(reduction_pct)
    if results is None:
        print("Could not load data for analysis")
        return None

    # Extract before/after PnL data
    before_pnl = []
    after_pnl = []
    net_benefits = []

    # Access individual trader results
    individual_results = results.get('individual_results', {})

    for trader_id, trader_data in individual_results.items():
        if isinstance(trader_data, dict) and 'metrics' in trader_data:
            metrics = trader_data['metrics']
            before_pnl.append(metrics['total_actual_pnl'])
            after_pnl.append(metrics['total_adjusted_pnl'])
            net_benefits.append(metrics['net_benefit'])

    before_pnl = np.array(before_pnl)
    after_pnl = np.array(after_pnl)
    net_benefits = np.array(net_benefits)

    print(f"Sample size: {len(before_pnl)} traders")
    print(f"Mean net benefit: ${np.mean(net_benefits):,.2f}")
    print(f"Median net benefit: ${np.median(net_benefits):,.2f}")
    print(f"Std deviation: ${np.std(net_benefits, ddof=1):,.2f}")

    # Paired t-test
    t_stat, p_value, cohens_d, n_valid = perform_paired_t_test(before_pnl, after_pnl, f"{reduction_pct}%")

    if p_value is not None:
        print(f"\nPAIRED T-TEST RESULTS:")
        print(f"t-statistic: {t_stat:.4f}")
        print(f"p-value: {p_value:.6f}")
        print(f"Cohen's d (effect size): {cohens_d:.4f}")
        print(f"Valid pairs: {n_valid}")

        # Interpret significance
        if p_value < 0.001:
            significance = "highly significant (p < 0.001)"
        elif p_value < 0.01:
            significance = "very significant (p < 0.01)"
        elif p_value < 0.05:
            significance = "significant (p < 0.05)"
        elif p_value < 0.1:
            significance = "marginally significant (p < 0.1)"
        else:
            significance = "not significant (p ≥ 0.1)"

        print(f"Result: {significance}")

        # Interpret effect size
        if abs(cohens_d) < 0.2:
            effect_size = "small"
        elif abs(cohens_d) < 0.5:
            effect_size = "small to medium"
        elif abs(cohens_d) < 0.8:
            effect_size = "medium to large"
        else:
            effect_size = "large"

        print(f"Effect size: {effect_size}")

    # Wilcoxon signed-rank test (non-parametric)
    w_stat, w_p_value = perform_wilcoxon_test(before_pnl, after_pnl)
    if w_p_value is not None:
        print(f"\nWILCOXON SIGNED-RANK TEST:")
        print(f"Statistic: {w_stat}")
        print(f"p-value: {w_p_value:.6f}")
        w_significance = "significant" if w_p_value < 0.05 else "not significant"
        print(f"Result: {w_significance}")

    # Bootstrap confidence interval for net benefit
    ci_lower, ci_upper = bootstrap_confidence_interval(net_benefits, n_bootstrap=1000)
    if ci_lower is not None:
        print(f"\n95% CONFIDENCE INTERVAL FOR NET BENEFIT:")
        print(f"${ci_lower:,.2f} to ${ci_upper:,.2f}")

        # Check if CI includes zero
        if ci_lower > 0:
            print("CI does not include zero - suggests significant positive effect")
        elif ci_upper < 0:
            print("CI does not include zero - suggests significant negative effect")
        else:
            print("CI includes zero - effect may not be significant")

    # Success rate analysis
    positive_improvements = np.sum(net_benefits > 0)
    total_traders = len(net_benefits)
    success_rate = positive_improvements / total_traders

    print(f"\nSUCCESS RATE ANALYSIS:")
    print(f"Positive improvements: {positive_improvements}/{total_traders} ({success_rate:.1%})")

    # Binomial test for success rate
    binomial_p = stats.binomtest(positive_improvements, total_traders, p=0.5, alternative='two-sided').pvalue
    print(f"Binomial test p-value: {binomial_p:.6f}")
    binomial_significance = "significant" if binomial_p < 0.05 else "not significant"
    print(f"Success rate vs 50% chance: {binomial_significance}")

    return {
        'reduction_pct': reduction_pct,
        'sample_size': len(before_pnl),
        'mean_net_benefit': np.mean(net_benefits),
        'median_net_benefit': np.median(net_benefits),
        'std_net_benefit': np.std(net_benefits, ddof=1),
        'ttest_p_value': p_value,
        'ttest_significance': significance if p_value is not None else "N/A",
        'cohens_d': cohens_d,
        'effect_size': effect_size if p_value is not None else "N/A",
        'wilcoxon_p_value': w_p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'success_rate': success_rate,
        'binomial_p_value': binomial_p,
        'positive_improvements': positive_improvements,
        'total_traders': total_traders
    }

def main():
    """Main analysis function."""
    print("STATISTICAL SIGNIFICANCE ANALYSIS")
    print("Risk Management System Performance")
    print("=" * 60)

    # Analyze each scenario
    scenarios = [25, 50, 70, 90]
    results_summary = []

    for reduction_pct in scenarios:
        result = analyze_scenario_significance(reduction_pct)
        if result:
            results_summary.append(result)

    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY OF STATISTICAL SIGNIFICANCE ACROSS ALL SCENARIOS")
    print(f"{'='*80}")

    print(f"{'Scenario':<10} {'p-value':<12} {'Significance':<20} {'Effect Size':<15} {'Success Rate':<12}")
    print("-" * 80)

    for result in results_summary:
        scenario = f"{result['reduction_pct']}%"
        p_val = f"{result['ttest_p_value']:.6f}" if result['ttest_p_value'] is not None else "N/A"
        significance = result['ttest_significance']
        effect_size = result['effect_size']
        success_rate = f"{result['success_rate']:.1%}"

        print(f"{scenario:<10} {p_val:<12} {significance:<20} {effect_size:<15} {success_rate:<12}")

    print(f"\n{'='*80}")
    print("RECOMMENDATIONS:")
    print(f"{'='*80}")

    # Find best performing scenario
    valid_results = [r for r in results_summary if r['ttest_p_value'] is not None]
    if valid_results:
        best_scenario = max(valid_results, key=lambda x: x['mean_net_benefit'])
        most_significant = min(valid_results, key=lambda x: x['ttest_p_value'])

        print(f"• Best performing scenario: {best_scenario['reduction_pct']}% reduction")
        print(f"  - Mean net benefit: ${best_scenario['mean_net_benefit']:,.2f}")
        print(f"  - 95% CI: ${best_scenario['ci_lower']:,.2f} to ${best_scenario['ci_upper']:,.2f}")
        print(f"  - Statistical significance: {best_scenario['ttest_significance']}")

        print(f"\n• Most statistically significant: {most_significant['reduction_pct']}% reduction")
        print(f"  - p-value: {most_significant['ttest_p_value']:.6f}")
        print(f"  - Effect size: {most_significant['effect_size']}")

        # Overall assessment
        significant_scenarios = [r for r in valid_results if r['ttest_p_value'] < 0.05]
        if significant_scenarios:
            print(f"\n• {len(significant_scenarios)} out of {len(valid_results)} scenarios show significant results")
            print("• The risk management system demonstrates statistically significant benefits")
        else:
            print("\n• No scenarios show statistically significant results at p < 0.05")
            print("• Consider larger sample size or longer evaluation period")

    return results_summary

if __name__ == "__main__":
    results = main()
