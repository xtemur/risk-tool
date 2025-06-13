"""
Advanced Causal Impact Analysis Demo

Comprehensive demonstration of state-of-the-art causal inference methods
for analyzing trading model impact. Shows how to use:

1. Double Machine Learning (DML) for robust treatment effect estimation
2. Synthetic Control Method for counterfactual analysis
3. Causal validation and robustness testing
4. Model interpretability with causal warnings
5. Comprehensive business impact assessment

This demo showcases how to prove model utility using modern causal inference techniques.
"""

import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add the src directory to the Python path and change working directory
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))
os.chdir(project_root)

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

# Import our advanced causal analysis modules
from modeling.advanced_causal_analyzer import AdvancedCausalAnalyzer
from modeling.double_ml_estimator import DoubleMachineLearningEstimator, TradingDMLAnalyzer
from modeling.synthetic_control_analyzer import SyntheticControlAnalyzer
from modeling.causal_validation_suite import CausalValidationSuite
from modeling.causal_interpretability import CausalInterpretabilityAnalyzer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_realistic_trading_data(n_days=200, n_traders=8, random_state=42):
    """
    Generate realistic trading data with complex relationships for demo

    Args:
        n_days: Number of trading days
        n_traders: Number of traders
        random_state: Random seed

    Returns:
        Tuple of (panel_data, features_data, model_confidence)
    """
    np.random.seed(random_state)

    # Generate dates
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='B')  # Business days

    # Generate trader IDs
    trader_ids = [f'TRADER_{i+1:03d}' for i in range(n_traders)]

    # Create multi-index for panel data
    index = pd.MultiIndex.from_product([dates, trader_ids], names=['date', 'trader_id'])

    # Generate features with realistic trading characteristics
    n_obs = len(index)

    # Market-level features (same for all traders on each day)
    market_volatility = np.repeat(np.random.gamma(2, 0.05, n_days), n_traders)
    market_return = np.repeat(np.random.normal(0.001, 0.02, n_days), n_traders)
    market_volume = np.repeat(np.random.lognormal(15, 0.3, n_days), n_traders)

    # Trader-specific features (vary by trader)
    trader_skill = np.tile(np.random.normal(0, 0.1, n_traders), n_days)
    trader_risk_appetite = np.tile(np.random.uniform(0.5, 2.0, n_traders), n_days)
    trader_experience = np.tile(np.random.exponential(2, n_traders), n_days)

    # Position-level features
    position_size = np.random.lognormal(10, 0.5, n_obs)
    leverage = np.random.uniform(1, 5, n_obs)

    # Technical indicators
    momentum = np.random.normal(0, 0.1, n_obs)
    mean_reversion = np.random.normal(0, 0.05, n_obs)

    # Create features DataFrame
    features_data = pd.DataFrame({
        'market_volatility': market_volatility,
        'market_return': market_return,
        'market_volume': market_volume,
        'trader_skill': trader_skill,
        'trader_risk_appetite': trader_risk_appetite,
        'trader_experience': trader_experience,
        'position_size': position_size,
        'leverage': leverage,
        'momentum': momentum,
        'mean_reversion': mean_reversion
    }, index=index)

    # Generate model predictions with realistic patterns
    # Model captures some real signal but also has biases
    true_signal = (
        0.3 * features_data['trader_skill'] +
        0.2 * features_data['momentum'] +
        0.1 * features_data['market_return'] +
        -0.15 * features_data['market_volatility']
    )

    model_noise = np.random.normal(0, 0.02, n_obs)
    model_bias = 0.01  # Slightly optimistic model

    predicted_returns = true_signal + model_noise + model_bias
    predicted_pnl = predicted_returns * features_data['position_size']

    # Generate actual PnL with model signal + additional noise
    actual_signal_strength = 0.4  # Model captures 40% of predictable signal

    actual_returns = (
        actual_signal_strength * true_signal +
        (1 - actual_signal_strength) * np.random.normal(0, 0.03, n_obs) +
        0.1 * features_data['trader_skill'] * np.random.normal(0, 0.01, n_obs)  # Skill-dependent noise
    )

    actual_pnl = actual_returns * features_data['position_size']

    # Generate model confidence (higher for more certain predictions)
    model_confidence = np.random.beta(2, 2, n_obs) * 0.8 + 0.2  # Between 0.2 and 1.0

    # Create panel data structure
    panel_data = pd.DataFrame({
        'actual_pnl': actual_pnl,
        'predicted_pnl': predicted_pnl,
        'actual_returns': actual_returns,
        'predicted_returns': predicted_returns,
        'model_confidence': model_confidence
    }, index=index)

    # Add lagged features
    for trader in trader_ids:
        trader_mask = features_data.index.get_level_values('trader_id') == trader
        features_data.loc[trader_mask, 'lagged_pnl'] = panel_data.loc[trader_mask, 'actual_pnl'].shift(1).fillna(0)
        features_data.loc[trader_mask, 'lagged_volatility'] = panel_data.loc[trader_mask, 'actual_pnl'].rolling(5).std().fillna(0.01)

    return panel_data, features_data, model_confidence


def demo_double_ml_analysis():
    """Demonstrate Double Machine Learning analysis"""

    print("\n" + "="*80)
    print("DOUBLE MACHINE LEARNING (DML) ANALYSIS DEMO")
    print("="*80)

    # Generate data
    print("\n1. Generating realistic trading data...")
    panel_data, features_data, model_confidence = generate_realistic_trading_data(n_days=150, n_traders=6)

    print(f"✓ Generated {len(panel_data)} observations")
    print(f"✓ Features: {list(features_data.columns)}")
    print(f"✓ Date range: {panel_data.index.get_level_values('date').min()} to {panel_data.index.get_level_values('date').max()}")

    # Initialize DML analyzer
    print("\n2. Initializing Double Machine Learning analyzer...")
    dml_analyzer = TradingDMLAnalyzer()

    # Run DML analysis
    print("\n3. Running DML causal impact analysis...")
    dml_results = dml_analyzer.analyze_model_impact(
        actual_pnl=panel_data['actual_pnl'],
        predicted_pnl=panel_data['predicted_pnl'],
        features=features_data,
        model_confidence=panel_data['model_confidence'],
        trader_id=panel_data.index.get_level_values('trader_id')
    )

    # Display results
    print("\n4. DML Results Summary:")
    print("-" * 40)

    causal_effect = dml_results['dml_causal_effect']
    print(f"Average Treatment Effect (ATE): ${causal_effect['ate_estimate']:.2f}")
    print(f"Standard Error: ±${causal_effect['ate_standard_error']:.2f}")
    print(f"95% Confidence Interval: [${causal_effect['ate_confidence_interval'][0]:.2f}, ${causal_effect['ate_confidence_interval'][1]:.2f}]")
    print(f"P-value: {causal_effect['ate_p_value']:.4f}")
    print(f"Statistically Significant: {'Yes' if causal_effect['is_significant'] else 'No'}")

    print(f"\nBusiness Impact:")
    business = dml_results['trading_metrics']['causal_impact']
    print(f"Daily Causal Effect: ${business['per_day_effect']:.2f}")
    print(f"Total Causal Impact: ${business['total_effect']:.2f}")
    print(f"Relative Improvement: {business['relative_improvement']:.1f}%")

    print(f"\nInterpretation:")
    interp = dml_results['interpretation']
    print(f"• {interp['main_finding']}")
    print(f"• {interp['statistical_significance']}")
    print(f"• {interp['business_impact']}")

    return dml_results


def demo_synthetic_control_analysis():
    """Demonstrate Synthetic Control Method analysis"""

    print("\n" + "="*80)
    print("SYNTHETIC CONTROL METHOD ANALYSIS DEMO")
    print("="*80)

    # Generate panel data suitable for synthetic control
    print("\n1. Generating panel data for synthetic control...")
    panel_data, features_data, _ = generate_realistic_trading_data(n_days=120, n_traders=8)

    # Reshape data for synthetic control (traders as columns, dates as rows)
    trader_pnl = panel_data['actual_pnl'].unstack('trader_id')
    model_usage = (panel_data['predicted_pnl'] > 0).unstack('trader_id').astype(int)

    print(f"✓ Panel data shape: {trader_pnl.shape}")
    print(f"✓ Traders: {list(trader_pnl.columns)}")

    # Initialize Synthetic Control analyzer
    print("\n2. Initializing Synthetic Control analyzer...")
    sc_analyzer = SyntheticControlAnalyzer(matching_period_length=30)

    # Run synthetic control analysis
    print("\n3. Running Synthetic Control analysis...")
    sc_results = sc_analyzer.analyze_trading_impact(
        trader_pnl=trader_pnl,
        model_usage=model_usage
    )

    # Display results
    if sc_results and 'aggregate_effects' in sc_results:
        print("\n4. Synthetic Control Results Summary:")
        print("-" * 45)

        agg = sc_results['aggregate_effects']['average_treatment_effect']
        print(f"Average Treatment Effect: ${agg['difference']:.2f}")
        print(f"Cumulative Effect: ${agg['cumulative_effect']:.2f}")
        print(f"Relative Improvement: {agg['relative_improvement']:.1f}%")

        if 'individual_effects' in sc_results:
            print(f"\nIndividual Trader Effects:")
            for trader_id, effects in list(sc_results['individual_effects'].items())[:3]:
                post_effect = effects['post_treatment_effect']['mean']
                print(f"• {trader_id}: ${post_effect:.2f} daily effect")

        if 'robustness_tests' in sc_results:
            print(f"\nRobustness Tests:")
            rob = sc_results['robustness_tests']
            if 'leave_one_out' in rob:
                loo = rob['leave_one_out']
                print(f"• Leave-one-out range: ${loo['min_effect']:.2f} to ${loo['max_effect']:.2f}")

        if 'placebo_tests' in sc_results:
            print(f"\nPlacebo Tests:")
            placebo = sc_results['placebo_tests']
            if 'actual_vs_placebo' in placebo:
                pvp = placebo['actual_vs_placebo']
                print(f"• Actual effect: ${pvp['actual_effect']:.2f}")
                print(f"• P-value vs placebo: {pvp['p_value']:.3f}")
                print(f"• Significant: {'Yes' if pvp['is_significant'] else 'No'}")
    else:
        print("\n4. Synthetic Control analysis completed with limited results")
        print("   (This may happen with simulated data)")

    return sc_results


def demo_comprehensive_analysis():
    """Demonstrate comprehensive advanced causal analysis"""

    print("\n" + "="*80)
    print("COMPREHENSIVE ADVANCED CAUSAL ANALYSIS DEMO")
    print("="*80)

    # Generate comprehensive dataset
    print("\n1. Generating comprehensive trading dataset...")
    panel_data, features_data, model_confidence = generate_realistic_trading_data(n_days=180, n_traders=6)

    # Initialize advanced analyzer
    print("\n2. Initializing Advanced Causal Analyzer...")
    advanced_analyzer = AdvancedCausalAnalyzer(
        enable_dml=True,
        enable_synthetic_control=True,
        enable_traditional_analysis=True,
        random_state=42
    )

    # Run comprehensive analysis
    print("\n3. Running comprehensive causal impact analysis...")
    print("   This may take a few minutes as it runs multiple sophisticated methods...")

    # Prepare data for analysis
    trader_pnl_panel = panel_data['actual_pnl'].unstack('trader_id')
    predicted_pnl_panel = panel_data['predicted_pnl'].unstack('trader_id')

    comprehensive_results = advanced_analyzer.analyze_comprehensive_impact(
        actual_pnl=trader_pnl_panel,
        predicted_pnl=predicted_pnl_panel,
        features=features_data,
        trader_ids=panel_data.index.get_level_values('trader_id'),
        dates=panel_data.index.get_level_values('date').unique(),
        model_confidence=panel_data['model_confidence']
    )

    # Display results
    print("\n4. Comprehensive Analysis Results:")
    print("=" * 50)

    # Synthesis results
    if 'synthesis_and_validation' in comprehensive_results:
        synthesis = comprehensive_results['synthesis_and_validation']

        print(f"\nMethod Comparison:")
        if 'method_comparison' in synthesis:
            comp = synthesis['method_comparison']
            print(f"• Effect estimates: {comp.get('effect_estimates', {})}")
            print(f"• Methods agree on direction: {'Yes' if comp.get('methods_agree', False) else 'No'}")

        if 'consistency_check' in synthesis:
            consistency = synthesis['consistency_check']
            print(f"• Consensus estimate: ${consistency.get('consensus_estimate', 0):.2f}")
            print(f"• Consensus direction: {consistency.get('consensus_direction', 'unknown')}")
            print(f"• Methods consistent: {'Yes' if consistency.get('is_consistent', False) else 'No'}")

    # Business impact
    if 'business_impact_assessment' in comprehensive_results:
        business = comprehensive_results['business_impact_assessment']

        print(f"\nBusiness Impact Assessment:")
        print(f"• Daily effect: ${business.get('consensus_daily_effect', 0):.2f}")
        print(f"• Total causal impact: ${business.get('total_causal_impact', 0):.2f}")
        print(f"• Relative improvement: {business.get('relative_improvement_pct', 0):.1f}%")

        if 'risk_adjusted_metrics' in business:
            risk = business['risk_adjusted_metrics']
            print(f"• Effect size: {risk.get('effect_size', 0):.2f}")
            print(f"• Practical significance: {risk.get('practical_significance', 'unknown')}")

        if 'business_value_assessment' in business:
            value = business['business_value_assessment']
            print(f"• Estimated annual impact: ${value.get('estimated_annual_impact', 0):.2f}")
            print(f"• ROI category: {value.get('roi_category', 'unknown')}")

    # Recommendations
    if 'recommendations_and_interpretation' in comprehensive_results:
        rec = comprehensive_results['recommendations_and_interpretation']

        print(f"\nRecommendations:")
        print(f"• Primary: {rec.get('primary_recommendation', 'No recommendation')}")
        print(f"• Confidence: {rec.get('confidence_level', 'unknown')}")

        if 'key_findings' in rec:
            print(f"\nKey Findings:")
            for finding in rec['key_findings'][:3]:
                print(f"• {finding}")

        if 'implementation_guidance' in rec:
            print(f"\nImplementation Guidance:")
            for guidance in rec['implementation_guidance'][:3]:
                print(f"• {guidance}")

    # Generate comprehensive report
    print("\n5. Generating comprehensive report...")

    try:
        report = advanced_analyzer.generate_comprehensive_report()

        # Save report
        os.makedirs("results/advanced_causal_demo", exist_ok=True)
        report_path = "results/advanced_causal_demo/comprehensive_causal_report.txt"

        with open(report_path, 'w') as f:
            f.write(report)

        print(f"✓ Comprehensive report saved to: {report_path}")

        # Show excerpt
        print(f"\nReport Excerpt:")
        print("-" * 30)
        report_lines = report.split('\n')[:25]  # First 25 lines
        for line in report_lines:
            if line.strip():
                print(line)
        print("... (see full report for complete analysis)")

    except Exception as e:
        print(f"Report generation failed: {e}")

    return comprehensive_results


def demo_causal_validation():
    """Demonstrate causal validation and robustness testing"""

    print("\n" + "="*80)
    print("CAUSAL VALIDATION AND ROBUSTNESS TESTING DEMO")
    print("="*80)

    # Generate data for validation
    print("\n1. Generating data for validation testing...")
    panel_data, features_data, _ = generate_realistic_trading_data(n_days=150, n_traders=5)

    # Prepare data for validation
    X = features_data.select_dtypes(include=[np.number])
    y = panel_data['actual_pnl']
    treatment = (panel_data['predicted_pnl'] > 0).astype(int)

    print(f"✓ Validation dataset: {len(X)} observations, {X.shape[1]} features")

    # Initialize validation suite
    print("\n2. Initializing Causal Validation Suite...")
    validator = CausalValidationSuite(n_bootstrap=50)  # Reduced for demo speed

    # Create a simple "causal estimator" for validation
    class MockCausalEstimator:
        def __init__(self):
            treated_outcomes = y[treatment == 1]
            control_outcomes = y[treatment == 0]
            self.ate_estimate_ = treated_outcomes.mean() - control_outcomes.mean()

    mock_estimator = MockCausalEstimator()

    # Run validation
    print("\n3. Running causal validation tests...")
    print("   This includes sensitivity analysis, placebo tests, and robustness checks...")

    validation_config = {
        'sensitivity_config': {'confounder_strengths': [0.1, 0.3, 0.5]},
        'placebo_config': {'n_placebo_tests': 20},  # Reduced for demo
        'bootstrap_config': {'n_bootstrap': 30},    # Reduced for demo
        'cv_config': {'n_folds': 3}
    }

    validation_results = validator.validate_causal_model(
        causal_estimator=mock_estimator,
        X=X,
        y=y,
        treatment=treatment,
        validation_config=validation_config
    )

    # Display validation results
    print("\n4. Validation Results Summary:")
    print("-" * 40)

    # Overall assessment
    if 'overall_assessment' in validation_results:
        assessment = validation_results['overall_assessment']
        print(f"Overall Score: {assessment.get('overall_score', 0):.2f}/1.0")
        print(f"Reliability Rating: {assessment.get('reliability_rating', 'unknown')}")

        if 'strengths' in assessment:
            print(f"\nStrengths:")
            for strength in assessment['strengths']:
                print(f"• {strength}")

        if 'key_concerns' in assessment:
            print(f"\nKey Concerns:")
            for concern in assessment['key_concerns']:
                print(f"• {concern}")

    # Sensitivity analysis
    if 'sensitivity_analysis' in validation_results and 'error' not in validation_results['sensitivity_analysis']:
        sens = validation_results['sensitivity_analysis']
        if 'robustness_metrics' in sens:
            rob = sens['robustness_metrics']
            print(f"\nSensitivity Analysis:")
            print(f"• Robustness score: {rob.get('robustness_score', 0):.2f}")
            print(f"• Robust to confounders: {'Yes' if rob.get('is_robust', False) else 'No'}")

    # Placebo tests
    if 'placebo_tests' in validation_results and 'error' not in validation_results['placebo_tests']:
        placebo = validation_results['placebo_tests']
        if 'statistical_assessment' in placebo:
            stat = placebo['statistical_assessment']
            print(f"\nPlacebo Tests:")
            print(f"• Mean placebo effect: ${stat.get('mean_placebo_effect', 0):.3f}")
            print(f"• Fraction significant: {stat.get('fraction_significant', 0):.1%}")

    return validation_results


def main():
    """Main demo function"""

    print("ADVANCED CAUSAL IMPACT ANALYSIS - COMPREHENSIVE DEMO")
    print("="*80)
    print("This demo showcases state-of-the-art causal inference methods for")
    print("analyzing trading model impact and proving model utility.")
    print("\nMethods demonstrated:")
    print("• Double Machine Learning (DML)")
    print("• Synthetic Control Method")
    print("• Comprehensive causal validation")
    print("• Advanced business impact assessment")

    try:
        # Demo 1: Double Machine Learning
        dml_results = demo_double_ml_analysis()

        # Demo 2: Synthetic Control
        sc_results = demo_synthetic_control_analysis()

        # Demo 3: Comprehensive Analysis
        comprehensive_results = demo_comprehensive_analysis()

        # Demo 4: Validation
        validation_results = demo_causal_validation()

        print("\n" + "="*80)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("="*80)

        print("\nKey Outcomes:")
        print("✓ Demonstrated Double Machine Learning for robust causal effect estimation")
        print("✓ Showed Synthetic Control Method for counterfactual analysis")
        print("✓ Ran comprehensive analysis combining multiple state-of-the-art methods")
        print("✓ Performed rigorous causal validation and robustness testing")

        print("\nBusiness Value:")
        print("• These methods provide scientifically rigorous proof of model utility")
        print("• Results can support regulatory requirements and business decisions")
        print("• Multiple methods provide robustness and confidence in findings")
        print("• Comprehensive validation ensures reliability of causal claims")

        print("\nNext Steps:")
        print("• Apply these methods to your actual trading data")
        print("• Customize validation tests for your specific use case")
        print("• Use results to optimize model deployment strategies")
        print("• Present findings to stakeholders with confidence")

        print(f"\nReports saved in: results/advanced_causal_demo/")

    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
