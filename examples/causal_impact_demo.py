"""
Causal Impact Analysis Demo

Demonstrates how to calculate and interpret causal impact of model predictions
on trader PnL, answering the key business question: "How much would PnL change
if traders had listened to the model?"
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
from modeling import CausalImpactAnalyzer
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_sample_data(n_days=100, trader_id='DEMO001'):
    """
    Generate sample trading data for demonstration

    Args:
        n_days: Number of trading days
        trader_id: Trader identifier

    Returns:
        Tuple of (actual_pnl, predicted_pnl, dates)
    """
    np.random.seed(42)  # For reproducible results

    # Generate dates
    dates = pd.date_range(start='2024-01-01', periods=n_days, freq='B')  # Business days

    # Generate actual PnL with some realistic patterns
    # Base performance with trend and volatility
    trend = np.linspace(-200, 500, n_days)  # Improving performance over time
    volatility = np.random.normal(0, 150, n_days)  # Daily noise
    momentum = np.cumsum(np.random.normal(0, 50, n_days)) * 0.1  # Momentum effects

    actual_pnl = trend + volatility + momentum

    # Generate predicted PnL that has some skill but isn't perfect
    # Model predictions have correlation with actual but with bias and noise
    prediction_skill = 0.3  # How much signal the model captures
    prediction_bias = 20    # Model tends to be slightly optimistic
    prediction_noise = np.random.normal(0, 80, n_days)  # Model uncertainty

    predicted_pnl = (prediction_skill * actual_pnl +
                    prediction_bias +
                    prediction_noise)

    return actual_pnl, predicted_pnl, dates


def demo_causal_impact_analysis():
    """
    Main demonstration of causal impact analysis
    """
    print("=" * 80)
    print("CAUSAL IMPACT ANALYSIS DEMO")
    print("=" * 80)

    print("\n1. Generating Sample Trading Data...")

    # Generate sample data for demonstration
    actual_pnl, predicted_pnl, dates = generate_sample_data(n_days=100, trader_id='DEMO001')

    print(f"✓ Generated {len(actual_pnl)} days of trading data")
    print(f"✓ Date range: {dates.min().date()} to {dates.max().date()}")
    print(f"✓ Actual total PnL: ${np.sum(actual_pnl):,.2f}")
    print(f"✓ Predicted total PnL: ${np.sum(predicted_pnl):,.2f}")
    print(f"✓ Prediction correlation: {np.corrcoef(actual_pnl, predicted_pnl)[0,1]:.3f}")

    print("\n2. Initializing Causal Impact Analyzer...")

    # Initialize the analyzer
    analyzer = CausalImpactAnalyzer()
    print("✓ Causal Impact Analyzer initialized")

    print("\n3. Calculating Trading Impact Scenarios...")

    # Calculate causal impact
    causal_results = analyzer.calculate_trading_impact(
        actual_pnl=actual_pnl,
        predicted_pnl=predicted_pnl,
        dates=dates,
        trader_id='DEMO001'
    )

    print("✓ Causal impact analysis completed")

    print("\n4. Results Summary...")

    # Extract key results
    baseline = causal_results['baseline_performance']
    scenarios = causal_results['causal_impact_scenarios']
    significance = causal_results['statistical_significance']

    print(f"\nBASELINE PERFORMANCE:")
    print(f"  Total PnL: ${baseline['actual_total_pnl']:,.2f}")
    print(f"  Average Daily PnL: ${baseline['actual_avg_daily_pnl']:,.2f}")
    print(f"  Sharpe Ratio: {baseline['actual_sharpe']:.3f}")
    print(f"  Win Rate: {baseline['win_rate']:.1%}")
    print(f"  Winning Days: {baseline['winning_days']}")
    print(f"  Losing Days: {baseline['losing_days']}")

    print(f"\nCAUSAL IMPACT SCENARIOS:")

    # Perfect following scenario
    perfect = scenarios['perfect_following']
    print(f"\n  1. PERFECT MODEL FOLLOWING:")
    print(f"     PnL Improvement: ${perfect['pnl_improvement']:,.2f}")
    print(f"     Improvement %: {perfect['pnl_improvement_pct']:+.1f}%")
    print(f"     Description: {perfect['description']}")

    # Directional trading scenario
    directional = scenarios['directional_trading']
    print(f"\n  2. DIRECTIONAL TRADING:")
    print(f"     PnL Improvement: ${directional['pnl_improvement']:,.2f}")
    print(f"     Improvement %: {directional['pnl_improvement_pct']:+.1f}%")
    print(f"     Trading Frequency: {directional['trading_frequency']:.1%}")
    print(f"     Trade Days: {directional['trade_days']}")
    print(f"     Skip Days: {directional['skip_days']}")
    print(f"     PnL from Trades: ${directional['pnl_from_trades']:,.2f}")
    print(f"     PnL Saved by Skipping: ${directional['pnl_saved']:,.2f}")
    print(f"     Description: {directional['description']}")

    # Risk-adjusted trading scenario
    risk_adj = scenarios['risk_adjusted_trading']
    print(f"\n  3. RISK-ADJUSTED TRADING:")
    print(f"     PnL Improvement: ${risk_adj['pnl_improvement']:,.2f}")
    print(f"     Improvement %: {risk_adj['pnl_improvement_pct']:+.1f}%")
    print(f"     Average Position Weight: {risk_adj['avg_position_weight']:.2f}")
    print(f"     Description: {risk_adj['description']}")

    # Selective trading scenario
    selective = scenarios['selective_trading']
    print(f"\n  4. SELECTIVE TRADING:")
    print(f"     PnL Improvement: ${selective['pnl_improvement']:,.2f}")
    print(f"     Improvement %: {selective['pnl_improvement_pct']:+.1f}%")
    print(f"     Trading Frequency: {selective['trading_frequency']:.1%}")
    print(f"     Confidence Threshold: {selective['confidence_threshold']:.0%}")
    print(f"     Description: {selective['description']}")

    print(f"\nSTATISTICAL SIGNIFICANCE:")
    print(f"  T-statistic: {significance['t_statistic']:.3f}")
    print(f"  P-value: {significance['p_value']:.6f}")
    print(f"  Is Significant: {'Yes' if significance['is_significant'] else 'No'}")
    print(f"  Effect Size: {significance['effect_size']:.3f}")
    print(f"  95% Confidence Interval: [{significance['confidence_interval_95'][0]:.2f}, {significance['confidence_interval_95'][1]:.2f}]")
    print(f"  Interpretation: {significance['interpretation']}")

    print("\n5. Generating Comprehensive Report...")

    # Generate full report
    report = analyzer.generate_impact_report(causal_results)

    # Save report
    os.makedirs("results/causal_impact_demo", exist_ok=True)
    report_path = "results/causal_impact_demo/causal_impact_report_demo.txt"

    with open(report_path, 'w') as f:
        f.write(report)

    print(f"✓ Comprehensive report saved to: {report_path}")

    # Show first part of report
    print(f"\n  Report Preview:")
    print(f"  {'-' * 50}")
    report_lines = report.split('\n')[:20]  # First 20 lines
    for line in report_lines:
        if line.strip():
            print(f"  {line}")
    print(f"  ... (see full report in {report_path})")
    print(f"  {'-' * 50}")

    print("\n6. Multi-Trader Simulation...")

    # Simulate multiple traders for aggregate analysis
    trader_results = {}

    print("  Generating data for 5 traders...")

    for i in range(5):
        trader_id = f"TRADER{i+1:03d}"

        # Generate different performance patterns for each trader
        np.random.seed(42 + i)  # Different seed for each trader
        actual, predicted, trader_dates = generate_sample_data(n_days=80 + i*10, trader_id=trader_id)

        # Calculate causal impact for this trader
        trader_result = analyzer.calculate_trading_impact(
            actual_pnl=actual,
            predicted_pnl=predicted,
            dates=trader_dates,
            trader_id=trader_id
        )

        trader_results[trader_id] = trader_result

        # Show summary for this trader
        baseline_pnl = trader_result['baseline_performance']['actual_total_pnl']
        improvement = trader_result['causal_impact_scenarios']['perfect_following']['pnl_improvement']
        improvement_pct = trader_result['causal_impact_scenarios']['perfect_following']['pnl_improvement_pct']

        print(f"    {trader_id}: Baseline ${baseline_pnl:,.2f}, Improvement ${improvement:,.2f} ({improvement_pct:+.1f}%)")

    print("\n  Calculating aggregate analysis...")

    # Aggregate analysis
    aggregate_results = analyzer.analyze_multiple_traders(trader_results)

    if aggregate_results:
        agg = aggregate_results['aggregate_metrics']

        print(f"\n  PORTFOLIO-WIDE RESULTS:")
        print(f"    Total Traders: {agg['total_traders']}")
        print(f"    Total Baseline PnL: ${agg['total_baseline_pnl']:,.2f}")
        print(f"    Total Potential Improvement: ${agg['total_improvement']:,.2f}")
        print(f"    Total Improvement %: {agg['total_improvement_pct']:+.1f}%")
        print(f"    Traders Improved: {agg['traders_improved']} of {agg['total_traders']}")
        print(f"    Success Rate: {agg['improvement_success_rate']:.1%}")
        print(f"    Average Improvement per Trader: ${agg['avg_improvement_per_trader']:,.2f}")
        print(f"    Median Improvement per Trader: ${agg['median_improvement_per_trader']:,.2f}")

        # Save aggregate report
        # CausalImpactAnalyzer already imported at top

        aggregate_report = f"""
PORTFOLIO CAUSAL IMPACT ANALYSIS
================================

Total Portfolio Summary:
- Number of Traders: {agg['total_traders']}
- Combined Baseline PnL: ${agg['total_baseline_pnl']:,.2f}
- Combined Potential Improvement: ${agg['total_improvement']:,.2f} ({agg['total_improvement_pct']:+.1f}%)

Success Metrics:
- Traders Who Would Improve: {agg['traders_improved']} out of {agg['total_traders']}
- Portfolio Success Rate: {agg['improvement_success_rate']:.1%}
- Average Improvement per Trader: ${agg['avg_improvement_per_trader']:,.2f}

Distribution Analysis:
- 25th Percentile: ${aggregate_results['distribution_stats']['improvement_percentiles']['25th']:,.2f}
- 50th Percentile: ${aggregate_results['distribution_stats']['improvement_percentiles']['50th']:,.2f}
- 75th Percentile: ${aggregate_results['distribution_stats']['improvement_percentiles']['75th']:,.2f}
- Standard Deviation: ${aggregate_results['distribution_stats']['improvement_std']:,.2f}

BUSINESS IMPACT:
{agg['improvement_success_rate']:.0%} of traders would benefit from following model recommendations.
The model shows potential to improve portfolio PnL by ${agg['total_improvement']:,.2f} ({agg['total_improvement_pct']:+.1f}%).
"""

        aggregate_path = "results/causal_impact_demo/portfolio_aggregate_report.txt"
        with open(aggregate_path, 'w') as f:
            f.write(aggregate_report)

        print(f"✓ Portfolio aggregate report saved to: {aggregate_path}")

    print("\n" + "=" * 80)
    print("CAUSAL IMPACT DEMO COMPLETED")
    print("=" * 80)

    print("\nKEY INSIGHTS FROM DEMO:")
    print("✓ Model predictions can be translated into business value")
    print("✓ Multiple trading strategies show different risk/reward profiles")
    print("✓ Statistical significance testing validates impact estimates")
    print("✓ Portfolio-wide analysis shows aggregate potential")
    print("✓ Comprehensive reports provide actionable business insights")

    print("\nPRACTICAL APPLICATIONS:")
    print("• Risk management: Understand model limitations and opportunities")
    print("• Strategy optimization: Choose best model deployment approach")
    print("• ROI calculation: Quantify value of model development investment")
    print("• Trader training: Show concrete examples of model guidance value")
    print("• Regulatory reporting: Demonstrate model effectiveness with statistics")

    print(f"\nREPORTS GENERATED:")
    print(f"• Individual trader report: {report_path}")
    print(f"• Portfolio aggregate report: {aggregate_path}")


if __name__ == "__main__":
    demo_causal_impact_analysis()
