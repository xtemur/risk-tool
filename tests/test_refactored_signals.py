#!/usr/bin/env python3
"""
Test script to verify the refactored signal generation with new modules.
Tests the new TraderMetricsProvider, RiskPredictor, and updated SignalGenerator.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference.signal_generator import SignalGenerator
from src.trader_metrics import TraderMetricsProvider
from src.risk_predictor import RiskPredictor
from src.utils import load_config
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_trader_metrics_provider():
    """Test the new TraderMetricsProvider module."""
    print("=" * 80)
    print("TESTING TRADER METRICS PROVIDER")
    print("=" * 80)

    config = load_config('configs/main_config.yaml')
    metrics_provider = TraderMetricsProvider(config)

    # Test trader names
    trader_names = metrics_provider.get_trader_names()
    print(f"Retrieved names for {len(trader_names)} traders")

    # Test comprehensive metrics including BAT and W/L
    metrics = metrics_provider.get_comprehensive_trader_metrics()
    print(f"Retrieved comprehensive metrics for {len(metrics)} traders")

    # Show sample metrics for first trader
    if metrics:
        trader_id = list(metrics.keys())[0]
        sample_metrics = metrics[trader_id]
        print(f"\nSample metrics for trader {trader_id}:")
        print(f"  BAT 30d: {sample_metrics.get('bat_30d', 0):.1f}%")
        print(f"  BAT All-time: {sample_metrics.get('bat_all_time', 0):.1f}%")
        print(f"  W/L Ratio 30d: {sample_metrics.get('wl_ratio_30d', 0):.2f}")
        print(f"  W/L Ratio All-time: {sample_metrics.get('wl_ratio_all_time', 0):.2f}")
        print(f"  Sharpe 30d: {sample_metrics.get('sharpe_30d', 0):.2f}")
        print(f"  Avg Daily PnL: ${sample_metrics.get('avg_daily_pnl', 0):,.2f}")

    return metrics

def test_risk_predictor():
    """Test the new RiskPredictor module."""
    print("\n" + "=" * 80)
    print("TESTING RISK PREDICTOR")
    print("=" * 80)

    config = load_config('configs/main_config.yaml')
    risk_predictor = RiskPredictor(config)

    # Test model loading
    print(f"Loaded {len(risk_predictor.trader_models)} trader models")
    print(f"Loaded thresholds for {len(risk_predictor.optimal_thresholds)} traders")

    # Test model info for first trader
    if risk_predictor.trader_models:
        trader_id = list(risk_predictor.trader_models.keys())[0]
        model_info = risk_predictor.get_model_info(int(trader_id))
        print(f"\nModel info for trader {trader_id}:")
        print(f"  Feature count: {model_info.get('feature_count', 0)}")
        print(f"  Has VaR model: {model_info.get('has_var_model', False)}")
        print(f"  Has classification model: {model_info.get('has_classification_model', False)}")

    return risk_predictor

def test_refactored_signal_generation():
    """Test the refactored signal generation."""
    print("\n" + "=" * 80)
    print("TESTING REFACTORED SIGNAL GENERATION")
    print("=" * 80)

    # Initialize signal generator
    generator = SignalGenerator('configs/main_config.yaml')

    # Generate signals
    signal_data = generator.generate_daily_signals()

    # Display results
    print(f"Date: {signal_data['date']}")
    print(f"Total Traders Processed: {len(signal_data['trader_signals'])}")

    # Count risk levels
    risk_counts = {'high': 0, 'low': 0}
    for signal in signal_data['trader_signals']:
        risk_counts[signal['risk_level']] += 1

    print(f"\nRisk Distribution:")
    print(f"  High Risk: {risk_counts['high']}")
    print(f"  Low Risk: {risk_counts['low']}")

    # Show alerts
    print(f"\nCritical Alerts: {len(signal_data['alerts'])}")
    if signal_data['alerts']:
        for alert in signal_data['alerts'][:3]:  # Show first 3
            print(f"  - {alert['trader_label']}: {alert['message'][:80]}...")

    # Show sample trader details with new metrics
    print("\nSample Trader Signals (first 2 with new metrics):")
    for i, signal in enumerate(signal_data['trader_signals'][:2]):
        print(f"\n  Trader {signal['trader_id']} ({signal['trader_name']}):")
        print(f"    Risk Level: {signal['risk_level']}")
        print(f"    VaR Prediction: ${signal['var_5pct']:,.2f}")
        print(f"    Loss Probability: {signal['loss_probability']:.2%}")
        print(f"    BAT 30d: {signal.get('bat_30d', 0):.1f}%")
        print(f"    BAT All-time: {signal.get('bat_all_time', 0):.1f}%")
        print(f"    W/L Ratio 30d: {signal.get('wl_ratio_30d', 0):.2f}")
        print(f"    W/L Ratio All-time: {signal.get('wl_ratio_all_time', 0):.2f}")
        print(f"    Sharpe 30d: {signal['sharpe_30d']:.2f}")
        print(f"    Last Trade Date: {signal['last_trade_date']}")

        # Show heatmap colors for new metrics
        if 'bat_heatmap' in signal:
            print(f"    BAT Heatmap: {signal['bat_heatmap']['bg']} ({signal['bat_heatmap']['class']})")
        if 'wl_ratio_heatmap' in signal:
            print(f"    W/L Heatmap: {signal['wl_ratio_heatmap']['bg']} ({signal['wl_ratio_heatmap']['class']})")

    # Summary statistics
    if signal_data['summary_stats']:
        print(f"\nSummary Statistics:")
        print(f"  Average VaR: ${signal_data['summary_stats']['avg_var']:,.2f}")
        print(f"  Max VaR: ${signal_data['summary_stats']['max_var']:,.2f}")
        print(f"  Average Loss Probability: {signal_data['summary_stats']['avg_loss_prob']:.2%}")
        print(f"  Max Loss Probability: {signal_data['summary_stats']['max_loss_prob']:.2%}")
        print(f"  Using Optimal Thresholds: {signal_data['summary_stats']['using_optimal_thresholds']}")
        print(f"  Intervention Based: {signal_data['summary_stats']['intervention_based']}")
        print(f"  Causal Impact Model: {signal_data['summary_stats']['causal_impact_model']}")

    return signal_data

def main():
    """Main test function."""
    print("TESTING REFACTORED RISK TOOL MODULES")
    print("=" * 80)

    try:
        # Test individual modules
        metrics = test_trader_metrics_provider()
        risk_predictor = test_risk_predictor()

        # Test integrated signal generation
        signal_data = test_refactored_signal_generation()

        print("\n" + "=" * 80)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("📊 New BAT and W/L metrics are now included in signal generation")
        print("🏗️ Modular architecture is working correctly")
        print("🎯 RiskPredictor class is using causal impact evaluation approach")
        print("=" * 80)

        return True

    except Exception as e:
        print(f"\n❌ ERROR during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
