#!/usr/bin/env python3
"""
Run causal impact analysis on the test period results.
Shows what would have happened if traders followed the model's risk signals.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.main_pipeline import RiskManagementPipeline
from src.evaluation.causal_impact import CausalImpactAnalyzer

def run_causal_impact_analysis():
    """Run complete causal impact analysis."""

    print("Starting Causal Impact Analysis...")
    print("="*60)

    # Step 1: Run the pipeline to get predictions
    print("1. Running risk management pipeline...")
    pipeline = RiskManagementPipeline(test_cutoff_date='2025-04-01')

    # Load and prepare data
    pipeline.load_and_prepare_data()

    # Use simple default parameters for speed
    pipeline.model.best_params = pipeline.model.get_default_params()

    # Train final model
    pipeline.train_final_model()

    # Get test predictions
    X_test, y_test = pipeline.validator.create_feature_target_split(
        pipeline.test_features, target_col='target'
    )

    if len(X_test) == 0:
        print("ERROR: No test data available")
        return

    # Make predictions
    y_pred_test = pipeline.model.predict(X_test)
    risk_signals = pipeline.model.generate_risk_signals(y_pred_test)

    # Prepare test data with predictions
    test_with_target = pipeline.test_features.dropna(subset=['target'])

    predictions_df = pd.DataFrame({
        'account_id': test_with_target['account_id'].values,
        'trade_date': test_with_target['trade_date'].values,
        'actual_target': y_test.values,
        'predicted_target': y_pred_test,
        'actual_pnl': test_with_target['next_day_realized_pnl'].values,
        'risk_signal': risk_signals
    })

    print(f"Test period: {predictions_df['trade_date'].min()} to {predictions_df['trade_date'].max()}")
    print(f"Total test samples: {len(predictions_df)}")
    print(f"Risk signal distribution:")
    print(f"  High Risk (0): {np.sum(risk_signals == 0)} days ({np.mean(risk_signals == 0):.1%})")
    print(f"  Neutral (1): {np.sum(risk_signals == 1)} days ({np.mean(risk_signals == 1):.1%})")
    print(f"  Low Risk (2): {np.sum(risk_signals == 2)} days ({np.mean(risk_signals == 2):.1%})")

    # Step 2: Run causal impact analysis
    print("\n2. Analyzing causal impact of risk-based strategies...")
    analyzer = CausalImpactAnalyzer()

    # Analyze different trading strategies
    impact_results = analyzer.analyze_trading_strategy_impact(predictions_df)

    # Print comprehensive report
    analyzer.print_causal_impact_report(impact_results)

    # Step 3: Generate daily analysis
    print("\n3. Generating daily impact analysis...")
    daily_analysis = analyzer.generate_daily_impact_analysis(predictions_df)

    # Show key statistics
    print(f"\nDAILY PERFORMANCE COMPARISON:")
    print(f"Baseline Total PnL: ${daily_analysis['baseline_pnl'].sum():,.2f}")
    print(f"Position Sizing Total PnL: ${daily_analysis['position_sizing_pnl'].sum():,.2f}")
    print(f"Trade Filtering Total PnL: ${daily_analysis['filtered_pnl'].sum():,.2f}")
    print(f"Combined Strategy Total PnL: ${daily_analysis['combined_pnl'].sum():,.2f}")

    # Show best and worst days
    print(f"\nBEST PREDICTION DAYS (Top 5):")
    best_days = daily_analysis.nlargest(5, 'combined_improvement')[
        ['trade_date', 'risk_signal', 'actual_pnl', 'combined_improvement']
    ]
    for _, row in best_days.iterrows():
        signal_name = ['High Risk', 'Neutral', 'Low Risk'][row['risk_signal']]
        print(f"  {row['trade_date'].strftime('%Y-%m-%d')}: {signal_name} | "
              f"Actual: ${row['actual_pnl']:,.2f} | "
              f"Improvement: ${row['combined_improvement']:,.2f}")

    print(f"\nWORST PREDICTION DAYS (Bottom 5):")
    worst_days = daily_analysis.nsmallest(5, 'combined_improvement')[
        ['trade_date', 'risk_signal', 'actual_pnl', 'combined_improvement']
    ]
    for _, row in worst_days.iterrows():
        signal_name = ['High Risk', 'Neutral', 'Low Risk'][row['risk_signal']]
        print(f"  {row['trade_date'].strftime('%Y-%m-%d')}: {signal_name} | "
              f"Actual: ${row['actual_pnl']:,.2f} | "
              f"Improvement: ${row['combined_improvement']:,.2f}")

    # Step 4: Analyze by risk signal effectiveness
    print(f"\nRISK SIGNAL EFFECTIVENESS:")

    for signal in [0, 1, 2]:
        signal_data = daily_analysis[daily_analysis['risk_signal'] == signal]
        signal_name = ['High Risk', 'Neutral', 'Low Risk'][signal]

        if len(signal_data) > 0:
            avg_actual_pnl = signal_data['actual_pnl'].mean()
            total_days = len(signal_data)
            win_rate = (signal_data['actual_pnl'] > 0).mean()

            print(f"  {signal_name}: {total_days} days | "
                  f"Avg PnL: ${avg_actual_pnl:,.2f} | "
                  f"Win Rate: {win_rate:.1%}")

    # Step 5: Generate summary insights
    print(f"\n" + "="*60)
    print("KEY INSIGHTS:")
    print("="*60)

    # Model prediction accuracy
    high_risk_days = daily_analysis[daily_analysis['risk_signal'] == 0]
    low_risk_days = daily_analysis[daily_analysis['risk_signal'] == 2]

    if len(high_risk_days) > 0:
        high_risk_avg_pnl = high_risk_days['actual_pnl'].mean()
        high_risk_loss_rate = (high_risk_days['actual_pnl'] < 0).mean()
        print(f"🔴 High Risk Signal Validation:")
        print(f"   Average PnL on high-risk days: ${high_risk_avg_pnl:,.2f}")
        print(f"   Loss rate on high-risk days: {high_risk_loss_rate:.1%}")

    if len(low_risk_days) > 0:
        low_risk_avg_pnl = low_risk_days['actual_pnl'].mean()
        low_risk_win_rate = (low_risk_days['actual_pnl'] > 0).mean()
        print(f"🟢 Low Risk Signal Validation:")
        print(f"   Average PnL on low-risk days: ${low_risk_avg_pnl:,.2f}")
        print(f"   Win rate on low-risk days: {low_risk_win_rate:.1%}")

    # Trading frequency impact
    baseline_active_days = (daily_analysis['baseline_pnl'] != 0).sum()
    filtered_active_days = (daily_analysis['filtered_pnl'] != 0).sum()

    print(f"📊 Trading Activity Impact:")
    print(f"   Baseline trading days: {baseline_active_days}")
    print(f"   Filtered trading days: {filtered_active_days}")
    print(f"   Activity reduction: {(1 - filtered_active_days/baseline_active_days):.1%}")

    print("="*60)
    print("Analysis complete! The model's risk signals show clear patterns.")

    return daily_analysis, impact_results

if __name__ == "__main__":
    daily_analysis, impact_results = run_causal_impact_analysis()
