#!/usr/bin/env python3
"""
Causal Impact Analysis for Risk Model
Analyzes the economic impact of using the VaR prediction model in production
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(filepath):
    """Load and prepare the backtesting results data"""
    print("Loading backtesting results...")
    df = pd.read_csv(filepath)

    # Convert trade_date to datetime
    df['trade_date'] = pd.to_datetime(df['trade_date'])

    # Sort by fold and date for proper time series analysis
    df = df.sort_values(['fold', 'trade_date', 'account_id']).reset_index(drop=True)

    print(f"Loaded {len(df):,} predictions")
    print(f"Date range: {df['trade_date'].min()} to {df['trade_date'].max()}")
    print(f"Unique traders: {df['account_id'].nunique()}")
    print(f"Walk-forward folds: {df['fold'].nunique()}")

    return df

def economic_impact_analysis(df):
    """Calculate economic impact of VaR model usage"""
    print("\n" + "="*60)
    print("1. ECONOMIC IMPACT ANALYSIS")
    print("="*60)

    # Basic P&L statistics
    total_pnl = df['true_pnl'].sum()
    avg_daily_pnl = df['true_pnl'].mean()

    print(f"Total Actual P&L: ${total_pnl:,.2f}")
    print(f"Average Daily P&L: ${avg_daily_pnl:,.2f}")

    # VaR breach analysis
    var_breaches = df[df['true_large_loss'] == 1]
    print(f"\nVaR Breaches: {len(var_breaches)} out of {len(df)} predictions ({len(var_breaches)/len(df)*100:.1f}%)")

    # Money lost on VaR breach days
    breach_losses = var_breaches['true_pnl'].sum()
    print(f"Total P&L on VaR breach days: ${breach_losses:,.2f}")
    print(f"Average loss per breach day: ${var_breaches['true_pnl'].mean():,.2f}")

    # Simulate strict VaR adherence
    # Assume traders would limit positions to VaR amount when model predicts high risk
    high_risk_threshold = 0.15  # 15% probability threshold
    high_risk_days = df[df['pred_loss_proba'] > high_risk_threshold]

    print(f"\nHigh Risk Days (prob > {high_risk_threshold:.0%}): {len(high_risk_days)}")

    # Calculate potential profit lost if positions were limited to VaR
    # For high-risk days with positive P&L, this would be opportunity cost
    high_risk_positive_pnl = high_risk_days[high_risk_days['true_pnl'] > 0]['true_pnl'].sum()
    high_risk_negative_pnl = high_risk_days[high_risk_days['true_pnl'] < 0]['true_pnl'].sum()

    print(f"Positive P&L on high-risk days: ${high_risk_positive_pnl:,.2f}")
    print(f"Negative P&L on high-risk days: ${high_risk_negative_pnl:,.2f}")

    # Net benefit calculation
    # Assume VaR compliance would reduce losses by 70% but also cap gains by 50%
    loss_reduction = abs(high_risk_negative_pnl) * 0.7
    opportunity_cost = high_risk_positive_pnl * 0.5
    net_benefit = loss_reduction - opportunity_cost

    print(f"\nEstimated loss reduction (70%): ${loss_reduction:,.2f}")
    print(f"Estimated opportunity cost (50%): ${opportunity_cost:,.2f}")
    print(f"Net Economic Benefit: ${net_benefit:,.2f}")

    return {
        'total_pnl': total_pnl,
        'breach_losses': breach_losses,
        'net_benefit': net_benefit,
        'high_risk_days': len(high_risk_days),
        'var_breaches': len(var_breaches)
    }

def position_sizing_analysis(df):
    """Simulate different position sizing strategies"""
    print("\n" + "="*60)
    print("2. POSITION SIZING IMPACT ANALYSIS")
    print("="*60)

    # Baseline: Original P&L (no model)
    baseline_pnl = df['true_pnl'].sum()

    # Conservative strategy: Reduce position by 50% when loss probability > 30%
    conservative_threshold = 0.30
    df['conservative_pnl'] = df['true_pnl'].copy()
    conservative_mask = df['pred_loss_proba'] > conservative_threshold
    df.loc[conservative_mask, 'conservative_pnl'] = df.loc[conservative_mask, 'true_pnl'] * 0.5
    conservative_pnl = df['conservative_pnl'].sum()

    # Aggressive strategy: Increase position by 25% when loss probability < 10%
    aggressive_threshold = 0.10
    df['aggressive_pnl'] = df['true_pnl'].copy()
    aggressive_mask = df['pred_loss_proba'] < aggressive_threshold
    df.loc[aggressive_mask, 'aggressive_pnl'] = df.loc[aggressive_mask, 'true_pnl'] * 1.25
    aggressive_pnl = df['aggressive_pnl'].sum()

    # Combined strategy
    df['combined_pnl'] = df['true_pnl'].copy()
    # Apply conservative adjustment first
    df.loc[conservative_mask, 'combined_pnl'] = df.loc[conservative_mask, 'true_pnl'] * 0.5
    # Apply aggressive adjustment (excluding conservative days)
    combined_aggressive_mask = aggressive_mask & ~conservative_mask
    df.loc[combined_aggressive_mask, 'combined_pnl'] = df.loc[combined_aggressive_mask, 'true_pnl'] * 1.25
    combined_pnl = df['combined_pnl'].sum()

    print(f"Baseline P&L (no model): ${baseline_pnl:,.2f}")
    print(f"Conservative Strategy P&L: ${conservative_pnl:,.2f} ({((conservative_pnl/baseline_pnl-1)*100):+.1f}%)")
    print(f"Aggressive Strategy P&L: ${aggressive_pnl:,.2f} ({((aggressive_pnl/baseline_pnl-1)*100):+.1f}%)")
    print(f"Combined Strategy P&L: ${combined_pnl:,.2f} ({((combined_pnl/baseline_pnl-1)*100):+.1f}%)")

    # Days affected by each strategy
    conservative_days = conservative_mask.sum()
    aggressive_days = aggressive_mask.sum()
    combined_aggressive_days = combined_aggressive_mask.sum()

    print(f"\nDays with reduced positions (conservative): {conservative_days} ({conservative_days/len(df)*100:.1f}%)")
    print(f"Days with increased positions (aggressive): {aggressive_days} ({aggressive_days/len(df)*100:.1f}%)")
    print(f"Days with increased positions (combined): {combined_aggressive_days} ({combined_aggressive_days/len(df)*100:.1f}%)")

    return {
        'baseline_pnl': baseline_pnl,
        'conservative_pnl': conservative_pnl,
        'aggressive_pnl': aggressive_pnl,
        'combined_pnl': combined_pnl,
        'conservative_improvement': (conservative_pnl/baseline_pnl-1)*100,
        'aggressive_improvement': (aggressive_pnl/baseline_pnl-1)*100,
        'combined_improvement': (combined_pnl/baseline_pnl-1)*100
    }

def risk_adjusted_performance(df):
    """Calculate risk-adjusted performance metrics"""
    print("\n" + "="*60)
    print("3. RISK-ADJUSTED PERFORMANCE ANALYSIS")
    print("="*60)

    # Daily returns by trader and date
    daily_returns = df.groupby(['trade_date'])['true_pnl'].sum().reset_index()
    daily_returns = daily_returns.sort_values('trade_date')

    # Conservative strategy daily returns
    conservative_returns = df.groupby(['trade_date'])['conservative_pnl'].sum().reset_index()
    conservative_returns = conservative_returns.sort_values('trade_date')

    # Calculate Sharpe ratios (assuming daily returns, annualized)
    baseline_sharpe = (daily_returns['true_pnl'].mean() / daily_returns['true_pnl'].std()) * np.sqrt(252) if daily_returns['true_pnl'].std() > 0 else 0
    conservative_sharpe = (conservative_returns['conservative_pnl'].mean() / conservative_returns['conservative_pnl'].std()) * np.sqrt(252) if conservative_returns['conservative_pnl'].std() > 0 else 0

    # Maximum drawdown calculation
    baseline_cumsum = daily_returns['true_pnl'].cumsum()
    baseline_running_max = baseline_cumsum.expanding().max()
    baseline_drawdown = (baseline_cumsum - baseline_running_max).min()

    conservative_cumsum = conservative_returns['conservative_pnl'].cumsum()
    conservative_running_max = conservative_cumsum.expanding().max()
    conservative_drawdown = (conservative_cumsum - conservative_running_max).min()

    # Tail risk (95th percentile losses)
    baseline_var95 = daily_returns['true_pnl'].quantile(0.05)  # 5th percentile (worst 5%)
    conservative_var95 = conservative_returns['conservative_pnl'].quantile(0.05)

    print(f"Baseline Sharpe Ratio: {baseline_sharpe:.3f}")
    print(f"Conservative Strategy Sharpe Ratio: {conservative_sharpe:.3f}")
    print(f"Sharpe Ratio Improvement: {((conservative_sharpe/baseline_sharpe-1)*100):+.1f}%" if baseline_sharpe != 0 else "N/A")

    print(f"\nBaseline Maximum Drawdown: ${baseline_drawdown:,.2f}")
    print(f"Conservative Maximum Drawdown: ${conservative_drawdown:,.2f}")
    print(f"Drawdown Reduction: ${baseline_drawdown - conservative_drawdown:,.2f} ({((conservative_drawdown/baseline_drawdown-1)*100):+.1f}%)" if baseline_drawdown != 0 else "N/A")

    print(f"\nBaseline 95% VaR (daily): ${baseline_var95:,.2f}")
    print(f"Conservative 95% VaR (daily): ${conservative_var95:,.2f}")
    print(f"VaR Improvement: ${baseline_var95 - conservative_var95:,.2f} ({((conservative_var95/baseline_var95-1)*100):+.1f}%)" if baseline_var95 != 0 else "N/A")

    # Model accuracy metrics
    accuracy = ((df['pred_loss_proba'] > 0.15) == (df['true_large_loss'] == 1)).mean()
    precision = ((df['pred_loss_proba'] > 0.15) & (df['true_large_loss'] == 1)).sum() / (df['pred_loss_proba'] > 0.15).sum() if (df['pred_loss_proba'] > 0.15).sum() > 0 else 0
    recall = ((df['pred_loss_proba'] > 0.15) & (df['true_large_loss'] == 1)).sum() / (df['true_large_loss'] == 1).sum() if (df['true_large_loss'] == 1).sum() > 0 else 0

    print(f"\nModel Performance (15% threshold):")
    print(f"Accuracy: {accuracy:.1%}")
    print(f"Precision: {precision:.1%}")
    print(f"Recall: {recall:.1%}")

    return {
        'baseline_sharpe': baseline_sharpe,
        'conservative_sharpe': conservative_sharpe,
        'baseline_drawdown': baseline_drawdown,
        'conservative_drawdown': conservative_drawdown,
        'baseline_var95': baseline_var95,
        'conservative_var95': conservative_var95,
        'model_accuracy': accuracy,
        'model_precision': precision,
        'model_recall': recall
    }

def per_trader_analysis(df):
    """Analyze impact per individual trader"""
    print("\n" + "="*60)
    print("4. PER-TRADER ANALYSIS")
    print("="*60)

    # Calculate metrics by trader
    trader_stats = df.groupby('account_id').agg({
        'true_pnl': ['sum', 'mean', 'std', 'count'],
        'conservative_pnl': 'sum',
        'pred_loss_proba': 'mean',
        'true_large_loss': 'sum'
    }).round(2)

    # Flatten column names
    trader_stats.columns = ['total_pnl', 'avg_pnl', 'pnl_std', 'trade_count', 'conservative_total', 'avg_risk_prob', 'breach_count']

    # Calculate improvement per trader
    trader_stats['pnl_improvement'] = trader_stats['conservative_total'] - trader_stats['total_pnl']
    trader_stats['pnl_improvement_pct'] = ((trader_stats['conservative_total'] / trader_stats['total_pnl'] - 1) * 100).fillna(0)

    # Calculate Sharpe ratio per trader (approximation)
    trader_stats['sharpe_approx'] = (trader_stats['avg_pnl'] / trader_stats['pnl_std']).fillna(0)

    # Sort by improvement
    trader_stats = trader_stats.sort_values('pnl_improvement', ascending=False)

    print("Trader Performance Summary:")
    print("-" * 120)
    print(f"{'Trader':>8} {'Total P&L':>12} {'Conservative':>12} {'Improvement':>12} {'Improve %':>10} {'Avg Risk':>10} {'Breaches':>10} {'Trades':>8}")
    print("-" * 120)

    for trader_id, row in trader_stats.iterrows():
        print(f"{trader_id:>8,} ${row['total_pnl']:>11,.0f} ${row['conservative_total']:>11,.0f} ${row['pnl_improvement']:>11,.0f} {row['pnl_improvement_pct']:>9.1f}% {row['avg_risk_prob']:>9.1%} {row['breach_count']:>9.0f} {row['trade_count']:>7.0f}")

    # Summary statistics
    total_improvement = trader_stats['pnl_improvement'].sum()
    traders_improved = (trader_stats['pnl_improvement'] > 0).sum()
    best_trader = trader_stats.index[0]
    worst_trader = trader_stats.index[-1]

    print("-" * 120)
    print(f"Total Portfolio Improvement: ${total_improvement:,.2f}")
    print(f"Traders with Improvement: {traders_improved} out of {len(trader_stats)}")
    print(f"Best Performing Trader (with model): {best_trader} (${trader_stats.loc[best_trader, 'pnl_improvement']:,.2f} improvement)")
    print(f"Worst Performing Trader (with model): {worst_trader} (${trader_stats.loc[worst_trader, 'pnl_improvement']:,.2f} change)")

    return trader_stats

def generate_summary_report(economic_results, position_results, risk_results, trader_stats, df):
    """Generate executive summary of findings"""
    print("\n" + "="*60)
    print("EXECUTIVE SUMMARY - CAUSAL IMPACT ANALYSIS")
    print("="*60)

    print(f"\n📊 OVERALL IMPACT:")
    print(f"• Total Historical P&L: ${economic_results['total_pnl']:,.2f}")
    print(f"• Net Economic Benefit from Model: ${economic_results['net_benefit']:,.2f}")
    print(f"• Best Strategy Improvement: {position_results['conservative_improvement']:+.1f}% (${position_results['conservative_pnl'] - position_results['baseline_pnl']:,.2f})")

    print(f"\n🎯 RISK REDUCTION:")
    print(f"• VaR Breaches Occurred: {economic_results['var_breaches']} out of 4,950 predictions ({economic_results['var_breaches']/4950*100:.1f}%)")
    print(f"• Maximum Drawdown Reduction: ${risk_results['baseline_drawdown'] - risk_results['conservative_drawdown']:,.2f}")
    print(f"• 95% VaR Improvement: ${risk_results['baseline_var95'] - risk_results['conservative_var95']:,.2f}")
    print(f"• Sharpe Ratio Improvement: {((risk_results['conservative_sharpe']/risk_results['baseline_sharpe']-1)*100):+.1f}%" if risk_results['baseline_sharpe'] != 0 else "N/A")

    print(f"\n👥 TRADER IMPACT:")
    print(f"• Traders Benefiting from Model: {(trader_stats['pnl_improvement'] > 0).sum()} out of {len(trader_stats)}")
    print(f"• Average Improvement per Trader: ${trader_stats['pnl_improvement'].mean():,.2f}")
    print(f"• Top Performer Gain: ${trader_stats['pnl_improvement'].max():,.2f}")

    print(f"\n🔍 MODEL PERFORMANCE:")
    print(f"• Accuracy: {risk_results['model_accuracy']:.1%}")
    print(f"• Precision: {risk_results['model_precision']:.1%}")
    print(f"• Recall: {risk_results['model_recall']:.1%}")

    print(f"\n💡 PRODUCTION RECOMMENDATIONS:")
    print(f"• Implement Conservative Strategy: Reduce positions by 50% when loss probability > 30%")
    print(f"• Expected Annual Benefit: ${(position_results['conservative_pnl'] - position_results['baseline_pnl']) * (252/len(pd.to_datetime(df['trade_date']).dt.date.unique())):,.2f}")
    print(f"• Focus on High-Risk Traders: Prioritize model adoption for traders with frequent VaR breaches")
    print(f"• Monitor Model Performance: Current accuracy of {risk_results['model_accuracy']:.1%} suggests room for improvement")

def main():
    """Main analysis function"""
    filepath = "/Users/temurbekkhujaev/Repos/risk-tool/models/production_model_artifacts/strict_walk_forward_results.csv"

    # Load data
    df = load_and_prepare_data(filepath)

    # Run analyses
    economic_results = economic_impact_analysis(df)
    position_results = position_sizing_analysis(df)
    risk_results = risk_adjusted_performance(df)
    trader_stats = per_trader_analysis(df)

    # Generate summary
    generate_summary_report(economic_results, position_results, risk_results, trader_stats, df)

    print(f"\n✅ Analysis complete! Results based on {len(df):,} predictions across {df['fold'].nunique()} time folds.")

if __name__ == "__main__":
    main()
