#!/usr/bin/env python3
"""
Final Comprehensive Trader Analysis - Individual PnL Impact Report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11

def extract_individual_trader_data():
    """Extract individual trader results from all strategies"""
    print("=== EXTRACTING INDIVIDUAL TRADER DATA ===")

    # Load strategy results
    strategy_results = pd.read_pickle('results/data/strategy_results.pkl')

    all_trader_data = []

    for strategy_name, strategy_data in strategy_results.items():
        print(f"\nProcessing {strategy_name} strategy...")

        if 'trader_results' in strategy_data:
            trader_results = strategy_data['trader_results']
            print(f"  Found {len(trader_results)} traders")

            # Handle both list and dict formats
            if isinstance(trader_results, list):
                trader_items = trader_results
            else:
                trader_items = trader_results.items()

            for trader_item in trader_items:
                if isinstance(trader_item, dict):
                    # List format: each item is a trader dict
                    trader_data = trader_item
                    trader_id = trader_data.get('trader_id')
                else:
                    # Dict format: (trader_id, trader_data) tuple
                    trader_id, trader_data = trader_item
                record = {
                    'trader_id': trader_id,
                    'strategy': strategy_name,
                    'baseline_pnl': trader_data.get('baseline_pnl', 0),
                    'strategy_pnl': trader_data.get('strategy_pnl', 0),
                    'baseline_sharpe': trader_data.get('baseline_sharpe', 0),
                    'strategy_sharpe': trader_data.get('strategy_sharpe', 0),
                    'total_observations': trader_data.get('total_days', trader_data.get('total_observations', 0)),
                }

                # Handle different risk day fields by strategy
                if strategy_name == 'trade_filtering':
                    # Trade filtering tracks filtered_days (days avoided)
                    record['high_risk_days'] = trader_data.get('filtered_days', 0)
                    record['low_risk_days'] = 0  # Not tracked for trade filtering
                    record['filtered_days'] = trader_data.get('filtered_days', 0)
                else:
                    # Position sizing and combined track high/low risk days
                    record['high_risk_days'] = trader_data.get('high_risk_days', 0)
                    record['low_risk_days'] = trader_data.get('low_risk_days', 0)
                    record['filtered_days'] = 0

                # Use pre-calculated improvements or calculate if not available
                record['pnl_improvement'] = trader_data.get('pnl_improvement', record['strategy_pnl'] - record['baseline_pnl'])
                record['sharpe_improvement'] = trader_data.get('sharpe_improvement', record['strategy_sharpe'] - record['baseline_sharpe'])
                record['has_positive_impact'] = record['pnl_improvement'] > 0

                # Calculate risk day ratios
                total_days = record['total_observations']
                record['high_risk_ratio'] = record['high_risk_days'] / total_days if total_days > 0 else 0
                record['low_risk_ratio'] = record['low_risk_days'] / total_days if total_days > 0 else 0

                # For trade filtering, handle the different PnL calculation
                if strategy_name == 'trade_filtering':
                    # Trade filtering uses avoided_pnl which represents losses avoided
                    avoided_pnl = trader_data.get('avoided_pnl', 0)
                    record['avoided_losses'] = avoided_pnl
                    # The pnl_improvement in the data is already calculated correctly
                    # It can be positive (avoided losses) or negative (missed gains)
                    actual_improvement = trader_data.get('pnl_improvement', 0)
                    # For display purposes, we want to show the absolute avoided losses as positive
                    if avoided_pnl > 0:  # Positive avoided_pnl means we avoided losses
                        record['pnl_improvement'] = abs(avoided_pnl)
                    elif avoided_pnl < 0:  # Negative avoided_pnl means we missed gains
                        record['pnl_improvement'] = avoided_pnl
                    else:
                        record['pnl_improvement'] = actual_improvement

                all_trader_data.append(record)

    df = pd.DataFrame(all_trader_data)
    print(f"\n✓ Extracted data for {len(df)} trader-strategy combinations")
    print(f"✓ Unique traders: {df['trader_id'].nunique()}")
    print(f"✓ Strategies: {df['strategy'].nunique()}")

    return df

def analyze_individual_performance(trader_df):
    """Analyze individual trader performance across strategies"""
    print("\n=== INDIVIDUAL TRADER PERFORMANCE ANALYSIS ===")

    # Summary by trader
    trader_summary = []

    for trader_id in trader_df['trader_id'].unique():
        trader_data = trader_df[trader_df['trader_id'] == trader_id]

        # Find best strategy for this trader
        best_strategy_idx = trader_data['pnl_improvement'].idxmax()
        best_strategy = trader_data.loc[best_strategy_idx]

        summary = {
            'trader_id': trader_id,
            'best_strategy': best_strategy['strategy'],
            'best_pnl_improvement': best_strategy['pnl_improvement'],
            'best_sharpe_improvement': best_strategy['sharpe_improvement'],
            'total_observations': best_strategy['total_observations'],
            'baseline_pnl': best_strategy['baseline_pnl'],
            'positive_strategies': (trader_data['pnl_improvement'] > 0).sum(),
            'total_strategies': len(trader_data)
        }

        trader_summary.append(summary)

    summary_df = pd.DataFrame(trader_summary)

    print(f"Individual Trader Summary:")
    for _, row in summary_df.iterrows():
        print(f"  Trader {row['trader_id']}: Best={row['best_strategy']}, "
              f"Impact=${row['best_pnl_improvement']:,.0f}, "
              f"Positive strategies={row['positive_strategies']}/{row['total_strategies']}")

    return summary_df

def create_trader_level_visualizations(trader_df, summary_df):
    """Create comprehensive trader-level visualizations"""
    print("\n=== CREATING TRADER-LEVEL VISUALIZATIONS ===")

    Path("results/reports").mkdir(parents=True, exist_ok=True)

    # 1. Individual Trader Impact Heatmap
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Individual Trader Risk Management Impact Analysis', fontsize=16, fontweight='bold')

    # Create pivot table for heatmap
    pivot_df = trader_df.pivot(index='trader_id', columns='strategy', values='pnl_improvement')

    # Heatmap of PnL improvements
    sns.heatmap(pivot_df, annot=True, fmt=',.0f', cmap='RdYlGn', center=0,
               ax=axes[0,0], cbar_kws={'label': 'PnL Improvement ($)'})
    axes[0,0].set_title('PnL Impact by Trader and Strategy')
    axes[0,0].set_xlabel('Strategy')
    axes[0,0].set_ylabel('Trader ID')

    # Individual trader best performance
    best_improvements = summary_df.set_index('trader_id')['best_pnl_improvement']
    colors = ['green' if x > 0 else 'red' for x in best_improvements.values]
    bars = axes[0,1].bar(range(len(best_improvements)), best_improvements.values,
                        color=colors, alpha=0.7)
    axes[0,1].set_title('Best Strategy Impact per Trader')
    axes[0,1].set_xlabel('Trader ID')
    axes[0,1].set_ylabel('Best PnL Improvement ($)')
    axes[0,1].set_xticks(range(len(best_improvements)))
    axes[0,1].set_xticklabels(best_improvements.index, rotation=45)
    axes[0,1].grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, value in zip(bars, best_improvements.values):
        height = bar.get_height()
        axes[0,1].text(bar.get_x() + bar.get_width()/2.,
                      height + (abs(height)*0.02 if height > 0 else -abs(height)*0.02),
                      f'${value:,.0f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)

    # Strategy effectiveness distribution
    strategy_effectiveness = trader_df.groupby('strategy').agg({
        'pnl_improvement': ['sum', 'mean', 'std'],
        'has_positive_impact': 'mean'
    }).round(2)

    strategies = trader_df['strategy'].unique()
    x_pos = np.arange(len(strategies))

    # Total impact bars
    total_impacts = [trader_df[trader_df['strategy'] == s]['pnl_improvement'].sum() for s in strategies]
    colors = ['green' if x > 0 else 'red' for x in total_impacts]
    bars = axes[1,0].bar(x_pos, total_impacts, color=colors, alpha=0.7)
    axes[1,0].set_title('Total Strategy Impact Across All Traders')
    axes[1,0].set_xlabel('Strategy')
    axes[1,0].set_ylabel('Total PnL Impact ($)')
    axes[1,0].set_xticks(x_pos)
    axes[1,0].set_xticklabels([s.replace('_', ' ').title() for s in strategies], rotation=45)
    axes[1,0].grid(True, alpha=0.3)

    # Success rate by strategy
    success_rates = [trader_df[trader_df['strategy'] == s]['has_positive_impact'].mean() for s in strategies]
    colors_success = ['green' if x > 0.5 else 'red' for x in success_rates]
    bars_success = axes[1,1].bar(x_pos, success_rates, color=colors_success, alpha=0.7)
    axes[1,1].set_title('Success Rate by Strategy')
    axes[1,1].set_xlabel('Strategy')
    axes[1,1].set_ylabel('Fraction of Traders with Positive Impact')
    axes[1,1].set_xticks(x_pos)
    axes[1,1].set_xticklabels([s.replace('_', ' ').title() for s in strategies], rotation=45)
    axes[1,1].set_ylim(0, 1)
    axes[1,1].grid(True, alpha=0.3)

    # Add percentage labels
    for bar, rate in zip(bars_success, success_rates):
        height = bar.get_height()
        axes[1,1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                      f'{rate*100:.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('results/reports/individual_trader_impact_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved individual trader impact analysis")

    # 2. Detailed Performance Comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Detailed Performance Metrics Comparison', fontsize=16, fontweight='bold')

    # Sharpe ratio improvements
    sharpe_pivot = trader_df.pivot(index='trader_id', columns='strategy', values='sharpe_improvement')
    sns.heatmap(sharpe_pivot, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
               ax=axes[0,0], cbar_kws={'label': 'Sharpe Ratio Improvement'})
    axes[0,0].set_title('Sharpe Ratio Improvement by Trader and Strategy')

    # Risk exposure analysis - show risk/filtered days by strategy
    # Create a custom metric that shows high risk days for position sizing and filtered days for trade filtering
    risk_data = []
    for _, row in trader_df.iterrows():
        if row['strategy'] == 'trade_filtering':
            # For trade filtering, show filtered days (days avoided)
            risk_metric = row.get('filtered_days', 0)
        else:
            # For other strategies, show high risk days
            risk_metric = row['high_risk_days']
        risk_data.append({
            'trader_id': row['trader_id'],
            'strategy': row['strategy'],
            'risk_metric': risk_metric
        })

    risk_df = pd.DataFrame(risk_data)
    risk_pivot = risk_df.pivot(index='trader_id', columns='strategy', values='risk_metric')

    sns.heatmap(risk_pivot, annot=True, fmt=',.0f', cmap='YlOrRd',
               ax=axes[0,1], cbar_kws={'label': 'Risk/Filtered Days'})
    axes[0,1].set_title('Risk Exposure by Strategy\n(High Risk Days for Sizing, Filtered Days for Trade Filtering)')

    # Baseline vs Strategy PnL scatter
    for strategy in strategies:
        strategy_data = trader_df[trader_df['strategy'] == strategy]
        axes[1,0].scatter(strategy_data['baseline_pnl'], strategy_data['strategy_pnl'],
                         label=strategy.replace('_', ' ').title(), alpha=0.7, s=100)

    # Add diagonal line for reference
    min_pnl = min(trader_df['baseline_pnl'].min(), trader_df['strategy_pnl'].min())
    max_pnl = max(trader_df['baseline_pnl'].max(), trader_df['strategy_pnl'].max())
    axes[1,0].plot([min_pnl, max_pnl], [min_pnl, max_pnl], 'k--', alpha=0.5, label='No change line')

    axes[1,0].set_xlabel('Baseline PnL ($)')
    axes[1,0].set_ylabel('Strategy PnL ($)')
    axes[1,0].set_title('Baseline vs Strategy Performance')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)

    # Performance vs observation count
    axes[1,1].scatter(trader_df['total_observations'], trader_df['pnl_improvement'],
                     c=trader_df['strategy'].astype('category').cat.codes,
                     s=100, alpha=0.7, cmap='tab10')
    axes[1,1].set_xlabel('Total Observations')
    axes[1,1].set_ylabel('PnL Improvement ($)')
    axes[1,1].set_title('Performance vs Data Size')
    axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/reports/detailed_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved detailed performance comparison")

def create_executive_dashboard(trader_df, summary_df):
    """Create executive dashboard with key insights"""
    print("\n=== CREATING EXECUTIVE DASHBOARD ===")

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    fig.suptitle('Executive Dashboard - Risk Management System Performance', fontsize=18, fontweight='bold')

    # 1. Overall System Performance
    ax1 = fig.add_subplot(gs[0, 0])
    total_traders = summary_df['trader_id'].nunique()
    positive_traders = (summary_df['best_pnl_improvement'] > 0).sum()
    negative_traders = total_traders - positive_traders

    ax1.pie([positive_traders, negative_traders],
           labels=[f'Positive Impact\\n({positive_traders} traders)', f'Negative Impact\\n({negative_traders} traders)'],
           colors=['green', 'red'], autopct='%1.1f%%', startangle=90)
    ax1.set_title('Overall Trader Impact Distribution')

    # 2. Best Strategy Distribution
    ax2 = fig.add_subplot(gs[0, 1])
    strategy_counts = summary_df['best_strategy'].value_counts()
    colors_strat = plt.cm.Set3(np.linspace(0, 1, len(strategy_counts)))
    ax2.pie(strategy_counts.values,
           labels=[s.replace('_', ' ').title() for s in strategy_counts.index],
           colors=colors_strat, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Best Strategy Distribution')

    # 3. Total Impact by Strategy
    ax3 = fig.add_subplot(gs[0, 2])
    strategy_totals = trader_df.groupby('strategy')['pnl_improvement'].sum()
    colors = ['green' if x > 0 else 'red' for x in strategy_totals.values]
    bars = ax3.bar(range(len(strategy_totals)), strategy_totals.values, color=colors, alpha=0.7)
    ax3.set_title('Total PnL Impact by Strategy')
    ax3.set_xlabel('Strategy')
    ax3.set_ylabel('Total PnL Impact ($)')
    ax3.set_xticks(range(len(strategy_totals)))
    ax3.set_xticklabels([s.replace('_', ' ').title() for s in strategy_totals.index], rotation=45)
    ax3.grid(True, alpha=0.3)

    # Add value labels
    for bar, value in zip(bars, strategy_totals.values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2.,
                height + (abs(height)*0.02 if height > 0 else -abs(height)*0.02),
                f'${value:,.0f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)

    # 4. Individual Trader Performance Ranking
    ax4 = fig.add_subplot(gs[1, :])
    sorted_traders = summary_df.sort_values('best_pnl_improvement', ascending=True)
    colors_rank = ['green' if x > 0 else 'red' for x in sorted_traders['best_pnl_improvement']]

    bars = ax4.barh(range(len(sorted_traders)), sorted_traders['best_pnl_improvement'],
                   color=colors_rank, alpha=0.7)
    ax4.set_title('Individual Trader Performance Ranking (Best Strategy)')
    ax4.set_xlabel('PnL Improvement ($)')
    ax4.set_ylabel('Trader ID')
    ax4.set_yticks(range(len(sorted_traders)))
    ax4.set_yticklabels(sorted_traders['trader_id'])
    ax4.grid(True, alpha=0.3)

    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, sorted_traders['best_pnl_improvement'])):
        width = bar.get_width()
        ax4.text(width + (abs(width)*0.01 if width > 0 else -abs(width)*0.01),
                bar.get_y() + bar.get_height()/2.,
                f'${value:,.0f}', ha='left' if width > 0 else 'right', va='center', fontsize=9)

    # 5. Key Metrics Summary
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.axis('off')

    # Calculate key metrics
    best_total_impact = strategy_totals.max()
    best_strategy_name = strategy_totals.idxmax()
    avg_improvement = summary_df['best_pnl_improvement'].mean()

    metrics_text = f"""
KEY METRICS

Total Traders Analyzed: {total_traders}
Traders with Positive Impact: {positive_traders} ({positive_traders/total_traders*100:.1f}%)

Best Strategy: {best_strategy_name.replace('_', ' ').title()}
Best Total Impact: ${best_total_impact:,.0f}

Average Best Improvement: ${avg_improvement:,.0f}
Range: ${summary_df['best_pnl_improvement'].min():,.0f} to ${summary_df['best_pnl_improvement'].max():,.0f}
    """

    ax5.text(0.1, 0.9, metrics_text, transform=ax5.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace')

    # 6. Recommendation
    ax6 = fig.add_subplot(gs[2, 1:])
    ax6.axis('off')

    # Determine recommendation
    if best_total_impact > 100000:  # >$100k improvement
        recommendation = "✅ STRONG RECOMMENDATION: DEPLOY SYSTEM"
        rec_color = 'green'
        details = f"""
System shows strong positive impact across traders.
Deploy {best_strategy_name.replace('_', ' ')} strategy with full monitoring.
Expected total benefit: ${best_total_impact:,.0f}
        """
    elif best_total_impact > 0:  # Positive but small
        recommendation = "⚠️ CAUTIOUS RECOMMENDATION: PILOT DEPLOYMENT"
        rec_color = 'orange'
        details = f"""
System shows modest positive impact.
Consider pilot deployment with enhanced monitoring.
Expected total benefit: ${best_total_impact:,.0f}
        """
    elif best_total_impact > -50000:  # Small negative impact
        recommendation = "⚠️ CAUTION: REQUIRES IMPROVEMENT"
        rec_color = 'orange'
        details = f"""
System shows marginal negative impact.
Requires model refinement before deployment.
Current impact: ${best_total_impact:,.0f}
        """
    else:  # Large negative impact
        recommendation = "❌ DO NOT DEPLOY"
        rec_color = 'red'
        details = f"""
System shows significant negative impact.
Fundamental redesign required.
Current impact: ${best_total_impact:,.0f}
        """

    ax6.text(0.1, 0.8, 'DEPLOYMENT RECOMMENDATION', transform=ax6.transAxes,
            fontsize=14, fontweight='bold')
    ax6.text(0.1, 0.6, recommendation, transform=ax6.transAxes,
            fontsize=12, color=rec_color, fontweight='bold')
    ax6.text(0.1, 0.2, details, transform=ax6.transAxes,
            fontsize=10, verticalalignment='top')

    plt.savefig('results/reports/executive_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved executive dashboard")

def create_comprehensive_report(trader_df, summary_df):
    """Create comprehensive text report"""
    print("\n=== CREATING COMPREHENSIVE REPORT ===")

    report_path = Path("results/reports/final_trader_analysis_report.txt")

    with open(report_path, 'w') as f:
        f.write("=" * 120 + "\n")
        f.write("FINAL COMPREHENSIVE TRADER RISK MANAGEMENT SYSTEM ANALYSIS\n")
        f.write("Individual Trader Performance and Causal Impact Assessment\n")
        f.write("=" * 120 + "\n\n")

        # Executive Summary
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-" * 60 + "\n")

        total_traders = summary_df['trader_id'].nunique()
        positive_traders = (summary_df['best_pnl_improvement'] > 0).sum()

        best_total_impact = trader_df.groupby('strategy')['pnl_improvement'].sum().max()
        best_strategy = trader_df.groupby('strategy')['pnl_improvement'].sum().idxmax()

        f.write(f"• System analyzed: {total_traders} individual traders across 3 strategies\n")
        f.write(f"• Traders with positive impact: {positive_traders}/{total_traders} ({positive_traders/total_traders*100:.1f}%)\n")
        f.write(f"• Best performing strategy: {best_strategy.replace('_', ' ').title()}\n")
        f.write(f"• Total financial impact: ${best_total_impact:,.2f}\n")
        f.write(f"• Average trader improvement: ${summary_df['best_pnl_improvement'].mean():,.2f}\n\n")

        # Individual Trader Results
        f.write("INDIVIDUAL TRADER ANALYSIS\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Trader ID':<12} {'Best Strategy':<18} {'PnL Impact':<15} {'Sharpe Δ':<10} {'Observations':<12} {'Status':<8}\n")
        f.write("-" * 85 + "\n")

        for _, row in summary_df.sort_values('best_pnl_improvement', ascending=False).iterrows():
            status = "✓" if row['best_pnl_improvement'] > 0 else "❌"
            sharpe_improvement = trader_df[
                (trader_df['trader_id'] == row['trader_id']) &
                (trader_df['strategy'] == row['best_strategy'])
            ]['sharpe_improvement'].iloc[0]

            f.write(f"{row['trader_id']:<12} {row['best_strategy']:<18} "
                   f"${row['best_pnl_improvement']:<14,.0f} {sharpe_improvement:<10.3f} "
                   f"{row['total_observations']:<12} {status:<8}\n")

        f.write("\n" + "=" * 120 + "\n\n")

        # Strategy Comparison
        f.write("STRATEGY PERFORMANCE COMPARISON\n")
        f.write("-" * 60 + "\n")

        for strategy in trader_df['strategy'].unique():
            strategy_data = trader_df[trader_df['strategy'] == strategy]
            f.write(f"\n{strategy.upper().replace('_', ' ')} STRATEGY:\n")
            f.write(f"  Total PnL Impact: ${strategy_data['pnl_improvement'].sum():,.2f}\n")
            f.write(f"  Success Rate: {(strategy_data['pnl_improvement'] > 0).mean()*100:.1f}%\n")
            f.write(f"  Average Impact per Trader: ${strategy_data['pnl_improvement'].mean():,.2f}\n")
            f.write(f"  Average Sharpe Improvement: {strategy_data['sharpe_improvement'].mean():.4f}\n")
            f.write(f"  Traders Analyzed: {len(strategy_data)}\n")

            f.write(f"  Individual Results:\n")
            for _, row in strategy_data.iterrows():
                status = "✓" if row['pnl_improvement'] > 0 else "❌"
                f.write(f"    Trader {row['trader_id']}: ${row['pnl_improvement']:,.0f} {status}\n")

        f.write("\n" + "=" * 120 + "\n\n")

        # Final Recommendation
        f.write("DEPLOYMENT RECOMMENDATION\n")
        f.write("-" * 60 + "\n")

        if best_total_impact > 100000:
            f.write("✅ RECOMMENDATION: PROCEED WITH FULL DEPLOYMENT\n\n")
            f.write(f"The risk management system demonstrates strong positive impact:\n")
            f.write(f"• Total expected benefit: ${best_total_impact:,.2f}\n")
            f.write(f"• Best strategy: {best_strategy.replace('_', ' ').title()}\n")
            f.write(f"• Success rate: {positive_traders/total_traders*100:.1f}% of traders benefit\n\n")
            f.write("Deployment Guidelines:\n")
            f.write("• Implement comprehensive monitoring dashboard\n")
            f.write("• Set up real-time performance tracking\n")
            f.write("• Establish monthly model performance reviews\n")
            f.write("• Configure circuit breakers for negative performance\n")
        elif best_total_impact > 0:
            f.write("⚠️ RECOMMENDATION: PILOT DEPLOYMENT WITH CAUTION\n\n")
            f.write(f"The system shows modest positive impact requiring careful monitoring:\n")
            f.write(f"• Total expected benefit: ${best_total_impact:,.2f}\n")
            f.write(f"• Consider gradual rollout to subset of traders first\n")
            f.write(f"• Enhanced monitoring and frequent performance reviews required\n")
        else:
            f.write("❌ RECOMMENDATION: DO NOT DEPLOY\n\n")
            f.write(f"The system shows negative financial impact:\n")
            f.write(f"• Total expected loss: ${best_total_impact:,.2f}\n")
            f.write(f"• Only {positive_traders}/{total_traders} traders would benefit\n")
            f.write(f"• System requires fundamental redesign before deployment\n\n")
            f.write("Required Improvements:\n")
            f.write("• Review feature engineering approach\n")
            f.write("• Consider alternative modeling techniques\n")
            f.write("• Analyze data quality and preprocessing steps\n")
            f.write("• Reassess target variable definition\n")

        f.write("\n" + "=" * 120 + "\n")

    print(f"✓ Saved comprehensive report to {report_path}")

def main():
    """Main analysis function"""
    print("=" * 120)
    print("FINAL COMPREHENSIVE TRADER RISK MANAGEMENT ANALYSIS")
    print("=" * 120)

    # Extract individual trader data
    trader_df = extract_individual_trader_data()

    if trader_df.empty:
        print("❌ No trader data available for analysis")
        return

    # Analyze individual performance
    summary_df = analyze_individual_performance(trader_df)

    # Create visualizations
    create_trader_level_visualizations(trader_df, summary_df)
    create_executive_dashboard(trader_df, summary_df)

    # Create comprehensive report
    create_comprehensive_report(trader_df, summary_df)

    print("\n" + "=" * 120)
    print("FINAL ANALYSIS COMPLETE")
    print("=" * 120)

    # Print executive summary
    total_traders = summary_df['trader_id'].nunique()
    positive_traders = (summary_df['best_pnl_improvement'] > 0).sum()
    best_total_impact = trader_df.groupby('strategy')['pnl_improvement'].sum().max()
    best_strategy = trader_df.groupby('strategy')['pnl_improvement'].sum().idxmax()

    print(f"\nEXECUTIVE SUMMARY:")
    print(f"• Traders analyzed: {total_traders}")
    print(f"• Traders with positive impact: {positive_traders}/{total_traders} ({positive_traders/total_traders*100:.1f}%)")
    print(f"• Best strategy: {best_strategy.replace('_', ' ').title()}")
    print(f"• Total financial impact: ${best_total_impact:,.0f}")

    if best_total_impact > 0:
        print(f"• ✅ SYSTEM RECOMMENDATION: {'DEPLOY' if best_total_impact > 100000 else 'PILOT'}")
    else:
        print(f"• ❌ SYSTEM RECOMMENDATION: DO NOT DEPLOY")

    print(f"\nGenerated Reports:")
    print(f"• results/reports/individual_trader_impact_analysis.png")
    print(f"• results/reports/detailed_performance_comparison.png")
    print(f"• results/reports/executive_dashboard.png")
    print(f"• results/reports/final_trader_analysis_report.txt")

if __name__ == "__main__":
    main()
