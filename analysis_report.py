#!/usr/bin/env python3
"""
Comprehensive Analysis Report for Trader Risk Management System
Evaluates individual trader performance, overfitting, and causal impact
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def load_data():
    """Load all pipeline results"""
    print("Loading pipeline results...")

    # Try to load backtest results from different locations
    backtest_results = {}
    for backtest_file in [Path("data/backtest_results.pkl"), Path("outputs/backtest_results.json"), Path("data/backtest_summary.json")]:
        if backtest_file.exists():
            try:
                if backtest_file.suffix == '.pkl':
                    backtest_results = pd.read_pickle(backtest_file)
                    if hasattr(backtest_results, 'to_dict'):
                        backtest_results = backtest_results.to_dict()
                    print(f"✓ Loaded backtest results from {backtest_file}")
                    break
                else:
                    with open(backtest_file, 'r') as f:
                        backtest_results = json.load(f)
                    print(f"✓ Loaded backtest results from {backtest_file}")
                    break
            except Exception as e:
                print(f"⚠️ Error loading {backtest_file}: {e}")
                continue

    if not backtest_results:
        print("❌ No backtest results found")

    # Load causal impact results
    causal_results = {}
    causal_file = Path("data/causal_impact_results.json")
    if causal_file.exists():
        try:
            with open(causal_file, 'r') as f:
                causal_results = json.load(f)
            print("✓ Loaded causal impact results")
        except Exception as e:
            print(f"❌ Error loading causal impact results: {e}")
    else:
        print("❌ Causal impact results not found")

    # Load daily data
    daily_data = pd.DataFrame()
    daily_file = Path("data/daily_aggregated.pkl")
    if daily_file.exists():
        try:
            daily_data = pd.read_pickle(daily_file)
            print("✓ Loaded daily aggregated data")
        except Exception as e:
            print(f"❌ Error loading daily data: {e}")
    else:
        print("❌ Daily data not found")

    return backtest_results, causal_results, daily_data

def analyze_overfitting(backtest_results):
    """Analyze overfitting by comparing train vs test performance"""
    print("\n=== OVERFITTING ANALYSIS ===")

    if not backtest_results:
        print("❌ No backtest results to analyze")
        return pd.DataFrame()

    analysis_data = []

    for trader_id, results in backtest_results.items():
        if isinstance(results, dict) and 'performance_metrics' in results:
            metrics = results['performance_metrics']

            # Extract train and test accuracies
            train_acc = metrics.get('train_accuracy', 0)
            test_acc = metrics.get('test_accuracy', 0)

            # Calculate overfitting metrics
            acc_diff = train_acc - test_acc
            overfitting_ratio = acc_diff / train_acc if train_acc > 0 else 0

            analysis_data.append({
                'trader_id': trader_id,
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'accuracy_difference': acc_diff,
                'overfitting_ratio': overfitting_ratio,
                'is_overfitting': acc_diff > 0.1,  # >10% difference suggests overfitting
                'test_observations': results.get('test_size', 0),
                'train_observations': results.get('train_size', 0)
            })

    if not analysis_data:
        print("❌ No valid performance metrics found")
        return pd.DataFrame()

    df = pd.DataFrame(analysis_data)

    # Summary statistics
    print(f"✓ Analyzed {len(df)} traders")
    print(f"✓ Average train accuracy: {df['train_accuracy'].mean():.4f}")
    print(f"✓ Average test accuracy: {df['test_accuracy'].mean():.4f}")
    print(f"✓ Average accuracy difference: {df['accuracy_difference'].mean():.4f}")
    print(f"✓ Traders with overfitting: {df['is_overfitting'].sum()}/{len(df)} ({df['is_overfitting'].mean()*100:.1f}%)")

    return df

def analyze_individual_causal_impact(causal_results, daily_data):
    """Analyze causal impact for each individual trader"""
    print("\n=== INDIVIDUAL TRADER CAUSAL IMPACT ===")

    if not causal_results:
        print("❌ No causal impact results to analyze")
        return pd.DataFrame()

    trader_impacts = []

    # Extract strategy results
    strategies = causal_results.get('strategy_results', {})

    for strategy_name, strategy_data in strategies.items():
        if 'trader_results' in strategy_data:
            for trader_id, trader_data in strategy_data['trader_results'].items():
                trader_impacts.append({
                    'trader_id': trader_id,
                    'strategy': strategy_name,
                    'pnl_improvement': trader_data.get('pnl_improvement', 0),
                    'sharpe_improvement': trader_data.get('sharpe_improvement', 0),
                    'has_positive_impact': trader_data.get('pnl_improvement', 0) > 0,
                    'baseline_pnl': trader_data.get('baseline_total_pnl', 0),
                    'strategy_pnl': trader_data.get('strategy_total_pnl', 0),
                    'total_trades': trader_data.get('total_observations', 0)
                })

    if not trader_impacts:
        print("❌ No trader impact data found")
        return pd.DataFrame()

    df = pd.DataFrame(trader_impacts)

    # Summary by strategy
    for strategy in df['strategy'].unique():
        strategy_df = df[df['strategy'] == strategy]
        print(f"\n{strategy.upper()} Strategy:")
        print(f"  ✓ Total PnL improvement: ${strategy_df['pnl_improvement'].sum():,.2f}")
        print(f"  ✓ Traders with positive impact: {strategy_df['has_positive_impact'].sum()}/{len(strategy_df)} ({strategy_df['has_positive_impact'].mean()*100:.1f}%)")
        print(f"  ✓ Average Sharpe improvement: {strategy_df['sharpe_improvement'].mean():.4f}")

    return df

def create_visualizations(overfitting_df, causal_impact_df, daily_data):
    """Create comprehensive visual reports"""
    print("\n=== CREATING VISUAL REPORTS ===")

    # Create output directory
    Path("outputs/reports").mkdir(parents=True, exist_ok=True)

    # 1. Overfitting Analysis Plots
    if not overfitting_df.empty:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Overfitting Analysis - Train vs Test Performance', fontsize=16, fontweight='bold')

        # Train vs Test Accuracy Scatter
        axes[0,0].scatter(overfitting_df['train_accuracy'], overfitting_df['test_accuracy'],
                         c=overfitting_df['is_overfitting'], cmap='RdYlBu_r', s=100, alpha=0.7)
        axes[0,0].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect fit line')
        axes[0,0].set_xlabel('Train Accuracy')
        axes[0,0].set_ylabel('Test Accuracy')
        axes[0,0].set_title('Train vs Test Accuracy')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)

        # Accuracy Difference Distribution
        axes[0,1].hist(overfitting_df['accuracy_difference'], bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0,1].axvline(0.1, color='red', linestyle='--', label='Overfitting threshold (0.1)')
        axes[0,1].set_xlabel('Train - Test Accuracy Difference')
        axes[0,1].set_ylabel('Number of Traders')
        axes[0,1].set_title('Distribution of Accuracy Differences')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)

        # Individual Trader Performance
        trader_ids = overfitting_df['trader_id'].astype(str)
        x_pos = np.arange(len(trader_ids))

        axes[1,0].bar(x_pos - 0.2, overfitting_df['train_accuracy'], width=0.4,
                     label='Train Accuracy', alpha=0.7, color='lightblue')
        axes[1,0].bar(x_pos + 0.2, overfitting_df['test_accuracy'], width=0.4,
                     label='Test Accuracy', alpha=0.7, color='lightcoral')
        axes[1,0].set_xlabel('Trader ID')
        axes[1,0].set_ylabel('Accuracy')
        axes[1,0].set_title('Individual Trader Performance')
        axes[1,0].set_xticks(x_pos)
        axes[1,0].set_xticklabels(trader_ids, rotation=45)
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)

        # Test Data Size vs Performance
        axes[1,1].scatter(overfitting_df['test_observations'], overfitting_df['test_accuracy'],
                         c=overfitting_df['accuracy_difference'], cmap='RdYlBu_r', s=100, alpha=0.7)
        axes[1,1].set_xlabel('Test Observations')
        axes[1,1].set_ylabel('Test Accuracy')
        axes[1,1].set_title('Test Size vs Performance')
        axes[1,1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('outputs/reports/overfitting_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved overfitting analysis chart")

    # 2. Causal Impact Analysis Plots
    if not causal_impact_df.empty:
        # Create pivot table for better visualization
        pivot_df = causal_impact_df.pivot(index='trader_id', columns='strategy', values='pnl_improvement')

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Causal Impact Analysis - Individual Trader PnL Impact', fontsize=16, fontweight='bold')

        # Heatmap of PnL improvements by trader and strategy
        sns.heatmap(pivot_df, annot=True, fmt=',.0f', cmap='RdYlGn', center=0,
                   ax=axes[0,0], cbar_kws={'label': 'PnL Improvement ($)'})
        axes[0,0].set_title('PnL Improvement by Trader and Strategy')
        axes[0,0].set_xlabel('Strategy')
        axes[0,0].set_ylabel('Trader ID')

        # Strategy comparison boxplot
        box_data = [causal_impact_df[causal_impact_df['strategy'] == strategy]['pnl_improvement']
                   for strategy in causal_impact_df['strategy'].unique()]
        axes[0,1].boxplot(box_data, labels=causal_impact_df['strategy'].unique())
        axes[0,1].set_title('PnL Improvement Distribution by Strategy')
        axes[0,1].set_ylabel('PnL Improvement ($)')
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].tick_params(axis='x', rotation=45)

        # Individual trader impact across strategies
        for strategy in causal_impact_df['strategy'].unique():
            strategy_data = causal_impact_df[causal_impact_df['strategy'] == strategy]
            axes[1,0].bar(strategy_data['trader_id'].astype(str), strategy_data['pnl_improvement'],
                         alpha=0.7, label=strategy)
        axes[1,0].set_title('Individual Trader PnL Impact by Strategy')
        axes[1,0].set_xlabel('Trader ID')
        axes[1,0].set_ylabel('PnL Improvement ($)')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].tick_params(axis='x', rotation=45)

        # Success rate by strategy
        success_rates = causal_impact_df.groupby('strategy')['has_positive_impact'].mean()
        colors = ['green' if x > 0.5 else 'red' for x in success_rates.values]
        axes[1,1].bar(success_rates.index, success_rates.values, color=colors, alpha=0.7)
        axes[1,1].set_title('Success Rate by Strategy')
        axes[1,1].set_ylabel('Fraction of Traders with Positive Impact')
        axes[1,1].set_ylim(0, 1)
        axes[1,1].grid(True, alpha=0.3)
        axes[1,1].tick_params(axis='x', rotation=45)

        # Add percentage labels on bars
        for i, v in enumerate(success_rates.values):
            axes[1,1].text(i, v + 0.02, f'{v*100:.1f}%', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig('outputs/reports/causal_impact_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved causal impact analysis chart")

    # 3. Combined Summary Report
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Risk Management System - Executive Summary', fontsize=16, fontweight='bold')

    if not overfitting_df.empty:
        # Model quality summary
        quality_labels = ['No Overfitting', 'Overfitting']
        quality_counts = [len(overfitting_df) - overfitting_df['is_overfitting'].sum(),
                         overfitting_df['is_overfitting'].sum()]
        colors_quality = ['green', 'red']
        axes[0,0].pie(quality_counts, labels=quality_labels, colors=colors_quality, autopct='%1.1f%%')
        axes[0,0].set_title('Model Quality Assessment')

    if not causal_impact_df.empty:
        # Best strategy summary
        strategy_summary = causal_impact_df.groupby('strategy').agg({
            'pnl_improvement': 'sum',
            'has_positive_impact': 'mean'
        }).round(2)

        axes[0,1].bar(strategy_summary.index, strategy_summary['pnl_improvement'], alpha=0.7)
        axes[0,1].set_title('Total PnL Impact by Strategy')
        axes[0,1].set_ylabel('Total PnL Improvement ($)')
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].tick_params(axis='x', rotation=45)

        # Performance vs Test Size
        if not overfitting_df.empty:
            merged_df = overfitting_df.merge(
                causal_impact_df[causal_impact_df['strategy'] == 'position_sizing'][['trader_id', 'pnl_improvement']],
                on='trader_id', how='left'
            )

            axes[1,0].scatter(merged_df['test_accuracy'], merged_df['pnl_improvement'],
                             s=100, alpha=0.7, c='blue')
            axes[1,0].set_xlabel('Test Accuracy')
            axes[1,0].set_ylabel('PnL Improvement ($)')
            axes[1,0].set_title('Model Accuracy vs Financial Impact')
            axes[1,0].grid(True, alpha=0.3)

    # System recommendation
    axes[1,1].text(0.1, 0.7, 'SYSTEM RECOMMENDATION', fontsize=14, fontweight='bold',
                   transform=axes[1,1].transAxes)

    # Determine recommendation based on results
    recommendation = "❌ DO NOT DEPLOY"
    reason = "Negative financial impact"
    color = 'red'

    if not causal_impact_df.empty:
        best_strategy = causal_impact_df.groupby('strategy')['pnl_improvement'].sum()
        if best_strategy.max() > 0:
            recommendation = "✅ PROCEED WITH DEPLOYMENT"
            reason = f"Positive impact: ${best_strategy.max():,.0f}"
            color = 'green'

    axes[1,1].text(0.1, 0.5, recommendation, fontsize=12, color=color, fontweight='bold',
                   transform=axes[1,1].transAxes)
    axes[1,1].text(0.1, 0.3, reason, fontsize=10, transform=axes[1,1].transAxes)
    axes[1,1].set_xlim(0, 1)
    axes[1,1].set_ylim(0, 1)
    axes[1,1].axis('off')

    plt.tight_layout()
    plt.savefig('outputs/reports/executive_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved executive summary chart")

def create_detailed_report(overfitting_df, causal_impact_df):
    """Create detailed text report"""
    print("\n=== CREATING DETAILED REPORT ===")

    report_path = Path("outputs/reports/detailed_analysis_report.txt")

    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TRADER RISK MANAGEMENT SYSTEM - DETAILED ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")

        # Overfitting Analysis
        f.write("OVERFITTING ANALYSIS\n")
        f.write("-" * 40 + "\n")

        if not overfitting_df.empty:
            f.write(f"Total Traders Analyzed: {len(overfitting_df)}\n")
            f.write(f"Average Train Accuracy: {overfitting_df['train_accuracy'].mean():.4f}\n")
            f.write(f"Average Test Accuracy: {overfitting_df['test_accuracy'].mean():.4f}\n")
            f.write(f"Average Accuracy Drop: {overfitting_df['accuracy_difference'].mean():.4f}\n")
            f.write(f"Traders with Overfitting (>10% drop): {overfitting_df['is_overfitting'].sum()}/{len(overfitting_df)}\n\n")

            f.write("Individual Trader Performance:\n")
            for _, row in overfitting_df.iterrows():
                f.write(f"  Trader {row['trader_id']}: Train={row['train_accuracy']:.3f}, Test={row['test_accuracy']:.3f}, "
                       f"Diff={row['accuracy_difference']:.3f} {'⚠️' if row['is_overfitting'] else '✓'}\n")
        else:
            f.write("No overfitting data available\n")

        f.write("\n" + "=" * 80 + "\n\n")

        # Causal Impact Analysis
        f.write("CAUSAL IMPACT ANALYSIS\n")
        f.write("-" * 40 + "\n")

        if not causal_impact_df.empty:
            for strategy in causal_impact_df['strategy'].unique():
                strategy_data = causal_impact_df[causal_impact_df['strategy'] == strategy]
                f.write(f"\n{strategy.upper().replace('_', ' ')} STRATEGY:\n")
                f.write(f"  Total PnL Impact: ${strategy_data['pnl_improvement'].sum():,.2f}\n")
                f.write(f"  Success Rate: {strategy_data['has_positive_impact'].mean()*100:.1f}%\n")
                f.write(f"  Average Sharpe Improvement: {strategy_data['sharpe_improvement'].mean():.4f}\n")

                f.write("  Individual Results:\n")
                for _, row in strategy_data.iterrows():
                    f.write(f"    Trader {row['trader_id']}: ${row['pnl_improvement']:,.2f} "
                           f"({'✓' if row['has_positive_impact'] else '❌'})\n")
        else:
            f.write("No causal impact data available\n")

        f.write("\n" + "=" * 80 + "\n\n")

        # Final Recommendation
        f.write("FINAL RECOMMENDATION\n")
        f.write("-" * 40 + "\n")

        if not causal_impact_df.empty:
            total_impacts = causal_impact_df.groupby('strategy')['pnl_improvement'].sum()
            best_strategy = total_impacts.idxmax()
            best_impact = total_impacts.max()

            if best_impact > 0:
                f.write("✅ RECOMMENDATION: PROCEED WITH DEPLOYMENT\n")
                f.write(f"   Best Strategy: {best_strategy.replace('_', ' ').title()}\n")
                f.write(f"   Expected Impact: ${best_impact:,.2f}\n")
            else:
                f.write("❌ RECOMMENDATION: DO NOT DEPLOY\n")
                f.write(f"   Best Strategy Impact: ${best_impact:,.2f}\n")
                f.write("   System shows negative financial impact\n")
        else:
            f.write("❌ RECOMMENDATION: INSUFFICIENT DATA FOR DEPLOYMENT\n")

        f.write("\n" + "=" * 80 + "\n")

    print(f"✓ Saved detailed report to {report_path}")

def main():
    """Main analysis function"""
    print("=" * 80)
    print("COMPREHENSIVE RISK MANAGEMENT SYSTEM ANALYSIS")
    print("=" * 80)

    # Load data
    backtest_results, causal_results, daily_data = load_data()

    # Analyze overfitting
    overfitting_df = analyze_overfitting(backtest_results)

    # Analyze individual causal impact
    causal_impact_df = analyze_individual_causal_impact(causal_results, daily_data)

    # Create visualizations
    create_visualizations(overfitting_df, causal_impact_df, daily_data)

    # Create detailed report
    create_detailed_report(overfitting_df, causal_impact_df)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("Reports saved to outputs/reports/")
    print("- overfitting_analysis.png")
    print("- causal_impact_analysis.png")
    print("- executive_summary.png")
    print("- detailed_analysis_report.txt")

if __name__ == "__main__":
    main()
