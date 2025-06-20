#!/usr/bin/env python3
"""
Comprehensive Analysis Report for Trader Risk Management System
Evaluates individual trader performance, overfitting, and causal impact using actual data structure
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

def load_all_data():
    """Load all available data files"""
    print("Loading all data files...")

    data = {}

    # Load backtest results
    try:
        data['backtest'] = pd.read_pickle('data/backtest_results.pkl')
        print(f"✓ Loaded backtest results for {len(data['backtest'])} traders")
    except Exception as e:
        print(f"❌ Error loading backtest results: {e}")
        data['backtest'] = {}

    # Load causal impact results
    try:
        with open('data/causal_impact_results.json', 'r') as f:
            data['causal_impact'] = json.load(f)
        print("✓ Loaded causal impact summary")
    except Exception as e:
        print(f"❌ Error loading causal impact: {e}")
        data['causal_impact'] = {}

    # Load strategy results
    try:
        data['strategy_results'] = pd.read_pickle('data/strategy_results.pkl')
        print(f"✓ Loaded strategy results for {len(data['strategy_results'])} strategies")
    except Exception as e:
        print(f"❌ Error loading strategy results: {e}")
        data['strategy_results'] = {}

    # Load training results
    try:
        with open('data/training_results.json', 'r') as f:
            data['training'] = json.load(f)
        print(f"✓ Loaded training results for {len(data['training'])} traders")
    except Exception as e:
        print(f"❌ Error loading training results: {e}")
        data['training'] = {}

    # Load daily data
    try:
        data['daily'] = pd.read_pickle('data/daily_aggregated.pkl')
        print(f"✓ Loaded daily data: {len(data['daily'])} observations")
    except Exception as e:
        print(f"❌ Error loading daily data: {e}")
        data['daily'] = pd.DataFrame()

    return data

def analyze_model_performance(data):
    """Analyze individual trader model performance and overfitting"""
    print("\n=== MODEL PERFORMANCE ANALYSIS ===")

    backtest = data['backtest']
    training = data['training']

    if not backtest or not training:
        print("❌ Missing backtest or training data")
        return pd.DataFrame()

    performance_data = []

    for trader_id in backtest.keys():
        trader_id_str = str(trader_id)

        # Get test performance
        test_accuracy = backtest[trader_id]['test_accuracy']
        test_samples = backtest[trader_id]['test_samples']

        # Get training performance if available
        train_accuracy = 0
        train_samples = 0
        if trader_id_str in training:
            train_data = training[trader_id_str]
            train_accuracy = train_data.get('train_accuracy', 0)
            train_samples = train_data.get('train_samples', 0)

        # Calculate overfitting metrics
        accuracy_diff = train_accuracy - test_accuracy
        overfitting_ratio = accuracy_diff / train_accuracy if train_accuracy > 0 else 0
        is_overfitting = accuracy_diff > 0.1  # >10% difference

        # Get class-specific accuracies
        class_accuracies = {
            'class_0_accuracy': backtest[trader_id].get('class_0_accuracy', 0),
            'class_1_accuracy': backtest[trader_id].get('class_1_accuracy', 0),
            'class_2_accuracy': backtest[trader_id].get('class_2_accuracy', 0)
        }

        performance_data.append({
            'trader_id': trader_id,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'accuracy_difference': accuracy_diff,
            'overfitting_ratio': overfitting_ratio,
            'is_overfitting': is_overfitting,
            'test_samples': test_samples,
            'train_samples': train_samples,
            **class_accuracies
        })

    df = pd.DataFrame(performance_data)

    # Summary statistics
    print(f"✓ Analyzed {len(df)} traders")
    print(f"✓ Average train accuracy: {df['train_accuracy'].mean():.4f}")
    print(f"✓ Average test accuracy: {df['test_accuracy'].mean():.4f}")
    print(f"✓ Average accuracy difference: {df['accuracy_difference'].mean():.4f}")
    print(f"✓ Traders with overfitting: {df['is_overfitting'].sum()}/{len(df)} ({df['is_overfitting'].mean()*100:.1f}%)")

    return df

def analyze_strategy_impact(data):
    """Analyze individual trader impact for each strategy"""
    print("\n=== STRATEGY IMPACT ANALYSIS ===")

    strategy_results = data['strategy_results']
    causal_impact = data['causal_impact']

    if not strategy_results:
        print("❌ No strategy results available")
        return pd.DataFrame()

    impact_data = []

    for strategy_name, strategy_data in strategy_results.items():
        if not isinstance(strategy_data, dict):
            continue

        print(f"\nAnalyzing {strategy_name} strategy...")

        # Get overall strategy metrics
        total_improvement = causal_impact.get(f'{strategy_name}_improvement', 0)
        success_rate = causal_impact.get(f'{strategy_name}_success_rate', 0)

        print(f"  Total improvement: ${total_improvement:,.2f}")
        print(f"  Success rate: {success_rate*100:.1f}%")

        # Try to extract individual trader results
        for trader_id, trader_data in strategy_data.items():
            if isinstance(trader_data, dict):
                # Extract trader-specific metrics
                baseline_pnl = trader_data.get('baseline_total_pnl', 0)
                strategy_pnl = trader_data.get('strategy_total_pnl', baseline_pnl)
                pnl_improvement = strategy_pnl - baseline_pnl

                impact_data.append({
                    'trader_id': trader_id,
                    'strategy': strategy_name,
                    'baseline_pnl': baseline_pnl,
                    'strategy_pnl': strategy_pnl,
                    'pnl_improvement': pnl_improvement,
                    'has_positive_impact': pnl_improvement > 0,
                    'sharpe_improvement': trader_data.get('sharpe_improvement', 0),
                    'total_trades': trader_data.get('total_observations', 0)
                })

    if not impact_data:
        print("❌ No individual trader impact data found")
        return pd.DataFrame()

    df = pd.DataFrame(impact_data)

    # Summary by strategy
    for strategy in df['strategy'].unique():
        strategy_df = df[df['strategy'] == strategy]
        print(f"\n{strategy.upper()} Strategy Summary:")
        print(f"  ✓ Traders analyzed: {len(strategy_df)}")
        print(f"  ✓ Total PnL improvement: ${strategy_df['pnl_improvement'].sum():,.2f}")
        print(f"  ✓ Positive impact rate: {strategy_df['has_positive_impact'].mean()*100:.1f}%")
        print(f"  ✓ Average Sharpe improvement: {strategy_df['sharpe_improvement'].mean():.4f}")

    return df

def create_comprehensive_visualizations(performance_df, impact_df, data):
    """Create comprehensive visual reports"""
    print("\n=== CREATING COMPREHENSIVE VISUALIZATIONS ===")

    # Create output directory
    Path("outputs/reports").mkdir(parents=True, exist_ok=True)

    # 1. Model Performance and Overfitting Analysis
    if not performance_df.empty:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Analysis - Overfitting Detection', fontsize=16, fontweight='bold')

        # Individual trader performance comparison
        trader_ids = performance_df['trader_id'].astype(str)
        x_pos = np.arange(len(trader_ids))

        axes[0,0].bar(x_pos - 0.2, performance_df['train_accuracy'], width=0.4,
                     label='Train Accuracy', alpha=0.7, color='lightblue')
        axes[0,0].bar(x_pos + 0.2, performance_df['test_accuracy'], width=0.4,
                     label='Test Accuracy', alpha=0.7, color='lightcoral')
        axes[0,0].set_xlabel('Trader ID')
        axes[0,0].set_ylabel('Accuracy')
        axes[0,0].set_title('Train vs Test Accuracy by Trader')
        axes[0,0].set_xticks(x_pos)
        axes[0,0].set_xticklabels(trader_ids, rotation=45)
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)

        # Overfitting detection scatter plot
        colors = ['red' if x else 'green' for x in performance_df['is_overfitting']]
        axes[0,1].scatter(performance_df['train_accuracy'], performance_df['test_accuracy'],
                         c=colors, s=100, alpha=0.7)
        axes[0,1].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect fit line')
        axes[0,1].set_xlabel('Train Accuracy')
        axes[0,1].set_ylabel('Test Accuracy')
        axes[0,1].set_title('Overfitting Detection (Red = Overfitting)')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)

        # Accuracy difference distribution
        axes[1,0].hist(performance_df['accuracy_difference'], bins=10, alpha=0.7,
                      color='skyblue', edgecolor='black')
        axes[1,0].axvline(0.1, color='red', linestyle='--', label='Overfitting threshold')
        axes[1,0].set_xlabel('Train - Test Accuracy Difference')
        axes[1,0].set_ylabel('Number of Traders')
        axes[1,0].set_title('Distribution of Performance Differences')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)

        # Class-specific performance
        class_cols = ['class_0_accuracy', 'class_1_accuracy', 'class_2_accuracy']
        class_data = performance_df[class_cols].values.T

        box_data = [class_data[i] for i in range(len(class_cols))]
        axes[1,1].boxplot(box_data, labels=['Loss', 'Neutral', 'Win'])
        axes[1,1].set_title('Class-Specific Accuracy Distribution')
        axes[1,1].set_ylabel('Accuracy')
        axes[1,1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('outputs/reports/model_performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved model performance analysis")

    # 2. Strategy Impact Analysis
    if not impact_df.empty:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Strategy Impact Analysis - Individual Trader Results', fontsize=16, fontweight='bold')

        # PnL improvement by trader and strategy
        pivot_df = impact_df.pivot(index='trader_id', columns='strategy', values='pnl_improvement')

        sns.heatmap(pivot_df, annot=True, fmt=',.0f', cmap='RdYlGn', center=0,
                   ax=axes[0,0], cbar_kws={'label': 'PnL Improvement ($)'})
        axes[0,0].set_title('PnL Improvement Heatmap')
        axes[0,0].set_xlabel('Strategy')
        axes[0,0].set_ylabel('Trader ID')

        # Strategy comparison boxplot
        strategies = impact_df['strategy'].unique()
        box_data = [impact_df[impact_df['strategy'] == strategy]['pnl_improvement']
                   for strategy in strategies]
        bp = axes[0,1].boxplot(box_data, labels=[s.replace('_', ' ').title() for s in strategies])
        axes[0,1].set_title('PnL Improvement Distribution by Strategy')
        axes[0,1].set_ylabel('PnL Improvement ($)')
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].tick_params(axis='x', rotation=45)

        # Individual trader results
        width = 0.25
        x_pos = np.arange(len(impact_df['trader_id'].unique()))

        for i, strategy in enumerate(strategies):
            strategy_data = impact_df[impact_df['strategy'] == strategy]
            # Sort by trader_id to ensure consistent ordering
            strategy_data = strategy_data.sort_values('trader_id')
            axes[1,0].bar(x_pos + i*width, strategy_data['pnl_improvement'],
                         width, label=strategy.replace('_', ' ').title(), alpha=0.7)

        axes[1,0].set_title('Individual Trader PnL Impact by Strategy')
        axes[1,0].set_xlabel('Trader ID')
        axes[1,0].set_ylabel('PnL Improvement ($)')
        axes[1,0].set_xticks(x_pos + width)
        axes[1,0].set_xticklabels(sorted(impact_df['trader_id'].unique()), rotation=45)
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)

        # Success rate by strategy
        success_rates = impact_df.groupby('strategy')['has_positive_impact'].mean()
        colors = ['green' if x > 0.5 else 'red' for x in success_rates.values]
        bars = axes[1,1].bar(range(len(success_rates)), success_rates.values, color=colors, alpha=0.7)
        axes[1,1].set_title('Success Rate by Strategy')
        axes[1,1].set_ylabel('Fraction with Positive Impact')
        axes[1,1].set_xticks(range(len(success_rates)))
        axes[1,1].set_xticklabels([s.replace('_', ' ').title() for s in success_rates.index], rotation=45)
        axes[1,1].set_ylim(0, 1)
        axes[1,1].grid(True, alpha=0.3)

        # Add percentage labels
        for bar, rate in zip(bars, success_rates.values):
            height = bar.get_height()
            axes[1,1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                          f'{rate*100:.1f}%', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig('outputs/reports/strategy_impact_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved strategy impact analysis")

    # 3. Executive Summary Dashboard
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Executive Summary - Risk Management System Performance', fontsize=16, fontweight='bold')

    # Model quality pie chart
    if not performance_df.empty:
        overfitting_counts = [
            len(performance_df) - performance_df['is_overfitting'].sum(),
            performance_df['is_overfitting'].sum()
        ]
        labels = ['No Overfitting', 'Overfitting Detected']
        colors = ['green', 'red']
        axes[0,0].pie(overfitting_counts, labels=labels, colors=colors, autopct='%1.1f%%')
        axes[0,0].set_title(f'Model Quality Assessment\\n({len(performance_df)} traders)')

    # Overall strategy performance
    if not impact_df.empty:
        strategy_totals = impact_df.groupby('strategy')['pnl_improvement'].sum()
        colors = ['green' if x > 0 else 'red' for x in strategy_totals.values]
        bars = axes[0,1].bar(range(len(strategy_totals)), strategy_totals.values, color=colors, alpha=0.7)
        axes[0,1].set_title('Total PnL Impact by Strategy')
        axes[0,1].set_ylabel('Total PnL Improvement ($)')
        axes[0,1].set_xticks(range(len(strategy_totals)))
        axes[0,1].set_xticklabels([s.replace('_', ' ').title() for s in strategy_totals.index], rotation=45)
        axes[0,1].grid(True, alpha=0.3)

        # Add value labels
        for bar, value in zip(bars, strategy_totals.values):
            height = bar.get_height()
            axes[0,1].text(bar.get_x() + bar.get_width()/2., height + (abs(height)*0.01 if height > 0 else -abs(height)*0.01),
                          f'${value:,.0f}', ha='center', va='bottom' if height > 0 else 'top')

    # Performance vs sample size
    if not performance_df.empty:
        axes[1,0].scatter(performance_df['test_samples'], performance_df['test_accuracy'],
                         c='blue', s=100, alpha=0.7)
        axes[1,0].set_xlabel('Test Sample Size')
        axes[1,0].set_ylabel('Test Accuracy')
        axes[1,0].set_title('Model Accuracy vs Test Sample Size')
        axes[1,0].grid(True, alpha=0.3)

    # Final recommendation
    axes[1,1].text(0.1, 0.8, 'SYSTEM RECOMMENDATION', fontsize=14, fontweight='bold',
                   transform=axes[1,1].transAxes)

    # Determine recommendation
    recommendation = "❌ DO NOT DEPLOY"
    reason = "No positive financial impact detected"
    color = 'red'

    if not impact_df.empty:
        best_total = impact_df.groupby('strategy')['pnl_improvement'].sum().max()
        if best_total > 0:
            recommendation = "✅ CONSIDER DEPLOYMENT"
            reason = f"Best strategy shows ${best_total:,.0f} improvement"
            color = 'green'
        elif best_total > -50000:  # Small negative impact might be acceptable
            recommendation = "⚠️ CAUTION REQUIRED"
            reason = f"Marginal negative impact: ${best_total:,.0f}"
            color = 'orange'

    axes[1,1].text(0.1, 0.6, recommendation, fontsize=12, color=color, fontweight='bold',
                   transform=axes[1,1].transAxes)
    axes[1,1].text(0.1, 0.4, reason, fontsize=10, transform=axes[1,1].transAxes, wrap=True)

    # Add key metrics
    if not performance_df.empty:
        avg_test_acc = performance_df['test_accuracy'].mean()
        axes[1,1].text(0.1, 0.2, f'Average Test Accuracy: {avg_test_acc:.3f}',
                      fontsize=10, transform=axes[1,1].transAxes)

    if not impact_df.empty:
        total_traders = len(impact_df['trader_id'].unique())
        axes[1,1].text(0.1, 0.1, f'Traders Analyzed: {total_traders}',
                      fontsize=10, transform=axes[1,1].transAxes)

    axes[1,1].set_xlim(0, 1)
    axes[1,1].set_ylim(0, 1)
    axes[1,1].axis('off')

    plt.tight_layout()
    plt.savefig('outputs/reports/executive_summary_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved comprehensive executive summary")

def create_detailed_report(performance_df, impact_df, data):
    """Create comprehensive text report"""
    print("\n=== CREATING DETAILED REPORT ===")

    report_path = Path("outputs/reports/comprehensive_analysis_report.txt")

    with open(report_path, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("COMPREHENSIVE TRADER RISK MANAGEMENT SYSTEM ANALYSIS REPORT\n")
        f.write("=" * 100 + "\n\n")

        # Executive Summary
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-" * 50 + "\n")

        if not performance_df.empty:
            avg_test_acc = performance_df['test_accuracy'].mean()
            overfitting_rate = performance_df['is_overfitting'].mean() * 100
            f.write(f"• Models trained for {len(performance_df)} individual traders\n")
            f.write(f"• Average test accuracy: {avg_test_acc:.3f}\n")
            f.write(f"• Overfitting detected in {overfitting_rate:.1f}% of models\n")

        if not impact_df.empty:
            strategies = impact_df['strategy'].unique()
            f.write(f"• {len(strategies)} trading strategies evaluated\n")

            for strategy in strategies:
                strategy_data = impact_df[impact_df['strategy'] == strategy]
                total_impact = strategy_data['pnl_improvement'].sum()
                success_rate = strategy_data['has_positive_impact'].mean() * 100
                f.write(f"• {strategy.replace('_', ' ').title()}: ${total_impact:,.0f} impact, {success_rate:.1f}% success rate\n")

        f.write("\n" + "=" * 100 + "\n\n")

        # Model Performance Analysis
        f.write("MODEL PERFORMANCE ANALYSIS\n")
        f.write("-" * 50 + "\n")

        if not performance_df.empty:
            f.write("Individual Trader Model Performance:\n\n")
            f.write(f"{'Trader ID':<10} {'Train Acc':<10} {'Test Acc':<10} {'Difference':<12} {'Overfitting':<12} {'Test Samples':<12}\n")
            f.write("-" * 70 + "\n")

            for _, row in performance_df.iterrows():
                overfitting_status = "YES ⚠️" if row['is_overfitting'] else "NO ✓"
                f.write(f"{row['trader_id']:<10} {row['train_accuracy']:<10.3f} {row['test_accuracy']:<10.3f} "
                       f"{row['accuracy_difference']:<12.3f} {overfitting_status:<12} {row['test_samples']:<12}\n")

            f.write(f"\nSummary Statistics:\n")
            f.write(f"• Average train accuracy: {performance_df['train_accuracy'].mean():.4f}\n")
            f.write(f"• Average test accuracy: {performance_df['test_accuracy'].mean():.4f}\n")
            f.write(f"• Standard deviation of test accuracy: {performance_df['test_accuracy'].std():.4f}\n")
            f.write(f"• Models with overfitting: {performance_df['is_overfitting'].sum()}/{len(performance_df)}\n")

            # Class-specific performance
            f.write(f"\nClass-Specific Performance:\n")
            for class_col in ['class_0_accuracy', 'class_1_accuracy', 'class_2_accuracy']:
                class_name = class_col.replace('class_', '').replace('_accuracy', '')
                class_labels = {'0': 'Loss', '1': 'Neutral', '2': 'Win'}
                avg_acc = performance_df[class_col].mean()
                f.write(f"• {class_labels.get(class_name, class_name)} class average accuracy: {avg_acc:.4f}\n")

        f.write("\n" + "=" * 100 + "\n\n")

        # Strategy Impact Analysis
        f.write("STRATEGY IMPACT ANALYSIS\n")
        f.write("-" * 50 + "\n")

        if not impact_df.empty:
            for strategy in impact_df['strategy'].unique():
                strategy_data = impact_df[impact_df['strategy'] == strategy]
                f.write(f"\n{strategy.upper().replace('_', ' ')} STRATEGY:\n")
                f.write(f"{'Trader ID':<10} {'Baseline PnL':<15} {'Strategy PnL':<15} {'Improvement':<15} {'Impact':<8}\n")
                f.write("-" * 65 + "\n")

                for _, row in strategy_data.iterrows():
                    impact_status = "✓" if row['has_positive_impact'] else "❌"
                    f.write(f"{row['trader_id']:<10} ${row['baseline_pnl']:<14,.0f} ${row['strategy_pnl']:<14,.0f} "
                           f"${row['pnl_improvement']:<14,.0f} {impact_status:<8}\n")

                total_improvement = strategy_data['pnl_improvement'].sum()
                success_rate = strategy_data['has_positive_impact'].mean() * 100
                avg_sharpe = strategy_data['sharpe_improvement'].mean()

                f.write(f"\nStrategy Summary:\n")
                f.write(f"• Total PnL improvement: ${total_improvement:,.2f}\n")
                f.write(f"• Success rate: {success_rate:.1f}%\n")
                f.write(f"• Average Sharpe improvement: {avg_sharpe:.4f}\n")
                f.write(f"• Traders with positive impact: {strategy_data['has_positive_impact'].sum()}/{len(strategy_data)}\n")

        f.write("\n" + "=" * 100 + "\n\n")

        # Causal Impact Summary
        causal_impact = data.get('causal_impact', {})
        if causal_impact:
            f.write("CAUSAL IMPACT VALIDATION\n")
            f.write("-" * 50 + "\n")
            f.write(f"• Signal validation passed: {'✓' if causal_impact.get('signal_validation_passed') else '❌'}\n")
            f.write(f"• Strategies tested: {causal_impact.get('strategies_tested', 0)}\n")
            f.write(f"• Best strategy: {causal_impact.get('best_strategy', 'None')}\n")
            f.write(f"• Deployment viable: {'✓' if causal_impact.get('deployment_viable') else '❌'}\n")
            f.write(f"• Total traders tested: {causal_impact.get('total_traders_tested', 0)}\n")

        f.write("\n" + "=" * 100 + "\n\n")

        # Final Recommendation
        f.write("FINAL RECOMMENDATION\n")
        f.write("-" * 50 + "\n")

        if not impact_df.empty:
            best_strategy = impact_df.groupby('strategy')['pnl_improvement'].sum().idxmax()
            best_impact = impact_df.groupby('strategy')['pnl_improvement'].sum().max()

            if best_impact > 0:
                f.write("✅ RECOMMENDATION: PROCEED WITH DEPLOYMENT\n")
                f.write(f"   • Best performing strategy: {best_strategy.replace('_', ' ').title()}\n")
                f.write(f"   • Expected financial impact: ${best_impact:,.2f}\n")
                f.write(f"   • Deploy with appropriate safeguards and monitoring\n")
            elif best_impact > -50000:
                f.write("⚠️ RECOMMENDATION: PROCEED WITH EXTREME CAUTION\n")
                f.write(f"   • Marginal negative impact: ${best_impact:,.2f}\n")
                f.write(f"   • Consider additional model refinement\n")
                f.write(f"   • Implement strict monitoring and circuit breakers\n")
            else:
                f.write("❌ RECOMMENDATION: DO NOT DEPLOY\n")
                f.write(f"   • Significant negative impact: ${best_impact:,.2f}\n")
                f.write(f"   • System would harm trading performance\n")
                f.write(f"   • Requires fundamental redesign\n")
        else:
            f.write("❌ RECOMMENDATION: INSUFFICIENT DATA FOR DEPLOYMENT\n")
            f.write("   • Unable to validate positive financial impact\n")

        f.write("\n" + "=" * 100 + "\n")

    print(f"✓ Saved comprehensive report to {report_path}")

def main():
    """Main analysis function"""
    print("=" * 100)
    print("COMPREHENSIVE TRADER RISK MANAGEMENT SYSTEM ANALYSIS")
    print("=" * 100)

    # Load all data
    data = load_all_data()

    # Analyze model performance and overfitting
    performance_df = analyze_model_performance(data)

    # Analyze strategy impact
    impact_df = analyze_strategy_impact(data)

    # Create comprehensive visualizations
    create_comprehensive_visualizations(performance_df, impact_df, data)

    # Create detailed report
    create_detailed_report(performance_df, impact_df, data)

    print("\n" + "=" * 100)
    print("COMPREHENSIVE ANALYSIS COMPLETE")
    print("=" * 100)
    print("Generated reports:")
    print("• outputs/reports/model_performance_analysis.png")
    print("• outputs/reports/strategy_impact_analysis.png")
    print("• outputs/reports/executive_summary_comprehensive.png")
    print("• outputs/reports/comprehensive_analysis_report.txt")

    # Print key findings
    if not performance_df.empty:
        print(f"\nKey Findings:")
        print(f"• Average test accuracy: {performance_df['test_accuracy'].mean():.3f}")
        print(f"• Overfitting rate: {performance_df['is_overfitting'].mean()*100:.1f}%")

    if not impact_df.empty:
        best_strategy_impact = impact_df.groupby('strategy')['pnl_improvement'].sum().max()
        print(f"• Best strategy impact: ${best_strategy_impact:,.0f}")

        if best_strategy_impact > 0:
            print("• ✅ System shows positive financial impact")
        else:
            print("• ❌ System shows negative financial impact")

if __name__ == "__main__":
    main()
