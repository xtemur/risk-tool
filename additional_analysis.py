#!/usr/bin/env python3
"""
Additional Analysis and Visualizations
====================================

This script creates additional visualizations and statistical analysis
to complement the comprehensive report.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_risk_return_heatmap():
    """Create a risk-return heatmap"""
    # Load comparison data
    df = pd.read_csv('results/causal_impact_comparison/comparison_table.csv')

    # Create risk-return matrix
    risk_levels = ['25%', '50%', '70%', '90%']
    metrics = ['Net Benefit', 'Overall Improvement %', 'Mean Intervention Rate %', 'Mean Volatility Reduction %']

    # Prepare data for heatmap
    data_matrix = df[['Net Benefit', 'Overall Improvement %', 'Mean Intervention Rate %', 'Mean Volatility Reduction %']].values

    # Normalize data for better visualization
    data_normalized = (data_matrix - data_matrix.min(axis=0)) / (data_matrix.max(axis=0) - data_matrix.min(axis=0))

    fig, ax = plt.subplots(figsize=(12, 8))

    # Create heatmap
    im = ax.imshow(data_normalized.T, cmap='RdYlGn', aspect='auto')

    # Set ticks and labels
    ax.set_xticks(range(len(risk_levels)))
    ax.set_xticklabels(risk_levels)
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(metrics)

    # Add text annotations
    for i in range(len(metrics)):
        for j in range(len(risk_levels)):
            text = ax.text(j, i, f'{data_matrix[j, i]:.0f}',
                         ha="center", va="center", color="black", fontweight='bold')

    ax.set_title('Risk Management Performance Heatmap', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Risk Reduction Level', fontsize=12)
    ax.set_ylabel('Performance Metrics', fontsize=12)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Normalized Performance Score', rotation=270, labelpad=20)

    plt.tight_layout()
    plt.savefig('risk_return_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_trader_performance_scatter():
    """Create trader performance scatter plot"""
    # Parse evaluation report to extract trader data
    eval_text = open('results/causal_impact_evaluation/evaluation_report.txt').read()

    # Extract trader data (simplified parsing)
    traders = []
    lines = eval_text.split('\n')
    current_trader = None

    for line in lines:
        if line.startswith('Trader '):
            current_trader = line.split(':')[0].replace('Trader ', '')
        elif 'Net Benefit:' in line and current_trader:
            net_benefit = float(line.split('$')[1].replace(',', ''))
        elif 'Improvement:' in line and current_trader:
            improvement = float(line.split(': ')[1].replace('%', ''))
        elif 'Intervention Rate:' in line and current_trader:
            intervention_rate = float(line.split(': ')[1].replace('%', ''))
            traders.append({
                'trader_id': current_trader,
                'net_benefit': net_benefit,
                'improvement': improvement,
                'intervention_rate': intervention_rate
            })

    if traders:
        df = pd.DataFrame(traders)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Scatter plot: Intervention Rate vs Net Benefit
        colors = ['green' if x >= 0 else 'red' for x in df['net_benefit']]
        ax1.scatter(df['intervention_rate'], df['net_benefit'], c=colors, s=100, alpha=0.7)
        ax1.set_xlabel('Intervention Rate (%)')
        ax1.set_ylabel('Net Benefit ($)')
        ax1.set_title('Intervention Rate vs Net Benefit')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

        # Scatter plot: Intervention Rate vs Improvement
        ax2.scatter(df['intervention_rate'], df['improvement'], c='blue', s=100, alpha=0.7)
        ax2.set_xlabel('Intervention Rate (%)')
        ax2.set_ylabel('Improvement (%)')
        ax2.set_title('Intervention Rate vs Improvement')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)

        # Bar chart: Net Benefit by Trader
        colors = ['green' if x >= 0 else 'red' for x in df['net_benefit']]
        bars = ax3.bar(df['trader_id'], df['net_benefit'], color=colors, alpha=0.7)
        ax3.set_xlabel('Trader ID')
        ax3.set_ylabel('Net Benefit ($)')
        ax3.set_title('Net Benefit by Trader')
        ax3.tick_params(axis='x', rotation=45)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

        # Bar chart: Improvement by Trader
        colors = ['green' if x >= 0 else 'red' for x in df['improvement']]
        ax4.bar(df['trader_id'], df['improvement'], color=colors, alpha=0.7)
        ax4.set_xlabel('Trader ID')
        ax4.set_ylabel('Improvement (%)')
        ax4.set_title('Improvement Percentage by Trader')
        ax4.tick_params(axis='x', rotation=45)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)

        plt.tight_layout()
        plt.savefig('trader_performance_detailed.png', dpi=300, bbox_inches='tight')
        plt.show()

def create_efficiency_analysis():
    """Create efficiency analysis charts"""
    df = pd.read_csv('results/causal_impact_comparison/comparison_table.csv')

    # Calculate efficiency metrics
    df['Risk_Return_Efficiency'] = df['Net Benefit'] / df['Mean Intervention Rate %']
    df['Loss_Mitigation_Ratio'] = df['Avoided Losses'] / df['Missed Gains']

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Risk-Return Efficiency
    ax1.plot(df['Reduction %'], df['Risk_Return_Efficiency'],
             marker='o', linewidth=3, markersize=10, color='darkblue')
    ax1.set_xlabel('Risk Reduction Level')
    ax1.set_ylabel('Risk-Return Efficiency')
    ax1.set_title('Risk-Return Efficiency by Reduction Level', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(len(df)))
    ax1.set_xticklabels(df['Reduction %'])

    # Loss Mitigation Ratio
    ax2.bar(range(len(df)), df['Loss_Mitigation_Ratio'],
            color='green', alpha=0.7, width=0.6)
    ax2.set_xlabel('Risk Reduction Level')
    ax2.set_ylabel('Loss Mitigation Ratio')
    ax2.set_title('Loss Mitigation Ratio (Avoided/Missed)', fontweight='bold')
    ax2.set_xticks(range(len(df)))
    ax2.set_xticklabels(df['Reduction %'])
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Break-even')
    ax2.legend()

    # Cumulative Benefit
    cumulative_benefit = df['Net Benefit'].cumsum()
    ax3.plot(range(len(df)), cumulative_benefit,
             marker='s', linewidth=3, markersize=8, color='green')
    ax3.set_xlabel('Risk Reduction Level')
    ax3.set_ylabel('Cumulative Net Benefit ($)')
    ax3.set_title('Cumulative Net Benefit', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(range(len(df)))
    ax3.set_xticklabels(df['Reduction %'])
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    # Volatility vs Return
    ax4.scatter(df['Mean Volatility Reduction %'], df['Overall Improvement %'],
               s=df['Net Benefit']/2000, alpha=0.7, color='purple')
    ax4.set_xlabel('Mean Volatility Reduction (%)')
    ax4.set_ylabel('Overall Improvement (%)')
    ax4.set_title('Volatility Reduction vs Return Improvement', fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # Add labels for each point
    for i, row in df.iterrows():
        ax4.annotate(row['Reduction %'],
                    (row['Mean Volatility Reduction %'], row['Overall Improvement %']),
                    xytext=(5, 5), textcoords='offset points', fontsize=10)

    plt.tight_layout()
    plt.savefig('efficiency_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_summary_statistics():
    """Generate comprehensive summary statistics"""
    df = pd.read_csv('results/causal_impact_comparison/comparison_table.csv')

    print("\n" + "="*100)
    print("COMPREHENSIVE STATISTICAL ANALYSIS")
    print("="*100)

    # Basic statistics
    print("\n📊 DESCRIPTIVE STATISTICS:")
    print(f"Mean Net Benefit: ${df['Net Benefit'].mean():,.2f}")
    print(f"Median Net Benefit: ${df['Net Benefit'].median():,.2f}")
    print(f"Standard Deviation: ${df['Net Benefit'].std():,.2f}")
    print(f"Min Net Benefit: ${df['Net Benefit'].min():,.2f}")
    print(f"Max Net Benefit: ${df['Net Benefit'].max():,.2f}")

    # Efficiency metrics
    efficiency = df['Net Benefit'] / df['Mean Intervention Rate %']
    print(f"\n⚡ EFFICIENCY METRICS:")
    print(f"Mean Efficiency: {efficiency.mean():.2f}")
    print(f"Best Efficiency: {efficiency.max():.2f} (at {df.loc[efficiency.idxmax(), 'Reduction %']} reduction)")
    print(f"Efficiency Growth Rate: {((efficiency.iloc[-1] / efficiency.iloc[0]) - 1) * 100:.1f}%")

    # Risk metrics
    print(f"\n🛡️ RISK METRICS:")
    print(f"Mean Volatility Reduction: {df['Mean Volatility Reduction %'].mean():.1f}%")
    print(f"Max Volatility Reduction: {df['Mean Volatility Reduction %'].max():.1f}%")
    print(f"Risk-Adjusted Return: {df['Overall Improvement %'].mean():.1f}%")

    # Loss mitigation
    loss_ratio = df['Avoided Losses'] / df['Missed Gains']
    print(f"\n💰 LOSS MITIGATION:")
    print(f"Total Avoided Losses: ${df['Avoided Losses'].sum():,.2f}")
    print(f"Total Missed Gains: ${df['Missed Gains'].sum():,.2f}")
    print(f"Average Loss Mitigation Ratio: {loss_ratio.mean():.2f}")
    print(f"Best Loss Mitigation: {loss_ratio.max():.2f} (at {df.loc[loss_ratio.idxmax(), 'Reduction %']} reduction)")

    # Optimal configuration
    optimal_idx = efficiency.idxmax()
    print(f"\n🎯 OPTIMAL CONFIGURATION:")
    print(f"Optimal Risk Reduction Level: {df.loc[optimal_idx, 'Reduction %']}")
    print(f"Expected Net Benefit: ${df.loc[optimal_idx, 'Net Benefit']:,.2f}")
    print(f"Efficiency Score: {efficiency.iloc[optimal_idx]:.2f}")
    print(f"Intervention Rate: {df.loc[optimal_idx, 'Mean Intervention Rate %']:.1f}%")
    print(f"Volatility Reduction: {df.loc[optimal_idx, 'Mean Volatility Reduction %']:.1f}%")

    print("\n" + "="*100)

if __name__ == "__main__":
    print("Generating additional analysis and visualizations...")

    # Create visualizations
    create_risk_return_heatmap()
    create_trader_performance_scatter()
    create_efficiency_analysis()

    # Generate statistics
    generate_summary_statistics()

    print("\nAdditional analysis complete!")
    print("Generated files:")
    print("- risk_return_heatmap.png")
    print("- trader_performance_detailed.png")
    print("- efficiency_analysis.png")
