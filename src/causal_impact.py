# src/causal_impact.py

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_causal_impact(model, df: pd.DataFrame, config: Dict, model_type: str = 'var') -> Dict:
    """
    Analyze the causal impact of model predictions on truly unseen data.

    This analysis helps understand:
    1. How well the model generalizes to completely new data
    2. Whether predictions have meaningful impact on risk management
    3. Statistical significance of the model's predictions

    Args:
        model: Trained model
        df: Feature DataFrame with unseen data
        config: Configuration dictionary
        model_type: 'var' or 'loss'

    Returns:
        Dict containing causal impact metrics
    """
    logger.info(f"Starting causal impact analysis for {model_type} model...")

    # Load selected features from model metadata
    import json
    import os
    metadata_path = os.path.join(config['paths']['model_dir'], 'model_metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Use only the features the model was trained on
    feature_cols = metadata['selected_features']
    logger.info(f"Using {len(feature_cols)} selected features for causal impact analysis")

    X = df[feature_cols]

    if model_type == 'var':
        y_true = df['target_pnl']
        predictions = model.predict(X)
        impact_results = analyze_var_impact(df, predictions, config)
    else:
        y_true = df['target_large_loss']
        predictions = model.predict_proba(X)[:, 1]
        impact_results = analyze_loss_impact(df, predictions, config)

    return impact_results


def analyze_var_impact(df: pd.DataFrame, var_predictions: np.ndarray, config: Dict) -> Dict:
    """
    Analyze the causal impact of VaR predictions.

    Key questions:
    - Would traders who followed the VaR limits have better outcomes?
    - What is the economic impact of the predictions?
    """
    results = {
        'model_type': 'var',
        'analysis_period': f"{df['trade_date'].min()} to {df['trade_date'].max()}",
        'n_observations': len(df)
    }

    # Add predictions to dataframe
    analysis_df = df.copy()
    analysis_df['var_prediction'] = var_predictions

    # 1. Violation Analysis
    alpha = config['production_model']['var_model']['alpha']
    violations = (analysis_df['target_pnl'] < analysis_df['var_prediction'])
    violation_rate = violations.mean()

    results['violation_metrics'] = {
        'expected_violation_rate': alpha,
        'actual_violation_rate': violation_rate,
        'violation_ratio': violation_rate / alpha,
        'n_violations': violations.sum()
    }

    # Statistical test for violation rate
    n_trials = len(violations)
    n_violations = violations.sum()
    from scipy.stats import binomtest
    test_result = binomtest(n_violations, n_trials, alpha, alternative='two-sided')
    results['violation_metrics']['statistical_significance'] = test_result.pvalue

    # 2. Economic Impact Analysis
    # Simulate what would happen if traders respected VaR limits
    analysis_df['would_trade'] = True  # Assume all trades happened
    analysis_df['hypothetical_would_trade'] = analysis_df['var_prediction'] > -1000  # Risk threshold

    # Actual vs hypothetical PnL
    actual_total_pnl = analysis_df['target_pnl'].sum()
    hypothetical_pnl = analysis_df[analysis_df['hypothetical_would_trade']]['target_pnl'].sum()
    avoided_losses = analysis_df[~analysis_df['hypothetical_would_trade'] & (analysis_df['target_pnl'] < 0)]['target_pnl'].sum()
    missed_gains = analysis_df[~analysis_df['hypothetical_would_trade'] & (analysis_df['target_pnl'] > 0)]['target_pnl'].sum()

    results['economic_impact'] = {
        'actual_total_pnl': actual_total_pnl,
        'hypothetical_total_pnl': hypothetical_pnl,
        'pnl_difference': hypothetical_pnl - actual_total_pnl,
        'avoided_losses': abs(avoided_losses),
        'missed_gains': missed_gains,
        'net_benefit': abs(avoided_losses) - missed_gains
    }

    # 3. Per-Trader Impact
    trader_impacts = []
    for trader_id in analysis_df['account_id'].unique():
        trader_data = analysis_df[analysis_df['account_id'] == trader_id]

        trader_impact = {
            'account_id': trader_id,
            'n_days': len(trader_data),
            'violation_rate': (trader_data['target_pnl'] < trader_data['var_prediction']).mean(),
            'actual_total_pnl': trader_data['target_pnl'].sum(),
            'worst_day_pnl': trader_data['target_pnl'].min(),
            'worst_day_var': trader_data.loc[trader_data['target_pnl'].idxmin(), 'var_prediction'],
            'var_breach_severity': []
        }

        # Calculate breach severity when violations occur
        breach_mask = trader_data['target_pnl'] < trader_data['var_prediction']
        if breach_mask.any():
            breaches = trader_data[breach_mask]
            trader_impact['var_breach_severity'] = (
                (breaches['var_prediction'] - breaches['target_pnl']) / abs(breaches['var_prediction'])
            ).mean()

        trader_impacts.append(trader_impact)

    results['trader_impacts'] = pd.DataFrame(trader_impacts)

    # 4. Temporal Stability
    # Check if model performance degrades over time
    analysis_df['month'] = pd.to_datetime(analysis_df['trade_date']).dt.to_period('M')
    monthly_violations = analysis_df.groupby('month').apply(
        lambda x: (x['target_pnl'] < x['var_prediction']).mean()
    )

    # Test for trend in violations over time
    if len(monthly_violations) > 2:
        months_numeric = np.arange(len(monthly_violations))
        slope, intercept, r_value, p_value, std_err = stats.linregress(months_numeric, monthly_violations.values)

        results['temporal_stability'] = {
            'trend_slope': slope,
            'trend_p_value': p_value,
            'r_squared': r_value ** 2,
            'performance_degrading': slope > 0 and p_value < 0.05
        }

    return results


def analyze_loss_impact(df: pd.DataFrame, loss_predictions: np.ndarray, config: Dict) -> Dict:
    """
    Analyze the causal impact of large loss predictions.

    Key questions:
    - Could the model have prevented large losses?
    - What is the precision/recall trade-off in practice?
    """
    results = {
        'model_type': 'loss',
        'analysis_period': f"{df['trade_date'].min()} to {df['trade_date'].max()}",
        'n_observations': len(df)
    }

    # Add predictions to dataframe
    analysis_df = df.copy()
    analysis_df['loss_probability'] = loss_predictions

    # 1. Classification Performance at Different Thresholds
    thresholds = [0.3, 0.5, 0.7]
    threshold_metrics = []

    for threshold in thresholds:
        pred_binary = (analysis_df['loss_probability'] > threshold).astype(int)

        tp = ((pred_binary == 1) & (analysis_df['target_large_loss'] == 1)).sum()
        fp = ((pred_binary == 1) & (analysis_df['target_large_loss'] == 0)).sum()
        tn = ((pred_binary == 0) & (analysis_df['target_large_loss'] == 0)).sum()
        fn = ((pred_binary == 0) & (analysis_df['target_large_loss'] == 1)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        # Economic impact of using this threshold
        # Assume trading is reduced when high loss probability is predicted
        high_risk_days = analysis_df[pred_binary == 1]
        prevented_losses = high_risk_days[high_risk_days['target_pnl'] < 0]['target_pnl'].sum()
        missed_profits = high_risk_days[high_risk_days['target_pnl'] > 0]['target_pnl'].sum()

        threshold_metrics.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1_score': 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0,
            'true_positives': tp,
            'false_positives': fp,
            'prevented_losses': abs(prevented_losses),
            'missed_profits': missed_profits,
            'net_benefit': abs(prevented_losses) - missed_profits
        })

    results['threshold_analysis'] = pd.DataFrame(threshold_metrics)

    # 2. Feature Importance for High-Risk Predictions
    high_risk_mask = analysis_df['loss_probability'] > 0.7
    if high_risk_mask.any():
        high_risk_days = analysis_df[high_risk_mask]

        # Identify which features were most extreme on high-risk days
        feature_cols = [col for col in df.columns if col not in [
            'account_id', 'trade_date', 'target_pnl', 'target_large_loss',
            'daily_pnl', 'large_loss_threshold', 'loss_probability'
        ]]

        feature_analysis = {}
        for feature in feature_cols:
            if feature in high_risk_days.columns:
                # Compare high-risk vs normal days
                high_risk_mean = high_risk_days[feature].mean()
                normal_mean = analysis_df[~high_risk_mask][feature].mean()

                # Statistical test
                if len(high_risk_days) > 1 and len(analysis_df[~high_risk_mask]) > 1:
                    t_stat, p_value = stats.ttest_ind(
                        high_risk_days[feature].dropna(),
                        analysis_df[~high_risk_mask][feature].dropna()
                    )

                    feature_analysis[feature] = {
                        'high_risk_mean': high_risk_mean,
                        'normal_mean': normal_mean,
                        'difference': high_risk_mean - normal_mean,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    }

        results['high_risk_feature_analysis'] = pd.DataFrame(feature_analysis).T

    # 3. Consecutive High-Risk Days Analysis
    # Are high-risk predictions clustered (indicating regime changes)?
    analysis_df['high_risk'] = (analysis_df['loss_probability'] > 0.5).astype(int)

    consecutive_risks = []
    for trader_id in analysis_df['account_id'].unique():
        trader_data = analysis_df[analysis_df['account_id'] == trader_id].sort_values('trade_date')

        # Find consecutive high-risk periods
        trader_data['risk_group'] = (trader_data['high_risk'] != trader_data['high_risk'].shift()).cumsum()
        risk_periods = trader_data[trader_data['high_risk'] == 1].groupby('risk_group').agg({
            'trade_date': ['min', 'max', 'count'],
            'target_pnl': 'sum'
        })

        if len(risk_periods) > 0:
            consecutive_risks.append({
                'account_id': trader_id,
                'n_risk_periods': len(risk_periods),
                'avg_period_length': risk_periods[('trade_date', 'count')].mean(),
                'max_period_length': risk_periods[('trade_date', 'count')].max(),
                'total_pnl_during_risk': risk_periods[('target_pnl', 'sum')].sum()
            })

    if consecutive_risks:
        results['risk_clustering'] = pd.DataFrame(consecutive_risks)

    # 4. Counterfactual Analysis
    # What if traders had reduced position sizes on high-risk days?
    position_reduction_factors = [0.5, 0.25, 0.0]  # 50% reduction, 75% reduction, no trading

    counterfactual_results = []
    for factor in position_reduction_factors:
        modified_df = analysis_df.copy()
        high_risk_mask = modified_df['loss_probability'] > 0.5

        # Simulate reduced positions
        modified_df.loc[high_risk_mask, 'counterfactual_pnl'] = (
            modified_df.loc[high_risk_mask, 'target_pnl'] * factor
        )
        modified_df.loc[~high_risk_mask, 'counterfactual_pnl'] = modified_df.loc[~high_risk_mask, 'target_pnl']

        counterfactual_results.append({
            'position_factor': factor,
            'original_total_pnl': analysis_df['target_pnl'].sum(),
            'counterfactual_total_pnl': modified_df['counterfactual_pnl'].sum(),
            'pnl_difference': modified_df['counterfactual_pnl'].sum() - analysis_df['target_pnl'].sum(),
            'n_days_affected': high_risk_mask.sum()
        })

    results['counterfactual_analysis'] = pd.DataFrame(counterfactual_results)

    return results


def generate_causal_impact_report(var_impact: Dict, loss_impact: Dict, config: Dict) -> None:
    """
    Generate comprehensive causal impact report with visualizations.
    """
    logger.info("Generating causal impact report...")

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))

    # 1. VaR Violation Rate Analysis
    ax1 = plt.subplot(3, 3, 1)
    violation_data = var_impact['violation_metrics']
    bars = ax1.bar(['Expected', 'Actual'],
                    [violation_data['expected_violation_rate'], violation_data['actual_violation_rate']])
    ax1.set_ylabel('Violation Rate')
    ax1.set_title('VaR Violation Rate: Expected vs Actual')
    ax1.axhline(y=violation_data['expected_violation_rate'], color='r', linestyle='--', alpha=0.5)

    # Add significance annotation
    if violation_data['statistical_significance'] < 0.05:
        ax1.text(0.5, max(violation_data['expected_violation_rate'], violation_data['actual_violation_rate']) * 1.1,
                f"p-value: {violation_data['statistical_significance']:.4f}*", ha='center')

    # 2. Economic Impact
    ax2 = plt.subplot(3, 3, 2)
    economic_data = var_impact['economic_impact']
    categories = ['Avoided\nLosses', 'Missed\nGains', 'Net\nBenefit']
    values = [economic_data['avoided_losses'], economic_data['missed_gains'], economic_data['net_benefit']]
    colors = ['green', 'red', 'blue']
    ax2.bar(categories, values, color=colors)
    ax2.set_ylabel('PnL Impact')
    ax2.set_title('Economic Impact of VaR Model')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    # 3. Per-Trader Violation Rates
    ax3 = plt.subplot(3, 3, 3)
    trader_data = var_impact['trader_impacts']
    ax3.bar(trader_data['account_id'].astype(str), trader_data['violation_rate'])
    ax3.set_xlabel('Trader ID')
    ax3.set_ylabel('Violation Rate')
    ax3.set_title('Violation Rates by Trader')
    ax3.axhline(y=violation_data['expected_violation_rate'], color='r', linestyle='--', alpha=0.5, label='Expected')
    ax3.tick_params(axis='x', rotation=45)
    ax3.legend()

    # 4. Loss Model Threshold Analysis
    ax4 = plt.subplot(3, 3, 4)
    threshold_data = loss_impact['threshold_analysis']
    ax4.plot(threshold_data['threshold'], threshold_data['precision'], 'b-o', label='Precision')
    ax4.plot(threshold_data['threshold'], threshold_data['recall'], 'r-o', label='Recall')
    ax4.plot(threshold_data['threshold'], threshold_data['f1_score'], 'g-o', label='F1-Score')
    ax4.set_xlabel('Probability Threshold')
    ax4.set_ylabel('Score')
    ax4.set_title('Loss Model Performance by Threshold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Net Benefit by Threshold
    ax5 = plt.subplot(3, 3, 5)
    ax5.bar(threshold_data['threshold'].astype(str), threshold_data['net_benefit'])
    ax5.set_xlabel('Probability Threshold')
    ax5.set_ylabel('Net Benefit (PnL)')
    ax5.set_title('Economic Benefit by Risk Threshold')

    # 6. Counterfactual Analysis
    ax6 = plt.subplot(3, 3, 6)
    counter_data = loss_impact['counterfactual_analysis']
    scenarios = ['100%', '50%', '25%', '0%'][:len(counter_data)]
    pnl_diff = counter_data['pnl_difference'].values
    ax6.bar(scenarios, pnl_diff)
    ax6.set_xlabel('Position Size on High-Risk Days')
    ax6.set_ylabel('PnL Difference')
    ax6.set_title('Counterfactual Analysis: Position Sizing Impact')
    ax6.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    # 7. Temporal Stability (if available)
    if 'temporal_stability' in var_impact:
        ax7 = plt.subplot(3, 3, 7)
        temp_data = var_impact['temporal_stability']
        ax7.text(0.1, 0.8, f"Trend Slope: {temp_data['trend_slope']:.4f}", transform=ax7.transAxes)
        ax7.text(0.1, 0.6, f"P-value: {temp_data['trend_p_value']:.4f}", transform=ax7.transAxes)
        ax7.text(0.1, 0.4, f"R²: {temp_data['r_squared']:.4f}", transform=ax7.transAxes)
        ax7.text(0.1, 0.2, f"Degrading: {'Yes' if temp_data['performance_degrading'] else 'No'}",
                transform=ax7.transAxes, color='red' if temp_data['performance_degrading'] else 'green')
        ax7.set_title('Model Temporal Stability')
        ax7.axis('off')

    # 8. Summary Statistics
    ax8 = plt.subplot(3, 3, 8)
    summary_text = [
        f"VaR Model Performance:",
        f"  Violation Ratio: {violation_data['violation_ratio']:.2f}",
        f"  Economic Benefit: ${economic_data['net_benefit']:,.0f}",
        "",
        f"Loss Model Performance:",
        f"  Best F1-Score: {threshold_data['f1_score'].max():.3f}",
        f"  Best Net Benefit: ${threshold_data['net_benefit'].max():,.0f}",
        "",
        f"Analysis Period: {var_impact['analysis_period']}",
        f"Total Observations: {var_impact['n_observations']:,}"
    ]

    for i, text in enumerate(summary_text):
        ax8.text(0.05, 0.9 - i*0.1, text, transform=ax8.transAxes, fontsize=10)
    ax8.set_title('Summary Statistics')
    ax8.axis('off')

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(config['paths']['report_dir'], 'causal_impact_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Causal impact report saved to {plot_path}")

    # Generate text report
    report_lines = [
        "Causal Impact Analysis Report",
        "=" * 80,
        "",
        "VAR MODEL IMPACT",
        "-" * 40,
        f"Expected violation rate: {violation_data['expected_violation_rate']:.3f}",
        f"Actual violation rate: {violation_data['actual_violation_rate']:.3f}",
        f"Statistical significance: p={violation_data['statistical_significance']:.4f}",
        f"Economic net benefit: ${economic_data['net_benefit']:,.2f}",
        "",
        "LOSS MODEL IMPACT",
        "-" * 40,
        "Optimal threshold analysis:",
    ]

    for _, row in threshold_data.iterrows():
        report_lines.append(
            f"  Threshold {row['threshold']}: F1={row['f1_score']:.3f}, "
            f"Net Benefit=${row['net_benefit']:,.2f}"
        )

    # Save text report
    text_path = os.path.join(config['paths']['report_dir'], 'causal_impact_summary.txt')
    with open(text_path, 'w') as f:
        f.write('\n'.join(report_lines))

    logger.info(f"Text summary saved to {text_path}")
