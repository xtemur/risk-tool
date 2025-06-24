# src/monitoring.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from typing import Dict
import logging
import os
from .causal_impact import analyze_causal_impact, generate_causal_impact_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_feature_drift_report(training_df: pd.DataFrame, recent_df: pd.DataFrame, config: Dict) -> None:
    """
    Generate feature drift report comparing training and recent data distributions.

    Args:
        training_df: Original training data
        recent_df: Recent production data
        config: Configuration dictionary
    """
    logger.info("Generating feature drift report...")

    # Get feature columns
    feature_cols = [col for col in training_df.columns if col not in [
        'account_id', 'trade_date', 'target_pnl', 'target_large_loss',
        'daily_pnl', 'large_loss_threshold'
    ]]

    # Calculate statistics for both datasets
    drift_report = []

    for feature in feature_cols:
        if feature in training_df.columns and feature in recent_df.columns:
            # Skip boolean columns for quantile calculations
            if training_df[feature].dtype == bool or recent_df[feature].dtype == bool:
                train_stats = {
                    'feature': feature,
                    'train_mean': training_df[feature].astype(float).mean(),
                    'train_std': training_df[feature].astype(float).std(),
                    'train_p25': training_df[feature].astype(float).quantile(0.25),
                    'train_p50': training_df[feature].astype(float).quantile(0.50),
                    'train_p75': training_df[feature].astype(float).quantile(0.75),
                    'recent_mean': recent_df[feature].astype(float).mean(),
                    'recent_std': recent_df[feature].astype(float).std(),
                    'recent_p25': recent_df[feature].astype(float).quantile(0.25),
                    'recent_p50': recent_df[feature].astype(float).quantile(0.50),
                    'recent_p75': recent_df[feature].astype(float).quantile(0.75)
                }
            else:
                train_stats = {
                    'feature': feature,
                    'train_mean': training_df[feature].mean(),
                    'train_std': training_df[feature].std(),
                    'train_p25': training_df[feature].quantile(0.25),
                    'train_p50': training_df[feature].quantile(0.50),
                    'train_p75': training_df[feature].quantile(0.75),
                    'recent_mean': recent_df[feature].mean(),
                    'recent_std': recent_df[feature].std(),
                    'recent_p25': recent_df[feature].quantile(0.25),
                    'recent_p50': recent_df[feature].quantile(0.50),
                    'recent_p75': recent_df[feature].quantile(0.75)
                }

            # Calculate drift metrics
            train_stats['mean_drift'] = abs(train_stats['recent_mean'] - train_stats['train_mean']) / (train_stats['train_std'] + 1e-8)
            train_stats['std_ratio'] = train_stats['recent_std'] / (train_stats['train_std'] + 1e-8)

            drift_report.append(train_stats)

    # Convert to DataFrame
    drift_df = pd.DataFrame(drift_report)

    # Sort by mean drift
    drift_df = drift_df.sort_values('mean_drift', ascending=False)

    # Create report text
    report_lines = [
        "Feature Drift Report",
        "=" * 80,
        f"Training data period: {training_df['trade_date'].min()} to {training_df['trade_date'].max()}",
        f"Recent data period: {recent_df['trade_date'].min()} to {recent_df['trade_date'].max()}",
        f"Training samples: {len(training_df)}",
        f"Recent samples: {len(recent_df)}",
        "",
        "Top 10 Features with Highest Drift:",
        "-" * 80
    ]

    # Add top drifting features
    for _, row in drift_df.head(10).iterrows():
        report_lines.append(
            f"{row['feature']:30s} | "
            f"Mean Drift: {row['mean_drift']:6.2f} | "
            f"Std Ratio: {row['std_ratio']:6.2f} | "
            f"Train Mean: {row['train_mean']:10.2f} | "
            f"Recent Mean: {row['recent_mean']:10.2f}"
        )

    report_lines.extend([
        "",
        "Features with Significant Drift (mean_drift > 0.5):",
        "-" * 80
    ])

    # Identify features with significant drift
    significant_drift = drift_df[drift_df['mean_drift'] > 0.5]
    for _, row in significant_drift.iterrows():
        report_lines.append(f"- {row['feature']}: drift = {row['mean_drift']:.2f}")

    # Save report
    report_path = os.path.join(config['paths']['report_dir'], 'feature_drift_report.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))

    logger.info(f"Feature drift report saved to {report_path}")

    # Also save detailed CSV
    csv_path = os.path.join(config['paths']['report_dir'], 'feature_drift_details.csv')
    drift_df.to_csv(csv_path, index=False)

    # Print summary
    logger.info(f"Features with significant drift: {len(significant_drift)}")
    logger.info(f"Average mean drift: {drift_df['mean_drift'].mean():.3f}")


def generate_model_stability_report(model, training_df: pd.DataFrame, recent_df: pd.DataFrame, config: Dict, model_type: str = 'var') -> None:
    """
    Generate model stability report using SHAP values.

    Args:
        model: Trained model
        training_df: Original training data
        recent_df: Recent production data
        config: Configuration dictionary
        model_type: 'var' or 'loss' to specify which model
    """
    logger.info(f"Generating model stability report for {model_type} model...")

    # Get feature columns
    feature_cols = [col for col in training_df.columns if col not in [
        'account_id', 'trade_date', 'target_pnl', 'target_large_loss',
        'daily_pnl', 'large_loss_threshold'
    ]]

    # Sample data for SHAP analysis (to avoid memory issues)
    n_samples = min(1000, len(training_df), len(recent_df))
    train_sample = training_df[feature_cols].sample(n=n_samples, random_state=42)
    recent_sample = recent_df[feature_cols].sample(n=n_samples, random_state=42)

    # Calculate SHAP values
    logger.info("Calculating SHAP values for training data...")
    explainer = shap.TreeExplainer(model)
    shap_values_train = explainer.shap_values(train_sample)

    logger.info("Calculating SHAP values for recent data...")
    shap_values_recent = explainer.shap_values(recent_sample)

    # For classification models, use positive class SHAP values
    if model_type == 'loss' and isinstance(shap_values_train, list):
        shap_values_train = shap_values_train[1]
        shap_values_recent = shap_values_recent[1]

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Plot SHAP summary for training data
    plt.sca(ax1)
    shap.summary_plot(shap_values_train, train_sample, show=False, max_display=20)
    ax1.set_title(f'SHAP Summary - Training Data ({model_type.upper()} Model)', fontsize=14)

    # Plot SHAP summary for recent data
    plt.sca(ax2)
    shap.summary_plot(shap_values_recent, recent_sample, show=False, max_display=20)
    ax2.set_title(f'SHAP Summary - Recent Data ({model_type.upper()} Model)', fontsize=14)

    # Save plot
    plot_path = os.path.join(config['paths']['report_dir'], f'model_stability_{model_type}.png')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Model stability plot saved to {plot_path}")

    # Calculate feature importance correlation
    train_importance = np.abs(shap_values_train).mean(axis=0)
    recent_importance = np.abs(shap_values_recent).mean(axis=0)

    importance_correlation = np.corrcoef(train_importance, recent_importance)[0, 1]
    logger.info(f"Feature importance correlation between training and recent: {importance_correlation:.3f}")

    # Create importance comparison DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'train_importance': train_importance,
        'recent_importance': recent_importance,
        'importance_change': recent_importance - train_importance
    })

    importance_df = importance_df.sort_values('importance_change', key=abs, ascending=False)

    # Save importance comparison
    importance_path = os.path.join(config['paths']['report_dir'], f'feature_importance_change_{model_type}.csv')
    importance_df.to_csv(importance_path, index=False)

    logger.info("Top 5 features with largest importance changes:")
    logger.info(importance_df[['feature', 'importance_change']].head())

    return importance_correlation
