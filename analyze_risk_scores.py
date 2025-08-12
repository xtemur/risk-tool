#!/usr/bin/env python3
"""
Analyze risk score distributions to understand why only one Medium Risk is generated.
"""

import sys
import pandas as pd
import numpy as np
import json
from pathlib import Path
from inference.signal_generator import SignalGenerator

def analyze_risk_distribution():
    """Analyze the distribution of risk scores and classifications."""

    # Initialize signal generator
    signal_gen = SignalGenerator()

    # Generate daily signals to get predictions
    signals = signal_gen.generate_daily_signals()

    # Print first trader signal to see structure
    if signals['trader_signals']:
        print(f"Sample trader signal structure: {list(signals['trader_signals'][0].keys())}")

    # Extract predictions from signals
    predictions = {}
    for trader_signal in signals['trader_signals']:
        trader_id = int(trader_signal['trader_id'])
        predictions[trader_id] = {
            'var_prediction': trader_signal.get('var_5pct', 0),
            'loss_probability': trader_signal.get('loss_probability', 0),
            'risk_score': trader_signal.get('risk_score', 0),
            'risk_level': trader_signal.get('risk_level', 'unknown')
        }

    # Analyze risk scores
    risk_analysis = []

    for trader_id, pred_data in predictions.items():
        if pred_data:
            var_pred = pred_data.get('var_prediction', 0)
            loss_prob = pred_data.get('loss_probability', 0)

            # Calculate risk score using default parameters
            risk_score = signal_gen._calculate_risk_score(
                var_pred, loss_prob,
                alpha=0.6, beta=0.4,
                var_range=(-50000, 0)
            )

            # Get risk classification
            risk_level = signal_gen.classify_risk_level(
                trader_id, var_pred, loss_prob,
                use_weighted_formula=True
            )

            risk_analysis.append({
                'trader_id': trader_id,
                'var_prediction': var_pred,
                'loss_probability': loss_prob,
                'risk_score': risk_score,
                'risk_level': risk_level,
                'actual_risk_score': pred_data.get('risk_score', 0),
                'actual_risk_level': pred_data.get('risk_level', 'unknown'),
                'normalized_var': (0 - var_pred) / (0 - (-50000)) if var_pred <= 0 else 0,
                'normalized_loss_prob': loss_prob
            })

    # Convert to DataFrame for analysis
    df = pd.DataFrame(risk_analysis)

    print("=== RISK SCORE ANALYSIS ===\n")
    print(f"Total traders analyzed: {len(df)}")
    print(f"\nRisk Level Distribution:")
    print(df['risk_level'].value_counts().sort_index())

    print(f"\nRisk Score Statistics:")
    print(f"Mean: {df['risk_score'].mean():.4f}")
    print(f"Std: {df['risk_score'].std():.4f}")
    print(f"Min: {df['risk_score'].min():.4f}")
    print(f"Max: {df['risk_score'].max():.4f}")
    print(f"25th percentile: {df['risk_score'].quantile(0.25):.4f}")
    print(f"50th percentile: {df['risk_score'].quantile(0.50):.4f}")
    print(f"75th percentile: {df['risk_score'].quantile(0.75):.4f}")

    print(f"\nTraders by Risk Level:")
    for level in ['high', 'medium', 'low', 'neutral']:
        traders = df[df['risk_level'] == level]['trader_id'].tolist()
        if traders:
            print(f"\n{level.upper()} RISK: {traders}")
            level_df = df[df['risk_level'] == level]
            print(f"  Risk scores: {level_df['risk_score'].tolist()}")

    print(f"\n=== COMPONENT ANALYSIS ===")
    print(f"\nVaR Component (normalized):")
    print(f"Mean: {df['normalized_var'].mean():.4f}")
    print(f"Std: {df['normalized_var'].std():.4f}")

    print(f"\nLoss Probability Component:")
    print(f"Mean: {df['normalized_loss_prob'].mean():.4f}")
    print(f"Std: {df['normalized_loss_prob'].std():.4f}")

    # Suggest better thresholds based on distribution
    print(f"\n=== SUGGESTED THRESHOLDS ===")

    # Calculate percentiles for better distribution
    p20 = df['risk_score'].quantile(0.20)
    p50 = df['risk_score'].quantile(0.50)
    p80 = df['risk_score'].quantile(0.80)

    print(f"\nCurrent thresholds:")
    print(f"High: >= 0.70")
    print(f"Medium: >= 0.50")
    print(f"Low: >= 0.30")

    print(f"\nSuggested thresholds (based on percentiles):")
    print(f"High: >= {p80:.3f} (top 20%)")
    print(f"Medium: >= {p50:.3f} (middle 30%)")
    print(f"Low: >= {p20:.3f} (bottom 30%)")
    print(f"Neutral: < {p20:.3f} (bottom 20%)")

    return df, {
        'high_threshold': p80,
        'medium_threshold': p50,
        'low_threshold': p20
    }

if __name__ == "__main__":
    df, suggested_thresholds = analyze_risk_distribution()

    # Save analysis results
    df.to_csv('risk_score_analysis.csv', index=False)
    print(f"\nDetailed analysis saved to risk_score_analysis.csv")

    # Save suggested thresholds
    with open('suggested_risk_thresholds.json', 'w') as f:
        json.dump(suggested_thresholds, f, indent=2)
    print(f"Suggested thresholds saved to suggested_risk_thresholds.json")
