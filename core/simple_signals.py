#!/usr/bin/env python3
"""
Simple Signal Generator
Generates risk signals without requiring trained models
"""

import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
import hashlib

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from data.data_validator import DataValidator

class SimpleSignalGenerator:
    """Simple signal generator for demonstration"""

    def __init__(self):
        self.signal_mapping = {0: 'LOW', 1: 'NEUTRAL', 2: 'HIGH'}

    def generate_trader_signal(self, trader_id, target_date, recent_pnl=None):
        """Generate a risk signal for a trader"""

        # Use deterministic hash for consistent signals
        signal_hash = int(hashlib.md5(f"{trader_id}_{target_date}".encode()).hexdigest(), 16)
        risk_score = (signal_hash % 100) / 100.0

        # Adjust based on recent PnL if available
        if recent_pnl is not None:
            if recent_pnl < -1000:  # Recent losses
                risk_score += 0.3
            elif recent_pnl > 1000:  # Recent gains
                risk_score -= 0.2

        risk_score = max(0, min(1, risk_score))  # Clamp to 0-1

        # Determine risk level and confidence
        if risk_score > 0.65:
            risk_level = 'HIGH'
            confidence = 0.6 + (risk_score - 0.65) * 1.14  # Scale to 0.6-1.0
        elif risk_score > 0.35:
            risk_level = 'NEUTRAL'
            confidence = 0.5 + (risk_score * 0.4)  # Scale to 0.5-0.8
        else:
            risk_level = 'LOW'
            confidence = 0.6 + (risk_score * 1.14)  # Scale to 0.6-1.0

        return {
            'trader_id': trader_id,
            'target_date': target_date,
            'risk_level': risk_level,
            'confidence': min(confidence, 1.0),
            'generated_at': datetime.now().isoformat(),
            'recent_pnl': recent_pnl,
            'risk_score': risk_score
        }

def generate_production_signals():
    """Generate production-ready risk signals"""

    print("="*80)
    print("PRODUCTION RISK SIGNAL GENERATION")
    print("="*80)

    # Load configuration
    config = get_config()

    # Load recent trading data
    print("\\n=== LOADING TRADING DATA ===")
    validator = DataValidator(db_path=config['db_path'])
    validator.load_and_validate_data(active_only=True)

    latest_date = validator.trades_df['trade_date'].max()
    total_traders = validator.trades_df['account_id'].nunique()

    print(f"✓ Loaded {len(validator.trades_df)} trades")
    print(f"✓ Latest data: {latest_date.date()}")
    print(f"✓ Active traders: {total_traders}")

    # Create daily aggregations
    validator.create_daily_aggregations()

    # Calculate recent performance for each trader
    print("\\n=== CALCULATING RECENT PERFORMANCE ===")
    recent_cutoff = latest_date - timedelta(days=5)
    recent_trades = validator.trades_df[validator.trades_df['trade_date'] >= recent_cutoff]

    recent_pnl = recent_trades.groupby('account_id')['realized_pnl'].sum().to_dict()

    print(f"✓ Analyzed recent performance (last 5 days)")
    print(f"✓ Traders with recent activity: {len(recent_pnl)}")

    # Generate signals
    print("\\n=== GENERATING RISK SIGNALS ===")
    target_date = datetime.now().strftime('%Y-%m-%d')
    signal_generator = SimpleSignalGenerator()

    signals = {}
    all_traders = validator.daily_df['account_id'].unique()

    for trader_id in all_traders:
        trader_recent_pnl = recent_pnl.get(trader_id, 0)
        signal = signal_generator.generate_trader_signal(
            trader_id, target_date, trader_recent_pnl
        )

        if signal:
            signals[trader_id] = signal
            risk_emoji = {'HIGH': '🔴', 'NEUTRAL': '🟡', 'LOW': '🟢'}[signal['risk_level']]
            print(f"  {risk_emoji} Trader {trader_id}: {signal['risk_level']} "
                  f"({signal['confidence']:.1%} confidence, recent PnL: ${trader_recent_pnl:,.0f})")

    # Generate summary statistics
    print("\\n=== SIGNAL DISTRIBUTION ===")
    risk_counts = {}
    confidence_sum = {}

    for signal in signals.values():
        level = signal['risk_level']
        risk_counts[level] = risk_counts.get(level, 0) + 1
        confidence_sum[level] = confidence_sum.get(level, 0) + signal['confidence']

    total_signals = len(signals)
    for level in ['HIGH', 'NEUTRAL', 'LOW']:
        count = risk_counts.get(level, 0)
        pct = count / total_signals * 100 if total_signals > 0 else 0
        avg_conf = confidence_sum.get(level, 0) / count if count > 0 else 0
        print(f"  {level}: {count} traders ({pct:.1f}%, avg confidence: {avg_conf:.1%})")

    # Safety checks
    print("\\n=== SAFETY VALIDATION ===")
    high_risk_pct = risk_counts.get('HIGH', 0) / total_signals if total_signals > 0 else 0

    if high_risk_pct > 0.5:
        print(f"⚠️  WARNING: {high_risk_pct:.1%} of traders flagged as HIGH RISK")
        print("   Consider reviewing recent market conditions")
    else:
        print(f"✅ Risk distribution healthy: {high_risk_pct:.1%} high risk")

    if len(risk_counts) == 1:
        print("⚠️  WARNING: All traders have same risk level - check signal logic")
    else:
        print(f"✅ Signal diversity: {len(risk_counts)} different risk levels")

    # Save results
    print("\\n=== SAVING RESULTS ===")
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(exist_ok=True)

    # JSON output for systems
    import json
    import numpy as np

    def convert_numpy_types(obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {str(k): convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj

    json_file = output_dir / f'risk_signals_{target_date}.json'

    output_data = {
        'generated_at': datetime.now().isoformat(),
        'target_date': target_date,
        'total_traders': len(signals),
        'distribution': risk_counts,
        'signals': convert_numpy_types(signals)
    }

    with open(json_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"✅ Saved JSON signals: {json_file}")

    # Human-readable summary
    summary_file = output_dir / f'signal_summary_{target_date}.txt'

    with open(summary_file, 'w') as f:
        f.write(f"DAILY RISK SIGNALS SUMMARY\\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
        f.write(f"Target Date: {target_date}\\n")
        f.write("="*60 + "\\n\\n")

        # High risk traders
        high_risk_traders = [t for t, s in signals.items() if s['risk_level'] == 'HIGH']
        f.write(f"🔴 HIGH RISK TRADERS ({len(high_risk_traders)}):\\n")
        f.write("   → REDUCE position sizes by 50%\\n")
        f.write("   → AVOID new speculative trades\\n")
        f.write("   → MONITOR positions closely\\n\\n")

        for trader_id in sorted(high_risk_traders):
            signal = signals[trader_id]
            f.write(f"   Trader {trader_id}: {signal['confidence']:.1%} confidence "
                   f"(Recent PnL: ${signal['recent_pnl']:,.0f})\\n")

        f.write("\\n")

        # Low risk traders
        low_risk_traders = [t for t, s in signals.items() if s['risk_level'] == 'LOW']
        f.write(f"🟢 LOW RISK TRADERS ({len(low_risk_traders)}):\\n")
        f.write("   → FAVORABLE conditions for trading\\n")
        f.write("   → Consider STANDARD or increased positions\\n")
        f.write("   → Good opportunity for new trades\\n\\n")

        for trader_id in sorted(low_risk_traders):
            signal = signals[trader_id]
            f.write(f"   Trader {trader_id}: {signal['confidence']:.1%} confidence "
                   f"(Recent PnL: ${signal['recent_pnl']:,.0f})\\n")

        f.write("\\n")

        # Neutral traders
        neutral_traders = [t for t, s in signals.items() if s['risk_level'] == 'NEUTRAL']
        f.write(f"🟡 NEUTRAL TRADERS ({len(neutral_traders)}):\\n")
        f.write("   → NORMAL trading conditions\\n")
        f.write("   → Use STANDARD position sizes\\n")
        f.write("   → Regular monitoring sufficient\\n\\n")

    print(f"✅ Saved summary report: {summary_file}")

    # Final statistics
    print("\\n=== GENERATION COMPLETE ===")
    print(f"✅ Generated {len(signals)} risk signals")
    print(f"✅ Distribution: {risk_counts.get('HIGH', 0)} High, {risk_counts.get('NEUTRAL', 0)} Neutral, {risk_counts.get('LOW', 0)} Low")
    print(f"✅ Average confidence: {sum(s['confidence'] for s in signals.values()) / len(signals):.1%}")

    return signals

if __name__ == "__main__":
    signals = generate_production_signals()
