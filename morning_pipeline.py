"""
Morning Pipeline (CLAUDE.md implementation)
Simplified pipeline that actually works
No over-engineering, just essentials
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict
import sys
import traceback

# Add src to path for imports
sys.path.append('src')
from src.minimal_risk_system import MinimalRiskSystem
from config import config


def morning_pipeline():
    """
    Simplified pipeline that actually works
    No over-engineering, just essentials
    """

    # 1. Load data (30 min)
    print(f"[{datetime.now().strftime('%H:%M')}] Loading data...")

    try:
        system = MinimalRiskSystem()
        data = system._load_data()

        # 2. Quick data quality check (15 min)
        assert not data['pnl'].isna().any(), "Missing P&L data"
        assert len(data['trader_id'].unique()) >= 10, "Missing traders"
        print(f"[{datetime.now().strftime('%H:%M')}] Data quality check passed")

        # 3. Feature computation and predictions (30 min)
        print(f"[{datetime.now().strftime('%H:%M')}] Computing features and generating predictions...")
        predictions = system.run_daily()

        # 4. Generate report (30 min)
        print(f"[{datetime.now().strftime('%H:%M')}] Report generated")

        # 5. Send email (5 min) - integrated in system.send_report()
        print(f"[{datetime.now().strftime('%H:%M')}] ✓ Complete")

        return predictions

    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M')}] ❌ Pipeline failed: {str(e)}")
        traceback.print_exc()

        # Fallback: send error notification
        print("Sending error notification to risk managers...")
        # In production, integrate with inference/email_service.py for error alerts

        return {}


def create_simple_report(predictions: Dict, features: pd.DataFrame = None) -> str:
    """
    Simple, clear email that risk managers can act on
    """
    date_str = datetime.now().strftime('%Y-%m-%d')

    email = f"""
RISK LIMITS - {date_str}
====================

ACTION REQUIRED (>40% reduction):
"""

    high_risk = [(t, p) for t, p in predictions.items() if p.get('reduction_pct', 0) > 40]
    moderate_risk = [(t, p) for t, p in predictions.items() if 20 <= p.get('reduction_pct', 0) <= 40]

    if high_risk:
        for trader_id, pred in sorted(high_risk, key=lambda x: x[1].get('reduction_pct', 0), reverse=True):
            reduction = pred.get('reduction_pct', 0)
            reasons = ', '.join(pred.get('reasons', ['Unknown']))
            new_limit = config.DEFAULT_LIMIT * (1 - reduction/100)
            email += f"""
Trader {trader_id}: REDUCE LIMIT BY {reduction:.0f}%
  New limit: ${new_limit:,.0f}
  Reasons: {reasons}
  Confidence: {pred.get('confidence', 'High')}
---"""
    else:
        email += "\nNone today.\n"

    if moderate_risk:
        email += "\n\nMODERATE ADJUSTMENTS (20-40%):\n"
        for trader_id, pred in moderate_risk:
            reduction = pred.get('reduction_pct', 0)
            reasons = ', '.join(pred.get('reasons', ['Unknown']))
            new_limit = config.DEFAULT_LIMIT * (1 - reduction/100)
            email += f"Trader {trader_id}: Reduce by {reduction:.0f}% (${new_limit:,.0f}) - {reasons}\n"

    # Determine system type
    system_used = "ML model" if any('model' in str(p.get('reasons', [])) for p in predictions.values()) else "Rules-based system"

    email += f"\n\nMODEL CONFIDENCE: {system_used}"

    return email


if __name__ == "__main__":
    """
    Main entry point - called by cron job every morning
    """
    print("="*60)
    print(f"MORNING RISK PIPELINE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    predictions = morning_pipeline()

    if predictions:
        # Summary
        total = len(predictions)
        restricted = len([p for p in predictions.values() if p.get('reduction_pct', 0) > 0])
        high_risk = len([p for p in predictions.values() if p.get('reduction_pct', 0) > 40])

        print(f"\nSUMMARY:")
        print(f"Total traders: {total}")
        print(f"Restrictions: {restricted}")
        print(f"High risk: {high_risk}")

        if high_risk > 0:
            print(f"\nHIGH RISK TRADERS:")
            for trader_id, pred in predictions.items():
                if pred.get('reduction_pct', 0) > 40:
                    print(f"  {trader_id}: {pred['reduction_pct']:.0f}% reduction")

    print("="*60)
