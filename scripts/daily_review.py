#!/usr/bin/env python
"""
Daily Analytics Review Script
Automated flagging system based on interpretation guide
"""

import sys
import pandas as pd
from datetime import date
sys.path.append('src')

from analytics import TraderAnalytics
from database import Database

class DailyReviewSystem:
    """Automated daily review based on interpretation guide"""

    def __init__(self):
        self.analytics = TraderAnalytics()
        self.db = Database()

        # Thresholds from interpretation guide
        self.RED_FLAGS = {
            'total_pnl': -1000,
            'current_drawdown': -500,
            'losing_streak': 5,
            'win_rate': 45,
            'fee_efficiency': 5
        }

        self.YELLOW_FLAGS = {
            'sharpe_ratio': 0.5,
            'profit_factor': 1.2,
            'concentration_risk': 0.5
        }

        self.GREEN_LIGHTS = {
            'omega_ratio': 2.0,
            'kelly_criterion': 0.15,
            'sortino_ratio': 1.5
        }

    def generate_daily_flags(self, lookback_days=7) -> dict:
        """Generate flags for all traders"""

        traders_df = self.db.get_all_traders()

        red_flags = []
        yellow_flags = []
        green_lights = []

        for _, trader in traders_df.iterrows():
            account_id = str(trader['account_id'])
            trader_name = trader['trader_name']

            # Get analytics
            analytics = self.analytics.generate_trader_analytics(account_id, lookback_days)

            if 'error' in analytics:
                continue

            # Check red flags
            flags = self._check_red_flags(analytics)
            if flags:
                red_flags.append({
                    'trader': trader_name,
                    'account_id': account_id,
                    'flags': flags,
                    'metrics': analytics
                })

            # Check yellow flags
            flags = self._check_yellow_flags(analytics)
            if flags:
                yellow_flags.append({
                    'trader': trader_name,
                    'account_id': account_id,
                    'flags': flags,
                    'metrics': analytics
                })

            # Check green lights
            flags = self._check_green_lights(analytics)
            if flags:
                green_lights.append({
                    'trader': trader_name,
                    'account_id': account_id,
                    'flags': flags,
                    'metrics': analytics
                })

        return {
            'red_flags': red_flags,
            'yellow_flags': yellow_flags,
            'green_lights': green_lights,
            'review_date': date.today().strftime('%Y-%m-%d')
        }

    def _check_red_flags(self, analytics) -> list:
        """Check for immediate action items"""
        flags = []

        perf = analytics.get('performance', {})
        risk = analytics.get('risk', {})
        efficiency = analytics.get('efficiency', {})

        if perf.get('total_pnl', 0) < self.RED_FLAGS['total_pnl']:
            flags.append(f"P&L below ${self.RED_FLAGS['total_pnl']}: ${perf.get('total_pnl', 0):.2f}")

        if perf.get('current_drawdown', 0) < self.RED_FLAGS['current_drawdown']:
            flags.append(f"High drawdown: ${perf.get('current_drawdown', 0):.2f}")

        if risk.get('max_losing_streak', 0) > self.RED_FLAGS['losing_streak']:
            flags.append(f"Long losing streak: {risk.get('max_losing_streak', 0)} days")

        if perf.get('win_rate', 100) < self.RED_FLAGS['win_rate']:
            flags.append(f"Low win rate: {perf.get('win_rate', 0):.1f}%")

        if efficiency.get('fee_efficiency', 0) > self.RED_FLAGS['fee_efficiency']:
            flags.append(f"High fees: {efficiency.get('fee_efficiency', 0):.2f}%")

        return flags

    def _check_yellow_flags(self, analytics) -> list:
        """Check for monitoring items"""
        flags = []

        perf = analytics.get('performance', {})
        behavior = analytics.get('behavior', {})

        if perf.get('sharpe_ratio', 0) < self.YELLOW_FLAGS['sharpe_ratio']:
            flags.append(f"Low Sharpe ratio: {perf.get('sharpe_ratio', 0):.3f}")

        if perf.get('profit_factor', 0) < self.YELLOW_FLAGS['profit_factor']:
            flags.append(f"Low profit factor: {perf.get('profit_factor', 0):.2f}")

        if behavior.get('concentration_risk', 0) > self.YELLOW_FLAGS['concentration_risk']:
            flags.append(f"High concentration: {behavior.get('concentration_risk', 0):.2f}")

        return flags

    def _check_green_lights(self, analytics) -> list:
        """Check for opportunities"""
        flags = []

        perf = analytics.get('performance', {})
        advanced = analytics.get('advanced', {})

        if advanced.get('omega_ratio', 0) > self.GREEN_LIGHTS['omega_ratio']:
            flags.append(f"Excellent Omega ratio: {advanced.get('omega_ratio', 0):.2f}")

        if perf.get('kelly_criterion', 0) > self.GREEN_LIGHTS['kelly_criterion']:
            flags.append(f"Can increase positions: Kelly={perf.get('kelly_criterion', 0):.3f}")

        if perf.get('sortino_ratio', 0) > self.GREEN_LIGHTS['sortino_ratio']:
            flags.append(f"Great risk-adjusted returns: {perf.get('sortino_ratio', 0):.3f}")

        return flags

    def print_daily_summary(self, flags_data):
        """Print formatted daily summary"""

        print("🔴 RED FLAGS - IMMEDIATE ACTION REQUIRED")
        print("=" * 50)

        if flags_data['red_flags']:
            for item in flags_data['red_flags']:
                print(f"\n👤 {item['trader']} ({item['account_id']})")
                for flag in item['flags']:
                    print(f"   ⚠️ {flag}")
        else:
            print("✅ No immediate action items")

        print("\n🟡 YELLOW FLAGS - MONITOR THIS WEEK")
        print("=" * 50)

        if flags_data['yellow_flags']:
            for item in flags_data['yellow_flags']:
                print(f"\n👤 {item['trader']} ({item['account_id']})")
                for flag in item['flags']:
                    print(f"   ⚠️ {flag}")
        else:
            print("✅ No monitoring items")

        print("\n🟢 GREEN LIGHTS - OPPORTUNITIES")
        print("=" * 50)

        if flags_data['green_lights']:
            for item in flags_data['green_lights']:
                print(f"\n👤 {item['trader']} ({item['account_id']})")
                for flag in item['flags']:
                    print(f"   💡 {flag}")
        else:
            print("No optimization opportunities identified")

        print(f"\n📅 Review Date: {flags_data['review_date']}")
        print(f"📊 Total Traders: {len(flags_data['red_flags']) + len(flags_data['yellow_flags']) + len(flags_data['green_lights'])}")

def main():
    """Run daily review"""

    print("DAILY ANALYTICS REVIEW")
    print("=" * 60)

    review_system = DailyReviewSystem()
    flags = review_system.generate_daily_flags(lookback_days=7)
    review_system.print_daily_summary(flags)

    # Save to file for records
    import json
    with open(f"data/daily_reviews/review_{flags['review_date']}.json", "w") as f:
        json.dump(flags, f, indent=2, default=str)

    print(f"\n💾 Review saved to data/daily_reviews/review_{flags['review_date']}.json")

if __name__ == "__main__":
    # Create directory if needed
    from pathlib import Path
    Path("data/daily_reviews").mkdir(exist_ok=True)

    main()
