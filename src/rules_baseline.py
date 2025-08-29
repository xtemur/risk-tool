"""
Rules-based baseline system (CLAUDE.md implementation)
Simple rules that any risk manager would implement
This is the baseline to beat with ML
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from config import config


class RulesBasedRiskSystem:
    """
    Simple rules that any risk manager would implement
    This is your baseline to beat
    """

    def rules_based_risk_limit(self, trader_data: pd.DataFrame) -> Dict:
        """
        Simple rules that any risk manager would implement
        This is your baseline to beat with ML
        """
        latest = trader_data.iloc[-1]
        history = trader_data.iloc[-20:]  # Last 20 days

        reduction_pct = 0
        reasons = []

        # Rule 1: Consecutive losses
        loss_streak = self._calculate_loss_streak(history)
        if loss_streak >= 3:
            reduction_pct = 30
            reasons.append(f"{loss_streak} day loss streak")
        elif loss_streak >= 5:
            reduction_pct = 50
            reasons.append(f"Extended loss streak ({loss_streak} days)")

        # Rule 2: Drawdown
        current_dd = self._calculate_drawdown_pct(history)
        if current_dd > 15:
            reduction_pct = max(reduction_pct, 40)
            reasons.append(f"Drawdown {current_dd:.1f}%")
        elif current_dd > 25:
            reduction_pct = max(reduction_pct, 60)
            reasons.append(f"Severe drawdown {current_dd:.1f}%")

        # Rule 3: Volatility spike
        if len(history) >= 10:
            recent_vol = history['pnl'].iloc[-5:].std()
            normal_vol = history['pnl'].iloc[:-5].std()
            if recent_vol > normal_vol * 2:
                reduction_pct = max(reduction_pct, 25)
                reasons.append("Volatility 2x normal")

        # Rule 4: Large single-day loss
        if len(history) >= 10:
            loss_threshold = history['pnl'].quantile(0.05)
            if latest['pnl'] < loss_threshold:
                reduction_pct = max(reduction_pct, 20)
                reasons.append("Large loss yesterday")

        return {
            'reduction_pct': min(reduction_pct, config.MAX_REDUCTION),  # Cap at 80%
            'reasons': reasons
        }

    def get_all_predictions(self, data: pd.DataFrame) -> Dict[str, Dict]:
        """Get predictions for all traders"""
        predictions = {}

        for trader_id in data['trader_id'].unique():
            trader_data = data[data['trader_id'] == trader_id].sort_values('date')

            if len(trader_data) >= 10:  # Minimum history required
                prediction = self.rules_based_risk_limit(trader_data)
            else:
                prediction = {'reduction_pct': 0, 'reasons': ['Insufficient history']}

            predictions[trader_id] = prediction

        return predictions

    def _calculate_loss_streak(self, trader_data: pd.DataFrame) -> int:
        """Calculate consecutive loss days ending today"""
        pnl_values = trader_data['pnl'].values[::-1]  # Reverse to start from today
        streak = 0
        for daily_pnl in pnl_values:
            if daily_pnl < 0:
                streak += 1
            else:
                break
        return streak

    def _calculate_drawdown_pct(self, trader_data: pd.DataFrame) -> float:
        """Calculate current drawdown percentage from recent peak"""
        cumulative = trader_data['pnl'].cumsum()
        running_max = cumulative.expanding().max()
        current_drawdown = (cumulative.iloc[-1] - running_max.iloc[-1]) / (running_max.iloc[-1] + 1e-8)
        return abs(current_drawdown) * 100


# This baseline must be beaten by at least 10% to justify ML complexity
