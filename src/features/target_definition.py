"""
Target variable definitions for risk prediction.

This module defines what constitutes a "high-risk" trading day
with mathematical precision.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RiskTargetDefinition:
    """
    Defines high-risk trading days using multiple criteria.

    A high-risk day is defined as a day where ANY of the following occur:
    1. Large drawdown: Daily net P&L < 5th percentile of trader's historical P&L
    2. Excessive fees: Fees > 50% of gross P&L (churning indicator)
    3. Leverage spike: Position size > 3x average daily position size
    4. Execution degradation: Fill rate < 70% AND slippage > 2x average

    This multi-criteria approach captures different types of risk:
    - Financial risk (drawdowns)
    - Operational risk (high fees, poor execution)
    - Behavioral risk (overleveraging)
    """

    def __init__(
        self,
        drawdown_percentile: float = 5.0,
        fee_ratio_threshold: float = 0.5,
        leverage_multiplier: float = 3.0,
        fill_rate_threshold: float = 0.7,
        slippage_multiplier: float = 2.0,
        lookback_days: int = 60,
    ):
        """
        Initialize risk target definition.

        Args:
            drawdown_percentile: Percentile for drawdown threshold (default: 5th)
            fee_ratio_threshold: Max acceptable fees/gross P&L ratio
            leverage_multiplier: Multiplier for position size spike
            fill_rate_threshold: Minimum acceptable fill rate
            slippage_multiplier: Multiplier for slippage spike
            lookback_days: Days to look back for rolling calculations
        """
        self.drawdown_percentile = drawdown_percentile
        self.fee_ratio_threshold = fee_ratio_threshold
        self.leverage_multiplier = leverage_multiplier
        self.fill_rate_threshold = fill_rate_threshold
        self.slippage_multiplier = slippage_multiplier
        self.lookback_days = lookback_days

    def calculate_targets(
        self,
        daily_data: pd.DataFrame,
        account_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Calculate risk targets for daily data.

        Args:
            daily_data: DataFrame with columns from account_daily_summary table
            account_id: Optional account filter

        Returns:
            DataFrame with risk indicators and final target
        """
        if account_id:
            data = daily_data[daily_data['account_id'] == account_id].copy()
        else:
            data = daily_data.copy()

        # Sort by account and date
        data = data.sort_values(['account_id', 'date'])

        # Initialize risk indicators
        data['is_large_drawdown'] = False
        data['is_high_fee_ratio'] = False
        data['is_overleveraged'] = False
        data['is_poor_execution'] = False

        # Process each account separately
        for account in data['account_id'].unique():
            mask = data['account_id'] == account
            account_data = data[mask]

            # 1. Large drawdown detection
            net_pnl = account_data['net'].fillna(0)
            rolling_threshold = net_pnl.rolling(
                window=self.lookback_days,
                min_periods=30
            ).quantile(self.drawdown_percentile / 100)

            data.loc[mask, 'drawdown_threshold'] = rolling_threshold
            data.loc[mask, 'is_large_drawdown'] = net_pnl < rolling_threshold

            # 2. High fee ratio detection
            gross_pnl = account_data['gross'].fillna(0)
            total_fees = account_data['fees'].fillna(0)

            # Avoid division by zero, consider high fee if gross is near zero
            fee_ratio = np.where(
                np.abs(gross_pnl) < 1e-6,
                np.where(total_fees > 0, 1.0, 0.0),
                np.abs(total_fees / gross_pnl)
            )

            data.loc[mask, 'fee_ratio'] = fee_ratio
            data.loc[mask, 'is_high_fee_ratio'] = fee_ratio > self.fee_ratio_threshold

            # 3. Overleveraging detection
            position_size = np.abs(account_data['qty'].fillna(0))
            avg_position = position_size.rolling(
                window=self.lookback_days,
                min_periods=30
            ).mean()

            data.loc[mask, 'avg_position_size'] = avg_position
            data.loc[mask, 'is_overleveraged'] = (
                position_size > self.leverage_multiplier * avg_position
            )

            # 4. Poor execution detection
            fills = account_data['fills'].fillna(0)
            orders = account_data['orders'].fillna(1)  # Avoid div by zero
            fill_rate = fills / orders

            # Calculate average slippage (simplified - would need tick data for accurate calc)
            # Using gross-net as proxy for slippage + fees
            execution_cost = gross_pnl - net_pnl
            avg_execution_cost = execution_cost.rolling(
                window=self.lookback_days,
                min_periods=30
            ).mean()

            data.loc[mask, 'fill_rate'] = fill_rate
            data.loc[mask, 'execution_cost'] = execution_cost
            data.loc[mask, 'avg_execution_cost'] = avg_execution_cost

            data.loc[mask, 'is_poor_execution'] = (
                (fill_rate < self.fill_rate_threshold) &
                (execution_cost > self.slippage_multiplier * avg_execution_cost)
            )

        # Calculate final target: high-risk if ANY indicator is true
        data['is_high_risk'] = (
            data['is_large_drawdown'] |
            data['is_high_fee_ratio'] |
            data['is_overleveraged'] |
            data['is_poor_execution']
        )

        # Add risk score (number of risk indicators triggered)
        data['risk_score'] = (
            data['is_large_drawdown'].astype(int) +
            data['is_high_fee_ratio'].astype(int) +
            data['is_overleveraged'].astype(int) +
            data['is_poor_execution'].astype(int)
        )

        # Log summary statistics
        high_risk_pct = data['is_high_risk'].mean() * 100
        logger.info(f"High-risk days: {high_risk_pct:.1f}%")
        logger.info(f"Risk indicator breakdown:")
        logger.info(f"  - Large drawdown: {data['is_large_drawdown'].mean() * 100:.1f}%")
        logger.info(f"  - High fee ratio: {data['is_high_fee_ratio'].mean() * 100:.1f}%")
        logger.info(f"  - Overleveraged: {data['is_overleveraged'].mean() * 100:.1f}%")
        logger.info(f"  - Poor execution: {data['is_poor_execution'].mean() * 100:.1f}%")

        return data

    def calculate_forward_looking_target(
        self,
        daily_data: pd.DataFrame,
        forward_days: int = 1,
    ) -> pd.DataFrame:
        """
        Calculate target that predicts risk in the NEXT N days.

        This is useful for giving traders advance warning.

        Args:
            daily_data: DataFrame with risk indicators
            forward_days: Days to look forward (default: 1 = next day)

        Returns:
            DataFrame with forward-looking target
        """
        data = daily_data.copy()

        # First calculate current day risk
        if 'is_high_risk' not in data.columns:
            data = self.calculate_targets(data)

        # Sort by account and date
        data = data.sort_values(['account_id', 'date'])

        # Create forward-looking target
        data['is_high_risk_next'] = False

        for account in data['account_id'].unique():
            mask = data['account_id'] == account

            # Shift risk indicator forward
            data.loc[mask, 'is_high_risk_next'] = data.loc[mask, 'is_high_risk'].shift(
                -forward_days
            ).fillna(False)

        return data

    def get_target_statistics(
        self,
        data: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Calculate statistics about the target variable.

        Args:
            data: DataFrame with risk targets

        Returns:
            Dictionary of statistics
        """
        if 'is_high_risk' not in data.columns:
            raise ValueError("Data must contain 'is_high_risk' column")

        stats = {
            'total_days': len(data),
            'high_risk_days': data['is_high_risk'].sum(),
            'high_risk_pct': data['is_high_risk'].mean() * 100,
            'risk_score_mean': data['risk_score'].mean(),
            'risk_score_std': data['risk_score'].std(),
        }

        # Add per-indicator statistics
        for indicator in ['is_large_drawdown', 'is_high_fee_ratio',
                         'is_overleveraged', 'is_poor_execution']:
            if indicator in data.columns:
                stats[f'{indicator}_pct'] = data[indicator].mean() * 100

        # Add per-account statistics
        account_stats = data.groupby('account_id')['is_high_risk'].agg(['sum', 'mean'])
        stats['accounts_with_risk'] = (account_stats['sum'] > 0).sum()
        stats['avg_risk_per_account'] = account_stats['mean'].mean() * 100

        return stats
