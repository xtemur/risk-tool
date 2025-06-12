"""
Minimal feature set for risk prediction.

This module implements a small, robust set of 5-10 features derived from
account_daily_summary table that have proven predictive power.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MinimalRiskFeatures:
    """
    Minimal feature set focusing on robust, interpretable risk indicators.

    Features:
    1. profit_per_volume: net / qty - Profit efficiency
    2. execution_efficiency: fills / orders - Order execution quality
    3. leverage_ratio: unrealized / end_balance - Risk exposure
    4. sharpe_ratio: Rolling Sharpe ratio of daily returns
    5. sortino_ratio: Rolling Sortino ratio (downside risk)
    6. fee_burden: fees / abs(gross) - Cost efficiency
    7. daily_volatility: Rolling standard deviation of returns
    8. max_drawdown: Rolling maximum drawdown
    9. win_rate: Proportion of profitable days (rolling)
    10. risk_adjusted_return: Return per unit of risk taken
    """

    def __init__(
        self,
        lookback_days: int = 20,
        min_periods: int = 10,
    ):
        """
        Initialize minimal features calculator.

        Args:
            lookback_days: Days to look back for rolling calculations
            min_periods: Minimum periods required for rolling calculations
        """
        self.lookback_days = lookback_days
        self.min_periods = min_periods

    def calculate_features(
        self,
        daily_data: pd.DataFrame,
        target_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """
        Calculate minimal feature set from daily data.

        Args:
            daily_data: DataFrame with account_daily_summary data
            target_date: Optional date cutoff for point-in-time calculation

        Returns:
            DataFrame with calculated features
        """
        # Filter by date if specified (point-in-time consistency)
        if target_date:
            data = daily_data[daily_data['date'] <= target_date].copy()
        else:
            data = daily_data.copy()

        # Sort by account and date
        data = data.sort_values(['account_id', 'date'])

        # Initialize feature columns
        feature_names = [
            'profit_per_volume',
            'execution_efficiency',
            'leverage_ratio',
            'sharpe_ratio',
            'sortino_ratio',
            'fee_burden',
            'daily_volatility',
            'max_drawdown',
            'win_rate',
            'risk_adjusted_return',
        ]

        for feature in feature_names:
            data[feature] = np.nan

        # Calculate features for each account
        for account in data['account_id'].unique():
            mask = data['account_id'] == account
            account_data = data[mask].copy()

            # 1. Profit per volume (efficiency metric)
            net_pnl = account_data['net'].fillna(0)
            volume = account_data['qty'].fillna(1)  # Avoid division by zero
            data.loc[mask, 'profit_per_volume'] = np.where(
                volume != 0,
                net_pnl / volume,
                0
            )

            # 2. Execution efficiency
            fills = account_data['fills'].fillna(0)
            orders = account_data['orders'].fillna(1)  # Avoid division by zero
            data.loc[mask, 'execution_efficiency'] = np.where(
                orders > 0,
                fills / orders,
                0
            )

            # 3. Leverage ratio (risk exposure)
            unrealized = account_data['unrealized'].fillna(0)
            end_balance = account_data['end_balance'].fillna(1)  # Avoid division by zero
            data.loc[mask, 'leverage_ratio'] = np.where(
                np.abs(end_balance) > 1,
                np.abs(unrealized / end_balance),
                0
            )

            # 4. Sharpe ratio (risk-adjusted returns)
            # Calculate daily returns
            returns = net_pnl / end_balance.shift(1)
            returns = returns.replace([np.inf, -np.inf], np.nan)

            # Rolling Sharpe ratio (annualized)
            rolling_mean = returns.rolling(
                window=self.lookback_days,
                min_periods=self.min_periods
            ).mean()

            rolling_std = returns.rolling(
                window=self.lookback_days,
                min_periods=self.min_periods
            ).std()

            # Annualize (252 trading days)
            sharpe = np.where(
                rolling_std > 0,
                rolling_mean / rolling_std * np.sqrt(252),
                0
            )
            data.loc[mask, 'sharpe_ratio'] = sharpe

            # 5. Sortino ratio (downside risk focus)
            # Calculate downside deviation
            negative_returns = returns.copy()
            negative_returns[negative_returns > 0] = 0

            downside_std = negative_returns.rolling(
                window=self.lookback_days,
                min_periods=self.min_periods
            ).std()

            sortino = np.where(
                downside_std > 0,
                rolling_mean / downside_std * np.sqrt(252),
                0
            )
            data.loc[mask, 'sortino_ratio'] = sortino

            # 6. Fee burden
            fees = account_data['fees'].fillna(0)
            gross = account_data['gross'].fillna(0)
            data.loc[mask, 'fee_burden'] = np.where(
                np.abs(gross) > 1,
                np.abs(fees / gross),
                0
            )

            # 7. Daily volatility
            data.loc[mask, 'daily_volatility'] = returns.rolling(
                window=self.lookback_days,
                min_periods=self.min_periods
            ).std()

            # 8. Maximum drawdown
            # Calculate cumulative returns
            cum_returns = (1 + returns.fillna(0)).cumprod()

            # Rolling maximum
            rolling_max = cum_returns.rolling(
                window=self.lookback_days,
                min_periods=self.min_periods
            ).max()

            # Drawdown from peak
            drawdown = np.where(
                rolling_max > 0,
                (cum_returns - rolling_max) / rolling_max,
                0
            )

            # Maximum drawdown in window
            max_dd = pd.Series(drawdown).rolling(
                window=self.lookback_days,
                min_periods=self.min_periods
            ).min()

            data.loc[mask, 'max_drawdown'] = np.abs(max_dd)

            # 9. Win rate (proportion of profitable days)
            profitable_days = (net_pnl > 0).astype(float)
            data.loc[mask, 'win_rate'] = profitable_days.rolling(
                window=self.lookback_days,
                min_periods=self.min_periods
            ).mean()

            # 10. Risk-adjusted return (return per unit of risk)
            # Using coefficient of variation inverse
            abs_mean = np.abs(rolling_mean)
            data.loc[mask, 'risk_adjusted_return'] = np.where(
                (rolling_std > 0) & (abs_mean > 0),
                rolling_mean / rolling_std,
                0
            )

        # Handle infinities and NaNs
        for feature in feature_names:
            data[feature] = data[feature].replace([np.inf, -np.inf], np.nan)
            data[feature] = data[feature].fillna(0)

        # Log feature statistics
        logger.info("Minimal feature statistics:")
        for feature in feature_names:
            logger.info(f"  {feature}: mean={data[feature].mean():.4f}, "
                       f"std={data[feature].std():.4f}")

        return data

    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return [
            'profit_per_volume',
            'execution_efficiency',
            'leverage_ratio',
            'sharpe_ratio',
            'sortino_ratio',
            'fee_burden',
            'daily_volatility',
            'max_drawdown',
            'win_rate',
            'risk_adjusted_return',
        ]

    def validate_predictive_power(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        dates: pd.Series,
    ) -> Dict[str, float]:
        """
        Validate predictive power of features using univariate analysis.

        Args:
            features: DataFrame with calculated features
            target: Binary target series (0=normal, 1=high-risk)
            dates: Date series for temporal ordering

        Returns:
            Dictionary of feature importance scores
        """
        from sklearn.metrics import roc_auc_score
        from scipy.stats import mannwhitneyu

        feature_names = self.get_feature_names()
        importance_scores = {}

        # Only use data where we have valid targets
        valid_mask = target.notna()

        for feature in feature_names:
            if feature not in features.columns:
                continue

            # Get feature values
            X = features.loc[valid_mask, feature].values
            y = target.loc[valid_mask].values

            # Skip if no variance
            if np.std(X) == 0 or len(np.unique(y)) < 2:
                importance_scores[feature] = 0.5
                continue

            try:
                # Calculate AUC for each feature individually
                auc = roc_auc_score(y, X)
                # Convert to 0.5-1 scale (0.5 = no predictive power)
                auc = max(auc, 1 - auc)

                # Also calculate Mann-Whitney U statistic
                normal_values = X[y == 0]
                risk_values = X[y == 1]

                if len(normal_values) > 0 and len(risk_values) > 0:
                    statistic, p_value = mannwhitneyu(
                        normal_values, risk_values, alternative='two-sided'
                    )
                    # Convert p-value to importance (lower p = higher importance)
                    mw_importance = 1 - p_value
                else:
                    mw_importance = 0.5

                # Combine metrics
                importance_scores[feature] = (auc + mw_importance) / 2

            except Exception as e:
                logger.warning(f"Error calculating importance for {feature}: {e}")
                importance_scores[feature] = 0.5

        # Sort by importance
        sorted_features = sorted(
            importance_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        logger.info("Feature importance scores:")
        for feature, score in sorted_features:
            logger.info(f"  {feature}: {score:.4f}")

        return importance_scores
