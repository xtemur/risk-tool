"""
Day Trading Feature Engineering Module

Implements volatility regime detection, position sizing signals, and risk management features
specifically designed for day trading applications.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DayTradingFeatureEngineer:
    """
    Enhanced feature engineering focused on day trading signals.

    Key improvements:
    1. Volatility regime detection (low/normal/high)
    2. Position sizing multiplier based on current conditions
    3. Daily risk budget tracking
    4. Momentum quality assessment
    5. Session-based performance metrics
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}

        # Default parameters for day trading features
        self.volatility_lookback = 20  # Days for volatility calculation
        self.momentum_lookback = 5     # Days for momentum assessment
        self.risk_percentile = 0.05    # For VaR calculation (5th percentile)

    def create_day_trading_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive day trading features.

        Args:
            df: DataFrame with basic trading data

        Returns:
            DataFrame with enhanced day trading features
        """
        logger.info("Creating day trading features...")

        # Start with copy of input data
        enhanced_df = df.copy()

        # 1. Volatility Features
        enhanced_df = self._add_volatility_features(enhanced_df)

        # 2. Position Sizing Features
        enhanced_df = self._add_position_sizing_features(enhanced_df)

        # 3. Risk Management Features
        enhanced_df = self._add_risk_management_features(enhanced_df)

        # 4. Momentum and Trend Features
        enhanced_df = self._add_momentum_features(enhanced_df)

        # 5. Session and Market Condition Features
        enhanced_df = self._add_session_features(enhanced_df)

        # 6. Enhanced Target Variables for Day Trading
        enhanced_df = self._create_day_trading_targets(enhanced_df)

        logger.info(f"Created {len(enhanced_df.columns)} total features for day trading")

        return enhanced_df

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility regime detection features"""

        # Rolling volatility (standard deviation of returns)
        df['volatility_5d'] = df['daily_pnl'].rolling(5, min_periods=1).std()
        df['volatility_10d'] = df['daily_pnl'].rolling(10, min_periods=1).std()
        df['volatility_20d'] = df['daily_pnl'].rolling(20, min_periods=1).std()

        # Volatility percentiles (regime detection)
        df['vol_percentile_20d'] = df['volatility_20d'].rolling(60, min_periods=20).rank(pct=True)
        df['vol_percentile_60d'] = df['volatility_20d'].rolling(120, min_periods=30).rank(pct=True)

        # Volatility regime classification (handle NaN values)
        df['volatility_regime'] = 1  # Normal (default)
        df.loc[(df['vol_percentile_60d'] <= 0.33) & df['vol_percentile_60d'].notna(), 'volatility_regime'] = 0  # Low volatility
        df.loc[(df['vol_percentile_60d'] >= 0.67) & df['vol_percentile_60d'].notna(), 'volatility_regime'] = 2  # High volatility

        # Volatility trend (increasing/decreasing)
        df['vol_trend'] = (df['volatility_5d'] - df['volatility_20d']) / (df['volatility_20d'] + 1e-6)

        # Range expansion/contraction
        df['daily_range'] = abs(df['daily_gross'])  # Proxy for intraday range
        df['range_expansion'] = df['daily_range'].rolling(5).mean() / df['daily_range'].rolling(20).mean()

        return df

    def _add_position_sizing_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features for dynamic position sizing"""

        # Recent performance metrics
        df['win_rate_10d'] = (df['daily_pnl'] > 0).rolling(10, min_periods=1).mean()
        df['win_rate_20d'] = (df['daily_pnl'] > 0).rolling(20, min_periods=1).mean()

        # Average win/loss size
        df['avg_win_10d'] = df['daily_pnl'].where(df['daily_pnl'] > 0).rolling(10, min_periods=1).mean()
        df['avg_loss_10d'] = df['daily_pnl'].where(df['daily_pnl'] < 0).rolling(10, min_periods=1).mean()

        # Risk-adjusted performance
        df['sharpe_10d'] = df['daily_pnl'].rolling(10, min_periods=1).mean() / (df['daily_pnl'].rolling(10, min_periods=1).std() + 1e-6)
        df['sharpe_20d'] = df['daily_pnl'].rolling(20, min_periods=1).mean() / (df['daily_pnl'].rolling(20, min_periods=1).std() + 1e-6)

        # Position sizing multiplier based on recent performance and volatility
        # Base multiplier on recent Sharpe ratio and volatility regime
        base_multiplier = 1.0

        # Adjust based on Sharpe ratio (recent performance)
        sharpe_adjustment = np.clip(df['sharpe_10d'] / 2.0, 0.25, 2.0)  # 0.25x to 2x based on Sharpe

        # Adjust based on volatility regime
        vol_adjustment = pd.Series(1.0, index=df.index)
        vol_adjustment[df['volatility_regime'] == 0] = 1.5  # Low vol = increase size
        vol_adjustment[df['volatility_regime'] == 2] = 0.5  # High vol = decrease size

        # Combined position sizing multiplier
        df['position_size_multiplier'] = np.clip(sharpe_adjustment * vol_adjustment, 0.1, 3.0)

        return df

    def _add_risk_management_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add risk management and daily budget features"""

        # Daily VaR estimates
        df['var_5pct_10d'] = df['daily_pnl'].rolling(10, min_periods=1).quantile(0.05)
        df['var_5pct_20d'] = df['daily_pnl'].rolling(20, min_periods=1).quantile(0.05)

        # Maximum drawdown metrics
        df['running_peak'] = df['cumulative_pnl'].expanding().max()
        df['current_drawdown'] = df['cumulative_pnl'] - df['running_peak']
        df['max_drawdown_20d'] = df['current_drawdown'].rolling(20, min_periods=1).min()

        # Consecutive loss/win streaks
        df['pnl_sign'] = np.sign(df['daily_pnl'])
        df['streak'] = df.groupby((df['pnl_sign'] != df['pnl_sign'].shift()).cumsum())['pnl_sign'].cumsum()
        df['loss_streak'] = np.where(df['pnl_sign'] < 0, -df['streak'], 0)
        df['win_streak'] = np.where(df['pnl_sign'] > 0, df['streak'], 0)

        # Risk budget utilization (how much of daily risk budget is used)
        df['daily_risk_budget'] = -df['var_5pct_20d'] * 2  # 2x VaR as daily risk budget
        df['risk_budget_utilization'] = np.where(
            df['daily_risk_budget'] > 0,
            np.clip(-df['daily_pnl'] / df['daily_risk_budget'], 0, 5),  # Cap at 5x budget
            0
        )

        return df

    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum and trend quality features"""

        # Price momentum (using cumulative PnL as proxy for account value)
        df['momentum_3d'] = df['daily_pnl'].rolling(3, min_periods=1).sum()
        df['momentum_5d'] = df['daily_pnl'].rolling(5, min_periods=1).sum()
        df['momentum_10d'] = df['daily_pnl'].rolling(10, min_periods=1).sum()

        # Trend consistency (what % of recent days were positive)
        df['trend_consistency_5d'] = (df['daily_pnl'] > 0).rolling(5, min_periods=1).mean()
        df['trend_consistency_10d'] = (df['daily_pnl'] > 0).rolling(10, min_periods=1).mean()

        # Momentum quality score (combines trend strength and consistency)
        momentum_strength = np.abs(df['momentum_5d']) / (df['volatility_5d'] * 5 + 1e-6)
        trend_consistency = df['trend_consistency_5d']
        df['momentum_quality'] = np.clip(momentum_strength * trend_consistency, 0, 2)

        # Volume-price relationship (higher volume on winning days is good)
        df['volume_on_wins'] = df['daily_volume'].where(df['daily_pnl'] > 0).rolling(10, min_periods=1).mean()
        df['volume_on_losses'] = df['daily_volume'].where(df['daily_pnl'] < 0).rolling(10, min_periods=1).mean()
        df['volume_selectivity'] = df['volume_on_wins'] / (df['volume_on_losses'] + 1e-6)

        return df

    def _add_session_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add session-based and market condition features"""

        # Day of week performance (some traders perform better on certain days)
        for day in range(5):  # Monday = 0, Friday = 4
            day_mask = df['day_of_week'] == day
            df[f'avg_pnl_day_{day}'] = df['daily_pnl'].where(day_mask).expanding().mean()
            df[f'win_rate_day_{day}'] = (df['daily_pnl'] > 0).where(day_mask).expanding().mean()

        # Days since last significant win/loss
        significant_win = df['daily_pnl'] > df['daily_pnl'].rolling(20, min_periods=1).quantile(0.8)
        significant_loss = df['daily_pnl'] < df['daily_pnl'].rolling(20, min_periods=1).quantile(0.2)

        df['days_since_big_win'] = 0
        df['days_since_big_loss'] = 0

        last_big_win = 0
        last_big_loss = 0

        for i in range(len(df)):
            if significant_win.iloc[i]:
                last_big_win = i
            if significant_loss.iloc[i]:
                last_big_loss = i

            df.iloc[i, df.columns.get_loc('days_since_big_win')] = i - last_big_win
            df.iloc[i, df.columns.get_loc('days_since_big_loss')] = i - last_big_loss

        return df

    def _create_day_trading_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create better target variables for day trading"""

        # Original targets (keep for compatibility)
        loss_threshold = df['daily_pnl'].quantile(0.1)
        df['target_large_loss'] = (df['daily_pnl'] < loss_threshold).astype(int)
        df['target_pnl'] = df['daily_pnl'].shift(-1)

        # NEW: Better targets for day trading

        # 1. Poor trading condition target (next day should reduce position size)
        # Combines high volatility + poor recent performance
        poor_conditions = (
            (df['volatility_regime'] == 2) |  # High volatility
            (df['sharpe_10d'] < -0.5) |       # Poor recent Sharpe
            (df['loss_streak'] <= -3) |       # 3+ consecutive losses
            (df['risk_budget_utilization'] > 1.5)  # Excessive risk taking
        )
        df['target_reduce_size'] = poor_conditions.shift(-1).fillna(0).astype(int)

        # 2. Favorable trading conditions (next day can increase position size)
        favorable_conditions = (
            (df['volatility_regime'] == 0) &  # Low volatility
            (df['sharpe_20d'] > 0.5) &        # Good recent performance
            (df['win_streak'] >= 2) &         # Winning streak
            (df['momentum_quality'] > 1.0)    # Strong momentum quality
        )
        df['target_increase_size'] = favorable_conditions.shift(-1).fillna(0).astype(int)

        # 3. Stop trading signal (next day should stop)
        stop_trading = (
            (df['loss_streak'] <= -5) |       # 5+ consecutive losses
            (df['risk_budget_utilization'] > 3.0) |  # Massive risk budget breach
            (df['current_drawdown'] < df['max_drawdown_20d'] * 1.5)  # New significant drawdown
        )
        df['target_stop_trading'] = stop_trading.shift(-1).fillna(0).astype(int)

        # 4. Next day volatility regime (for regime prediction)
        df['target_vol_regime'] = df['volatility_regime'].shift(-1).fillna(1).astype(int)

        # 5. Next day performance category (better than historical targets)
        # Instead of just "large loss", use performance relative to recent average
        recent_avg = df['daily_pnl'].rolling(20, min_periods=1).mean()
        recent_std = df['daily_pnl'].rolling(20, min_periods=1).std()

        next_day_pnl = df['daily_pnl'].shift(-1)
        df['target_underperform'] = (next_day_pnl < (recent_avg - 0.5 * recent_std)).fillna(0).astype(int)
        df['target_outperform'] = (next_day_pnl > (recent_avg + 0.5 * recent_std)).fillna(0).astype(int)

        return df

    def calculate_feature_importance_scores(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate feature importance for day trading signals"""

        feature_scores = {}

        # Core day trading features with importance scores
        day_trading_features = {
            'volatility_regime': 0.95,
            'position_size_multiplier': 0.90,
            'sharpe_10d': 0.85,
            'momentum_quality': 0.80,
            'risk_budget_utilization': 0.75,
            'vol_trend': 0.70,
            'win_rate_10d': 0.65,
            'loss_streak': 0.60,
            'trend_consistency_5d': 0.55,
            'var_5pct_10d': 0.50
        }

        # Check which features are available and calculate scores
        for feature, importance in day_trading_features.items():
            if feature in df.columns:
                # Adjust importance based on feature quality (non-null rate)
                non_null_rate = 1 - df[feature].isnull().mean()
                adjusted_importance = importance * non_null_rate
                feature_scores[feature] = adjusted_importance

        return feature_scores


def enhance_trader_data_with_day_trading_features(df: pd.DataFrame, config: Dict = None) -> pd.DataFrame:
    """
    Convenience function to enhance trader data with day trading features.

    Args:
        df: Basic trader data DataFrame
        config: Configuration dictionary

    Returns:
        Enhanced DataFrame with day trading features
    """
    engineer = DayTradingFeatureEngineer(config)
    return engineer.create_day_trading_features(df)
