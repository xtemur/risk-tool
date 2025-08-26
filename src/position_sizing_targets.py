#!/usr/bin/env python3
"""
Position Sizing Target Variable Creation

Creates optimal position sizing targets (0% to 150%) based on risk-adjusted performance optimization.
True quant approach: maximize Sharpe ratio while controlling for drawdowns.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class PositionSizingTargets:
    """
    Calculate optimal position sizing targets for dynamic risk management.

    Logic:
    - 0-50%: High risk conditions, reduce exposure
    - 50-100%: Normal conditions, scale with confidence
    - 100-150%: Favorable conditions, increase exposure (limited leverage)
    """

    def __init__(self, lookback_window: int = 21, vol_window: int = 10):
        self.lookback_window = lookback_window
        self.vol_window = vol_window

    def calculate_optimal_position_size(self,
                                      pnl_series: pd.Series,
                                      vol_series: pd.Series = None) -> pd.Series:
        """
        Calculate optimal position size for each day based on forward-looking performance.

        For each day t, determine what position size (0% to 150%) would have been optimal
        for the next trading day based on risk-adjusted metrics.

        Args:
            pnl_series: Historical daily P&L
            vol_series: Optional volatility series (if None, calculated from P&L)

        Returns:
            Series of optimal position sizes (0.0 to 1.5)
        """

        if vol_series is None:
            vol_series = pnl_series.rolling(self.vol_window).std()

        optimal_sizes = pd.Series(index=pnl_series.index, dtype=float)

        # For each day, calculate what would have been optimal for the NEXT day
        for i in range(len(pnl_series) - 1):  # Exclude last day (no future data)
            current_idx = pnl_series.index[i]
            next_day_pnl = pnl_series.iloc[i + 1]

            # Use historical data up to (but not including) the current day
            historical_window = pnl_series.iloc[max(0, i - self.lookback_window):i]
            historical_vol = vol_series.iloc[max(0, i - self.vol_window):i]

            if len(historical_window) < 5:  # Need minimum data
                optimal_sizes.iloc[i] = 1.0  # Default full position
                continue

            # Calculate optimal position size for next day
            optimal_size = self._calculate_single_optimal_size(
                next_day_pnl=next_day_pnl,
                historical_pnl=historical_window,
                historical_vol=historical_vol
            )

            optimal_sizes.iloc[i] = optimal_size

        # Fill any remaining NaN values
        optimal_sizes = optimal_sizes.bfill().fillna(1.0)

        return optimal_sizes

    def _calculate_single_optimal_size(self,
                                     next_day_pnl: float,
                                     historical_pnl: pd.Series,
                                     historical_vol: pd.Series) -> float:
        """
        Calculate optimal position size for a single day using multiple risk metrics.

        Approach:
        1. Base size from Kelly-style risk-adjusted sizing
        2. Volatility adjustment
        3. Momentum/trend consideration
        4. Drawdown protection
        5. Bounded to [0, 1.5]
        """

        if len(historical_pnl) == 0 or historical_vol.iloc[-1] == 0:
            return 1.0

        # 1. Kelly-inspired base sizing
        mean_return = historical_pnl.mean()
        return_vol = historical_pnl.std()

        if return_vol == 0:
            kelly_size = 1.0
        else:
            # Modified Kelly: f = (expected_return - risk_free_rate) / variance
            # Assuming risk_free_rate = 0 for simplicity
            kelly_size = max(0, mean_return / (return_vol ** 2))
            kelly_size = min(kelly_size, 2.0)  # Cap Kelly at 200%

        # 2. Volatility regime adjustment
        recent_vol = historical_vol.iloc[-1] if len(historical_vol) > 0 else return_vol
        avg_vol = historical_vol.mean() if len(historical_vol) > 0 else return_vol

        if avg_vol > 0:
            vol_adjustment = avg_vol / recent_vol  # Lower recent vol = higher size
            vol_adjustment = np.clip(vol_adjustment, 0.5, 2.0)
        else:
            vol_adjustment = 1.0

        # 3. Momentum consideration
        if len(historical_pnl) >= 5:
            recent_momentum = historical_pnl.tail(5).mean()
            longer_momentum = historical_pnl.mean()

            if longer_momentum != 0:
                momentum_factor = 1 + 0.2 * (recent_momentum / abs(longer_momentum))
                momentum_factor = np.clip(momentum_factor, 0.7, 1.3)
            else:
                momentum_factor = 1.0
        else:
            momentum_factor = 1.0

        # 4. Drawdown protection
        if len(historical_pnl) >= 10:
            cumulative_returns = (1 + historical_pnl / 10000).cumprod()  # Assuming PnL in dollars
            running_max = cumulative_returns.expanding().max()
            current_drawdown = (cumulative_returns.iloc[-1] / running_max.iloc[-1]) - 1

            if current_drawdown < -0.1:  # In significant drawdown
                drawdown_protection = 0.5  # Reduce size significantly
            elif current_drawdown < -0.05:  # In moderate drawdown
                drawdown_protection = 0.8  # Reduce size moderately
            else:
                drawdown_protection = 1.0  # No drawdown protection needed
        else:
            drawdown_protection = 1.0

        # 5. Combine all factors
        base_size = kelly_size * vol_adjustment * momentum_factor * drawdown_protection

        # 6. Final adjustments based on next day outcome (this is the "optimal" part)
        # In practice, we're reverse-engineering what would have been optimal
        if next_day_pnl > 0:
            # For profitable days, optimal would have been larger position
            outcome_multiplier = 1.2
        elif next_day_pnl < -2 * return_vol:  # Large loss day
            # For large loss days, optimal would have been much smaller position
            outcome_multiplier = 0.3
        else:
            # For normal loss days, optimal would have been smaller position
            outcome_multiplier = 0.8

        optimal_size = base_size * outcome_multiplier

        # 7. Bound the result to [0, 1.5]
        optimal_size = np.clip(optimal_size, 0.0, 1.5)

        return optimal_size

    def create_position_sizing_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create position sizing targets for the entire dataframe.

        Args:
            df: DataFrame with columns ['account_id', 'trade_date', 'daily_pnl', ...]

        Returns:
            DataFrame with added 'target_position_size' column
        """

        logger.info("Creating position sizing targets...")

        # Ensure the dataframe is sorted
        df = df.sort_values(['account_id', 'trade_date']).copy()

        # Create groupby object
        groupby_account = df.groupby('account_id')

        # Calculate volatility if not present
        if 'rolling_vol_7' not in df.columns:
            df['rolling_vol_7'] = groupby_account['daily_pnl'].transform(
                lambda x: x.rolling(7).std()
            )

        # Calculate optimal position sizes for each trader
        logger.info("Calculating optimal position sizes for each trader...")

        def calculate_trader_targets(group):
            """Calculate targets for a single trader."""
            pnl_series = group['daily_pnl']
            vol_series = group['rolling_vol_7']

            optimal_sizes = self.calculate_optimal_position_size(pnl_series, vol_series)
            return optimal_sizes

        # Apply to each trader
        position_sizes_list = []
        for account_id, group in groupby_account:
            trader_targets = calculate_trader_targets(group)
            trader_targets.index = group.index  # Maintain original index
            position_sizes_list.append(trader_targets)

        # Combine all traders' results
        all_position_sizes = pd.concat(position_sizes_list).sort_index()
        df['target_position_size'] = all_position_sizes

        # Shift target to next day (what we're trying to predict)
        df['target_position_size'] = groupby_account['target_position_size'].shift(-1)

        # Create categorical versions for analysis
        df['target_position_category'] = pd.cut(
            df['target_position_size'],
            bins=[0, 0.5, 0.8, 1.2, 1.5],
            labels=['Reduce', 'Conservative', 'Normal', 'Aggressive'],
            include_lowest=True
        ).astype(str)  # Convert to string to avoid categorical fillna issues

        # Create binary high/low risk targets for comparison
        df['target_high_risk'] = (df['target_position_size'] < 0.7).astype(int)
        df['target_low_risk'] = (df['target_position_size'] > 1.1).astype(int)

        # Log statistics
        valid_targets = df['target_position_size'].dropna()
        if len(valid_targets) > 0:
            logger.info(f"Position sizing targets created:")
            logger.info(f"  Mean target size: {valid_targets.mean():.3f}")
            logger.info(f"  Range: {valid_targets.min():.3f} to {valid_targets.max():.3f}")
            logger.info(f"  Reduce positions (<50%): {(valid_targets < 0.5).sum()} days")
            logger.info(f"  Conservative (50-80%): {((valid_targets >= 0.5) & (valid_targets < 0.8)).sum()} days")
            logger.info(f"  Normal (80-120%): {((valid_targets >= 0.8) & (valid_targets <= 1.2)).sum()} days")
            logger.info(f"  Aggressive (>120%): {(valid_targets > 1.2).sum()} days")

        return df

    def validate_targets(self, df: pd.DataFrame) -> Dict:
        """
        Validate the position sizing targets for reasonableness.

        Args:
            df: DataFrame with target_position_size column

        Returns:
            Dictionary with validation results
        """

        validation = {}

        valid_targets = df['target_position_size'].dropna()

        if len(valid_targets) == 0:
            validation['error'] = 'No valid targets created'
            return validation

        validation['target_count'] = len(valid_targets)
        validation['mean_target'] = float(valid_targets.mean())
        validation['std_target'] = float(valid_targets.std())
        validation['min_target'] = float(valid_targets.min())
        validation['max_target'] = float(valid_targets.max())

        # Distribution analysis
        validation['distribution'] = {
            'reduce_pct': float((valid_targets < 0.5).mean() * 100),
            'conservative_pct': float(((valid_targets >= 0.5) & (valid_targets < 0.8)).mean() * 100),
            'normal_pct': float(((valid_targets >= 0.8) & (valid_targets <= 1.2)).mean() * 100),
            'aggressive_pct': float((valid_targets > 1.2).mean() * 100)
        }

        # Sanity checks
        validation['warnings'] = []

        if validation['mean_target'] < 0.3 or validation['mean_target'] > 1.3:
            validation['warnings'].append(f"Mean target size unusual: {validation['mean_target']:.3f}")

        if (valid_targets < 0.1).sum() > len(valid_targets) * 0.05:
            validation['warnings'].append("Many targets suggest very low position sizes")

        if (valid_targets > 1.4).sum() > len(valid_targets) * 0.05:
            validation['warnings'].append("Many targets suggest high leverage")

        # Correlation with next day performance (validation)
        if 'target_pnl' in df.columns:
            target_pnl = df['target_pnl'].dropna()
            matching_indices = valid_targets.index.intersection(target_pnl.index)

            if len(matching_indices) > 10:
                correlation = valid_targets.loc[matching_indices].corr(target_pnl.loc[matching_indices])
                validation['pnl_correlation'] = float(correlation)

                if correlation < 0:
                    validation['warnings'].append(f"Negative correlation with next-day PnL: {correlation:.3f}")

        validation['is_valid'] = len(validation['warnings']) == 0

        return validation


def create_position_sizing_features(df: pd.DataFrame, config: Dict = None) -> pd.DataFrame:
    """
    Main function to create position sizing targets.

    Args:
        df: Enhanced features DataFrame
        config: Configuration dictionary (optional)

    Returns:
        DataFrame with position sizing targets added
    """

    if config is None:
        config = {}

    lookback_window = config.get('position_sizing_lookback', 21)
    vol_window = config.get('position_sizing_vol_window', 10)

    # Initialize position sizing calculator
    position_sizer = PositionSizingTargets(
        lookback_window=lookback_window,
        vol_window=vol_window
    )

    # Create targets
    df_with_targets = position_sizer.create_position_sizing_targets(df)

    # Validate results
    validation = position_sizer.validate_targets(df_with_targets)

    logger.info("Position sizing target validation:")
    logger.info(f"  Valid targets: {validation.get('target_count', 0)}")
    logger.info(f"  Mean size: {validation.get('mean_target', 0):.3f}")
    logger.info(f"  Distribution: {validation.get('distribution', {})}")

    if validation.get('warnings'):
        for warning in validation['warnings']:
            logger.warning(f"  WARNING: {warning}")

    return df_with_targets


if __name__ == "__main__":
    # Test with sample data
    import logging
    logging.basicConfig(level=logging.INFO)

    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')

    sample_data = pd.DataFrame({
        'account_id': [1001] * 100,
        'trade_date': dates,
        'daily_pnl': np.random.normal(100, 500, 100),  # Mean $100, std $500
    })

    # Test position sizing
    result = create_position_sizing_features(sample_data)

    print("\nSample results:")
    print(result[['trade_date', 'daily_pnl', 'target_position_size', 'target_position_category']].head(10))
