# src/enhanced_feature_engineering.py

import pandas as pd
import numpy as np
from typing import Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_enhanced_features(panel_df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Build comprehensive feature set including traditional and fills-based features.

    Args:
        panel_df: Enhanced panel DataFrame from enhanced_data_processing
        config: Configuration dictionary

    Returns:
        pd.DataFrame: Panel with comprehensive engineered features
    """
    logger.info("Starting enhanced feature engineering with fills data...")

    # Create a copy to avoid modifying original
    df = panel_df.copy()

    # Sort by account_id and trade_date
    df = df.sort_values(['account_id', 'trade_date'])

    # Handle missing data from inactivity
    groupby_account = df.groupby('account_id')

    # Forward-fill these columns within each account
    fill_cols = ['daily_gross', 'daily_fees', 'gross_profit', 'gross_loss']
    for col in fill_cols:
        if col in df.columns:
            df[col] = groupby_account[col].ffill()

    # Fill daily_pnl and n_trades with 0 for inactive days
    df['daily_pnl'] = df['daily_pnl'].fillna(0)
    df['n_trades'] = df['n_trades'].fillna(0)
    df['daily_volume'] = df['daily_volume'].fillna(0)

    # Shift for lookahead prevention - all features based on lagged data
    logger.info("Applying time shifts to prevent lookahead bias...")

    # Generate base features (existing functionality)
    feature_params = config['feature_params']

    # Traditional EWMA features
    for span in feature_params['ewma_spans']:
        df[f'ewm_pnl_{span}'] = groupby_account['daily_pnl'].apply(
            lambda x: x.shift(1).ewm(span=span, adjust=False).mean()
        ).reset_index(level=0, drop=True)

        df[f'ewm_volume_{span}'] = groupby_account['daily_volume'].apply(
            lambda x: x.shift(1).ewm(span=span, adjust=False).mean()
        ).reset_index(level=0, drop=True)

    # Traditional rolling volatility features
    for window in feature_params['rolling_vol_windows']:
        df[f'rolling_vol_{window}'] = groupby_account['daily_pnl'].apply(
            lambda x: x.shift(1).rolling(window=window).std()
        ).reset_index(level=0, drop=True)

    # ENHANCED: Fills-based EWMA features
    logger.info("Calculating fills-based EWMA features...")

    fills_ewma_features = [
        'fills_count', 'fills_total_notional', 'fills_mean_fee_rate',
        'fills_adding_liquidity_rate', 'fills_removing_liquidity_rate',
        'price_impact_mean', 'trading_aggressiveness', 'liquidity_provision_score'
    ]

    for feature in fills_ewma_features:
        if feature in df.columns:
            for span in [5, 10, 21]:
                df[f'ewm_{feature}_{span}'] = groupby_account[feature].apply(
                    lambda x: x.shift(1).ewm(span=span, adjust=False).mean()
                ).reset_index(level=0, drop=True)

    # ENHANCED: Fills-based rolling features
    logger.info("Calculating fills-based rolling features...")

    fills_rolling_features = [
        'fills_count', 'avg_fill_size', 'cost_per_share',
        'effective_spread_proxy', 'orders_count', 'fills_per_order_actual'
    ]

    for feature in fills_rolling_features:
        if feature in df.columns:
            for window in [7, 14, 21]:
                # Mean
                df[f'rolling_{feature}_mean_{window}'] = groupby_account[feature].apply(
                    lambda x: x.shift(1).rolling(window=window).mean()
                ).reset_index(level=0, drop=True)

                # Standard deviation
                df[f'rolling_{feature}_std_{window}'] = groupby_account[feature].apply(
                    lambda x: x.shift(1).rolling(window=window).std()
                ).reset_index(level=0, drop=True)

    # ENHANCED: Advanced execution quality features
    logger.info("Calculating advanced execution quality features...")

    # Execution efficiency trends
    df['execution_efficiency_21d'] = groupby_account['cost_per_share'].apply(
        lambda x: calculate_efficiency_trend(x.shift(1), window=21)
    ).reset_index(level=0, drop=True)

    # Liquidity provision consistency
    df['liquidity_consistency_14d'] = groupby_account['liquidity_provision_score'].apply(
        lambda x: x.shift(1).rolling(window=14).std()
    ).reset_index(level=0, drop=True)

    # Market impact trend
    df['market_impact_trend_10d'] = groupby_account['price_impact_mean'].apply(
        lambda x: calculate_trend_slope(x.shift(1), window=10)
    ).reset_index(level=0, drop=True)

    # ENHANCED: Cross-feature interactions
    logger.info("Calculating cross-feature interactions...")

    # Volume vs fills efficiency
    df['volume_fills_efficiency'] = df['daily_volume'] / (df['fills_count'] + 1e-6)
    df['ewm_volume_fills_efficiency_10'] = groupby_account['volume_fills_efficiency'].apply(
        lambda x: x.shift(1).ewm(span=10, adjust=False).mean()
    ).reset_index(level=0, drop=True)

    # PnL per unit of market impact
    df['pnl_impact_ratio'] = df['daily_pnl'] / (df['price_impact_mean'] + 1e-6)
    df['rolling_pnl_impact_ratio_14'] = groupby_account['pnl_impact_ratio'].apply(
        lambda x: x.shift(1).rolling(window=14).mean()
    ).reset_index(level=0, drop=True)

    # Trading intensity vs performance
    df['intensity_performance_ratio'] = df['daily_pnl'] / (df['fills_count'] + 1e-6)
    df['ewm_intensity_performance_7'] = groupby_account['intensity_performance_ratio'].apply(
        lambda x: x.shift(1).ewm(span=7, adjust=False).mean()
    ).reset_index(level=0, drop=True)

    # Traditional advanced risk features (keeping existing)
    logger.info("Calculating traditional advanced risk features...")

    # Rolling Sortino ratio
    sortino_window = feature_params['sortino_window']
    df['rolling_sortino_ratio'] = groupby_account['daily_pnl'].apply(
        lambda x: calculate_sortino_ratio(x.shift(1), sortino_window)
    ).reset_index(level=0, drop=True)

    # Rolling profit factor
    profit_factor_window = feature_params['profit_factor_window']
    df['rolling_profit_factor'] = groupby_account.apply(
        lambda x: calculate_profit_factor(
            x['gross_profit'].shift(1),
            x['gross_loss'].shift(1),
            profit_factor_window
        )
    ).reset_index(level=0, drop=True)

    # Rolling max drawdown
    drawdown_window = feature_params['drawdown_window']
    df['rolling_max_drawdown'] = groupby_account['daily_pnl'].apply(
        lambda x: calculate_max_drawdown(x.shift(1), drawdown_window)
    ).reset_index(level=0, drop=True)

    # ENHANCED: Behavioral features with fills data
    logger.info("Calculating enhanced behavioral features...")

    # Calculate 21-day average trades and fills
    df['avg_trades_21d'] = groupby_account['n_trades'].apply(
        lambda x: x.shift(1).rolling(window=21).mean()
    ).reset_index(level=0, drop=True)

    df['avg_fills_21d'] = groupby_account['fills_count'].apply(
        lambda x: x.shift(1).rolling(window=21).mean()
    ).reset_index(level=0, drop=True)

    # Large loss flag (for revenge trading detection)
    large_loss_quantile = config['target_variable_params']['large_loss_quantile']
    df['daily_loss_flag'] = df['daily_pnl'] < 0

    # Calculate quantile threshold for large losses using ONLY historical data
    df['large_loss_threshold'] = groupby_account['daily_pnl'].apply(
        lambda x: calculate_expanding_quantile_threshold(x, large_loss_quantile)
    ).reset_index(level=0, drop=True)

    df['large_loss_recent'] = groupby_account.apply(
        lambda x: (x['daily_pnl'].shift(1) < x['large_loss_threshold']).rolling(
            window=feature_params['large_loss_lookback']
        ).max()
    ).reset_index(level=0, drop=True)

    # ENHANCED: Revenge trading proxy with fills data
    df['revenge_trading_proxy'] = (
        (df['n_trades'] > 1.5 * df['avg_trades_21d']) &
        (df['fills_count'] > 1.5 * df['avg_fills_21d']) &
        (df['large_loss_recent'] == 1)
    ).astype(int)

    # ENHANCED: Execution stress indicators (only if columns exist)
    stress_conditions = []

    if 'price_impact_mean' in df.columns and 'rolling_price_impact_mean_mean_21' in df.columns:
        stress_conditions.append(df['price_impact_mean'] > df['rolling_price_impact_mean_mean_21'])

    if 'cost_per_share' in df.columns and 'rolling_cost_per_share_mean_21' in df.columns:
        stress_conditions.append(df['cost_per_share'] > df['rolling_cost_per_share_mean_21'])

    if 'trading_aggressiveness' in df.columns and 'ewm_trading_aggressiveness_10' in df.columns:
        stress_conditions.append(df['trading_aggressiveness'] > df['ewm_trading_aggressiveness_10'])

    if stress_conditions:
        df['execution_stress_indicator'] = pd.concat(stress_conditions, axis=1).all(axis=1).astype(int)
    else:
        df['execution_stress_indicator'] = 0

    # Generate target variables with strict temporal validation
    logger.info("Creating target variables with leakage prevention...")

    # Target PnL (next day's PnL) - for validation and analysis
    df['target_pnl'] = groupby_account['daily_pnl'].shift(-1)

    # ENHANCED: Primary target - OPTIMAL POSITION SIZE (0% to 150%)
    logger.info("Creating position sizing targets for dynamic risk management...")
    logger.info(f"  Input DataFrame shape: {df.shape}")
    logger.info(f"  Config keys: {list(config.keys())}")

    try:
        try:
            from .position_sizing_targets import create_position_sizing_features
        except ImportError:
            from position_sizing_targets import create_position_sizing_features

        # Create position sizing targets (main prediction target)
        df_before = df.copy()
        df = create_position_sizing_features(df, config)

        logger.info(f"  Output DataFrame shape: {df.shape}")
        pos_cols = [col for col in df.columns if 'position' in col]
        logger.info(f"  Position columns created: {pos_cols}")

        if 'target_position_size' in df.columns:
            valid_pos = df['target_position_size'].dropna()
            logger.info(f"  ✓ Position sizing targets: {len(valid_pos)} valid values")
            if len(valid_pos) > 0:
                logger.info(f"    Range: {valid_pos.min():.3f} to {valid_pos.max():.3f}")
                logger.info(f"    Mean: {valid_pos.mean():.3f}")
        else:
            logger.error("  ✗ target_position_size column was NOT created!")

    except Exception as e:
        logger.error(f"  ✗ Error creating position sizing targets: {e}")
        import traceback
        logger.error(traceback.format_exc())

    # Legacy targets (for comparison and analysis)
    df['target_large_loss'] = groupby_account.apply(
        lambda x: (x['daily_pnl'].shift(-1) < x['large_loss_threshold']).astype(int)
    ).reset_index(level=0, drop=True)

    # ENHANCED: Additional target variables for fills-based modeling
    if 'price_impact_mean' in df.columns:
        df['target_high_impact'] = groupby_account['price_impact_mean'].apply(
            lambda x: (x.shift(-1) > x.shift(1).rolling(window=21).quantile(0.8)).astype(int)
        ).reset_index(level=0, drop=True)
    else:
        df['target_high_impact'] = 0

    if 'cost_per_share' in df.columns:
        df['target_high_cost'] = groupby_account['cost_per_share'].apply(
            lambda x: (x.shift(-1) > x.shift(1).rolling(window=21).quantile(0.8)).astype(int)
        ).reset_index(level=0, drop=True)
    else:
        df['target_high_cost'] = 0

    # Validation: Ensure no feature can predict current day's PnL
    logger.info("Performing temporal leakage validation...")

    # Check each feature's correlation with same-day PnL
    feature_cols_to_check = [col for col in df.columns if any(col.startswith(prefix) for prefix in
        ['ewm_', 'rolling_', 'avg_', 'win_rate', 'cumulative', 'execution_', 'liquidity_', 'market_impact'])]

    leakage_warnings = []
    for feature in feature_cols_to_check:
        if feature in df.columns and df[feature].dtype in ['int64', 'float64']:
            same_day_corr = df[feature].corr(df['daily_pnl'])
            if abs(same_day_corr) > 0.7:  # High correlation suggests potential leakage
                leakage_warnings.append(f"Feature {feature} has high same-day correlation: {same_day_corr:.3f}")

    if leakage_warnings:
        logger.warning("Potential temporal leakage detected:")
        for warning in leakage_warnings[:5]:  # Show first 5
            logger.warning(f"  {warning}")
    else:
        logger.info("Temporal leakage validation passed - no high correlations found")

    # Additional traditional features (keeping existing functionality)
    df['cumulative_pnl'] = groupby_account['daily_pnl'].apply(
        lambda x: x.shift(1).cumsum()
    ).reset_index(level=0, drop=True)

    # Win rate
    df['win_rate_21d'] = groupby_account['daily_pnl'].apply(
        lambda x: (x.shift(1) > 0).rolling(window=21).mean()
    ).reset_index(level=0, drop=True)

    # Average win/loss
    df['avg_win_21d'] = groupby_account['daily_pnl'].apply(
        lambda x: x.shift(1).where(x.shift(1) > 0).rolling(window=21).mean()
    ).reset_index(level=0, drop=True)

    df['avg_loss_21d'] = groupby_account['daily_pnl'].apply(
        lambda x: x.shift(1).where(x.shift(1) < 0).rolling(window=21).mean()
    ).reset_index(level=0, drop=True)

    # Drop initial rows with NaN values from rolling calculations
    max_window = max([21, sortino_window, profit_factor_window, drawdown_window] +
                    feature_params['rolling_vol_windows'] + [21])  # Include fills windows

    df = df.groupby('account_id').apply(
        lambda x: x.iloc[max_window:]
    ).reset_index(drop=True)

    # Drop any remaining NaN target values
    df = df.dropna(subset=['target_pnl', 'target_large_loss'])

    # Comprehensive data quality validation
    logger.info("Performing final enhanced data quality validation...")

    validation_results = validate_enhanced_feature_data_quality(df, config)

    # Save processed features with validation metadata
    logger.info(f"Saving enhanced features to {config['paths']['processed_features']}")

    # Try to save as parquet first, fall back to pickle if needed
    try:
        df.to_parquet(config['paths']['processed_features'], index=False)
        logger.info(f"Saved enhanced features as parquet: {config['paths']['processed_features']}")
    except Exception as e:
        logger.warning(f"Parquet save failed ({str(e)}), using pickle instead")
        import pickle
        pickle_path = config['paths']['processed_features'].replace('.parquet', '_enhanced.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(df, f)
        logger.info(f"Saved enhanced features to {pickle_path}")

    # Save validation results
    validation_path = config['paths']['processed_features'].replace('.parquet', '_enhanced_validation.json').replace('.pkl', '_enhanced_validation.json')
    import json
    with open(validation_path, 'w') as f:
        json.dump(validation_results, f, indent=2, default=str)
    logger.info(f"Enhanced validation results saved to {validation_path}")

    logger.info(f"Enhanced feature engineering complete. Final shape: {df.shape}")
    logger.info(f"Traditional features: {len([col for col in df.columns if any(col.startswith(prefix) for prefix in ['ewm_', 'rolling_', 'avg_', 'win_rate', 'cumulative']) and not any(col.startswith(fills_prefix) for fills_prefix in ['ewm_fills_', 'rolling_fills_'])])}")
    logger.info(f"Fills-based features: {len([col for col in df.columns if 'fills' in col or 'order' in col or col in ['price_impact_mean', 'trading_aggressiveness', 'liquidity_provision_score']])}")
    logger.info(f"Cross-feature interactions: {len([col for col in df.columns if any(suffix in col for suffix in ['_efficiency', '_ratio', '_trend', '_consistency', '_stress'])])}")
    logger.info(f"Data quality issues found: {validation_results['n_issues']}")

    return df


def validate_enhanced_feature_data_quality(df: pd.DataFrame, config: Dict) -> Dict:
    """
    Comprehensive validation of enhanced feature data quality including fills features.
    """
    logger.info("Validating enhanced feature data quality...")

    validation_results = {
        'validation_timestamp': pd.Timestamp.now().isoformat(),
        'issues': [],
        'warnings': [],
        'data_summary': {},
        'feature_statistics': {}
    }

    # Enhanced data summary
    validation_results['data_summary'] = {
        'n_rows': len(df),
        'n_traders': df['account_id'].nunique(),
        'date_range': [df['trade_date'].min().isoformat(), df['trade_date'].max().isoformat()],

        # Feature counts by type
        'traditional_features': len([col for col in df.columns if any(col.startswith(prefix) for prefix in
            ['ewm_', 'rolling_', 'avg_', 'win_rate', 'cumulative']) and not 'fills' in col]),
        'fills_features': len([col for col in df.columns if 'fills' in col or 'order' in col]),
        'execution_features': len([col for col in df.columns if any(keyword in col for keyword in
            ['impact', 'liquidity', 'aggressiveness', 'efficiency'])]),
        'cross_features': len([col for col in df.columns if any(suffix in col for suffix in
            ['_ratio', '_trend', '_consistency', '_stress'])]),

        # Target variables
        'target_pnl_range': [float(df['target_pnl'].min()), float(df['target_pnl'].max())],
        'large_loss_rate': float(df['target_large_loss'].mean()),
        'high_impact_rate': float(df['target_high_impact'].mean()) if 'target_high_impact' in df.columns else None,
        'high_cost_rate': float(df['target_high_cost'].mean()) if 'target_high_cost' in df.columns else None
    }

    # Check fills data coverage
    fills_coverage = (df['fills_count'] > 0).mean()
    validation_results['data_summary']['fills_data_coverage'] = fills_coverage

    if fills_coverage < 0.8:
        validation_results['warnings'].append(f"Low fills data coverage: {fills_coverage:.1%} of days have fills data")

    # Check missing data patterns
    missing_data = df.isnull().sum()
    high_missing = missing_data[missing_data > len(df) * 0.05]

    if len(high_missing) > 0:
        validation_results['warnings'].append(f"Features with >5% missing data: {high_missing.to_dict()}")

    # Enhanced feature stability checks
    feature_categories = {
        'traditional': [col for col in df.columns if any(col.startswith(prefix) for prefix in
            ['ewm_', 'rolling_']) and not 'fills' in col],
        'fills_based': [col for col in df.columns if 'fills' in col or 'order' in col],
        'execution_quality': [col for col in df.columns if any(keyword in col for keyword in
            ['impact', 'liquidity', 'aggressiveness', 'efficiency'])]
    }

    for category, features in feature_categories.items():
        category_warnings = []
        for feature in features[:5]:  # Check first 5 in each category
            if feature in df.columns and df[feature].dtype in ['int64', 'float64']:
                trader_means = df.groupby('account_id')[feature].mean()
                if trader_means.std() > 0:
                    mean_ratio = trader_means.max() / (trader_means.min() + 1e-6)
                    if mean_ratio > 100:
                        category_warnings.append(f"{feature}: ratio {mean_ratio:.1f}")

        if category_warnings:
            validation_results['warnings'].append(f"{category} features with extreme trader differences: {category_warnings[:2]}")

    # Validate enhanced target variables
    target_issues = []

    # Check all target variables
    for target in ['target_pnl', 'target_large_loss', 'target_high_impact', 'target_high_cost']:
        if target in df.columns:
            if df[target].isna().all():
                target_issues.append(f"{target} is all NaN")
            elif target.endswith('_loss') or target.endswith('_impact') or target.endswith('_cost'):
                # Binary target checks
                rate = df[target].mean()
                if rate < 0.05 or rate > 0.25:
                    target_issues.append(f"Unusual {target} rate: {rate:.1%}")

    validation_results['issues'].extend(target_issues)

    # Feature correlation analysis
    correlation_warnings = []
    execution_features = [col for col in df.columns if any(keyword in col for keyword in
        ['impact', 'cost', 'efficiency', 'liquidity'])]

    if len(execution_features) >= 2:
        # Check for very high correlations between execution features
        for i, feat1 in enumerate(execution_features[:5]):
            for feat2 in execution_features[i+1:6]:
                if feat1 in df.columns and feat2 in df.columns:
                    corr = df[feat1].corr(df[feat2])
                    if abs(corr) > 0.95:
                        correlation_warnings.append(f"{feat1} - {feat2}: {corr:.3f}")

    if correlation_warnings:
        validation_results['warnings'].append(f"Very high correlations between features: {correlation_warnings[:3]}")

    validation_results['n_issues'] = len(validation_results['issues'])
    validation_results['n_warnings'] = len(validation_results['warnings'])

    # Log enhanced validation results
    if validation_results['issues']:
        logger.warning(f"Found {validation_results['n_issues']} enhanced data quality issues")
        for issue in validation_results['issues'][:5]:
            logger.warning(f"  ISSUE: {issue}")

    if validation_results['warnings']:
        logger.info(f"Found {validation_results['n_warnings']} enhanced data quality warnings")
        for warning in validation_results['warnings'][:3]:
            logger.info(f"  WARNING: {warning}")

    if not validation_results['issues'] and not validation_results['warnings']:
        logger.info("All enhanced data quality checks passed successfully")

    return validation_results


# Helper functions for enhanced features
def calculate_efficiency_trend(cost_series: pd.Series, window: int) -> pd.Series:
    """Calculate trend in execution cost efficiency."""
    def trend_slope(x):
        if len(x) < window:
            return np.nan
        indices = np.arange(len(x))
        # Simple linear regression slope
        if np.std(indices) == 0:
            return 0
        return np.corrcoef(indices, x)[0, 1] * (np.std(x) / np.std(indices))

    return cost_series.rolling(window=window).apply(trend_slope)


def calculate_trend_slope(series: pd.Series, window: int) -> pd.Series:
    """Calculate rolling trend slope."""
    def trend_slope(x):
        if len(x) < window or x.isna().all():
            return np.nan
        indices = np.arange(len(x))
        valid_mask = ~x.isna()
        if valid_mask.sum() < 3:
            return np.nan
        x_clean = x[valid_mask]
        indices_clean = indices[valid_mask]
        if np.std(indices_clean) == 0 or np.std(x_clean) == 0:
            return 0
        return np.corrcoef(indices_clean, x_clean)[0, 1] * (np.std(x_clean) / np.std(indices_clean))

    return series.rolling(window=window).apply(trend_slope)


# Keep existing helper functions from original feature_engineering.py
def calculate_expanding_quantile_threshold(returns: pd.Series, quantile: float) -> pd.Series:
    """Calculate expanding window quantile threshold to prevent future leakage."""
    threshold_series = pd.Series(index=returns.index, dtype=float)

    for i in range(len(returns)):
        historical_data = returns.iloc[:i]
        negative_returns = historical_data[historical_data < 0]

        if len(negative_returns) >= 10:
            threshold_series.iloc[i] = negative_returns.quantile(quantile)
        else:
            threshold_series.iloc[i] = -1e6

    return threshold_series


def calculate_sortino_ratio(returns: pd.Series, window: int) -> pd.Series:
    """Calculate rolling Sortino ratio."""
    def sortino(x):
        if len(x) < window:
            return np.nan
        mean_return = x.mean()
        downside_returns = x[x < 0]
        if len(downside_returns) == 0:
            return np.inf if mean_return > 0 else 0
        downside_std = downside_returns.std()
        if downside_std == 0:
            return np.inf if mean_return > 0 else 0
        return mean_return / downside_std * np.sqrt(252)

    return returns.rolling(window=window).apply(sortino)


def calculate_profit_factor(gross_profit: pd.Series, gross_loss: pd.Series, window: int) -> pd.Series:
    """Calculate rolling profit factor."""
    rolling_profit = gross_profit.rolling(window=window).sum()
    rolling_loss = gross_loss.rolling(window=window).sum()

    profit_factor = rolling_profit / rolling_loss.replace(0, np.nan)
    profit_factor = profit_factor.fillna(np.inf)

    return profit_factor


def calculate_max_drawdown(returns: pd.Series, window: int) -> pd.Series:
    """Calculate rolling maximum drawdown."""
    def max_dd(x):
        if len(x) < window:
            return np.nan
        cumsum = x.cumsum()
        running_max = cumsum.expanding().max()
        drawdown = cumsum - running_max
        return drawdown.min()

    return returns.rolling(window=window).apply(max_dd)
