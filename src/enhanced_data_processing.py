# src/enhanced_data_processing.py

import pandas as pd
import numpy as np
import sqlite3
from typing import Dict, List, Tuple
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_fills_features(config: Dict) -> pd.DataFrame:
    """
    Extract and engineer features from the fills table.

    Args:
        config: Configuration dictionary containing paths and parameters

    Returns:
        pd.DataFrame: Daily fills-based features for each account
    """
    logger.info("Creating fills-based features...")

    con = sqlite3.connect(config['paths']['raw_data'])

    # Get active traders
    active_traders = config['active_traders']
    placeholders = ','.join('?' * len(active_traders))

    # Load fills data
    fills_query = f"""
    SELECT
        account_id,
        fill_datetime,
        side,
        qty,
        symbol,
        price,
        route,
        liq,
        comm,
        ecn_fee,
        sec,
        taf,
        nscc,
        clr,
        misc,
        order_id,
        fill_id
    FROM fills
    WHERE account_id IN ({placeholders})
    ORDER BY account_id, fill_datetime
    """

    fills_df = pd.read_sql_query(fills_query, con, params=active_traders)
    con.close()

    logger.info(f"Loaded {len(fills_df)} fills for feature engineering")

    # Parse datetime with mixed format handling
    fills_df['fill_datetime'] = pd.to_datetime(fills_df['fill_datetime'], format='mixed', dayfirst=False)
    fills_df['trade_date'] = fills_df['fill_datetime'].dt.date
    fills_df['trade_date'] = pd.to_datetime(fills_df['trade_date'])

    # Calculate fill-level metrics
    fills_df['notional_value'] = fills_df['qty'] * fills_df['price']
    fills_df['total_fees'] = fills_df[['comm', 'ecn_fee', 'sec', 'taf', 'nscc', 'clr', 'misc']].sum(axis=1)
    fills_df['fee_rate'] = fills_df['total_fees'] / (fills_df['notional_value'] + 1e-6)

    # Liquidity indicators
    fills_df['is_liquidity_adding'] = fills_df['liq'].isin(['A', 'ASB'])  # Adding liquidity
    fills_df['is_liquidity_removing'] = fills_df['liq'].isin(['R', '7'])  # Removing liquidity

    # Side indicators
    fills_df['is_buy'] = fills_df['side'] == 'B'
    fills_df['is_sell'] = fills_df['side'] == 'S'

    # Time-based features
    fills_df['hour'] = fills_df['fill_datetime'].dt.hour
    fills_df['minute'] = fills_df['fill_datetime'].dt.minute
    fills_df['is_market_open'] = (fills_df['hour'] >= 9) & (fills_df['hour'] < 16)
    fills_df['is_first_hour'] = (fills_df['hour'] == 9) & (fills_df['minute'] < 60)
    fills_df['is_last_hour'] = (fills_df['hour'] == 15)

    # Order fragmentation analysis
    order_stats = fills_df.groupby(['account_id', 'trade_date', 'order_id']).agg({
        'qty': ['count', 'sum'],
        'fill_datetime': ['min', 'max'],
        'price': ['min', 'max', 'mean', 'std']
    }).reset_index()

    # Flatten column names
    order_stats.columns = [
        'account_id', 'trade_date', 'order_id',
        'fills_per_order', 'total_qty',
        'first_fill_time', 'last_fill_time',
        'min_price', 'max_price', 'avg_price', 'price_std'
    ]

    # Calculate order duration
    order_stats['order_duration_seconds'] = (
        order_stats['last_fill_time'] - order_stats['first_fill_time']
    ).dt.total_seconds()

    # Price impact metrics
    order_stats['price_range'] = order_stats['max_price'] - order_stats['min_price']
    order_stats['price_impact'] = order_stats['price_range'] / (order_stats['avg_price'] + 1e-6)

    # Aggregate fills to daily level
    daily_fills = fills_df.groupby(['account_id', 'trade_date']).agg({
        # Volume and count metrics
        'qty': ['count', 'sum', 'mean', 'std'],
        'notional_value': ['sum', 'mean'],
        'price': ['mean', 'std', 'min', 'max'],

        # Cost metrics
        'total_fees': 'sum',
        'fee_rate': 'mean',
        'comm': 'sum',
        'ecn_fee': 'sum',

        # Liquidity metrics
        'is_liquidity_adding': ['sum', 'mean'],
        'is_liquidity_removing': ['sum', 'mean'],

        # Timing metrics
        'is_market_open': 'mean',
        'is_first_hour': 'mean',
        'is_last_hour': 'mean',
        'hour': ['min', 'max', 'mean'],

        # Side metrics
        'is_buy': ['sum', 'mean'],
        'is_sell': ['sum', 'mean'],

        # Venue metrics
        'route': 'nunique',
        'symbol': 'nunique'
    }).reset_index()

    # Flatten column names
    daily_fills.columns = [
        'account_id', 'trade_date',
        'fills_count', 'fills_total_qty', 'fills_mean_qty', 'fills_std_qty',
        'fills_total_notional', 'fills_mean_notional',
        'fills_mean_price', 'fills_std_price', 'fills_min_price', 'fills_max_price',
        'fills_total_fees', 'fills_mean_fee_rate', 'fills_total_comm', 'fills_total_ecn',
        'fills_adding_liquidity_count', 'fills_adding_liquidity_rate',
        'fills_removing_liquidity_count', 'fills_removing_liquidity_rate',
        'fills_market_open_rate', 'fills_first_hour_rate', 'fills_last_hour_rate',
        'fills_min_hour', 'fills_max_hour', 'fills_mean_hour',
        'fills_buy_count', 'fills_buy_rate',
        'fills_sell_count', 'fills_sell_rate',
        'fills_routes_used', 'fills_symbols_traded'
    ]

    # Order-level daily aggregation
    daily_orders = order_stats.groupby(['account_id', 'trade_date']).agg({
        'order_id': 'count',
        'fills_per_order': ['mean', 'max', 'std'],
        'total_qty': ['mean', 'std'],
        'order_duration_seconds': ['mean', 'max', 'std'],
        'price_impact': ['mean', 'max', 'std'],
        'price_range': ['mean', 'sum']
    }).reset_index()

    # Flatten column names
    daily_orders.columns = [
        'account_id', 'trade_date',
        'orders_count',
        'fills_per_order_mean', 'fills_per_order_max', 'fills_per_order_std',
        'order_qty_mean', 'order_qty_std',
        'order_duration_mean', 'order_duration_max', 'order_duration_std',
        'price_impact_mean', 'price_impact_max', 'price_impact_std',
        'price_range_mean', 'total_price_range'
    ]

    # Merge daily fills and orders data
    daily_features = pd.merge(daily_fills, daily_orders, on=['account_id', 'trade_date'], how='outer')

    # Calculate derived metrics
    daily_features['fills_per_order_actual'] = (
        daily_features['fills_count'] / (daily_features['orders_count'] + 1e-6)
    )

    daily_features['avg_fill_size'] = (
        daily_features['fills_total_qty'] / (daily_features['fills_count'] + 1e-6)
    )

    daily_features['cost_per_share'] = (
        daily_features['fills_total_fees'] / (daily_features['fills_total_qty'] + 1e-6)
    )

    daily_features['effective_spread_proxy'] = (
        daily_features['total_price_range'] / (daily_features['fills_mean_price'] + 1e-6)
    )

    # Liquidity provision score
    daily_features['liquidity_provision_score'] = (
        daily_features['fills_adding_liquidity_rate'] -
        daily_features['fills_removing_liquidity_rate']
    )

    # Trading aggressiveness metrics
    daily_features['trading_aggressiveness'] = (
        daily_features['fills_removing_liquidity_rate'] *
        daily_features['price_impact_mean']
    )

    # Fill NaN values with 0 for days with no fills
    daily_features = daily_features.fillna(0)

    logger.info(f"Created fills features with shape: {daily_features.shape}")
    logger.info(f"Fills features: {len([col for col in daily_features.columns if col not in ['account_id', 'trade_date']])} features")

    return daily_features


def create_enhanced_trader_day_panel(config: Dict) -> pd.DataFrame:
    """
    Create enhanced trader-day panel that includes both traditional and fills-based features.

    Args:
        config: Configuration dictionary containing paths and parameters

    Returns:
        pd.DataFrame: Enhanced panel DataFrame with trader-day observations
    """
    logger.info("Starting enhanced trader-day panel creation with fills features...")

    # Get traditional panel data (existing functionality)
    from src.data_processing import create_trader_day_panel
    traditional_panel = create_trader_day_panel(config)

    # Get fills-based features
    fills_features = create_fills_features(config)

    # Merge traditional panel with fills features
    enhanced_panel = pd.merge(
        traditional_panel,
        fills_features,
        on=['account_id', 'trade_date'],
        how='left'
    )

    # Fill NaN values for days with no fills
    fills_cols = [col for col in fills_features.columns if col not in ['account_id', 'trade_date']]
    enhanced_panel[fills_cols] = enhanced_panel[fills_cols].fillna(0)

    # Calculate additional cross-feature metrics
    # Efficiency metrics
    enhanced_panel['pnl_per_fill'] = (
        enhanced_panel['daily_pnl'] / (enhanced_panel['fills_count'] + 1e-6)
    )

    enhanced_panel['pnl_per_order'] = (
        enhanced_panel['daily_pnl'] / (enhanced_panel['orders_count'] + 1e-6)
    )

    enhanced_panel['volume_efficiency'] = (
        enhanced_panel['daily_volume'] / (enhanced_panel['fills_total_qty'] + 1e-6)
    )

    # Risk-adjusted metrics
    enhanced_panel['fees_to_pnl_ratio'] = (
        enhanced_panel['fills_total_fees'] / (abs(enhanced_panel['daily_pnl']) + 1e-6)
    )

    enhanced_panel['trades_to_fills_ratio'] = (
        enhanced_panel['n_trades'] / (enhanced_panel['fills_count'] + 1e-6)
    )

    logger.info(f"Enhanced panel created with shape: {enhanced_panel.shape}")
    logger.info(f"Total features: {len(enhanced_panel.columns)}")

    return enhanced_panel


def validate_enhanced_data_quality(df: pd.DataFrame, config: Dict) -> Dict:
    """
    Validate the quality of enhanced data including fills features.

    Args:
        df: Enhanced panel DataFrame
        config: Configuration dictionary

    Returns:
        Dict: Validation results
    """
    logger.info("Validating enhanced data quality...")

    validation_results = {
        'validation_timestamp': pd.Timestamp.now().isoformat(),
        'issues': [],
        'warnings': [],
        'data_summary': {}
    }

    # Basic data summary
    validation_results['data_summary'] = {
        'n_rows': len(df),
        'n_traders': df['account_id'].nunique(),
        'date_range': [df['trade_date'].min().isoformat(), df['trade_date'].max().isoformat()],
        'traditional_features': len([col for col in df.columns if col.startswith(('daily_', 'gross_', 'n_'))]),
        'fills_features': len([col for col in df.columns if col.startswith('fills_')]),
        'order_features': len([col for col in df.columns if col.startswith('order')]),
        'derived_features': len([col for col in df.columns if col.endswith(('_ratio', '_efficiency', '_score'))])
    }

    # Check fills data availability
    days_with_fills = (df['fills_count'] > 0).sum()
    total_days = len(df)
    fills_coverage = days_with_fills / total_days

    validation_results['data_summary']['fills_coverage'] = fills_coverage

    if fills_coverage < 0.7:
        validation_results['warnings'].append(f"Low fills coverage: {fills_coverage:.1%} of days have fills data")

    # Check for consistency between traditional and fills features
    consistency_checks = []

    # Volume consistency (allowing for some differences due to aggregation)
    if 'daily_volume' in df.columns and 'fills_total_qty' in df.columns:
        volume_corr = df['daily_volume'].corr(df['fills_total_qty'])
        if volume_corr < 0.8:
            consistency_checks.append(f"Volume correlation between trades and fills: {volume_corr:.3f}")

    # Trade count vs fills/orders
    if 'n_trades' in df.columns and 'fills_count' in df.columns:
        active_days = df[df['n_trades'] > 0]
        if len(active_days) > 0:
            fills_per_trade = (active_days['fills_count'] / active_days['n_trades']).mean()
            if fills_per_trade > 20 or fills_per_trade < 0.1:
                consistency_checks.append(f"Unusual fills per trade ratio: {fills_per_trade:.2f}")

    if consistency_checks:
        validation_results['warnings'].extend(consistency_checks)

    # Check for extreme values in fills features
    fills_cols = [col for col in df.columns if col.startswith('fills_')]
    extreme_values = []

    for col in fills_cols[:10]:  # Check first 10 fills columns
        if df[col].dtype in ['int64', 'float64']:
            q99 = df[col].quantile(0.99)
            q01 = df[col].quantile(0.01)
            if q99 > 1000 * (q01 + 1e-6):  # Very wide range
                extreme_values.append(f"{col}: range {q01:.2f} to {q99:.2f}")

    if extreme_values:
        validation_results['warnings'].append(f"Extreme value ranges in fills features: {extreme_values[:3]}")

    # Validation summary
    validation_results['n_issues'] = len(validation_results['issues'])
    validation_results['n_warnings'] = len(validation_results['warnings'])

    # Log results
    if validation_results['issues']:
        logger.warning(f"Found {validation_results['n_issues']} data quality issues")
        for issue in validation_results['issues']:
            logger.warning(f"  ISSUE: {issue}")

    if validation_results['warnings']:
        logger.info(f"Found {validation_results['n_warnings']} data quality warnings")
        for warning in validation_results['warnings'][:3]:
            logger.info(f"  WARNING: {warning}")

    if not validation_results['issues'] and not validation_results['warnings']:
        logger.info("All enhanced data quality checks passed successfully")

    return validation_results


if __name__ == "__main__":
    # Test the enhanced data processing
    import yaml

    with open('configs/main_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    enhanced_panel = create_enhanced_trader_day_panel(config)
    validation_results = validate_enhanced_data_quality(enhanced_panel, config)

    print(f"\nEnhanced Panel Summary:")
    print(f"Shape: {enhanced_panel.shape}")
    print(f"Features: {enhanced_panel.columns.tolist()}")
    print(f"Data Quality Issues: {validation_results['n_issues']}")
    print(f"Data Quality Warnings: {validation_results['n_warnings']}")
