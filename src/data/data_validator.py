"""
Data Validator for Risk Tool
Validates data integrity and provides stats
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging
from .database_manager import DatabaseManager

logger = logging.getLogger(__name__)


def validate_data(account_id: Optional[str] = None, fast_only: bool = False) -> Dict[str, Any]:
    """
    Validate trading data in database and return comprehensive stats

    Args:
        account_id: Optional specific account to validate
        fast_only: If True, only run fast aggregate validation

    Returns:
        Dictionary with validation results and stats
    """
    db = DatabaseManager()

    validation_result = {
        'success': True,
        'errors': [],
        'warnings': [],
        'stats': {
            'total_accounts': 0,
            'accounts_with_data': 0,
            'failed_validation': 0,
            'total_discrepancy_days': 0,
            'date_range': None,
            'by_account': {}
        }
    }

    try:
        # Get accounts to validate
        if account_id:
            accounts_df = db.get_accounts()
            accounts_df = accounts_df[accounts_df['account_id'] == account_id]
            if accounts_df.empty:
                validation_result['errors'].append(f"Account {account_id} not found")
                validation_result['success'] = False
                return validation_result
        else:
            accounts_df = db.get_accounts()

        if accounts_df.empty:
            validation_result['errors'].append("No accounts found in database")
            validation_result['success'] = False
            return validation_result

        validation_result['stats']['total_accounts'] = len(accounts_df)

        # Validate each account
        for _, account in accounts_df.iterrows():
            acc_id = account['account_id']
            acc_name = account['account_name']

            logger.info(f"Validating account {acc_name} ({acc_id})")

            account_stats = _validate_account(db, acc_id, fast_only)
            validation_result['stats']['by_account'][acc_id] = account_stats

            if account_stats['has_data']:
                validation_result['stats']['accounts_with_data'] += 1

            if account_stats['validation_errors'] > 0:
                validation_result['stats']['failed_validation'] += 1
                validation_result['warnings'].extend(account_stats['errors'])

            validation_result['stats']['total_discrepancy_days'] += account_stats.get('discrepancy_days', 0)

        # Overall database stats
        db_stats = db.get_database_stats()
        validation_result['stats']['database'] = db_stats

        # Calculate overall portfolio statistics
        if validation_result['stats']['by_account']:
            portfolio_stats = {
                'total_pnl': 0,
                'total_volume': 0,
                'total_trades': 0,
                'total_trading_days': 0,
                'total_profitable_days': 0,
                'total_losing_days': 0,
                'best_account_pnl': 0,
                'worst_account_pnl': 0,
                'best_day_pnl': 0,
                'worst_day_pnl': 0
            }

            for acc_id, acc_stats in validation_result['stats']['by_account'].items():
                if acc_stats.get('trading_stats'):
                    ts = acc_stats['trading_stats']
                    portfolio_stats['total_pnl'] += ts['total_pnl']
                    portfolio_stats['total_volume'] += ts['total_volume']
                    portfolio_stats['total_trades'] += ts['total_trades']
                    portfolio_stats['total_trading_days'] += ts['trading_days']
                    portfolio_stats['total_profitable_days'] += ts['profitable_days']
                    portfolio_stats['total_losing_days'] += ts['losing_days']

                    # Track best/worst account performance
                    if ts['total_pnl'] > portfolio_stats['best_account_pnl']:
                        portfolio_stats['best_account_pnl'] = ts['total_pnl']
                    if ts['total_pnl'] < portfolio_stats['worst_account_pnl']:
                        portfolio_stats['worst_account_pnl'] = ts['total_pnl']

                    # Track best/worst daily performance
                    if ts['max_daily_pnl'] > portfolio_stats['best_day_pnl']:
                        portfolio_stats['best_day_pnl'] = ts['max_daily_pnl']
                    if ts['min_daily_pnl'] < portfolio_stats['worst_day_pnl']:
                        portfolio_stats['worst_day_pnl'] = ts['min_daily_pnl']

            validation_result['stats']['portfolio'] = portfolio_stats

        # Date range
        summary_data = db.get_summary_data()
        if not summary_data.empty:
            validation_result['stats']['date_range'] = {
                'start': summary_data['date'].min().strftime('%Y-%m-%d'),
                'end': summary_data['date'].max().strftime('%Y-%m-%d'),
                'total_days': summary_data['date'].nunique()
            }

        logger.info(f"Validation complete: {validation_result['stats']['accounts_with_data']}/{validation_result['stats']['total_accounts']} accounts have data")

    except Exception as e:
        validation_result['success'] = False
        validation_result['errors'].append(f"Validation failed: {str(e)}")
        logger.error(f"Validation error: {e}")

    return validation_result


def _validate_account(db: DatabaseManager, account_id: str, fast_only: bool = False) -> Dict[str, Any]:
    """
    Validate data for a specific account

    Args:
        db: Database manager
        account_id: Account ID to validate
        fast_only: If True, only run fast aggregate validation

    Returns:
        Dictionary with account validation results
    """
    account_stats = {
        'has_data': False,
        'validation_errors': 0,
        'errors': [],
        'warnings': [],
        'summary_records': 0,
        'fills_records': 0,
        'date_range': None,
        'discrepancy_days': 0,
        'data_quality': {}
    }

    try:
        # Get summary data
        summary_data = db.get_summary_data(account_id=account_id)
        account_stats['summary_records'] = len(summary_data)

        # Get fills data
        fills_data = db.get_fills_data(account_id=account_id)
        account_stats['fills_records'] = len(fills_data)

        if account_stats['summary_records'] == 0 and account_stats['fills_records'] == 0:
            account_stats['warnings'].append(f"Account {account_id}: No data found")
            return account_stats

        account_stats['has_data'] = True

        # Date range validation
        if not summary_data.empty:
            account_stats['date_range'] = {
                'start': summary_data['date'].min().strftime('%Y-%m-%d'),
                'end': summary_data['date'].max().strftime('%Y-%m-%d'),
                'days': summary_data['date'].nunique()
            }

            # Calculate key trading statistics
            account_stats['trading_stats'] = {
                'total_pnl': float(summary_data['net'].sum()),
                'min_daily_pnl': float(summary_data['net'].min()),
                'max_daily_pnl': float(summary_data['net'].max()),
                'avg_daily_pnl': float(summary_data['net'].mean()),
                'profitable_days': int((summary_data['net'] > 0).sum()),
                'losing_days': int((summary_data['net'] < 0).sum()),
                'total_volume': float(summary_data['qty'].sum()),
                'total_trades': int(summary_data['fills'].sum()),
                'trading_days': int((summary_data['fills'] > 0).sum())
            }

            # Check for gaps in daily data
            date_range = pd.date_range(
                start=summary_data['date'].min(),
                end=summary_data['date'].max(),
                freq='D'
            )
            # Only count weekdays (trading days)
            weekdays = date_range[date_range.weekday < 5]
            actual_days = summary_data['date'].nunique()
            expected_days = len(weekdays)

            if actual_days < expected_days * 0.8:  # Allow for holidays
                account_stats['warnings'].append(
                    f"Account {account_id}: Possible missing trading days ({actual_days}/{expected_days})"
                )

        # Data quality checks
        if not summary_data.empty:
            account_stats['data_quality']['summary'] = _check_summary_quality(summary_data, account_id)

        if not fills_data.empty:
            account_stats['data_quality']['fills'] = _check_fills_quality(fills_data, account_id)

        # FAST VALIDATION: Aggregate comparison
        if not summary_data.empty and not fills_data.empty:
            fast_validation = _fast_aggregate_validation(summary_data, fills_data, account_id)
            account_stats['fast_validation'] = fast_validation
            if fast_validation['issues']:
                account_stats['warnings'].extend(fast_validation['issues'])
        elif not summary_data.empty and fills_data.empty:
            account_stats['warnings'].append(f"Account {account_id}: Has summary data but no fills data")

        # SLOW VALIDATION: Day-by-day comparison (only if not fast_only)
        if not fast_only and not summary_data.empty and not fills_data.empty:
            slow_validation = _slow_daily_validation(summary_data, fills_data, account_id)
            account_stats['slow_validation'] = slow_validation
            if slow_validation['issues']:
                account_stats['discrepancy_days'] = len(slow_validation['issues'])
                account_stats['warnings'].extend(slow_validation['issues'])

    except Exception as e:
        account_stats['validation_errors'] += 1
        account_stats['errors'].append(f"Account {account_id}: Validation error - {str(e)}")

    return account_stats


def _check_summary_quality(summary_data: pd.DataFrame, account_id: str) -> Dict[str, Any]:
    """Check quality of summary data"""
    quality = {
        'issues': [],
        'stats': {}
    }

    # Basic stats
    quality['stats'] = {
        'total_days': len(summary_data),
        'profitable_days': len(summary_data[summary_data['net'] > 0]),
        'losing_days': len(summary_data[summary_data['net'] < 0]),
        'total_pnl': summary_data['net'].sum(),
        'avg_daily_pnl': summary_data['net'].mean(),
        'max_daily_gain': summary_data['net'].max(),
        'max_daily_loss': summary_data['net'].min()
    }

    # Quality checks
    if (summary_data['net'] == 0).sum() > len(summary_data) * 0.5:
        quality['issues'].append(f"Account {account_id}: >50% of days have zero P&L")

    # Check for extreme values
    if summary_data['net'].abs().max() > 100000:
        quality['issues'].append(f"Account {account_id}: Extreme P&L values detected")

    # Check fills vs orders ratio
    if 'fills' in summary_data.columns and 'orders' in summary_data.columns:
        mask = summary_data['orders'] > 0
        if mask.any():
            fill_ratio = (summary_data.loc[mask, 'fills'] / summary_data.loc[mask, 'orders']).mean()
            if fill_ratio > 5:  # More than 5 fills per order on average seems high
                quality['issues'].append(f"Account {account_id}: High fills/orders ratio ({fill_ratio:.2f})")

    return quality


def _check_fills_quality(fills_data: pd.DataFrame, account_id: str) -> Dict[str, Any]:
    """Check quality of fills data"""
    quality = {
        'issues': [],
        'stats': {}
    }

    # Basic stats
    quality['stats'] = {
        'total_fills': len(fills_data),
        'unique_symbols': fills_data['symbol'].nunique(),
        'avg_price': fills_data['price'].mean(),
        'avg_qty': fills_data['qty'].mean(),
        'total_fees': fills_data['total_fee'].sum()
    }

    # Quality checks
    if (fills_data['price'] <= 0).any():
        invalid_prices = (fills_data['price'] <= 0).sum()
        quality['issues'].append(f"Account {account_id}: {invalid_prices} fills with invalid prices")

    if (fills_data['qty'] <= 0).any():
        invalid_qty = (fills_data['qty'] <= 0).sum()
        quality['issues'].append(f"Account {account_id}: {invalid_qty} fills with invalid quantities")

    # Check for missing symbols
    if fills_data['symbol'].isna().any() or (fills_data['symbol'] == '').any():
        missing_symbols = fills_data['symbol'].isna().sum() + (fills_data['symbol'] == '').sum()
        quality['issues'].append(f"Account {account_id}: {missing_symbols} fills missing symbols")

    # Check side distribution
    side_counts = fills_data['side'].value_counts()
    if len(side_counts) == 0:
        quality['issues'].append(f"Account {account_id}: No valid buy/sell sides found")
    elif len(side_counts) == 1:
        quality['issues'].append(f"Account {account_id}: Only {side_counts.index[0]} trades found")

    return quality


def _fast_aggregate_validation(summary_data: pd.DataFrame, fills_data: pd.DataFrame, account_id: str) -> Dict[str, Any]:
    """
    Fast validation: Compare total aggregates between summary and fills

    Args:
        summary_data: Daily summary DataFrame
        fills_data: Fills DataFrame
        account_id: Account ID

    Returns:
        Dictionary with validation results
    """
    validation = {
        'issues': [],
        'stats': {},
        'discrepancies': {}
    }

    try:
        # Aggregate totals from summary data
        summary_totals = {
            'total_fills': summary_data['fills'].sum(),
            'total_qty': summary_data['qty'].sum(),
            'total_fees': summary_data['trade_fees'].sum()
        }

        # Aggregate totals from fills data
        fills_totals = {
            'total_fills': len(fills_data),
            'total_qty': fills_data['qty'].sum(),
            'total_fees': fills_data['total_fee'].sum()
        }

        validation['stats'] = {
            'summary': summary_totals,
            'fills': fills_totals
        }

        # Compare aggregates with reasonable tolerances
        tolerances = {
            'fills': 0.02,  # 2% tolerance for fill count
            'qty': 0.05,    # 5% tolerance for quantity
            'fees': 0.10    # 10% tolerance for fees (more variation expected)
        }

        for metric in ['fills', 'qty', 'fees']:
            summary_val = summary_totals[f'total_{metric}']
            fills_val = fills_totals[f'total_{metric}']

            if summary_val == 0 and fills_val == 0:
                continue

            if summary_val == 0 or fills_val == 0:
                validation['issues'].append(
                    f"Account {account_id}: {metric.capitalize()} - one source has zero (Summary: {summary_val:,.0f}, Fills: {fills_val:,.0f})"
                )
                validation['discrepancies'][metric] = {
                    'summary': summary_val,
                    'fills': fills_val,
                    'diff_pct': float('inf')
                }
            else:
                diff_pct = abs(summary_val - fills_val) / max(summary_val, fills_val)
                validation['discrepancies'][metric] = {
                    'summary': summary_val,
                    'fills': fills_val,
                    'diff_pct': diff_pct
                }

                if diff_pct > tolerances[metric]:
                    validation['issues'].append(
                        f"Account {account_id}: {metric.capitalize()} discrepancy - Summary: {summary_val:,.0f}, Fills: {fills_val:,.0f} ({diff_pct:.1%} diff)"
                    )

    except Exception as e:
        validation['issues'].append(f"Account {account_id}: Error in fast validation - {str(e)}")

    return validation


def _slow_daily_validation(summary_data: pd.DataFrame, fills_data: pd.DataFrame, account_id: str) -> Dict[str, Any]:
    """
    Slow validation: Day-by-day comparison between summary and fills

    Args:
        summary_data: Daily summary DataFrame
        fills_data: Fills DataFrame
        account_id: Account ID

    Returns:
        Dictionary with validation results
    """
    validation = {
        'issues': [],
        'daily_discrepancies': [],
        'stats': {}
    }

    try:
        # Group fills by date
        fills_data['date'] = fills_data['datetime'].dt.date
        fills_daily = fills_data.groupby('date').agg({
            'qty': 'sum',
            'total_fee': 'sum'
        }).reset_index()
        fills_daily['fills_count'] = fills_data.groupby('date').size().values

        # Merge with summary data
        summary_data['date_only'] = summary_data['date'].dt.date
        merged = summary_data.merge(fills_daily, left_on='date_only', right_on='date', how='inner', suffixes=('_summary', '_fills'))

        if merged.empty:
            validation['issues'].append(f"Account {account_id}: No overlapping dates between summary and fills")
            return validation

        # Day-by-day validation
        discrepancy_days = []

        for _, row in merged.iterrows():
            date_str = row['date'].strftime('%Y-%m-%d')
            day_issues = []

            # Check fills count
            fills_diff = abs(row['fills'] - row['fills_count'])
            if fills_diff > max(row['fills'] * 0.05, 5):  # 5% or 5 fills tolerance
                day_issues.append(f"fills: {row['fills']} vs {row['fills_count']}")

            # Check quantity
            qty_diff = abs(row['qty'] - row['qty_summary'])
            if qty_diff > max(abs(row['qty']) * 0.1, 1000):  # 10% or 1000 shares tolerance
                day_issues.append(f"qty: {row['qty']:,.0f} vs {row['qty_summary']:,.0f}")

            # Check fees
            fee_diff = abs(row['trade_fees'] - row['total_fee'])
            if fee_diff > max(abs(row['trade_fees']) * 0.15, 50):  # 15% or $50 tolerance
                day_issues.append(f"fees: ${row['trade_fees']:.2f} vs ${row['total_fee']:.2f}")

            if day_issues:
                discrepancy_days.append({
                    'date': date_str,
                    'issues': day_issues
                })

        validation['daily_discrepancies'] = discrepancy_days
        validation['stats'] = {
            'total_compared_days': len(merged),
            'discrepancy_days': len(discrepancy_days),
            'discrepancy_rate': len(discrepancy_days) / len(merged) if len(merged) > 0 else 0
        }

        # Only report if significant number of discrepancies
        if len(discrepancy_days) > len(merged) * 0.05:  # More than 5% of days
            validation['issues'].append(
                f"Account {account_id}: Daily discrepancies on {len(discrepancy_days)} of {len(merged)} days ({len(discrepancy_days)/len(merged):.1%})"
            )

    except Exception as e:
        validation['issues'].append(f"Account {account_id}: Error in slow validation - {str(e)}")

    return validation


def validate_data_fast() -> Dict[str, Any]:
    """
    Run only fast aggregate validation for all accounts

    Returns:
        Dictionary with validation results
    """
    return validate_data(fast_only=True)


def validate_data_full() -> Dict[str, Any]:
    """
    Run full validation (fast + slow) for all accounts

    Returns:
        Dictionary with validation results
    """
    return validate_data(fast_only=False)


def get_data_summary() -> Dict[str, Any]:
    """Get high-level data summary"""
    db = DatabaseManager()

    summary = {
        'database_stats': db.get_database_stats(),
        'accounts': [],
        'date_coverage': {}
    }

    # Get account summaries
    accounts = db.get_accounts()
    for _, account in accounts.iterrows():
        acc_id = account['account_id']
        acc_summary = db.get_summary_data(account_id=acc_id)

        if not acc_summary.empty:
            account_info = {
                'account_id': acc_id,
                'account_name': account['account_name'],
                'days_of_data': len(acc_summary),
                'date_range': {
                    'start': acc_summary['date'].min().strftime('%Y-%m-%d'),
                    'end': acc_summary['date'].max().strftime('%Y-%m-%d')
                },
                'total_pnl': acc_summary['net'].sum(),
                'profitable_days': len(acc_summary[acc_summary['net'] > 0]),
                'avg_daily_pnl': acc_summary['net'].mean()
            }
            summary['accounts'].append(account_info)

    # Overall date coverage
    all_summary = db.get_summary_data()
    if not all_summary.empty:
        summary['date_coverage'] = {
            'earliest_date': all_summary['date'].min().strftime('%Y-%m-%d'),
            'latest_date': all_summary['date'].max().strftime('%Y-%m-%d'),
            'total_trading_days': all_summary['date'].nunique(),
            'accounts_count': all_summary['account_id'].nunique()
        }

    return summary
