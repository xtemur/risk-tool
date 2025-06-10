"""
Data Validator
Ensures data quality and integrity for trading systems
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

from src.core.constants import TradingConstants as TC, DataQualityLimits as DQL

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of data validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    stats: Dict[str, Any]

    def __str__(self):
        status = "VALID" if self.is_valid else "INVALID"
        return f"ValidationResult({status}, {len(self.errors)} errors, {len(self.warnings)} warnings)"


class DataValidator:
    """
    Comprehensive data validation for trading data
    Checks for common issues that can break models or cause bad predictions
    """

    def __init__(self, strict_mode: bool = False):
        """
        Initialize validator

        Args:
            strict_mode: If True, warnings become errors
        """
        self.strict_mode = strict_mode
        self.validation_stats = {}

    def validate_totals(self,
                       totals_df: pd.DataFrame,
                       account_id: Optional[str] = None) -> ValidationResult:
        """
        Validate daily totals data

        Args:
            totals_df: Daily totals DataFrame
            account_id: Optional account ID for specific validation

        Returns:
            ValidationResult with details
        """
        errors = []
        warnings = []
        stats = {}

        # Basic structure checks
        if totals_df.empty:
            errors.append("Totals DataFrame is empty")
            return ValidationResult(False, errors, warnings, stats)

        # Required columns
        required_cols = ['date', 'account_id', 'net_pnl', 'gross_pnl',
                        'orders_count', 'fills_count', 'total_fees']
        missing_cols = set(required_cols) - set(totals_df.columns)
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
            return ValidationResult(False, errors, warnings, stats)

        # Create a copy for validation
        df = totals_df.copy()

        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])

        # 1. Check for duplicate dates per account
        duplicates = df.groupby(['account_id', 'date']).size()
        duplicate_entries = duplicates[duplicates > 1]
        if not duplicate_entries.empty:
            errors.append(f"Duplicate date entries found: {len(duplicate_entries)} cases")
            stats['duplicate_dates'] = duplicate_entries.to_dict()

        # 2. Check for missing dates (gaps in time series)
        for acc_id in df['account_id'].unique():
            if account_id and acc_id != account_id:
                continue

            acc_df = df[df['account_id'] == acc_id].sort_values('date')
            date_range = pd.date_range(acc_df['date'].min(), acc_df['date'].max(), freq='D')
            trading_days = pd.bdate_range(acc_df['date'].min(), acc_df['date'].max(), freq='B')

            missing_days = set(trading_days) - set(acc_df['date'])
            if len(missing_days) > TC.MAX_CONSECUTIVE_MISSING:
                warnings.append(f"Account {acc_id} has {len(missing_days)} missing trading days")

            # Check for long gaps
            date_diffs = acc_df['date'].diff()
            max_gap = date_diffs.max()
            if pd.notna(max_gap) and max_gap.days > TC.MAX_CONSECUTIVE_MISSING:
                warnings.append(f"Account {acc_id} has gap of {max_gap.days} days")

        # 3. Check P&L consistency
        pnl_issues = []

        # Gross P&L should be >= Net P&L (fees reduce profit)
        incorrect_pnl = df[df['gross_pnl'] < df['net_pnl'] - 0.01]  # Small tolerance
        if not incorrect_pnl.empty:
            pnl_issues.append(f"{len(incorrect_pnl)} rows where gross P&L < net P&L")

        # Fees should be non-negative
        negative_fees = df[df['total_fees'] < 0]
        if not negative_fees.empty:
            warnings.append(f"{len(negative_fees)} rows with negative fees")

        # Check extreme P&L values
        extreme_pnl = df[
            (df['net_pnl'] > DQL.MAX_DAILY_PNL) |
            (df['net_pnl'] < DQL.MIN_DAILY_PNL)
        ]
        if not extreme_pnl.empty:
            warnings.append(f"{len(extreme_pnl)} rows with extreme P&L values")
            stats['extreme_pnl_examples'] = extreme_pnl[['date', 'account_id', 'net_pnl']].head().to_dict()

        # 4. Check fee ratios
        df['fee_ratio'] = df['total_fees'] / (df['gross_pnl'].abs() + 1e-8)
        high_fee_ratio = df[df['fee_ratio'] > DQL.MAX_FEE_PCT]
        if not high_fee_ratio.empty:
            warnings.append(f"{len(high_fee_ratio)} rows with fees > {DQL.MAX_FEE_PCT*100}% of gross P&L")

        # 5. Check order/fill consistency
        # Fills should not exceed orders
        incorrect_fills = df[df['fills_count'] > df['orders_count']]
        if not incorrect_fills.empty:
            errors.append(f"{len(incorrect_fills)} rows where fills > orders")

        # 6. Check for suspicious patterns
        # Same P&L for multiple days (might indicate stale data)
        for acc_id in df['account_id'].unique():
            acc_df = df[df['account_id'] == acc_id].sort_values('date')
            if len(acc_df) > 5:
                # Check for repeated values
                pnl_counts = acc_df['net_pnl'].value_counts()
                repeated_pnl = pnl_counts[pnl_counts > len(acc_df) * 0.1]  # Same P&L >10% of days
                if not repeated_pnl.empty and not (repeated_pnl.index == 0).any():
                    warnings.append(f"Account {acc_id} has repeated P&L values: {repeated_pnl.to_dict()}")

        # 7. Statistical checks
        stats['total_rows'] = len(df)
        stats['date_range'] = f"{df['date'].min()} to {df['date'].max()}"
        stats['accounts'] = df['account_id'].nunique()
        stats['avg_daily_pnl'] = df.groupby('account_id')['net_pnl'].mean().to_dict()
        stats['total_pnl'] = df.groupby('account_id')['net_pnl'].sum().to_dict()

        # Check data recency
        latest_date = df['date'].max()
        days_old = (datetime.now().date() - latest_date.date()).days
        if days_old > TC.MAX_DATA_AGE_DAYS:
            warnings.append(f"Latest data is {days_old} days old")

        # In strict mode, warnings become errors
        if self.strict_mode:
            errors.extend(warnings)
            warnings = []

        is_valid = len(errors) == 0
        return ValidationResult(is_valid, errors, warnings, stats)

    def validate_fills(self,
                      fills_df: pd.DataFrame,
                      account_id: Optional[str] = None) -> ValidationResult:
        """
        Validate fills (transaction) data

        Args:
            fills_df: Fills DataFrame
            account_id: Optional account ID for specific validation

        Returns:
            ValidationResult with details
        """
        errors = []
        warnings = []
        stats = {}

        # Basic structure checks
        if fills_df.empty:
            warnings.append("Fills DataFrame is empty (may be acceptable for some traders)")
            return ValidationResult(True, errors, warnings, stats)

        # Required columns
        required_cols = ['datetime', 'account_id', 'symbol', 'price', 'quantity']
        missing_cols = set(required_cols) - set(fills_df.columns)
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
            return ValidationResult(False, errors, warnings, stats)

        # Create a copy for validation
        df = fills_df.copy()

        # Ensure datetime column
        if not pd.api.types.is_datetime64_any_dtype(df['datetime']):
            df['datetime'] = pd.to_datetime(df['datetime'])

        # 1. Check for invalid prices
        invalid_prices = df[(df['price'] <= 0) | (df['price'] > DQL.MAX_PRICE)]
        if not invalid_prices.empty:
            errors.append(f"{len(invalid_prices)} fills with invalid prices")
            stats['invalid_price_examples'] = invalid_prices[['datetime', 'symbol', 'price']].head().to_dict()

        # 2. Check for invalid quantities
        zero_qty = df[df['quantity'] == 0]
        if not zero_qty.empty:
            errors.append(f"{len(zero_qty)} fills with zero quantity")

        # 3. Check timestamps
        # Fills should be during market hours (with some tolerance for pre/post market)
        df['hour'] = df['datetime'].dt.hour
        df['minute'] = df['datetime'].dt.minute

        # Define extended market hours (4 AM to 8 PM EST)
        outside_hours = df[
            (df['hour'] < 4) |
            (df['hour'] > 20) |
            ((df['hour'] == 20) & (df['minute'] > 0))
        ]
        if not outside_hours.empty:
            warnings.append(f"{len(outside_hours)} fills outside extended market hours")

        # 4. Check for duplicate fills
        # Same symbol, price, quantity, and timestamp might indicate duplicates
        potential_dupes = df.groupby(['account_id', 'datetime', 'symbol', 'price', 'quantity']).size()
        duplicates = potential_dupes[potential_dupes > 1]
        if not duplicates.empty:
            warnings.append(f"{len(duplicates)} potential duplicate fills")

        # 5. Check for suspicious trading patterns
        # Extremely high frequency (more than 1 trade per second on same symbol)
        for acc_id in df['account_id'].unique():
            if account_id and acc_id != account_id:
                continue

            acc_df = df[df['account_id'] == acc_id].sort_values('datetime')

            for symbol in acc_df['symbol'].unique():
                sym_df = acc_df[acc_df['symbol'] == symbol]
                time_diffs = sym_df['datetime'].diff()

                # Count trades within 1 second
                rapid_trades = time_diffs[time_diffs < timedelta(seconds=1)]
                if len(rapid_trades) > len(sym_df) * 0.1:  # >10% of trades
                    warnings.append(f"Account {acc_id} has rapid trading on {symbol}")

        # 6. Check for penny stocks or unusual symbols
        penny_stocks = df[df['price'] < DQL.MIN_PRICE]
        if not penny_stocks.empty:
            unique_penny = penny_stocks['symbol'].unique()
            warnings.append(f"{len(penny_stocks)} fills on {len(unique_penny)} penny stocks")

        # 7. Volume analysis
        df['trade_value'] = df['price'] * df['quantity'].abs()

        # Check for unusually large trades
        daily_values = df.groupby([df['datetime'].dt.date, 'account_id'])['trade_value'].sum()
        extreme_days = daily_values[daily_values > 1e6]  # Days with >$1M traded
        if not extreme_days.empty:
            warnings.append(f"{len(extreme_days)} days with >$1M in trades")

        # 8. Statistical summary
        stats['total_fills'] = len(df)
        stats['date_range'] = f"{df['datetime'].min()} to {df['datetime'].max()}"
        stats['unique_symbols'] = df['symbol'].nunique()
        stats['avg_trade_size'] = df['trade_value'].mean()
        stats['total_volume'] = df['trade_value'].sum()

        # Symbol distribution
        symbol_counts = df['symbol'].value_counts().head(10)
        stats['top_symbols'] = symbol_counts.to_dict()

        # In strict mode, warnings become errors
        if self.strict_mode:
            errors.extend(warnings)
            warnings = []

        is_valid = len(errors) == 0
        return ValidationResult(is_valid, errors, warnings, stats)

    def validate_combined(self,
                         totals_df: pd.DataFrame,
                         fills_df: pd.DataFrame) -> ValidationResult:
        """
        Validate consistency between totals and fills data

        Args:
            totals_df: Daily totals DataFrame
            fills_df: Fills DataFrame

        Returns:
            ValidationResult with details
        """
        errors = []
        warnings = []
        stats = {}

        # First validate each dataset individually
        totals_result = self.validate_totals(totals_df)
        fills_result = self.validate_fills(fills_df)

        # Combine individual results
        errors.extend(totals_result.errors)
        errors.extend(fills_result.errors)
        warnings.extend(totals_result.warnings)
        warnings.extend(fills_result.warnings)

        # If individual validations failed, skip cross-validation
        if not totals_result.is_valid or not fills_result.is_valid:
            return ValidationResult(False, errors, warnings, stats)

        # Cross-validation between totals and fills
        if not fills_df.empty and not totals_df.empty:
            # Aggregate fills to daily level
            fills_daily = fills_df.copy()
            fills_daily['date'] = pd.to_datetime(fills_daily['datetime']).dt.date
            fills_daily['date'] = pd.to_datetime(fills_daily['date'])

            fills_agg = fills_daily.groupby(['account_id', 'date']).agg({
                'price': 'count',  # Number of fills
                'quantity': 'sum',  # Total quantity
                'total_fees': 'sum' if 'total_fees' in fills_daily.columns else 'count'
            }).rename(columns={'price': 'fill_count_calc'})

            # Merge with totals
            merged = pd.merge(
                totals_df,
                fills_agg,
                on=['account_id', 'date'],
                how='inner'
            )

            if not merged.empty:
                # Check fill counts match
                fill_mismatch = merged[
                    np.abs(merged['fills_count'] - merged['fill_count_calc']) > 2
                ]
                if not fill_mismatch.empty:
                    pct_mismatch = len(fill_mismatch) / len(merged) * 100
                    warnings.append(f"{pct_mismatch:.1f}% of days have fill count mismatch")

                # Check quantities match (with tolerance)
                if 'quantity' in merged.columns and 'quantity_y' in merged.columns:
                    qty_mismatch = merged[
                        np.abs(merged['quantity_x'] - merged['quantity_y']) /
                        (merged['quantity_x'].abs() + 1) > 0.01
                    ]
                    if not qty_mismatch.empty:
                        warnings.append(f"{len(qty_mismatch)} days have quantity mismatch")

        # Update stats
        stats.update({
            'totals_stats': totals_result.stats,
            'fills_stats': fills_result.stats,
            'cross_validation_complete': not fills_df.empty
        })

        is_valid = len(errors) == 0
        return ValidationResult(is_valid, errors, warnings, stats)

    def generate_report(self, result: ValidationResult) -> str:
        """
        Generate a human-readable validation report

        Args:
            result: ValidationResult to report on

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 60)
        report.append("DATA VALIDATION REPORT")
        report.append("=" * 60)
        report.append(f"Status: {'PASSED' if result.is_valid else 'FAILED'}")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        if result.errors:
            report.append("ERRORS:")
            for i, error in enumerate(result.errors, 1):
                report.append(f"  {i}. {error}")
            report.append("")

        if result.warnings:
            report.append("WARNINGS:")
            for i, warning in enumerate(result.warnings, 1):
                report.append(f"  {i}. {warning}")
            report.append("")

        if result.stats:
            report.append("STATISTICS:")
            for key, value in result.stats.items():
                if isinstance(value, dict) and len(value) > 5:
                    report.append(f"  {key}: {len(value)} items")
                else:
                    report.append(f"  {key}: {value}")

        report.append("=" * 60)
        return "\n".join(report)
