"""
PropreReports Parser
Handles parsing of PropreReports CSV files with proper format detection
Updated to support new summaryByDate format
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class PropreReportsParser:
    """
    Parser for PropreReports CSV files
    Handles both summaryByDate and fills reports
    """

    def __init__(self):
        # Expected columns for different report types
        self.fills_columns = [
            'Date/Time', 'Account', 'B/S', 'Qty', 'Symbol', 'Price',
            'Route', 'Liq', 'Comm', 'Ecn Fee', 'SEC', 'ORF', 'CAT',
            'TAF', 'FTT', 'NSCC', 'Acc', 'Clr', 'Misc', 'Order Id',
            'Fill Id', 'Currency', 'ISIN', 'CUSIP', 'Status', 'PropReports Id'
        ]

        # Essential columns common to both equities and options summaryByDate reports
        self.summary_essential_columns = [
            'Date', 'Type', 'Orders', 'Fills', 'Qty', 'Gross', 'Comm',
            'Ecn Fee', 'SEC', 'ORF', 'CAT', 'TAF', 'FTT', 'NSCC', 'Acc',
            'Clr', 'Misc', 'Trade Fees', 'Net', 'Adj Fees', 'Adj Net',
            'Unrealized Δ', 'Total Δ', 'Transfer: Deposit', 'Transfers',
            'Cash', 'Unrealized', 'End Balance'
        ]

        # Optional columns that may vary between account types
        self.summary_optional_columns = [
            'Fee: Software & MD',  # Equities accounts
            'Fee: VAT',            # Equities accounts
            'Fee: Daily Interest'  # Options accounts
        ]

    def parse_csv_file(self, file_path: str) -> Tuple[pd.DataFrame, str]:
        """
        Parse a PropreReports CSV file and detect its type

        Args:
            file_path: Path to CSV file

        Returns:
            Tuple of (DataFrame, report_type)
            report_type is either 'fills' or 'summary'
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Try to detect file type from filename
        filename = file_path.name.lower()

        if 'fill' in filename:
            logger.info(f"Detected fills report from filename: {filename}")
            df = self.parse_fills_report(file_path)
            return df, 'fills'
        elif 'summary' in filename or 'summarybydate' in filename:
            logger.info(f"Detected summary report from filename: {filename}")
            df = self.parse_summary_report(file_path)
            return df, 'summary'
        else:
            # Try to detect from content
            logger.info("Detecting report type from content...")
            df = self.detect_and_parse(file_path)
            report_type = self._detect_report_type(df)
            return df, report_type

    def parse_fills_report(self, file_path: str) -> pd.DataFrame:
        """Parse fills report CSV"""
        try:
            # Read CSV
            df = pd.read_csv(file_path)

            # Validate columns
            missing_cols = set(self.fills_columns) - set(df.columns)
            if missing_cols:
                logger.warning(f"Missing expected columns: {missing_cols}")

            # Clean and convert data types
            df = self._clean_fills_data(df)

            logger.info(f"Parsed {len(df)} fills from {file_path}")
            return df

        except Exception as e:
            logger.error(f"Error parsing fills report: {e}")
            raise

    def parse_summary_report(self, file_path: str) -> pd.DataFrame:
        """Parse summaryByDate report CSV"""
        try:
            # Read CSV directly
            df = pd.read_csv(file_path)

            # Check if last row is "Equities" or "Options" summary and remove it
            if not df.empty and 'Date' in df.columns:
                last_date_value = str(df.iloc[-1]['Date']).strip()
                if last_date_value in ['Equities', 'Options', 'Eq', 'Op']:
                    logger.info(f"Removing '{last_date_value}' summary row")
                    df = df.iloc[:-1]

            # Log which columns we found
            found_columns = set(df.columns)
            essential_missing = set(self.summary_essential_columns) - found_columns
            if essential_missing:
                logger.warning(f"Missing essential columns: {essential_missing}")

            # Log any optional columns found
            optional_found = found_columns & set(self.summary_optional_columns)
            if optional_found:
                logger.info(f"Found optional columns: {optional_found}")

            # Clean and convert data types
            df = self._clean_summary_data(df)

            # Detect account type
            account_type = self.detect_account_type(df)
            logger.info(f"Detected account type: {account_type}")

            logger.info(f"Parsed {len(df)} daily summaries from {file_path}")
            return df

        except Exception as e:
            logger.error(f"Error parsing summary report: {e}")
            raise

    def _clean_summary_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize summary data"""
        # Convert date
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            # Remove any rows where Date conversion failed (like "Equities"/"Options" row)
            df = df.dropna(subset=['Date'])

        # List of all numeric columns (essential + optional)
        numeric_columns = [
            'Orders', 'Fills', 'Qty', 'Gross', 'Comm', 'Ecn Fee', 'SEC',
            'ORF', 'CAT', 'TAF', 'FTT', 'NSCC', 'Acc', 'Clr', 'Misc',
            'Trade Fees', 'Net', 'Adj Fees', 'Adj Net', 'Unrealized Δ',
            'Total Δ', 'Transfer: Deposit', 'Transfers', 'Cash', 'Unrealized',
            'End Balance',
            # Optional columns
            'Fee: Software & MD', 'Fee: VAT', 'Fee: Daily Interest'
        ]

        for col in numeric_columns:
            if col in df.columns:
                # Handle parentheses for negative numbers and remove commas
                df[col] = df[col].astype(str).str.replace(',', '')
                df[col] = df[col].str.replace(r'\(([0-9.]+)\)', r'-\1', regex=True)
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Clean Type column
        if 'Type' in df.columns:
            df['Type'] = df['Type'].str.strip()

        return df

    def _clean_fills_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize fills data"""
        # Convert numeric columns
        numeric_columns = ['Qty', 'Price', 'Comm', 'Ecn Fee', 'SEC', 'ORF',
                          'CAT', 'TAF', 'FTT', 'NSCC', 'Acc', 'Clr', 'Misc']

        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Convert datetime
        if 'Date/Time' in df.columns:
            df['Date/Time'] = pd.to_datetime(df['Date/Time'], errors='coerce')

        # Clean B/S column
        if 'B/S' in df.columns:
            df['B/S'] = df['B/S'].str.strip().str.upper()

        # Clean symbol
        if 'Symbol' in df.columns:
            df['Symbol'] = df['Symbol'].str.strip().str.upper()

        return df

    def detect_and_parse(self, file_path: str) -> pd.DataFrame:
        """Detect report type and parse accordingly"""
        # Try reading first few lines
        with open(file_path, 'r') as f:
            first_lines = [f.readline() for _ in range(5)]

        # Check for fills report signature
        if any('Date/Time' in line and 'Symbol' in line for line in first_lines):
            return self.parse_fills_report(file_path)
        # Check for summary report signature - look for common essential columns
        elif any('Date' in line and 'Type' in line and 'Orders' in line and 'Net' in line for line in first_lines):
            return self.parse_summary_report(file_path)
        else:
            # Try both and see which works
            try:
                return self.parse_summary_report(file_path)
            except:
                return self.parse_fills_report(file_path)

    def _detect_report_type(self, df: pd.DataFrame) -> str:
        """Detect report type from DataFrame columns"""
        if 'Date/Time' in df.columns and 'B/S' in df.columns:
            return 'fills'
        elif 'Date' in df.columns and 'Type' in df.columns and 'Net' in df.columns:
            return 'summary'
        else:
            raise ValueError("Unable to determine report type from columns")

    def validate_data(self, df: pd.DataFrame, report_type: str) -> Dict[str, Any]:
        """Validate parsed data and return validation results"""
        validation = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }

        # Check if empty
        if df.empty:
            validation['is_valid'] = False
            validation['errors'].append("No data found in file")
            return validation

        if report_type == 'fills':
            # Validate fills data
            required_cols = ['Date/Time', 'Symbol', 'Qty', 'Price', 'B/S']
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                validation['errors'].append(f"Missing required columns: {missing}")
                validation['is_valid'] = False

            # Check for invalid prices
            if 'Price' in df.columns:
                invalid_prices = df[df['Price'] <= 0]
                if not invalid_prices.empty:
                    validation['warnings'].append(f"{len(invalid_prices)} rows with invalid prices")

            # Stats
            if 'Symbol' in df.columns:
                validation['stats']['unique_symbols'] = df['Symbol'].nunique()
                validation['stats']['total_trades'] = len(df)

        elif report_type == 'summary':
            # Validate summary data - only check essential columns
            required_cols = ['Date', 'Type', 'Net']
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                validation['errors'].append(f"Missing required columns: {missing}")
                validation['is_valid'] = False

            # Check essential columns
            essential_missing = [col for col in self.summary_essential_columns if col not in df.columns]
            if essential_missing:
                validation['warnings'].append(f"Missing essential columns: {essential_missing}")

            # Stats
            if 'Date' in df.columns:
                validation['stats']['date_range'] = (df['Date'].min(), df['Date'].max())
                validation['stats']['total_days'] = df['Date'].nunique()

            if 'Net' in df.columns:
                validation['stats']['total_net_pnl'] = df['Net'].sum()

            if 'Type' in df.columns:
                validation['stats']['account_type'] = df['Type'].iloc[0] if not df.empty else 'Unknown'

        validation['stats']['total_rows'] = len(df)

        return validation

    def _parse_value(self, value: str) -> Any:
        """Parse a string value to appropriate type"""
        value = str(value).strip()

        # Empty or dash values
        if not value or value == '-':
            return None

        # Try to parse as number
        try:
            # Remove commas and parentheses for negative numbers
            clean_value = value.replace(',', '')
            if clean_value.startswith('(') and clean_value.endswith(')'):
                clean_value = '-' + clean_value[1:-1]

            # Try float
            if '.' in clean_value:
                return float(clean_value)
            else:
                return int(clean_value)
        except ValueError:
            # Return as string
            return value

    def detect_account_type(self, df: pd.DataFrame) -> str:
        """Detect whether account is Equities or Options based on columns"""
        if 'Fee: Software & MD' in df.columns or 'Fee: VAT' in df.columns:
            return 'Equities'
        elif 'Fee: Daily Interest' in df.columns:
            return 'Options'
        else:
            # Default based on Type column if available
            if 'Type' in df.columns and not df.empty:
                first_type = str(df['Type'].iloc[0]).strip()
                if first_type in ['Eq', 'Equities']:
                    return 'Equities'
                elif first_type in ['Op', 'Options']:
                    return 'Options'
            return 'Unknown'
