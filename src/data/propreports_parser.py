"""
PropreReports Parser
Handles parsing of PropreReports CSV files with proper format detection
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
    Handles both totals by date and fills reports
    """

    def __init__(self):
        # Expected columns for different report types
        self.fills_columns = [
            'Date/Time', 'Account', 'B/S', 'Qty', 'Symbol', 'Price',
            'Route', 'Liq', 'Comm', 'Ecn Fee', 'SEC', 'ORF', 'CAT',
            'TAF', 'FTT', 'NSCC', 'Acc', 'Clr', 'Misc', 'Order Id',
            'Fill Id', 'Currency', 'ISIN', 'CUSIP', 'Status', 'PropReports Id'
        ]

        self.totals_columns = [
            'Symbol', 'Orders', 'Fills', 'Shares', 'Gross P&L', 'Net P&L',
            'Unrealized', 'Total', 'Volume', 'High', 'Low', 'Open Shares',
            'Closed P&L', 'Trades'
        ]

    def parse_csv_file(self, file_path: str) -> Tuple[pd.DataFrame, str]:
        """
        Parse a PropreReports CSV file and detect its type

        Args:
            file_path: Path to CSV file

        Returns:
            Tuple of (DataFrame, report_type)
            report_type is either 'fills' or 'totals'
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
        elif 'tbd' in filename or 'total' in filename:
            logger.info(f"Detected totals report from filename: {filename}")
            df = self.parse_totals_report(file_path)
            return df, 'totals'
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

    def parse_totals_report(self, file_path: str) -> pd.DataFrame:
        """Parse totals by date report CSV"""
        try:
            # Read the file content first to handle multi-date format
            with open(file_path, 'r') as f:
                content = f.read()

            # Parse the complex format
            df = self._parse_totals_content(content)

            logger.info(f"Parsed {len(df)} daily totals from {file_path}")
            return df

        except Exception as e:
            logger.error(f"Error parsing totals report: {e}")
            raise

    def _parse_totals_content(self, content: str) -> pd.DataFrame:
        """
        Parse the totals by date content which has multiple dates
        Each date section starts with a date line (MM/DD/YYYY)
        """
        lines = content.strip().split('\n')
        all_data = []
        current_date = None
        date_pattern = r'^\d{1,2}/\d{1,2}/\d{4}$'

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Check if this line is a date
            if re.match(date_pattern, line):
                current_date = pd.to_datetime(line).date()
                logger.debug(f"Found date: {current_date}")
                i += 1
                continue

            # Look for the header line that starts with "Symbol"
            if line.startswith('Symbol,') and current_date:
                # Found header, parse the data section
                header = line.split(',')
                i += 1

                # Read data lines until we hit a summary or new date
                while i < len(lines):
                    data_line = lines[i].strip()

                    # Stop conditions
                    if (not data_line or
                        data_line.startswith('Fee:') or
                        data_line.startswith('Daily Total') or
                        re.match(date_pattern, data_line) or
                        'Page' in data_line):
                        break

                    # Parse data line
                    try:
                        values = data_line.split(',')
                        if len(values) >= len(header) and values[0] not in ['', 'Symbol']:
                            row_data = {
                                'Date': current_date,
                                **{header[j]: self._parse_value(values[j])
                                   for j in range(min(len(header), len(values)))}
                            }
                            all_data.append(row_data)
                    except Exception as e:
                        logger.debug(f"Skipping line: {data_line} - Error: {e}")

                    i += 1
            else:
                i += 1

        # Convert to DataFrame
        if all_data:
            df = pd.DataFrame(all_data)
            return df
        else:
            return pd.DataFrame()

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

    def _parse_value(self, value: str) -> Any:
        """Parse a string value to appropriate type"""
        value = value.strip()

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

    def detect_and_parse(self, file_path: str) -> pd.DataFrame:
        """Detect report type and parse accordingly"""
        # Try reading first few lines
        with open(file_path, 'r') as f:
            first_lines = [f.readline() for _ in range(5)]

        # Check for fills report signature
        if any('Date/Time' in line and 'Symbol' in line for line in first_lines):
            return self.parse_fills_report(file_path)
        else:
            # Assume totals report
            return self.parse_totals_report(file_path)

    def _detect_report_type(self, df: pd.DataFrame) -> str:
        """Detect report type from DataFrame columns"""
        if 'Date/Time' in df.columns and 'B/S' in df.columns:
            return 'fills'
        elif 'Date' in df.columns and 'Symbol' in df.columns:
            return 'totals'
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

        elif report_type == 'totals':
            # Validate totals data
            required_cols = ['Date', 'Symbol', 'Net P&L']
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                validation['errors'].append(f"Missing required columns: {missing}")
                validation['is_valid'] = False

            # Stats
            if 'Date' in df.columns:
                validation['stats']['date_range'] = (df['Date'].min(), df['Date'].max())
                validation['stats']['total_days'] = df['Date'].nunique()

        validation['stats']['total_rows'] = len(df)

        return validation
