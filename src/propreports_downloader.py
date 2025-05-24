import glob
import io
import logging
import os
import re
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# API Configuration
API_URL = "https://neo.propreports.com/api.php"
HEADERS = {"Content-Type": "application/x-www-form-urlencoded"}


class PropreportsDownloader:
    def __init__(self, token, config_path="config/config.yaml"):
        self.token = token
        self.config_path = config_path

        # Load trader accounts from config
        try:
            with open("config/trader_accounts.yaml", "r") as f:
                trader_config = yaml.safe_load(f)
                self.traders = trader_config["traders"]
            logger.info(f"Loaded {len(self.traders)} trader accounts from config")
        except FileNotFoundError:
            logger.error(
                "config/trader_accounts.yaml not found. Please create it first."
            )
            self.traders = []

        # Create necessary directories
        self.setup_directories()

    def setup_directories(self):
        """Create necessary directory structure"""
        directories = ["data/raw", "data/raw/totals_by_date", "data/raw/fills"]
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    def fetch_csv_page(self, form_data):
        """Send a POST request with the provided form data."""
        try:
            response = requests.post(API_URL, data=form_data, headers=HEADERS)
            logger.debug(
                f"Fetching page {form_data.get('page')} for account {form_data.get('accountId')}"
            )

            if response.status_code == 200:
                return response.text
            else:
                logger.error(
                    f"HTTP {response.status_code} for page {form_data.get('page')}"
                )
                return None
        except requests.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            return None

    def parse_csv_response(self, csv_text):
        """Parse CSV response and extract pagination info"""
        lines = csv_text.strip().splitlines()
        current_page = 1
        total_pages = 1

        if lines and re.match(r"Page\s+\d+/\d+", lines[-1].strip()):
            pagination_line = lines.pop().strip()
            m_pagination = re.search(r"Page\s+(\d+)\s*/\s*(\d+)", pagination_line)
            if m_pagination:
                current_page = int(m_pagination.group(1))
                total_pages = int(m_pagination.group(2))

        return lines, current_page, total_pages

    def fetch_pages(self, base_data):
        """Fetch all pages for a given request"""
        page_num = 0
        header = ""
        all_data_lines = []

        while True:
            page_num += 1
            base_data["page"] = str(page_num)

            csv_text = self.fetch_csv_page(base_data)
            if not csv_text:
                logger.warning(f"Skipping page {page_num} due to fetch error.")
                continue

            page_lines, current_page, total_pages = self.parse_csv_response(csv_text)
            if not page_lines:
                continue

            if header == "":
                header = page_lines[0]

            data_lines = page_lines[1:] if len(page_lines) > 1 else []
            all_data_lines.extend(data_lines)

            if current_page >= total_pages:
                break

        return header, all_data_lines

    def month_date_range(self, start_date, end_date):
        """Generate first day of each month between start_date and end_date"""
        current = start_date.replace(day=1)
        while current < end_date:
            yield current
            next_month = (current.replace(day=28) + timedelta(days=4)).replace(day=1)
            current = next_month

    def get_last_day_of_month(self, dt):
        """Get the last day of the month for given date"""
        next_month = (dt.replace(day=28) + timedelta(days=4)).replace(day=1)
        return next_month - timedelta(days=1)

    def get_account_df(self):
        """Get list of all accounts"""
        base_data = {"action": "accounts", "token": self.token}
        header, data_lines = self.fetch_pages(base_data)

        if header is None:
            logger.error("Failed to download accounts data.")
            return pd.DataFrame()

        df = pd.read_csv(io.StringIO("\n".join([header] + data_lines)))
        logger.info(f"Retrieved {len(df)} accounts from API")
        return df

    def download_totals_by_date(self, start_date, end_date, account_ids=None):
        """Download totals by date for specified accounts"""
        if account_ids is None:
            account_ids = [trader["account_id"] for trader in self.traders]

        logger.info(f"Downloading totals by date for {len(account_ids)} accounts")

        reports_data = {"action": "report", "type": "totalsByDate", "token": self.token}

        for account_id in account_ids:
            logger.info(f"Processing totals for account: {account_id}")
            base_data = reports_data.copy()
            base_data["accountId"] = account_id

            monthly_files = []

            for month_start in self.month_date_range(start_date, end_date):
                month_end = self.get_last_day_of_month(month_start)
                start_date_str = month_start.strftime("%Y-%m-%d")
                end_date_str = month_end.strftime("%Y-%m-%d")

                base_data["startDate"] = start_date_str
                base_data["endDate"] = end_date_str

                logger.debug(
                    f"Downloading {account_id} totals: {start_date_str} to {end_date_str}"
                )
                header, data_lines = self.fetch_pages(base_data)

                if header is None:
                    logger.warning(
                        f"Failed to download totals for {account_id}: {start_date_str} to {end_date_str}"
                    )
                    continue

                # Save monthly file
                csv_combined = "\n".join([header] + data_lines)
                month_dir = f"data/raw/totals_by_date/{account_id}"
                Path(month_dir).mkdir(parents=True, exist_ok=True)

                file_name = (
                    f"{month_dir}/tbd_{account_id}_{month_start.strftime('%Y_%m')}.csv"
                )
                with open(file_name, "w", encoding="utf-8") as f:
                    f.write(csv_combined)
                monthly_files.append(file_name)

                logger.debug(f"Saved: {file_name}")

            # Combine all monthly files into single totals file
            self.combine_monthly_files(account_id, monthly_files, "totals")

    def download_fills(self, start_date, end_date, account_ids=None):
        """Download fills data for specified accounts"""
        if account_ids is None:
            account_ids = [trader["account_id"] for trader in self.traders]

        logger.info(f"Downloading fills for {len(account_ids)} accounts")

        fills_data = {"action": "fills", "token": self.token}

        for account_id in account_ids:
            logger.info(f"Processing fills for account: {account_id}")
            base_data = fills_data.copy()
            base_data["accountId"] = account_id

            monthly_files = []

            for month_start in self.month_date_range(start_date, end_date):
                month_end = self.get_last_day_of_month(month_start)
                start_date_str = month_start.strftime("%Y-%m-%d")
                end_date_str = month_end.strftime("%Y-%m-%d")

                base_data["startDate"] = start_date_str
                base_data["endDate"] = end_date_str

                logger.debug(
                    f"Downloading {account_id} fills: {start_date_str} to {end_date_str}"
                )
                header, data_lines = self.fetch_pages(base_data)

                if header is None:
                    logger.warning(
                        f"Failed to download fills for {account_id}: {start_date_str} to {end_date_str}"
                    )
                    continue

                # Save monthly file
                csv_combined = "\n".join([header] + data_lines)
                month_dir = f"data/raw/fills/{account_id}"
                Path(month_dir).mkdir(parents=True, exist_ok=True)

                file_name = f"{month_dir}/fills_{account_id}_{month_start.strftime('%Y_%m')}.csv"
                with open(file_name, "w", encoding="utf-8") as f:
                    f.write(csv_combined)
                monthly_files.append(file_name)

                logger.debug(f"Saved: {file_name}")

            # Combine all monthly files into single fills file
            self.combine_monthly_files(account_id, monthly_files, "fills")

    def combine_monthly_files(self, account_id, monthly_files, data_type):
        """Combine monthly CSV files into single file for the risk-tool system"""
        if not monthly_files:
            logger.warning(f"No monthly files to combine for {account_id} {data_type}")
            return

        try:
            # Read and combine all monthly files
            dfs = []
            for file_path in monthly_files:
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    if not df.empty:
                        dfs.append(df)

            if not dfs:
                logger.warning(f"No valid data found for {account_id} {data_type}")
                return

            # Combine all dataframes
            combined_df = pd.concat(dfs, ignore_index=True)

            # Remove duplicates and sort by date
            if "Date" in combined_df.columns:
                combined_df["Date"] = pd.to_datetime(combined_df["Date"])
                combined_df = combined_df.drop_duplicates().sort_values("Date")

            # Save combined file in the format expected by risk-tool
            output_file = f"data/raw/{account_id}_{data_type}.csv"
            combined_df.to_csv(output_file, index=False)

            logger.info(
                f"Combined {len(dfs)} monthly files into {output_file} ({len(combined_df)} records)"
            )

        except Exception as e:
            logger.error(
                f"Error combining files for {account_id} {data_type}: {str(e)}"
            )

    def download_latest_data(self, days_back=30):
        """Download only recent data (for daily updates)"""
        end_date = date.today()
        start_date = end_date - timedelta(days=days_back)

        logger.info(f"Downloading latest data from {start_date} to {end_date}")

        account_ids = [trader["account_id"] for trader in self.traders]

        # Download recent data
        self.download_totals_by_date(start_date, end_date, account_ids)
        self.download_fills(start_date, end_date, account_ids)

        logger.info("Latest data download completed")

    def download_full_history(self, start_date=None, end_date=None):
        """Download full historical data"""
        if start_date is None:
            start_date = date(2023, 4, 1)  # Default start date
        if end_date is None:
            end_date = date.today()

        logger.info(f"Downloading full history from {start_date} to {end_date}")

        account_ids = [trader["account_id"] for trader in self.traders]

        # Download all data
        self.download_totals_by_date(start_date, end_date, account_ids)
        self.download_fills(start_date, end_date, account_ids)

        logger.info("Full history download completed")

    def verify_data_quality(self):
        """Verify downloaded data quality"""
        logger.info("Verifying data quality...")

        issues = []

        for trader in self.traders:
            account_id = trader["account_id"]

            # Check if files exist
            totals_file = f"data/raw/{account_id}_totals.csv"
            fills_file = f"data/raw/{account_id}_fills.csv"

            if not os.path.exists(totals_file):
                issues.append(f"Missing totals file for {account_id}")
                continue

            if not os.path.exists(fills_file):
                issues.append(f"Missing fills file for {account_id}")
                continue

            # Check data quality
            try:
                totals_df = pd.read_csv(totals_file)
                fills_df = pd.read_csv(fills_file)

                if totals_df.empty:
                    issues.append(f"Empty totals data for {account_id}")

                if fills_df.empty:
                    issues.append(f"Empty fills data for {account_id}")

                # Check required columns
                required_totals_cols = [
                    "Date",
                    "Net",
                    "Gross",
                    "Orders",
                    "Fills",
                    "Qty",
                ]
                missing_cols = [
                    col for col in required_totals_cols if col not in totals_df.columns
                ]
                if missing_cols:
                    issues.append(
                        f"Missing columns in {account_id} totals: {missing_cols}"
                    )

                logger.info(
                    f"{account_id}: {len(totals_df)} trading days, {len(fills_df)} fills"
                )

            except Exception as e:
                issues.append(f"Error reading data for {account_id}: {str(e)}")

        if issues:
            logger.warning("Data quality issues found:")
            for issue in issues:
                logger.warning(f"  - {issue}")
        else:
            logger.info("Data quality verification passed!")

        return len(issues) == 0


def main():
    """Main function for standalone execution"""
    # Configuration
    TOKEN = "0b8f3c5ee57fc7dd376af28ae83e4c2c:2523"  # Your API token

    # Initialize downloader
    downloader = PropreportsDownloader(TOKEN)

    if not downloader.traders:
        logger.error(
            "No traders configured. Please set up config/trader_accounts.yaml first."
        )
        return

    # Download full history (you can modify dates as needed)
    start_date = date(2023, 4, 1)
    end_date = date(2025, 4, 30)

    try:
        downloader.download_full_history(start_date, end_date)

        # Verify data quality
        downloader.verify_data_quality()

        logger.info("Data download completed successfully!")
        logger.info("You can now proceed with: python main.py --train")

    except Exception as e:
        logger.error(f"Download failed: {str(e)}")


if __name__ == "__main__":
    main()


# ================================
# Integration script for daily updates
# ================================


def download_for_risk_tool():
    """Function to be called by the risk management system"""
    TOKEN = "0b8f3c5ee57fc7dd376af28ae83e4c2c:2523"  # Move this to config or env

    downloader = PropreportsDownloader(TOKEN)

    # Download latest 30 days of data
    downloader.download_latest_data(days_back=30)

    # Verify data quality
    data_quality_ok = downloader.verify_data_quality()

    if not data_quality_ok:
        logger.error("Data quality issues detected!")
        return False

    logger.info("Data download and verification completed successfully")
    return True
