#!/usr/bin/env python
"""
Trader Analytics Report Generator
Generates comprehensive performance analytics and sends email reports
"""

import sys
import logging
from pathlib import Path
from datetime import date, timedelta

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.analytics import TraderAnalytics
from src.analytics_email import AnalyticsEmailService
from src.database import Database


def setup_logging():
    """Configure logging with debug level for analytics"""
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.DEBUG,  # Changed to DEBUG for troubleshooting
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/analytics.log')
        ]
    )


class AnalyticsReportGenerator:
    """Main class for generating analytics reports"""

    def __init__(self):
        self.analytics = TraderAnalytics()
        self.email_service = AnalyticsEmailService()
        self.db = Database()
        self.logger = logging.getLogger(__name__)

    def generate_weekly_report(self) -> bool:
        """Generate and send weekly analytics report"""

        self.logger.info("Generating Weekly Analytics Report")
        self.logger.info("=" * 50)

        # Weekly analysis (7 days)
        return self._generate_report(lookback_days=7, report_type="Weekly")

    def generate_monthly_report(self) -> bool:
        """Generate and send monthly analytics report"""

        self.logger.info("Generating Monthly Analytics Report")
        self.logger.info("=" * 50)

        # Monthly analysis (30 days)
        return self._generate_report(lookback_days=30, report_type="Monthly")

    def _generate_report(self, lookback_days: int, report_type: str) -> bool:
        """Generate comprehensive analytics report"""

        try:
            # Step 1: Get all traders
            traders_df = self.db.get_all_traders()

            if traders_df.empty:
                self.logger.error("No traders found in database")
                return False

            self.logger.info(f"Found {len(traders_df)} traders for analysis")

            # Step 2: Generate individual analytics
            self.logger.info("Generating individual trader analytics...")
            analytics_data = {}
            performance_charts = {}

            for _, trader in traders_df.iterrows():
                account_id = str(trader['account_id'])
                trader_name = trader['trader_name']

                self.logger.info(f"Analyzing {trader_name} ({account_id})...")

                # Generate analytics
                trader_analytics = self.analytics.generate_trader_analytics(account_id, lookback_days)

                if 'error' not in trader_analytics:
                    analytics_data[account_id] = trader_analytics

                    # Generate performance chart
                    try:
                        totals_df, _ = self.db.get_trader_data(
                            account_id,
                            (date.today() - timedelta(days=lookback_days)).strftime('%Y-%m-%d'),
                            date.today().strftime('%Y-%m-%d')
                        )

                        if not totals_df.empty:
                            chart_b64 = self.analytics.create_performance_chart(totals_df, trader_name)
                            performance_charts[account_id] = chart_b64

                    except Exception as e:
                        self.logger.warning(f"Could not generate chart for {trader_name}: {str(e)}")
                else:
                    self.logger.warning(f"No data available for {trader_name}")

            if not analytics_data:
                self.logger.error("No analytics data generated")
                return False

            # Step 3: Generate peer comparison
            self.logger.info("Generating peer comparison analytics...")
            comparison_data = self.analytics.generate_peer_comparison(lookback_days)

            # Step 4: Generate comparison chart
            comparison_chart = ""
            if comparison_data and 'comparison_data' in comparison_data:
                try:
                    comparison_chart = self.analytics.create_comparison_chart(comparison_data['comparison_data'])
                except Exception as e:
                    self.logger.warning(f"Could not generate comparison chart: {str(e)}")

            # Step 5: Generate summary statistics
            self._log_summary_stats(analytics_data, comparison_data)

            # Step 6: Send email report
            self.logger.info(f"Sending {report_type.lower()} analytics email report...")
            email_success = self.email_service.send_analytics_report(
                analytics_data,
                comparison_data,
                performance_charts,
                comparison_chart
            )

            if email_success:
                self.logger.info("✅ Analytics report sent successfully")
            else:
                self.logger.error("❌ Failed to send analytics report")

            # Step 7: Save analytics to files
            self._save_analytics_to_files(analytics_data, comparison_data, lookback_days)

            return email_success

        except Exception as e:
            self.logger.error(f"Analytics report generation failed: {str(e)}")
            return False

    def _log_summary_stats(self, analytics_data: dict, comparison_data: dict):
        """Log summary statistics"""

        total_traders = len(analytics_data)

        # Calculate portfolio metrics
        total_pnl = 0
        profitable_traders = 0
        total_win_rate = 0
        total_sharpe = 0

        for data in analytics_data.values():
            perf = data.get('performance', {})
            pnl = perf.get('total_pnl', 0)
            total_pnl += pnl

            if pnl > 0:
                profitable_traders += 1

            total_win_rate += perf.get('win_rate', 0)
            total_sharpe += perf.get('sharpe_ratio', 0)

        avg_win_rate = total_win_rate / total_traders if total_traders > 0 else 0
        avg_sharpe = total_sharpe / total_traders if total_traders > 0 else 0

        self.logger.info("📊 ANALYTICS SUMMARY")
        self.logger.info("-" * 30)
        self.logger.info(f"Total Traders Analyzed: {total_traders}")
        self.logger.info(f"Profitable Traders: {profitable_traders} ({profitable_traders/total_traders*100:.1f}%)")
        self.logger.info(f"Total Portfolio P&L: ${total_pnl:,.2f}")
        self.logger.info(f"Average Win Rate: {avg_win_rate:.1f}%")
        self.logger.info(f"Average Sharpe Ratio: {avg_sharpe:.3f}")

        # Top and bottom performers
        if comparison_data and 'top_performers' in comparison_data:
            top_pnl = comparison_data['top_performers']['by_pnl'][0]
            self.logger.info(f"Top Performer: {top_pnl['trader_name']} (${top_pnl['total_pnl']:,.2f})")

    def _save_analytics_to_files(self, analytics_data: dict, comparison_data: dict, lookback_days: int):
        """Save analytics data to CSV files for analysis"""

        try:
            results_dir = Path('data/analytics_results')
            results_dir.mkdir(exist_ok=True)

            timestamp = date.today().strftime('%Y%m%d')

            # Save individual analytics
            if analytics_data:
                analytics_list = []
                for account_id, data in analytics_data.items():
                    if 'error' not in data:
                        row = {'account_id': account_id}
                        row.update(data.get('performance', {}))
                        row.update(data.get('risk', {}))
                        row.update(data.get('behavior', {}))
                        row.update(data.get('efficiency', {}))
                        analytics_list.append(row)

                if analytics_list:
                    import pandas as pd
                    analytics_df = pd.DataFrame(analytics_list)
                    analytics_file = results_dir / f"analytics_{lookback_days}d_{timestamp}.csv"
                    analytics_df.to_csv(analytics_file, index=False)
                    self.logger.info(f"Analytics data saved to {analytics_file}")

            # Save comparison data
            if comparison_data and 'comparison_data' in comparison_data:
                import pandas as pd
                comparison_df = pd.DataFrame(comparison_data['comparison_data'])
                comparison_file = results_dir / f"comparison_{lookback_days}d_{timestamp}.csv"
                comparison_df.to_csv(comparison_file, index=False)
                self.logger.info(f"Comparison data saved to {comparison_file}")

        except Exception as e:
            self.logger.warning(f"Could not save analytics files: {str(e)}")

    def test_analytics_system(self) -> bool:
        """Test the analytics system with sample data"""

        self.logger.info("Testing Analytics System")
        self.logger.info("=" * 30)

        try:
            # Test email functionality
            self.logger.info("Testing email service...")
            email_test = self.email_service.send_test_analytics_email()

            if email_test:
                self.logger.info("✅ Email test successful")
            else:
                self.logger.error("❌ Email test failed")

            # Test analytics generation
            self.logger.info("Testing analytics generation...")
            traders_df = self.db.get_all_traders()

            if not traders_df.empty:
                test_trader = traders_df.iloc[0]
                account_id = str(test_trader['account_id'])

                analytics = self.analytics.generate_trader_analytics(account_id, 7)

                if 'error' not in analytics:
                    self.logger.info(f"✅ Analytics generation successful for {test_trader['trader_name']}")
                    return True
                else:
                    self.logger.error(f"❌ Analytics generation failed: {analytics['error']}")
                    return False
            else:
                self.logger.error("❌ No traders found for testing")
                return False

        except Exception as e:
            self.logger.error(f"❌ Analytics system test failed: {str(e)}")
            return False


def main():
    """Main function"""
    setup_logging()

    generator = AnalyticsReportGenerator()

    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Generate Trader Analytics Reports')
    parser.add_argument('--type', choices=['weekly', 'monthly', 'test'],
                       default='monthly', help='Type of report to generate')
    parser.add_argument('--test', action='store_true', help='Run system test')

    args = parser.parse_args()

    if args.test:
        success = generator.test_analytics_system()
    elif args.type == 'weekly':
        success = generator.generate_weekly_report()
    elif args.type == 'monthly':
        success = generator.generate_monthly_report()
    else:
        # Default to monthly
        success = generator.generate_monthly_report()

    if success:
        print("✅ Analytics report completed successfully")
        exit(0)
    else:
        print("❌ Analytics report failed")
        exit(1)


if __name__ == "__main__":
    main()
