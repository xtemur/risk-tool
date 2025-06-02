import os
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from typing import Dict, List
import pandas as pd
from dotenv import load_dotenv
import numpy as np

load_dotenv()
logger = logging.getLogger(__name__)


class AnalyticsEmailService:
    """Email service for analytics reports"""

    def __init__(self):
        self.from_email = os.getenv('EMAIL_FROM')
        self.password = os.getenv('EMAIL_PASSWORD')
        self.to_emails = os.getenv('EMAIL_TO').split(',')

        if not self.from_email or not self.password:
            logger.warning("Email credentials not configured")

    def create_analytics_html_report(self, analytics_data: Dict, comparison_data: Dict, performance_charts: Dict, comparison_chart: str = "") -> str:
        """Create professional HTML analytics report optimized for email clients"""

        # Get summary statistics
        total_traders = len([a for a in analytics_data.values() if 'error' not in a])
        period = list(analytics_data.values())[0].get('period_days', 30) if analytics_data else 30

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 10px; background-color: #f5f5f5; }}
                .container {{ max-width: 800px; margin: 0 auto; background-color: #ffffff; }}
                .header {{ background-color: #2c3e50; color: white; padding: 20px; text-align: center; }}
                .header h1 {{ margin: 0; font-size: 24px; }}
                .header p {{ margin: 5px 0 0 0; font-size: 14px; }}

                .summary-section {{ padding: 15px; background-color: #ecf0f1; border-bottom: 1px solid #bdc3c7; }}
                .summary-table {{ width: 100%; }}
                .summary-table td {{ padding: 8px; text-align: center; font-weight: bold; }}

                .section {{ margin: 0; padding: 15px; border-bottom: 1px solid #e1e8ed; }}
                .section-title {{ color: #2c3e50; font-size: 18px; margin: 0 0 15px 0; padding-bottom: 5px; border-bottom: 2px solid #3498db; }}

                .data-table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
                .data-table th {{ background-color: #34495e; color: white; padding: 10px 8px; text-align: left; font-size: 12px; }}
                .data-table td {{ padding: 8px; border-bottom: 1px solid #ddd; font-size: 12px; }}
                .data-table tr:nth-child(even) {{ background-color: #f8f9fa; }}

                .positive {{ color: #27ae60; font-weight: bold; }}
                .negative {{ color: #e74c3c; font-weight: bold; }}
                .neutral {{ color: #7f8c8d; }}

                .trader-section {{ margin: 15px 0; padding: 15px; border: 1px solid #e1e8ed; }}
                .trader-title {{ color: #2c3e50; font-size: 16px; margin: 0 0 10px 0; }}

                .metrics-table {{ width: 100%; }}
                .metrics-table td {{ padding: 5px 10px; font-size: 12px; }}
                .metrics-table .label {{ font-weight: bold; width: 40%; }}
                .metrics-table .value {{ text-align: right; width: 60%; }}

                .alert {{ background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 10px; margin: 10px 0; font-size: 12px; }}
                .alert-danger {{ background-color: #f8d7da; border-color: #f5c6cb; }}

                .chart-container {{ text-align: center; margin: 15px 0; }}
                .chart-container img {{ max-width: 100%; height: auto; }}

                .footer {{ padding: 15px; text-align: center; color: #7f8c8d; font-size: 11px; border-top: 1px solid #eee; }}

                @media only screen and (max-width: 600px) {{
                    .container {{ width: 100% !important; }}
                    .metrics-table td {{ padding: 3px 5px; font-size: 11px; }}
                    .data-table th, .data-table td {{ padding: 5px 4px; font-size: 10px; }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Trader Analytics Report</h1>
                    <p>{pd.Timestamp.now().strftime('%B %d, %Y')} • {period}-Day Analysis</p>
                </div>

                <div class="summary-section">
                    <table class="summary-table">
                        <tr>
                            <td>Active Traders<br><span style="font-size: 18px;">{total_traders}</span></td>
                            <td>Analysis Period<br><span style="font-size: 18px;">{period} Days</span></td>
                            <td>Report Date<br><span style="font-size: 18px;">{pd.Timestamp.now().strftime('%m/%d')}</span></td>
                        </tr>
                    </table>
                </div>
        """

        # ADD FLAGS SECTION
        trader_flags = comparison_data.get('trader_flags', {})
        portfolio_flags = comparison_data.get('portfolio_flags', {})

        if trader_flags or portfolio_flags:
            # Count flags
            red_flags = {k: v for k, v in trader_flags.items() if v['flags']['red_flags']}
            yellow_flags = {k: v for k, v in trader_flags.items() if v['flags']['yellow_flags']}
            green_lights = {k: v for k, v in trader_flags.items() if v['flags']['green_lights']}

            # Portfolio alerts
            if portfolio_flags.get('portfolio_flags'):
                html += """
                    <div style="padding: 15px; background-color: #fff3cd; border-left: 5px solid #ffc107; margin: 0;">
                        <h3 style="margin: 0 0 10px 0; color: #856404;">📊 Portfolio Alerts</h3>
                """
                for flag in portfolio_flags['portfolio_flags']:
                    html += f"<div style='margin: 5px 0;'>• {flag}</div>"
                html += "</div>"

            # Red flags
            if red_flags:
                html += f"""
                    <div style="padding: 15px; background-color: #f8d7da; border-left: 5px solid #dc3545; margin: 0;">
                        <h3 style="margin: 0 0 10px 0; color: #721c24;">🚨 IMMEDIATE ACTION REQUIRED ({len(red_flags)} traders)</h3>
                """
                for account_id, flag_data in red_flags.items():
                    html += f"<div style='margin: 8px 0; font-weight: bold;'>{flag_data['trader_name']} ({account_id})</div>"
                    for flag in flag_data['flags']['red_flags']:
                        html += f"<div style='margin: 3px 0 3px 15px; font-size: 12px;'>• {flag}</div>"
                html += "</div>"

            # Yellow flags
            if yellow_flags:
                html += f"""
                    <div style="padding: 15px; background-color: #fff3cd; border-left: 5px solid #ffc107; margin: 0;">
                        <h3 style="margin: 0 0 10px 0; color: #856404;">⚠️ MONITOR THIS WEEK ({len(yellow_flags)} traders)</h3>
                """
                for account_id, flag_data in yellow_flags.items():
                    html += f"<div style='margin: 8px 0; font-weight: bold;'>{flag_data['trader_name']} ({account_id})</div>"
                    for flag in flag_data['flags']['yellow_flags']:
                        html += f"<div style='margin: 3px 0 3px 15px; font-size: 12px;'>• {flag}</div>"
                html += "</div>"

            # Green lights
            if green_lights:
                html += f"""
                    <div style="padding: 15px; background-color: #d1e7dd; border-left: 5px solid #198754; margin: 0;">
                        <h3 style="margin: 0 0 10px 0; color: #0f5132;">💡 OPPORTUNITIES ({len(green_lights)} traders)</h3>
                """
                for account_id, flag_data in green_lights.items():
                    html += f"<div style='margin: 8px 0; font-weight: bold;'>{flag_data['trader_name']} ({account_id})</div>"
                    for flag in flag_data['flags']['green_lights']:
                        html += f"<div style='margin: 3px 0 3px 15px; font-size: 12px;'>• {flag}</div>"
                html += "</div>"

            # No flags message
            if not red_flags and not yellow_flags and not green_lights:
                html += """
                    <div style="padding: 15px; background-color: #d1e7dd; border-left: 5px solid #198754; margin: 0;">
                        <h3 style="margin: 0; color: #0f5132;">✅ All Clear - No Action Items</h3>
                        <p style="margin: 5px 0 0 0;">All traders are performing within normal parameters.</p>
                    </div>
                """

            # Quick Action Summary
            html += f"""
                <div style="padding: 15px; background-color: #f8f9fa; border-bottom: 1px solid #dee2e6; margin: 0;">
                    <h3 style="margin: 0 0 10px 0; color: #495057;">📋 Quick Action Summary</h3>
                    <table style="width: 100%; border-collapse: collapse;">
                        <tr>
                            <td style="padding: 8px; text-align: center; background-color: #dc3545; color: white; font-weight: bold;">
                                Immediate Action<br>{len(red_flags)} traders
                            </td>
                            <td style="padding: 8px; text-align: center; background-color: #ffc107; color: white; font-weight: bold;">
                                Monitor Closely<br>{len(yellow_flags)} traders
                            </td>
                            <td style="padding: 8px; text-align: center; background-color: #198754; color: white; font-weight: bold;">
                                Opportunities<br>{len(green_lights)} traders
                            </td>
                        </tr>
                    </table>
                </div>
            """

        # Portfolio Summary
        if comparison_data and 'comparison_data' in comparison_data:
            df = pd.DataFrame(comparison_data['comparison_data'])

            total_pnl = df['total_pnl'].sum()
            profitable_traders = len(df[df['total_pnl'] > 0])
            avg_win_rate = df['win_rate'].mean()
            avg_sharpe = df['sharpe_ratio'].mean()

            html += f"""
                <div class="section">
                    <h2 class="section-title">Portfolio Summary</h2>
                    <table class="data-table">
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                            <th>Benchmark</th>
                            <th>Status</th>
                        </tr>
                        <tr>
                            <td>Total Portfolio P&L</td>
                            <td class="{'positive' if total_pnl > 0 else 'negative'}">${total_pnl:,.2f}</td>
                            <td>$0.00</td>
                            <td>{'✓' if total_pnl > 0 else '✗'}</td>
                        </tr>
                        <tr>
                            <td>Profitable Traders</td>
                            <td>{profitable_traders}/{len(df)}</td>
                            <td>>50%</td>
                            <td>{'✓' if profitable_traders/len(df) > 0.5 else '✗'}</td>
                        </tr>
                        <tr>
                            <td>Average Win Rate</td>
                            <td>{avg_win_rate:.1f}%</td>
                            <td>>55%</td>
                            <td>{'✓' if avg_win_rate > 55 else '✗'}</td>
                        </tr>
                        <tr>
                            <td>Average Sharpe Ratio</td>
                            <td>{avg_sharpe:.3f}</td>
                            <td>>1.0</td>
                            <td>{'✓' if avg_sharpe > 1.0 else '✗'}</td>
                        </tr>
                    </table>
                </div>
            """

            # Top and Bottom Performers
            top_3 = df.nlargest(3, 'total_pnl')
            bottom_3 = df.nsmallest(3, 'total_pnl')

            html += f"""
                <div class="section">
                    <h2 class="section-title">Performance Rankings</h2>

                    <h3 style="color: #27ae60; margin: 10px 0 5px 0;">Top Performers</h3>
                    <table class="data-table">
                        <tr><th>Trader</th><th>P&L</th><th>Win Rate</th><th>Sharpe</th></tr>
            """

            for _, trader in top_3.iterrows():
                html += f"""
                    <tr>
                        <td>{trader['trader_name']}</td>
                        <td class="positive">${trader['total_pnl']:,.2f}</td>
                        <td>{trader['win_rate']:.1f}%</td>
                        <td>{trader['sharpe_ratio']:.3f}</td>
                    </tr>
                """

            html += """
                    </table>

                    <h3 style="color: #e74c3c; margin: 15px 0 5px 0;">Attention Required</h3>
                    <table class="data-table">
                        <tr><th>Trader</th><th>P&L</th><th>Win Rate</th><th>Max DD</th></tr>
            """

            for _, trader in bottom_3.iterrows():
                html += f"""
                    <tr>
                        <td>{trader['trader_name']}</td>
                        <td class="negative">${trader['total_pnl']:,.2f}</td>
                        <td>{trader['win_rate']:.1f}%</td>
                        <td class="negative">${trader['max_drawdown']:,.2f}</td>
                    </tr>
                """

            html += "</table></div>"

        # Individual Trader Details
        html += """
            <div class="section">
                <h2 class="section-title">Individual Trader Analysis</h2>
        """

        for account_id, data in analytics_data.items():
            if 'error' in data:
                continue

            trader_name = next((trader['trader_name'] for trader in comparison_data.get('comparison_data', [])
                            if trader['account_id'] == account_id), account_id)

            perf = data.get('performance', {})
            risk = data.get('risk', {})
            behavior = data.get('behavior', {})
            efficiency = data.get('efficiency', {})
            advanced = data.get('advanced', {})

            # Risk alerts
            alerts = []
            if perf.get('is_in_drawdown', False):
                alerts.append("Currently in drawdown")
            if risk.get('max_losing_streak', 0) > 5:
                alerts.append(f"Long losing streak: {risk.get('max_losing_streak')} days")
            if efficiency.get('fee_efficiency', 0) > 5:
                alerts.append("High fee ratio detected")
            if behavior.get('overtrading_pnl', 0) < -100:
                alerts.append("Potential overtrading")

            html += f"""
                <div class="trader-section">
                    <h3 class="trader-title">{trader_name} ({account_id})</h3>

                    <table class="metrics-table">
                        <tr>
                            <td class="label">Performance Metrics</td>
                            <td class="value"></td>
                        </tr>
                        <tr>
                            <td>Total P&L:</td>
                            <td class="value {'positive' if perf.get('total_pnl', 0) > 0 else 'negative'}">${perf.get('total_pnl', 0):,.2f}</td>
                        </tr>
                        <tr>
                            <td>Win Rate:</td>
                            <td class="value">{perf.get('win_rate', 0):.1f}%</td>
                        </tr>
                        <tr>
                            <td>Profit Factor:</td>
                            <td class="value">{perf.get('profit_factor', 0):.2f}</td>
                        </tr>
                        <tr>
                            <td>Sharpe Ratio:</td>
                            <td class="value">{perf.get('sharpe_ratio', 0):.3f}</td>
                        </tr>
                        <tr>
                            <td>Sortino Ratio:</td>
                            <td class="value">{perf.get('sortino_ratio', 0):.3f}</td>
                        </tr>
                        <tr>
                            <td>Calmar Ratio:</td>
                            <td class="value">{perf.get('calmar_ratio', 0):.3f}</td>
                        </tr>

                        <tr><td class="label" style="padding-top: 15px;">Risk Metrics</td><td></td></tr>
                        <tr>
                            <td>Max Drawdown:</td>
                            <td class="value negative">${perf.get('max_drawdown', 0):,.2f}</td>
                        </tr>
                        <tr>
                            <td>VaR (5%):</td>
                            <td class="value">${risk.get('var_5_percent', 0):,.2f}</td>
                        </tr>
                        <tr>
                            <td>CVaR (5%):</td>
                            <td class="value">${risk.get('cvar_5_percent', 0):,.2f}</td>
                        </tr>
                        <tr>
                            <td>Max Losing Streak:</td>
                            <td class="value">{risk.get('max_losing_streak', 0)} days</td>
                        </tr>

                        <tr><td class="label" style="padding-top: 15px;">Advanced Analytics</td><td></td></tr>
                        <tr>
                            <td>Omega Ratio:</td>
                            <td class="value">{advanced.get('omega_ratio', 0):.2f}</td>
                        </tr>
                        <tr>
                            <td>Kelly Criterion:</td>
                            <td class="value">{perf.get('kelly_criterion', 0):.3f}</td>
                        </tr>
                        <tr>
                            <td>Tail Expectation:</td>
                            <td class="value">${advanced.get('tail_expectation', 0):,.2f}</td>
                        </tr>

                        <tr><td class="label" style="padding-top: 15px;">Trading Behavior</td><td></td></tr>
                        <tr>
                            <td>Avg Daily Orders:</td>
                            <td class="value">{behavior.get('avg_daily_orders', 0):.1f}</td>
                        </tr>
                        <tr>
                            <td>Symbols Traded:</td>
                            <td class="value">{behavior.get('symbols_traded', 0)}</td>
                        </tr>
                        <tr>
                            <td>Fee Efficiency:</td>
                            <td class="value {'negative' if efficiency.get('fee_efficiency', 0) > 3 else 'neutral'}">{efficiency.get('fee_efficiency', 0):.2f}%</td>
                        </tr>
                    </table>
            """

            # Add alerts if any
            if alerts:
                html += f"""
                    <div class="alert alert-danger">
                        <strong>Risk Alerts:</strong><br>
                        • {'<br>• '.join(alerts)}
                    </div>
                """

            html += "</div>"

        html += f"""
                </div>

                <div class="footer">
                    <p>Generated by Analytics System with Flag Detection • {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p>🚨 Red = Act Today | ⚠️ Yellow = Monitor This Week | 💡 Green = Optimization Opportunity</p>
                </div>
            </div>
        </body>
        </html>
        """

        return html

    def send_analytics_report(self, analytics_data: Dict, comparison_data: Dict,
                            performance_charts: Dict, comparison_chart: str = "") -> bool:
        """Send comprehensive analytics report"""

        if not self.from_email or not self.password:
            logger.error("Email credentials not configured")
            return False

        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['From'] = self.from_email
            msg['To'] = ', '.join(self.to_emails)

            # Professional subject with key metrics
            total_traders = len([a for a in analytics_data.values() if 'error' not in a])
            period = list(analytics_data.values())[0].get('period_days', 30) if analytics_data else 30

            # Enhanced subject with flag summary
            flag_summary = comparison_data.get('flag_summary', {})
            report_type = flag_summary.get('report_type', 'Analytics')

            if flag_summary:
                red_count = flag_summary.get('red_count', 0)
                yellow_count = flag_summary.get('yellow_count', 0)
                green_count = flag_summary.get('green_count', 0)

                # Create flag indicator
                flag_indicator = ""
                if red_count > 0:
                    flag_indicator = f"🚨 {red_count} URGENT"
                elif yellow_count > 0:
                    flag_indicator = f"⚠️ {yellow_count} MONITOR"
                elif green_count > 0:
                    flag_indicator = f"✅ {green_count} OPPORTUNITIES"
                else:
                    flag_indicator = "✅ ALL CLEAR"

                # Get portfolio P&L for context
                if comparison_data and 'comparison_data' in comparison_data:
                    df = pd.DataFrame(comparison_data['comparison_data'])
                    total_pnl = df['total_pnl'].sum()
                    status = "POSITIVE" if total_pnl > 0 else "NEGATIVE"
                    subject = f"{report_type} Report | {flag_indicator} | {status} ${total_pnl:,.0f}"
                else:
                    subject = f"{report_type} Report | {flag_indicator}"
            else:
                # Fallback to original subject
                subject = f"Trading Analytics Report - {pd.Timestamp.now().strftime('%Y-%m-%d')}"

            msg['Subject'] = subject

            # Create HTML content
            html_content = self.create_analytics_html_report(
                analytics_data, comparison_data, performance_charts, comparison_chart
            )

            # Attach HTML
            html_part = MIMEText(html_content, 'html')
            msg.attach(html_part)

            # Send email
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login(self.from_email, self.password)
                server.send_message(msg)

            logger.info(f"Analytics report sent to {', '.join(self.to_emails)}")
            return True

        except Exception as e:
            logger.error(f"Failed to send analytics email: {str(e)}")
            return False

    def send_test_analytics_email(self) -> bool:
        """Send a test analytics email"""

        # Create sample data
        test_analytics = {
            '3946': {
                'account_id': '3946',
                'period_days': 30,
                'performance': {
                    'total_pnl': 2500.50,
                    'win_rate': 65.5,
                    'sharpe_ratio': 1.25,
                    'profit_factor': 1.45,
                    'max_drawdown': -850.25,
                    'avg_daily_pnl': 83.35
                },
                'risk': {
                    'var_5_percent': -145.30,
                    'max_losing_streak': 3
                },
                'behavior': {
                    'overtrading_pnl': -50.25
                },
                'efficiency': {
                    'consistency_score': 78.5,
                    'fee_efficiency': 2.1
                }
            }
        }

        test_comparison = {
            'total_traders': 1,
            'comparison_data': [{
                'account_id': '3946',
                'trader_name': 'Test Trader',
                'total_pnl': 2500.50,
                'win_rate': 65.5,
                'sharpe_ratio': 1.25,
                'profit_factor': 1.45,
                'max_drawdown': -850.25
            }],
            'top_performers': {
                'by_pnl': [{'trader_name': 'Test Trader', 'total_pnl': 2500.50}],
                'by_sharpe': [{'trader_name': 'Test Trader', 'sharpe_ratio': 1.25}],
                'by_consistency': [{'trader_name': 'Test Trader', 'consistency_score': 78.5}]
            }
        }

        logger.info("Sending test analytics email...")
        success = self.send_analytics_report(test_analytics, test_comparison, {})

        if success:
            logger.info("Test analytics email sent successfully!")
        else:
            logger.error("Test analytics email failed.")

        return success

    # Add this method to the AnalyticsEmailService class in src/analytics_email.py

def send_analytics_report_with_flags(self, analytics_data: Dict, comparison_data: Dict,
                                   trader_flags: Dict, portfolio_flags: Dict,
                                   performance_charts: Dict, comparison_chart: str = "",
                                   report_type: str = "Weekly") -> bool:
    """Send analytics report with integrated flag system"""

    if not self.from_email or not self.password:
        logger.error("Email credentials not configured")
        return False

    try:
        # Create message
        msg = MIMEMultipart('alternative')
        msg['From'] = self.from_email
        msg['To'] = ', '.join(self.to_emails)

        # Enhanced subject with flag summary
        total_traders = len([a for a in analytics_data.values() if 'error' not in a])
        period = list(analytics_data.values())[0].get('period_days', 30) if analytics_data else 30

        # Count flags
        red_count = sum(1 for flags in trader_flags.values() if flags['flags']['red_flags'])
        yellow_count = sum(1 for flags in trader_flags.values() if flags['flags']['yellow_flags'])
        green_count = sum(1 for flags in trader_flags.values() if flags['flags']['green_lights'])

        # Portfolio P&L
        if comparison_data and 'comparison_data' in comparison_data:
            df = pd.DataFrame(comparison_data['comparison_data'])
            total_pnl = df['total_pnl'].sum()
            status = "POSITIVE" if total_pnl > 0 else "NEGATIVE"
        else:
            total_pnl = 0
            status = "NEUTRAL"

        # Create comprehensive subject
        flag_summary = ""
        if red_count > 0:
            flag_summary = f"🚨 {red_count} URGENT"
        elif yellow_count > 0:
            flag_summary = f"⚠️ {yellow_count} MONITOR"
        elif green_count > 0:
            flag_summary = f"✅ {green_count} OPPORTUNITIES"

        subject = f"{report_type} Analytics | {flag_summary} | {status} ${total_pnl:,.0f} | {total_traders} Traders"

        msg['Subject'] = subject

        # Create HTML content with flags
        html_content = self.create_analytics_html_with_flags(
            analytics_data, comparison_data, trader_flags, portfolio_flags,
            performance_charts, comparison_chart, report_type
        )

        # Attach HTML
        html_part = MIMEText(html_content, 'html')
        msg.attach(html_part)

        # Send email
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(self.from_email, self.password)
            server.send_message(msg)

        logger.info(f"Analytics report with flags sent to {', '.join(self.to_emails)}")
        return True

    except Exception as e:
        logger.error(f"Failed to send analytics email with flags: {str(e)}")
        return False

    def create_analytics_html_with_flags(self, analytics_data: Dict, comparison_data: Dict,
                                    trader_flags: Dict, portfolio_flags: Dict,
                                    performance_charts: Dict, comparison_chart: str = "",
                                    report_type: str = "Weekly") -> str:
        """Create HTML report with prominent flag display"""

        # Get summary statistics
        total_traders = len([a for a in analytics_data.values() if 'error' not in a])
        period = list(analytics_data.values())[0].get('period_days', 30) if analytics_data else 30

        # Count flags
        red_flags = {k: v for k, v in trader_flags.items() if v['flags']['red_flags']}
        yellow_flags = {k: v for k, v in trader_flags.items() if v['flags']['yellow_flags']}
        green_lights = {k: v for k, v in trader_flags.items() if v['flags']['green_lights']}

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 10px; background-color: #f5f5f5; }}
                .container {{ max-width: 800px; margin: 0 auto; background-color: #ffffff; }}
                .header {{ background-color: #2c3e50; color: white; padding: 20px; text-align: center; }}
                .header h1 {{ margin: 0; font-size: 24px; }}
                .header p {{ margin: 5px 0 0 0; font-size: 14px; }}

                .flag-alert {{ margin: 0; padding: 15px; border-left: 5px solid; }}
                .red-alert {{ background-color: #fee; border-color: #e74c3c; }}
                .yellow-alert {{ background-color: #ffc; border-color: #f39c12; }}
                .green-alert {{ background-color: #efe; border-color: #27ae60; }}
                .portfolio-alert {{ background-color: #ffeaa7; border-color: #fdcb6e; }}

                .flag-section {{ padding: 15px; margin: 0; }}
                .flag-title {{ color: #2c3e50; font-size: 18px; margin: 0 0 10px 0; font-weight: bold; }}
                .trader-flag {{ margin: 8px 0; padding: 8px; background-color: #f8f9fa; border-radius: 4px; }}
                .trader-name {{ font-weight: bold; color: #2c3e50; }}
                .flag-list {{ margin: 5px 0 0 15px; font-size: 12px; }}
                .flag-item {{ margin: 3px 0; }}

                .summary-table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
                .summary-table td {{ padding: 8px; text-align: center; font-weight: bold; border: 1px solid #ddd; }}

                .section {{ margin: 0; padding: 15px; border-bottom: 1px solid #e1e8ed; }}
                .section-title {{ color: #2c3e50; font-size: 16px; margin: 0 0 10px 0; }}

                .positive {{ color: #27ae60; font-weight: bold; }}
                .negative {{ color: #e74c3c; font-weight: bold; }}
                .neutral {{ color: #7f8c8d; }}

                .footer {{ padding: 15px; text-align: center; color: #7f8c8d; font-size: 11px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>{report_type} Analytics Report with Action Items</h1>
                    <p>{pd.Timestamp.now().strftime('%B %d, %Y')} • {period}-Day Analysis • {total_traders} Traders</p>
                </div>
        """

        # Portfolio-level alerts
        if portfolio_flags.get('portfolio_flags'):
            html += """
                <div class="flag-alert portfolio-alert">
                    <h3 style="margin: 0 0 10px 0; color: #d35400;">📊 Portfolio-Level Alerts</h3>
            """
            for flag in portfolio_flags['portfolio_flags']:
                html += f"<div style='margin: 5px 0;'>• {flag}</div>"
            html += "</div>"

        # RED FLAGS - Immediate Action
        if red_flags:
            html += f"""
                <div class="flag-alert red-alert">
                    <h3 style="margin: 0 0 10px 0; color: #c0392b;">🚨 IMMEDIATE ACTION REQUIRED ({len(red_flags)} traders)</h3>
            """
            for account_id, flag_data in red_flags.items():
                html += f"""
                    <div class="trader-flag">
                        <div class="trader-name">{flag_data['trader_name']} ({account_id})</div>
                        <div class="flag-list">
                """
                for flag in flag_data['flags']['red_flags']:
                    html += f"<div class='flag-item'>• {flag}</div>"
                html += "</div></div>"
            html += "</div>"

        # YELLOW FLAGS - Monitor
        if yellow_flags:
            html += f"""
                <div class="flag-alert yellow-alert">
                    <h3 style="margin: 0 0 10px 0; color: #d68910;">⚠️ MONITOR THIS WEEK ({len(yellow_flags)} traders)</h3>
            """
            for account_id, flag_data in yellow_flags.items():
                html += f"""
                    <div class="trader-flag">
                        <div class="trader-name">{flag_data['trader_name']} ({account_id})</div>
                        <div class="flag-list">
                """
                for flag in flag_data['flags']['yellow_flags']:
                    html += f"<div class='flag-item'>• {flag}</div>"
                html += "</div></div>"
            html += "</div>"

        # GREEN LIGHTS - Opportunities
        if green_lights:
            html += f"""
                <div class="flag-alert green-alert">
                    <h3 style="margin: 0 0 10px 0; color: #239b56;">💡 OPPORTUNITIES ({len(green_lights)} traders)</h3>
            """
            for account_id, flag_data in green_lights.items():
                html += f"""
                    <div class="trader-flag">
                        <div class="trader-name">{flag_data['trader_name']} ({account_id})</div>
                        <div class="flag-list">
                """
                for flag in flag_data['flags']['green_lights']:
                    html += f"<div class='flag-item'>• {flag}</div>"
                html += "</div></div>"
            html += "</div>"

        # No flags message
        if not red_flags and not yellow_flags and not green_lights:
            html += """
                <div class="flag-alert green-alert">
                    <h3 style="margin: 0; color: #239b56;">✅ All Clear - No Action Items</h3>
                    <p style="margin: 5px 0 0 0;">All traders are performing within normal parameters.</p>
                </div>
            """

        # Quick Action Summary
        html += f"""
            <div class="section">
                <h2 class="section-title">📋 Quick Action Summary</h2>
                <table class="summary-table">
                    <tr>
                        <td style="background-color: #e74c3c; color: white;">Immediate Action<br>{len(red_flags)} traders</td>
                        <td style="background-color: #f39c12; color: white;">Monitor Closely<br>{len(yellow_flags)} traders</td>
                        <td style="background-color: #27ae60; color: white;">Opportunities<br>{len(green_lights)} traders</td>
                    </tr>
                </table>
            </div>
        """

        # Include existing analytics content (truncated for brevity)
        # You would include the existing portfolio summary and detailed analytics here
        # using the same format as your current create_analytics_html_report method

        if comparison_data and 'comparison_data' in comparison_data:
            df = pd.DataFrame(comparison_data['comparison_data'])
            total_pnl = df['total_pnl'].sum()
            profitable_traders = len(df[df['total_pnl'] > 0])

            html += f"""
                <div class="section">
                    <h2 class="section-title">📊 Portfolio Performance</h2>
                    <table class="summary-table">
                        <tr>
                            <td>Total P&L<br><span class="{'positive' if total_pnl > 0 else 'negative'}">${total_pnl:,.2f}</span></td>
                            <td>Profitable Traders<br>{profitable_traders}/{len(df)}</td>
                            <td>Success Rate<br>{profitable_traders/len(df)*100:.1f}%</td>
                        </tr>
                    </table>
                </div>
            """

        html += f"""
                <div class="footer">
                    <p>Generated by Analytics System with Flag Detection • {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p>🚨 Red = Act Today | ⚠️ Yellow = Monitor This Week | 💡 Green = Optimization Opportunity</p>
                </div>
            </div>
        </body>
        </html>
        """

        return html

    def send_analytics_report_with_flags(self, analytics_data: Dict, comparison_data: Dict,
                                   trader_flags: Dict, portfolio_flags: Dict,
                                   performance_charts: Dict, comparison_chart: str = "",
                                   report_type: str = "Weekly") -> bool:
        """Simple wrapper that adds flags to existing email system"""

        # Add flag summary to comparison data
        if comparison_data:
            comparison_data['trader_flags'] = trader_flags
            comparison_data['portfolio_flags'] = portfolio_flags

            # Count flags for subject line
            red_count = sum(1 for flags in trader_flags.values() if flags['flags']['red_flags'])
            yellow_count = sum(1 for flags in trader_flags.values() if flags['flags']['yellow_flags'])
            green_count = sum(1 for flags in trader_flags.values() if flags['flags']['green_lights'])

            comparison_data['flag_summary'] = {
                'red_count': red_count,
                'yellow_count': yellow_count,
                'green_count': green_count,
                'report_type': report_type
            }

        # Use existing email method
        return self.send_analytics_report(
            analytics_data,
            comparison_data,
            performance_charts,
            comparison_chart
        )
