# scripts/daily_risk_report.py
#!/usr/bin/env python3
"""
Generate daily risk assessment report
Usage: python scripts/daily_risk_report.py --date 2025-05-23
"""

import argparse
from datetime import datetime

from src.data.data_loader import DataLoader
from src.risk.risk_analyzer import RiskAnalyzer
from src.visualization.reports import ReportGenerator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--traders", nargs="+", help="Specific traders to analyze")
    parser.add_argument("--output", default="outputs/reports", help="Output directory")
    args = parser.parse_args()

    # Generate daily risk report
    generate_daily_report(args.date, args.traders, args.output)


if __name__ == "__main__":
    main()
