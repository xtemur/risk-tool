#!/usr/bin/env python3
"""
Risk Tool Health Check Script

Monitors the health of the automated risk management system.
Checks for recent data updates, model freshness, and system status.

Usage:
    python health_check.py [--alert-email email@example.com]
"""

import sqlite3
import argparse
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

def check_database_health():
    """Check database health and recent updates."""
    try:
        conn = sqlite3.connect('data/risk_tool.db')
        cursor = conn.cursor()

        # Check if database exists and has data
        cursor.execute("SELECT COUNT(*) FROM trades")
        total_trades = cursor.fetchone()[0]

        # Check for recent trades (last 7 days)
        week_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        cursor.execute("SELECT COUNT(*) FROM trades WHERE trade_date >= ?", (week_ago,))
        recent_trades = cursor.fetchone()[0]

        # Check accounts
        cursor.execute("SELECT COUNT(*) FROM accounts WHERE is_active = 1")
        active_accounts = cursor.fetchone()[0]

        conn.close()

        return {
            'status': 'healthy',
            'total_trades': total_trades,
            'recent_trades': recent_trades,
            'active_accounts': active_accounts,
            'issues': []
        }

    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'issues': [f"Database error: {e}"]
        }

def check_model_health():
    """Check model files and freshness."""
    model_dir = Path('models/production_model_artifacts')
    issues = []

    try:
        # Check if model files exist
        var_model = model_dir / 'lgbm_var_model.joblib'
        loss_model = model_dir / 'lgbm_loss_model.joblib'
        metadata_file = model_dir / 'model_metadata.json'

        if not var_model.exists():
            issues.append("VaR model file missing")
        if not loss_model.exists():
            issues.append("Loss model file missing")
        if not metadata_file.exists():
            issues.append("Model metadata missing")

        # Check model age
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            if 'training_date' in metadata:
                training_date = datetime.fromisoformat(metadata['training_date'].replace('Z', '+00:00'))
                model_age = (datetime.now(training_date.tzinfo) - training_date).days

                if model_age > 14:  # Models older than 2 weeks
                    issues.append(f"Models are {model_age} days old (consider retraining)")

        return {
            'status': 'healthy' if not issues else 'warning',
            'model_age_days': model_age if 'model_age' in locals() else None,
            'issues': issues
        }

    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'issues': [f"Model check error: {e}"]
        }

def check_logs():
    """Check recent log files for errors."""
    log_dir = Path('logs')
    issues = []

    try:
        if not log_dir.exists():
            issues.append("Log directory missing")
            return {'status': 'warning', 'issues': issues}

        # Check for recent automation logs
        today = datetime.now().strftime('%Y%m%d')
        automation_log = log_dir / f"quant_automation_{today}.log"

        if automation_log.exists():
            # Check for errors in today's log
            with open(automation_log, 'r') as f:
                log_content = f.read()
                error_count = log_content.count('ERROR')
                if error_count > 0:
                    issues.append(f"Found {error_count} errors in today's automation log")

        return {
            'status': 'healthy' if not issues else 'warning',
            'issues': issues
        }

    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'issues': [f"Log check error: {e}"]
        }

def main():
    """Main health check function."""
    parser = argparse.ArgumentParser(description='Risk Tool Health Check')
    parser.add_argument(
        '--alert-email',
        type=str,
        help='Email address for alerts if issues found'
    )

    args = parser.parse_args()

    print("🔍 Risk Tool Health Check")
    print("=" * 50)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    # Run health checks
    db_health = check_database_health()
    model_health = check_model_health()
    log_health = check_logs()

    # Print results
    print("📊 Database Health:")
    print(f"  Status: {db_health['status']}")
    if db_health['status'] == 'healthy':
        print(f"  Total trades: {db_health['total_trades']:,}")
        print(f"  Recent trades (7 days): {db_health['recent_trades']:,}")
        print(f"  Active accounts: {db_health['active_accounts']}")

    if db_health['issues']:
        print("  Issues:")
        for issue in db_health['issues']:
            print(f"    ⚠️  {issue}")
    print()

    print("🤖 Model Health:")
    print(f"  Status: {model_health['status']}")
    if model_health.get('model_age_days'):
        print(f"  Model age: {model_health['model_age_days']} days")

    if model_health['issues']:
        print("  Issues:")
        for issue in model_health['issues']:
            print(f"    ⚠️  {issue}")
    print()

    print("📝 Log Health:")
    print(f"  Status: {log_health['status']}")
    if log_health['issues']:
        print("  Issues:")
        for issue in log_health['issues']:
            print(f"    ⚠️  {issue}")
    print()

    # Overall status
    all_statuses = [db_health['status'], model_health['status'], log_health['status']]
    all_issues = db_health['issues'] + model_health['issues'] + log_health['issues']

    if 'error' in all_statuses:
        overall_status = 'ERROR'
        print("❌ Overall Status: CRITICAL - System has errors")
        exit_code = 2
    elif 'warning' in all_statuses or all_issues:
        overall_status = 'WARNING'
        print("⚠️  Overall Status: WARNING - System has issues")
        exit_code = 1
    else:
        overall_status = 'HEALTHY'
        print("✅ Overall Status: HEALTHY - All systems operational")
        exit_code = 0

    # TODO: Implement email alerting if needed
    if args.alert_email and exit_code > 0:
        print(f"\n📧 Alert would be sent to: {args.alert_email}")
        print("(Email alerting not yet implemented)")

    return exit_code

if __name__ == "__main__":
    exit(main())
