#!/usr/bin/env python3
"""
Send daily risk signals email report.

Usage:
    python send_daily_signals.py [--email recipient@example.com] [--save-only]

Options:
    --email: Email address to send report to (can be used multiple times)
    --save-only: Only save HTML file, don't send email
    --date: Specific date for report (YYYY-MM-DD), defaults to today
"""

import argparse
import logging
from datetime import datetime
import sys
import os

# Add directories to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Switch to temporally-aligned pooled model system (CLAUDE.md compliant)
from src.minimal_risk_system import MinimalRiskSystem
from inference.email_service import EmailService

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def convert_predictions_to_signals(predictions: dict, target_date: str = None) -> dict:
    """
    Convert MinimalRiskSystem predictions to signal format for email template
    Maintains compatibility with existing email system
    """
    from datetime import datetime

    trader_signals = []
    alerts = []

    for trader_id, pred_data in predictions.items():
        reduction_pct = pred_data.get('reduction_pct', 0)
        reasons = pred_data.get('reasons', ['Model prediction'])
        confidence = pred_data.get('confidence', 'model-based')

        # Map reduction percentage to risk levels
        if reduction_pct > 50:
            risk_level = 'high'
        elif reduction_pct > 20:
            risk_level = 'medium'
        elif reduction_pct > 0:
            risk_level = 'low'
        else:
            risk_level = 'neutral'

        # Convert reduction to position recommendation
        position_size = max(0, 1.0 - reduction_pct/100)

        signal = {
            'trader_id': str(trader_id),
            'trader_name': f'ID {trader_id}',  # Basic name
            'trader_label': f'{trader_id}',
            'risk_level': risk_level,
            'position_size': position_size,
            'loss_probability': reduction_pct / 100.0,  # Convert to 0-1 scale
            'model_confidence': 0.8 if confidence == 'model-based' else 0.6,
            'risk_score': reduction_pct / 100.0,
            'last_trade_date': 'N/A',
            'last_trading_day_pnl': 0,
            'sharpe_30d': 0,
            'avg_daily_pnl': 0,
            'avg_winning_trade': 0,
            'avg_losing_trade': 0,
            'highest_pnl': 0,
            'lowest_pnl': 0,
            'bat_30d': 0,
            'bat_all_time': 0,
            'wl_ratio_30d': 0,
            'wl_ratio_all_time': 0,
            'volatility': 0,
            'warning_signals': reasons,
            'optimal_thresholds': {},
            # Default heatmap colors (neutral)
            'position_heatmap': {'bg': '#F5F5F5', 'text': '#000000', 'class': 'neutral'},
            'loss_prob_heatmap': {'bg': '#F5F5F5', 'text': '#000000', 'class': 'neutral'},
            'last_day_pnl_heatmap': {'bg': '#F5F5F5', 'text': '#000000', 'class': 'neutral'},
            'sharpe_heatmap': {'bg': '#F5F5F5', 'text': '#000000', 'class': 'neutral'},
            'avg_daily_pnl_heatmap': {'bg': '#F5F5F5', 'text': '#000000', 'class': 'neutral'},
            'avg_winning_heatmap': {'bg': '#F5F5F5', 'text': '#000000', 'class': 'neutral'},
            'avg_losing_heatmap': {'bg': '#F5F5F5', 'text': '#000000', 'class': 'neutral'},
            'highest_pnl_heatmap': {'bg': '#F5F5F5', 'text': '#000000', 'class': 'neutral'},
            'lowest_pnl_heatmap': {'bg': '#F5F5F5', 'text': '#000000', 'class': 'neutral'},
            'bat_heatmap': {'bg': '#F5F5F5', 'text': '#000000', 'class': 'neutral'},
            'wl_ratio_heatmap': {'bg': '#F5F5F5', 'text': '#000000', 'class': 'neutral'}
        }

        trader_signals.append(signal)

        # Generate alerts for high risk
        if risk_level in ['high', 'medium']:
            alerts.append({
                'trader_id': str(trader_id),
                'trader_label': f'Trader {trader_id}',
                'message': f'{risk_level.upper()} RISK: Recommend {reduction_pct:.0f}% position reduction',
                'risk_level': risk_level,
                'risk_score': reduction_pct / 100.0
            })

    # Sort by risk level
    risk_order = {'high': 0, 'medium': 1, 'low': 2, 'neutral': 3}
    trader_signals.sort(key=lambda x: (risk_order.get(x['risk_level'], 3), -x['loss_probability']))

    return {
        'date': target_date or datetime.now().strftime('%Y-%m-%d'),
        'trader_signals': trader_signals,
        'alerts': alerts,
        'summary_stats': {
            'avg_position_size': sum(s['position_size'] for s in trader_signals) / len(trader_signals) if trader_signals else 1.0,
            'max_position_size': max((s['position_size'] for s in trader_signals), default=1.0),
            'avg_loss_prob': sum(s['loss_probability'] for s in trader_signals) / len(trader_signals) if trader_signals else 0.0,
            'max_loss_prob': max((s['loss_probability'] for s in trader_signals), default=0.0),
            'total_warning_signals': sum(len(s['warning_signals']) for s in trader_signals),
            'using_optimal_thresholds': True,
            'intervention_based': True,
            'causal_impact_model': False,  # We use pooled model, not causal impact
            'weighted_formula_enabled': False,
            'temporal_alignment': True,  # Key differentiator
            'pooled_model': True,        # Key differentiator
            'alpha': None,
            'beta': None,
            'risk_classification_levels': 4
        }
    }


def main():
    """Main function to generate and send daily signals."""
    parser = argparse.ArgumentParser(description='Send daily risk signals email')
    parser.add_argument(
        '--email',
        action='append',
        help='Email address to send report to (can be used multiple times)'
    )
    parser.add_argument(
        '--save-only',
        action='store_true',
        help='Only save HTML file without sending email'
    )
    parser.add_argument(
        '--date',
        type=str,
        default=None,
        help='Specific date for report (YYYY-MM-DD), defaults to today'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/main_config.yaml',
        help='Path to configuration file'
    )

    args = parser.parse_args()

    try:
        # Initialize temporally-aligned pooled model system
        logger.info("Initializing minimal risk system (pooled model)...")
        risk_system = MinimalRiskSystem()

        # Generate predictions using pooled model (prevents temporal leakage)
        logger.info("Generating daily signals with temporal alignment...")
        predictions = risk_system.run_daily()

        # Convert to signal data format for email
        signal_data = convert_predictions_to_signals(predictions, args.date)

        # Initialize email service (don't require credentials if save-only)
        email_service = EmailService(require_credentials=not args.save_only)

        # Prepare output filename
        date_str = signal_data['date']
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"inference/outputs/risk_signals_{date_str}_{timestamp}.html"

        # Determine recipients
        recipients = args.email if args.email else email_service.default_recipients

        if args.save_only:
            # Just save the file
            logger.info("Saving signals to file only (--save-only flag)...")
            html_content = email_service.render_daily_signals(signal_data)

            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as f:
                f.write(html_content)

            logger.info(f"Risk signals saved to: {output_file}")

        elif recipients:
            # Send email
            logger.info(f"Sending signals to: {', '.join(recipients)}")

            success = email_service.send_daily_signals(
                signal_data=signal_data,
                to_emails=recipients,
                save_to_file=output_file
            )

            if success:
                logger.info("Daily signals sent successfully!")
            else:
                logger.error("Failed to send daily signals email")
                sys.exit(1)

        else:
            # No recipients and not save-only
            logger.warning("No email recipients found. Set EMAIL_RECIPIENTS in .env or use --email flag.")
            logger.info("Saving signals to file instead...")

            html_content = email_service.render_daily_signals(signal_data)
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as f:
                f.write(html_content)

            logger.info(f"Risk signals saved to: {output_file}")

        # Print summary
        print("\n" + "="*60)
        print("DAILY RISK SIGNALS SUMMARY")
        print("="*60)
        print(f"Date: {signal_data['date']}")
        print(f"Total Traders: {len(signal_data['trader_signals'])}")

        position_counts = {'reduce': 0, 'conservative': 0, 'normal': 0, 'aggressive': 0}
        for signal in signal_data['trader_signals']:
            position_level = signal.get('position_level', 'normal')
            # Map old risk levels to position levels for backward compatibility
            if position_level in ['high', 'medium', 'low', 'neutral']:
                position_map = {'high': 'reduce', 'medium': 'conservative', 'low': 'normal', 'neutral': 'normal'}
                position_level = position_map.get(position_level, 'normal')
            if position_level in position_counts:
                position_counts[position_level] += 1

        print(f"Reduce Position: {position_counts['reduce']}")
        print(f"Conservative: {position_counts['conservative']}")
        print(f"Normal: {position_counts['normal']}")
        print(f"Aggressive: {position_counts['aggressive']}")
        # Position counts already printed above
        print(f"Critical Alerts: {len(signal_data['alerts'])}")

        if signal_data['alerts']:
            print("\nCRITICAL ALERTS:")
            for alert in signal_data['alerts']:
                trader_label = alert.get('trader_label')
                if not trader_label:
                    trader_label = f"Trader {alert['trader_id']}"
                print(f"  - {trader_label}: {alert['message']}")

        print("="*60)

    except Exception as e:
        logger.error(f"Error generating/sending signals: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
