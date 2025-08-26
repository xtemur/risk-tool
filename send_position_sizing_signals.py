#!/usr/bin/env python3
"""
Send daily position sizing signals email report.

Usage:
    python send_position_sizing_signals.py [--email recipient@example.com] [--save-only]

This script generates position sizing recommendations (0% to 150%) for each trader
based on the enhanced models trained with position sizing targets.
"""

import argparse
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
from pathlib import Path
import pickle

# Add directories to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils import load_config
from src.enhanced_data_processing import create_enhanced_trader_day_panel
from src.enhanced_feature_engineering import build_enhanced_features
from inference.email_service import EmailService

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PositionSizingSignalGenerator:
    """Generate position sizing signals using enhanced models."""

    def __init__(self, config_path: str = 'configs/main_config.yaml'):
        """Initialize the position sizing signal generator."""
        self.config = load_config(config_path)
        self.models_dir = Path('models/trader_specific')
        self.active_traders = self.config.get('active_traders', [])

    def load_trader_models(self, trader_id: int) -> tuple:
        """Load risk and position models for a trader."""
        trader_dir = self.models_dir / str(trader_id)

        # Try to load position sizing models first (new format)
        position_model_path = trader_dir / 'enhanced_position_model.pkl'
        risk_model_path = trader_dir / 'enhanced_risk_model.pkl'

        # Fallback to legacy models if new ones don't exist
        if not position_model_path.exists():
            position_model_path = trader_dir / 'enhanced_var_model.pkl'
        if not risk_model_path.exists():
            risk_model_path = trader_dir / 'enhanced_loss_model.pkl'

        try:
            with open(position_model_path, 'rb') as f:
                position_model = pickle.load(f)
            with open(risk_model_path, 'rb') as f:
                risk_model = pickle.load(f)

            return risk_model, position_model

        except FileNotFoundError as e:
            logger.warning(f"Models not found for trader {trader_id}: {e}")
            return None, None

    def prepare_latest_features(self, target_date: str = None) -> pd.DataFrame:
        """Prepare the latest features for prediction."""
        if target_date is None:
            target_date = datetime.now().strftime('%Y-%m-%d')

        logger.info(f"Preparing features for date: {target_date}")

        # Create enhanced panel with latest data
        panel_df = create_enhanced_trader_day_panel(self.config)

        # Build enhanced features
        features_df = build_enhanced_features(panel_df, self.config)

        # Filter to recent data (last 30 days for context)
        end_date = pd.to_datetime(target_date)
        start_date = end_date - timedelta(days=30)

        recent_data = features_df[
            (features_df['trade_date'] >= start_date) &
            (features_df['trade_date'] <= end_date)
        ].copy()

        return recent_data

    def generate_position_sizing_signals(self, target_date: str = None) -> dict:
        """Generate position sizing signals for all active traders."""
        if target_date is None:
            target_date = datetime.now().strftime('%Y-%m-%d')

        logger.info(f"Generating position sizing signals for {target_date}")

        # Prepare features
        features_df = self.prepare_latest_features(target_date)

        # Get feature columns (exclude metadata and targets)
        target_cols = [col for col in features_df.columns if col.startswith('target_')]
        feature_cols = [col for col in features_df.columns
                       if col not in ['account_id', 'trade_date'] + target_cols]

        trader_signals = []
        alerts = []

        for trader_id in self.active_traders:
            try:
                # Load trader models
                risk_model, position_model = self.load_trader_models(trader_id)

                if risk_model is None or position_model is None:
                    logger.warning(f"Skipping trader {trader_id} - models not available")
                    continue

                # Get latest data for this trader
                trader_data = features_df[features_df['account_id'] == trader_id].copy()

                if trader_data.empty:
                    logger.warning(f"No recent data for trader {trader_id}")
                    continue

                # Get the most recent row
                latest_row = trader_data.iloc[-1]
                latest_features = latest_row[feature_cols].values.reshape(1, -1)

                # Handle missing values
                latest_features = np.nan_to_num(latest_features, nan=0.0,
                                              posinf=0.0, neginf=0.0)

                # Generate predictions
                risk_prob = risk_model.predict_proba(latest_features)[0, 1]  # Probability of high risk
                position_size = position_model.predict(latest_features)[0]   # Optimal position size

                # Determine position sizing recommendation
                position_pct = position_size * 100

                if position_size < 0.3:
                    sizing_level = "REDUCE"
                    sizing_color = "red"
                    sizing_message = f"Significant risk detected. Reduce position to {position_pct:.1f}%"
                elif position_size < 0.7:
                    sizing_level = "CONSERVATIVE"
                    sizing_color = "orange"
                    sizing_message = f"Elevated risk. Conservative position at {position_pct:.1f}%"
                elif position_size > 1.2:
                    sizing_level = "AGGRESSIVE"
                    sizing_color = "green"
                    sizing_message = f"Favorable conditions. Increase position to {position_pct:.1f}%"
                else:
                    sizing_level = "NORMAL"
                    sizing_color = "blue"
                    sizing_message = f"Normal position sizing at {position_pct:.1f}%"

                # Create trader signal
                signal = {
                    'trader_id': trader_id,
                    'trader_label': f"Trader {trader_id}",
                    'date': target_date,
                    'risk_probability': risk_prob,
                    'position_size': position_size,
                    'position_percentage': position_pct,
                    'sizing_level': sizing_level,
                    'sizing_color': sizing_color,
                    'sizing_message': sizing_message,
                    'last_update': latest_row['trade_date'].strftime('%Y-%m-%d'),
                    'recent_pnl': latest_row.get('daily_pnl', 0),
                    'model_confidence': 1 - abs(position_size - 1.0)  # Higher when closer to 100%
                }

                trader_signals.append(signal)

                # Generate alerts for extreme positions
                if position_size < 0.2:
                    alerts.append({
                        'trader_id': trader_id,
                        'trader_label': f"Trader {trader_id}",
                        'type': 'CRITICAL_RISK',
                        'message': f"CRITICAL: Reduce position to {position_pct:.1f}% immediately",
                        'severity': 'critical'
                    })
                elif position_size > 1.4:
                    alerts.append({
                        'trader_id': trader_id,
                        'trader_label': f"Trader {trader_id}",
                        'type': 'HIGH_OPPORTUNITY',
                        'message': f"Strong opportunity: Consider increasing position to {position_pct:.1f}%",
                        'severity': 'opportunity'
                    })

            except Exception as e:
                logger.error(f"Error processing trader {trader_id}: {str(e)}")
                continue

        # Sort signals by position size (most critical first)
        trader_signals.sort(key=lambda x: x['position_size'])

        return {
            'date': target_date,
            'timestamp': datetime.now().isoformat(),
            'signal_type': 'position_sizing',
            'trader_signals': trader_signals,
            'alerts': alerts,
            'summary': {
                'total_traders': len(trader_signals),
                'reduce_positions': len([s for s in trader_signals if s['sizing_level'] == 'REDUCE']),
                'conservative_positions': len([s for s in trader_signals if s['sizing_level'] == 'CONSERVATIVE']),
                'normal_positions': len([s for s in trader_signals if s['sizing_level'] == 'NORMAL']),
                'aggressive_positions': len([s for s in trader_signals if s['sizing_level'] == 'AGGRESSIVE']),
                'critical_alerts': len([a for a in alerts if a['severity'] == 'critical']),
                'opportunities': len([a for a in alerts if a['severity'] == 'opportunity'])
            }
        }


def create_position_sizing_html_report(signal_data: dict) -> str:
    """Create HTML report for position sizing signals."""

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Daily Position Sizing Signals - {signal_data['date']}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .header {{ text-align: center; margin-bottom: 30px; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 8px; }}
            .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 30px; }}
            .summary-card {{ background: #f8f9fa; padding: 15px; border-radius: 6px; text-align: center; border-left: 4px solid #007bff; }}
            .summary-card .number {{ font-size: 2em; font-weight: bold; color: #007bff; }}
            .summary-card .label {{ color: #6c757d; margin-top: 5px; }}
            .signals-grid {{ display: grid; gap: 15px; }}
            .trader-card {{ border: 1px solid #dee2e6; border-radius: 6px; padding: 15px; background: white; }}
            .trader-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }}
            .trader-name {{ font-weight: bold; font-size: 1.1em; }}
            .position-badge {{ padding: 4px 12px; border-radius: 20px; color: white; font-weight: bold; text-transform: uppercase; font-size: 0.8em; }}
            .position-details {{ margin-top: 10px; }}
            .position-bar {{ width: 100%; height: 20px; background: #e9ecef; border-radius: 10px; margin: 10px 0; position: relative; }}
            .position-fill {{ height: 100%; border-radius: 10px; transition: width 0.3s ease; }}
            .position-text {{ position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); color: white; font-weight: bold; font-size: 0.8em; }}
            .alerts {{ margin-top: 30px; }}
            .alert {{ padding: 15px; margin-bottom: 10px; border-radius: 6px; border-left: 4px solid; }}
            .alert-critical {{ background: #f8d7da; border-color: #dc3545; color: #721c24; }}
            .alert-opportunity {{ background: #d1ecf1; border-color: #17a2b8; color: #0c5460; }}
            .reduce {{ background-color: #dc3545; }}
            .conservative {{ background-color: #fd7e14; }}
            .normal {{ background-color: #28a745; }}
            .aggressive {{ background-color: #007bff; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Daily Position Sizing Signals</h1>
                <p>Date: {signal_data['date']} | Generated: {datetime.fromisoformat(signal_data['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>

            <div class="summary">
                <div class="summary-card">
                    <div class="number">{signal_data['summary']['total_traders']}</div>
                    <div class="label">Total Traders</div>
                </div>
                <div class="summary-card">
                    <div class="number">{signal_data['summary']['reduce_positions']}</div>
                    <div class="label">Reduce Positions</div>
                </div>
                <div class="summary-card">
                    <div class="number">{signal_data['summary']['aggressive_positions']}</div>
                    <div class="label">Increase Positions</div>
                </div>
                <div class="summary-card">
                    <div class="number">{signal_data['summary']['critical_alerts']}</div>
                    <div class="label">Critical Alerts</div>
                </div>
            </div>

            <div class="signals-grid">
    """

    # Add trader signals
    for signal in signal_data['trader_signals']:
        position_pct = signal['position_percentage']
        fill_width = min(position_pct, 150)  # Cap display at 150%

        html_content += f"""
                <div class="trader-card">
                    <div class="trader-header">
                        <span class="trader-name">{signal['trader_label']}</span>
                        <span class="position-badge {signal['sizing_level'].lower()}">{signal['sizing_level']}</span>
                    </div>
                    <div class="position-details">
                        <strong>Recommended Position: {position_pct:.1f}%</strong>
                        <div class="position-bar">
                            <div class="position-fill {signal['sizing_level'].lower()}" style="width: {fill_width}%;"></div>
                            <div class="position-text">{position_pct:.1f}%</div>
                        </div>
                        <p>{signal['sizing_message']}</p>
                        <small>Risk Probability: {signal['risk_probability']:.1%} | Last Update: {signal['last_update']} | Recent P&L: ${signal['recent_pnl']:,.0f}</small>
                    </div>
                </div>
        """

    # Add alerts section
    if signal_data['alerts']:
        html_content += """
            </div>
            <div class="alerts">
                <h2>Critical Alerts</h2>
        """

        for alert in signal_data['alerts']:
            alert_class = 'alert-critical' if alert['severity'] == 'critical' else 'alert-opportunity'
            html_content += f"""
                <div class="alert {alert_class}">
                    <strong>{alert['trader_label']}</strong>: {alert['message']}
                </div>
            """

        html_content += "</div>"
    else:
        html_content += "</div>"

    html_content += """
        </div>
    </body>
    </html>
    """

    return html_content


def main():
    """Main function to generate and send position sizing signals."""
    parser = argparse.ArgumentParser(description='Send daily position sizing signals')
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
        # Initialize position sizing signal generator
        logger.info("Initializing position sizing signal generator...")
        generator = PositionSizingSignalGenerator(args.config)

        # Generate position sizing signals
        logger.info("Generating position sizing signals...")
        signal_data = generator.generate_position_sizing_signals(args.date)

        # Create HTML report
        html_content = create_position_sizing_html_report(signal_data)

        # Prepare output filename
        date_str = signal_data['date']
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"inference/outputs/position_sizing_signals_{date_str}_{timestamp}.html"

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Save HTML file
        with open(output_file, 'w') as f:
            f.write(html_content)
        logger.info(f"Position sizing signals saved to: {output_file}")

        # Handle email sending
        if not args.save_only and args.email:
            try:
                email_service = EmailService(require_credentials=True)

                # Send email with position sizing report
                success = email_service.send_email(
                    to_emails=args.email,
                    subject=f"Daily Position Sizing Signals - {date_str}",
                    html_content=html_content
                )

                if success:
                    logger.info(f"Position sizing signals sent to: {', '.join(args.email)}")
                else:
                    logger.error("Failed to send position sizing signals email")

            except Exception as e:
                logger.warning(f"Email sending failed: {e}")
                logger.info("Report saved to file only")

        # Print summary
        print("\n" + "="*70)
        print("DAILY POSITION SIZING SIGNALS SUMMARY")
        print("="*70)
        print(f"Date: {signal_data['date']}")
        print(f"Total Traders: {signal_data['summary']['total_traders']}")
        print(f"Reduce Positions: {signal_data['summary']['reduce_positions']}")
        print(f"Conservative Positions: {signal_data['summary']['conservative_positions']}")
        print(f"Normal Positions: {signal_data['summary']['normal_positions']}")
        print(f"Aggressive Positions: {signal_data['summary']['aggressive_positions']}")
        print(f"Critical Alerts: {signal_data['summary']['critical_alerts']}")
        print(f"Opportunities: {signal_data['summary']['opportunities']}")

        if signal_data['alerts']:
            print(f"\nALERTS:")
            for alert in signal_data['alerts']:
                print(f"  - {alert['trader_label']}: {alert['message']}")

        print(f"\nReport saved to: {output_file}")
        print("="*70)

    except Exception as e:
        logger.error(f"Error generating position sizing signals: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
