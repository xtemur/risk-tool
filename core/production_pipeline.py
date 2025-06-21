#!/usr/bin/env python3
"""
Production Risk Management Pipeline
Streamlined version focused on daily signal generation for live trading
"""

import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
import logging

# Add core to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from utils import get_logger, log_system_event, log_error
from data.data_validator import DataValidator
from models.signal_generator import DeploymentReadySignals

class ProductionRiskPipeline:
    """Production pipeline for daily risk signal generation"""

    def __init__(self):
        self.config = get_config()
        self.logger = get_logger(__name__)
        self.start_time = datetime.now()

        # Ensure output directories exist
        Path(self.config['output_dir']).mkdir(parents=True, exist_ok=True)

    def generate_daily_signals(self, target_date=None):
        """
        Generate risk signals for a specific date (default: today)

        Args:
            target_date (str, optional): Date in YYYY-MM-DD format. Defaults to today.

        Returns:
            dict: Signal results for all active traders
        """
        if target_date is None:
            target_date = datetime.now().strftime('%Y-%m-%d')

        print("="*80)
        print(f"PRODUCTION RISK SIGNAL GENERATION - {target_date}")
        print("="*80)

        try:
            # Step 1: Load and validate recent data
            signals = self._load_and_validate_data(target_date)

            # Step 2: Generate signals using trained models
            if signals:
                signals = self._generate_risk_signals(signals, target_date)

            # Step 3: Apply safety checks
            if signals:
                signals = self._apply_safety_checks(signals)

            # Step 4: Output results
            if signals:
                self._output_signals(signals, target_date)

            log_system_event(
                "daily_signals_generated",
                f"Generated signals for {len(signals) if signals else 0} traders",
                {"date": target_date, "trader_count": len(signals) if signals else 0}
            )

            return signals

        except Exception as e:
            log_error("signal_generation_failed", str(e), {"date": target_date})
            print(f"❌ Signal generation failed: {e}")
            return None

    def _load_and_validate_data(self, target_date):
        """Load and validate recent trading data"""
        print("\\n=== LOADING RECENT DATA ===")

        try:
            validator = DataValidator(db_path=self.config['db_path'])

            # Load data for active traders only
            validator.load_and_validate_data(active_only=True)

            # Check data freshness
            latest_date = validator.trades_df['trade_date'].max()
            days_old = (datetime.now() - latest_date).days

            if days_old > 7:
                print(f"⚠️  WARNING: Latest data is {days_old} days old")

            print(f"✓ Loaded {len(validator.trades_df)} trades")
            print(f"✓ Latest data: {latest_date.date()}")
            print(f"✓ Active traders: {validator.trades_df['account_id'].nunique()}")

            # Create daily aggregations
            validator.create_daily_aggregations()

            # Return trader IDs that have sufficient recent data
            recent_cutoff = datetime.strptime(target_date, '%Y-%m-%d') - timedelta(days=30)
            recent_data = validator.daily_df[validator.daily_df['trade_date'] >= recent_cutoff]

            viable_traders = recent_data.groupby('account_id').size()
            viable_traders = viable_traders[viable_traders >= self.config['min_trading_days']].index.tolist()

            print(f"✓ Viable traders for signals: {len(viable_traders)}")

            return viable_traders

        except Exception as e:
            print(f"❌ Data loading failed: {e}")
            return None

    def _generate_risk_signals(self, trader_ids, target_date):
        """Generate risk signals for viable traders"""
        print("\\n=== GENERATING RISK SIGNALS ===")

        try:
            # Check if trained models exist
            models_path = Path(self.config['output_dir']) / 'trained_models.pkl'
            if not models_path.exists():
                print("❌ No trained models found. Run full pipeline first.")
                return None

            deployment = DeploymentReadySignals()

            # Generate signals for each trader
            signals = {}
            for trader_id in trader_ids:
                try:
                    signal = deployment.generate_trader_signal(trader_id, target_date)
                    if signal is not None:
                        signals[trader_id] = signal
                        print(f"✓ Generated signal for trader {trader_id}: {signal['risk_level']}")
                except Exception as e:
                    print(f"⚠️  Failed to generate signal for trader {trader_id}: {e}")
                    continue

            print(f"✓ Generated signals for {len(signals)} traders")
            return signals

        except Exception as e:
            print(f"❌ Signal generation failed: {e}")
            return None

    def _apply_safety_checks(self, signals):
        """Apply safety checks to generated signals"""
        print("\\n=== APPLYING SAFETY CHECKS ===")

        try:
            # Check signal distribution
            risk_counts = {}
            for signal in signals.values():
                risk_level = signal['risk_level']
                risk_counts[risk_level] = risk_counts.get(risk_level, 0) + 1

            total_signals = len(signals)
            high_risk_pct = risk_counts.get('HIGH', 0) / total_signals if total_signals > 0 else 0

            # Safety check: Too many high risk signals
            if high_risk_pct > 0.5:
                print(f"⚠️  WARNING: {high_risk_pct:.1%} traders flagged as high risk")
                print("   Consider reviewing model performance")

            # Safety check: All traders same signal
            if len(risk_counts) == 1:
                print("⚠️  WARNING: All traders have same risk level")
                print("   Model may not be functioning correctly")

            # Log signal distribution
            print("Signal distribution:")
            for level, count in risk_counts.items():
                pct = count / total_signals * 100 if total_signals > 0 else 0
                print(f"  {level}: {count} traders ({pct:.1f}%)")

            print("✓ Safety checks completed")
            return signals

        except Exception as e:
            print(f"❌ Safety checks failed: {e}")
            return signals

    def _output_signals(self, signals, target_date):
        """Output signals to files and logs"""
        print("\\n=== OUTPUTTING SIGNALS ===")

        try:
            # Save to JSON file
            import json
            output_file = Path(self.config['output_dir']) / f'daily_signals_{target_date}.json'

            # Prepare output data
            output_data = {
                'generated_at': datetime.now().isoformat(),
                'target_date': target_date,
                'signal_count': len(signals),
                'signals': signals
            }

            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)

            print(f"✓ Saved signals to {output_file}")

            # Create trader summary report
            summary_file = Path(self.config['output_dir']) / f'signal_summary_{target_date}.txt'

            with open(summary_file, 'w') as f:
                f.write(f"DAILY RISK SIGNALS - {target_date}\\n")
                f.write("="*50 + "\\n\\n")

                # Sort by risk level
                high_risk = [tid for tid, s in signals.items() if s['risk_level'] == 'HIGH']
                neutral_risk = [tid for tid, s in signals.items() if s['risk_level'] == 'NEUTRAL']
                low_risk = [tid for tid, s in signals.items() if s['risk_level'] == 'LOW']

                f.write(f"HIGH RISK TRADERS ({len(high_risk)}):\\n")
                for trader_id in high_risk:
                    signal = signals[trader_id]
                    f.write(f"  Trader {trader_id}: {signal['confidence']:.1%} confidence\\n")

                f.write(f"\\nNEUTRAL RISK TRADERS ({len(neutral_risk)}):\\n")
                for trader_id in neutral_risk:
                    signal = signals[trader_id]
                    f.write(f"  Trader {trader_id}: {signal['confidence']:.1%} confidence\\n")

                f.write(f"\\nLOW RISK TRADERS ({len(low_risk)}):\\n")
                for trader_id in low_risk:
                    signal = signals[trader_id]
                    f.write(f"  Trader {trader_id}: {signal['confidence']:.1%} confidence\\n")

            print(f"✓ Created summary report: {summary_file}")

            # Log individual signals
            for trader_id, signal in signals.items():
                from utils import log_signal_generation
                log_signal_generation(
                    trader_id=str(trader_id),
                    signal=signal['risk_level'],
                    confidence=signal['confidence']
                )

            print("✓ Signal output completed")

        except Exception as e:
            print(f"❌ Signal output failed: {e}")

    def validate_model_performance(self):
        """Validate that models are performing within acceptable ranges"""
        print("\\n=== MODEL PERFORMANCE VALIDATION ===")

        try:
            # Check if recent backtest results exist
            backtest_file = Path(self.config['output_dir']) / 'latest_backtest.json'

            if not backtest_file.exists():
                print("⚠️  No recent backtest results found")
                return False

            import json
            with open(backtest_file, 'r') as f:
                backtest_data = json.load(f)

            # Check model accuracy
            avg_accuracy = backtest_data.get('average_accuracy', 0)
            if avg_accuracy < 0.5:
                print(f"❌ Model accuracy too low: {avg_accuracy:.1%}")
                return False

            print(f"✓ Model accuracy: {avg_accuracy:.1%}")

            # Check signal correlation with outcomes
            signal_accuracy = backtest_data.get('signal_accuracy', 0)
            if signal_accuracy < 0.5:
                print(f"❌ Signal accuracy too low: {signal_accuracy:.1%}")
                return False

            print(f"✓ Signal accuracy: {signal_accuracy:.1%}")
            print("✓ Model performance validation passed")

            return True

        except Exception as e:
            print(f"❌ Model validation failed: {e}")
            return False

def main():
    """Main entry point for production signal generation"""
    import argparse

    parser = argparse.ArgumentParser(description='Generate daily risk signals')
    parser.add_argument('--date', type=str, help='Target date (YYYY-MM-DD)', default=None)
    parser.add_argument('--validate', action='store_true', help='Validate model performance first')

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = ProductionRiskPipeline()

    # Validate models if requested
    if args.validate:
        if not pipeline.validate_model_performance():
            print("❌ Model validation failed. Signals may be unreliable.")
            return 1

    # Generate signals
    signals = pipeline.generate_daily_signals(args.date)

    if signals:
        print(f"\\n✅ Successfully generated {len(signals)} signals")
        return 0
    else:
        print("\\n❌ Signal generation failed")
        return 1

if __name__ == "__main__":
    exit(main())
