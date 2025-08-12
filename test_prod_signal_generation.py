#!/usr/bin/env python3
"""
Test production model signal generation and send email notification.
"""

import sys
import os
from datetime import datetime
import logging

# Add the project directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference.signal_generator import SignalGenerator
from inference.email_service import EmailService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_production_signal_generation():
    """Test signal generation with production models."""

    print("="*80)
    print("TESTING PRODUCTION MODEL SIGNAL GENERATION")
    print("="*80)

    try:
        # Initialize signal generator (now using production models)
        logger.info("Initializing SignalGenerator with production models...")
        generator = SignalGenerator()

        # Generate signals with enhanced weighted formula
        logger.info("Generating daily signals with production models...")
        signals = generator.generate_daily_signals(
            use_weighted_formula=True,
            alpha=0.6,
            beta=0.4
        )

        # Print summary
        print(f"\n✅ Successfully generated signals for {len(signals['trader_signals'])} traders")
        print(f"📊 Risk Level Distribution:")

        # Count risk levels
        risk_counts = {}
        for signal in signals['trader_signals']:
            level = signal['risk_level']
            risk_counts[level] = risk_counts.get(level, 0) + 1

        for level, count in risk_counts.items():
            print(f"   - {level}: {count} traders")

        print(f"🚨 Generated {len(signals['alerts'])} alerts")

        # Show model information
        stats = signals['summary_stats']
        print(f"\n📈 Model Configuration:")
        print(f"   - Using Production Models: ✅")
        print(f"   - Weighted Formula: {'✅' if stats['weighted_formula_enabled'] else '❌'}")
        print(f"   - Alpha (VaR weight): {stats['alpha']}")
        print(f"   - Beta (Loss Prob weight): {stats['beta']}")
        print(f"   - Classification Levels: {stats['risk_classification_levels']}")

        # Show top risk traders
        print(f"\n⚠️  Top Risk Traders:")
        for i, signal in enumerate(signals['trader_signals'][:5]):
            risk_score = signal.get('risk_score', 'N/A')
            if risk_score != 'N/A':
                risk_score = f"{risk_score:.3f}"
            print(f"   {i+1}. {signal['trader_label']}: {signal['risk_level']} (Risk Score: {risk_score})")

        # Test email sending
        print(f"\n📧 Sending email notification...")
        email_success = generator.send_email_signal(
            signals,
            to_emails=["temurbekkhujaev@gmail.com"]
        )

        if email_success:
            print("✅ Email sent successfully to temurbekkhujaev@gmail.com")
        else:
            print("❌ Failed to send email")

        # Save results for analysis
        import json
        from pathlib import Path

        results_dir = Path("results/prod_signal_generation")
        results_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save signal data
        with open(results_dir / f"prod_signals_{timestamp}.json", 'w') as f:
            json.dump(signals, f, indent=2, default=str)

        print(f"\n💾 Results saved to: {results_dir}")

        return {
            'success': True,
            'signals': signals,
            'email_sent': email_success,
            'timestamp': timestamp
        }

    except Exception as e:
        logger.error(f"Error during production signal generation: {e}")
        print(f"❌ Error: {e}")
        return {
            'success': False,
            'error': str(e)
        }

def send_deployment_notification():
    """Send notification about production model deployment."""

    try:
        email_service = EmailService()

        # Create deployment notification
        subject = "🚀 Production Models Deployed - Risk Tool Update"

        html_body = f"""
        <html>
        <body>
        <h2>🚀 Production Models Successfully Deployed</h2>

        <h3>📋 Deployment Summary</h3>
        <ul>
            <li><strong>Deployment Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</li>
            <li><strong>Model Type:</strong> Production Models (_tuned_validated_prod)</li>
            <li><strong>Signal Generation:</strong> Updated to use production models</li>
            <li><strong>Email Notifications:</strong> Active</li>
        </ul>

        <h3>🎯 Performance Improvements</h3>
        <ul>
            <li><strong>Net Benefit:</strong> +60% vs original models ($276.6K vs $174.7K)</li>
            <li><strong>Overall Improvement:</strong> 162.15% (vs 102.44% for original)</li>
            <li><strong>Intervention Rate:</strong> 37.6% (vs 16.8% for original)</li>
            <li><strong>Positive Trader Coverage:</strong> 10/11 traders (vs 8/11 for original)</li>
            <li><strong>Sharpe Improvement:</strong> 1.116 mean (vs 0.624 for original)</li>
        </ul>

        <h3>⚙️ Configuration</h3>
        <ul>
            <li><strong>Risk Formula:</strong> Weighted (α=0.6, β=0.4)</li>
            <li><strong>Optimal Reduction:</strong> 60% PnL reduction</li>
            <li><strong>Classification:</strong> 4-level risk system</li>
            <li><strong>Model Validation:</strong> Completed on test data</li>
        </ul>

        <h3>📊 Next Steps</h3>
        <ul>
            <li>Monitor daily signal generation</li>
            <li>Track real-time performance metrics</li>
            <li>Compare live performance against backtests</li>
            <li>Prepare for 60% reduction implementation</li>
        </ul>

        <p><strong>Status:</strong> ✅ Production models are now live and generating daily signals</p>

        <hr>
        <p><em>This is an automated notification from the Risk Tool system.</em></p>
        </body>
        </html>
        """

        success = email_service.send_email(
            subject=subject,
            html_body=html_body,
            to_emails=["temurbekkhujaev@gmail.com"]
        )

        if success:
            print("✅ Deployment notification sent successfully")
        else:
            print("❌ Failed to send deployment notification")

        return success

    except Exception as e:
        logger.error(f"Error sending deployment notification: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Testing Production Model Signal Generation...")

    # Test production signal generation
    result = test_production_signal_generation()

    if result['success']:
        print("\n🎉 Production model testing completed successfully!")

        # Send deployment notification
        print("\n📧 Sending deployment notification...")
        notification_sent = send_deployment_notification()

        if notification_sent:
            print("✅ All tasks completed successfully!")
        else:
            print("⚠️  Signal generation successful but notification failed")

    else:
        print(f"\n❌ Production model testing failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)
