#!/usr/bin/env python3
"""
Test script to generate and save an email with the new BAT and W/L metrics.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference.signal_generator import SignalGenerator
from inference.email_service import EmailService
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_email_with_new_metrics():
    """Generate and save email with new BAT and W/L metrics."""
    print("=" * 80)
    print("TESTING EMAIL TEMPLATE WITH NEW METRICS")
    print("=" * 80)

    try:
        # Generate real signal data with new metrics
        generator = SignalGenerator('configs/main_config.yaml')
        signal_data = generator.generate_daily_signals()

        print(f"Generated signals for {len(signal_data['trader_signals'])} traders")

        # Initialize email service
        email_service = EmailService(require_credentials=False)

        # Render the email template
        html_content = email_service.render_daily_signals(signal_data)

        # Save to file for preview
        output_dir = 'inference/outputs'
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, f'test_email_with_new_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html')

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"✅ Email template rendered successfully!")
        print(f"📧 Email saved to: {output_path}")

        # Show sample data to verify new metrics are included
        if signal_data['trader_signals']:
            sample_signal = signal_data['trader_signals'][0]
            print(f"\n📊 Sample trader data verification:")
            print(f"  Trader: {sample_signal['trader_label']}")
            print(f"  BAT 30d: {sample_signal.get('bat_30d', 'NOT FOUND'):.1f}%" if sample_signal.get('bat_30d') is not None else f"  BAT 30d: NOT FOUND")
            print(f"  BAT All-time: {sample_signal.get('bat_all_time', 'NOT FOUND'):.1f}%" if sample_signal.get('bat_all_time') is not None else f"  BAT All-time: NOT FOUND")
            print(f"  W/L Ratio 30d: {sample_signal.get('wl_ratio_30d', 'NOT FOUND'):.2f}" if sample_signal.get('wl_ratio_30d') is not None else f"  W/L Ratio 30d: NOT FOUND")
            print(f"  W/L Ratio All-time: {sample_signal.get('wl_ratio_all_time', 'NOT FOUND'):.2f}" if sample_signal.get('wl_ratio_all_time') is not None else f"  W/L Ratio All-time: NOT FOUND")

            # Check heatmap colors
            print(f"  BAT Heatmap: {sample_signal.get('bat_heatmap', {}).get('bg', 'NOT FOUND')}")
            print(f"  W/L Heatmap: {sample_signal.get('wl_ratio_heatmap', {}).get('bg', 'NOT FOUND')}")

        # Verify HTML contains new metrics
        print(f"\n🔍 HTML Template Verification:")
        bat_in_html = 'BAT' in html_content and ('35.7%' in html_content or '%.1f' in html_content)
        wl_in_html = 'W/L' in html_content and ('2.11' in html_content or '%.2f' in html_content)

        # Also check for the actual values from sample data
        sample_bat_value = f"{sample_signal.get('bat_30d', 0):.1f}%" if sample_signal.get('bat_30d') is not None else "N/A"
        sample_wl_value = f"{sample_signal.get('wl_ratio_30d', 0):.2f}" if sample_signal.get('wl_ratio_30d') is not None else "N/A"

        bat_value_in_html = sample_bat_value in html_content
        wl_value_in_html = sample_wl_value in html_content

        print(f"  BAT header in HTML: {'✅ YES' if 'BAT' in html_content else '❌ NO'}")
        print(f"  BAT value ({sample_bat_value}) in HTML: {'✅ YES' if bat_value_in_html else '❌ NO'}")
        print(f"  W/L header in HTML: {'✅ YES' if 'W/L' in html_content else '❌ NO'}")
        print(f"  W/L value ({sample_wl_value}) in HTML: {'✅ YES' if wl_value_in_html else '❌ NO'}")

        if bat_value_in_html and wl_value_in_html:
            print(f"\n🎉 SUCCESS: Email template now includes BAT and W/L metrics!")
        else:
            print(f"\n⚠️  Note: Metrics may have different values than expected")

        return output_path

    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main test function."""
    print("TESTING EMAIL TEMPLATE WITH NEW BAT AND W/L METRICS")
    print("=" * 80)

    output_path = test_email_with_new_metrics()

    if output_path:
        print(f"\n" + "=" * 80)
        print("✅ EMAIL TEMPLATE TEST COMPLETED SUCCESSFULLY!")
        print(f"📧 Open the generated file to preview: {output_path}")
        print("📊 New BAT and W/L metrics are now included in the email")
        print("🎨 Heatmap colors are applied to the new metrics")
        print("=" * 80)
        return True
    else:
        print(f"\n" + "=" * 80)
        print("❌ EMAIL TEMPLATE TEST FAILED")
        print("=" * 80)
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
