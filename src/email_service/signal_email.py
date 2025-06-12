"""
Signal Email Service

Specialized email service for sending trading signals and predictions.
"""

import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import logging
from jinja2 import Environment, FileSystemLoader, Template

from .email_sender import EmailSender
from .email_config import EmailConfig

logger = logging.getLogger(__name__)


class SignalEmailService:
    """
    Service for sending trading signal emails with predictions
    """

    def __init__(self, config: Optional[EmailConfig] = None):
        """
        Initialize signal email service

        Args:
            config: Email configuration object
        """
        self.config = config or EmailConfig.from_env()
        self.email_sender = EmailSender(self.config)

        # Setup Jinja2 environment for templates
        template_dir = Path(self.config.template_dir)
        if not template_dir.exists():
            template_dir.mkdir(parents=True, exist_ok=True)
            logger.warning(f"Template directory created: {template_dir}")

        self.jinja_env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=True
        )

    def send_signal_email(self,
                         predictions: Dict[str, Any],
                         to_emails: Optional[Union[str, List[str]]] = None,
                         subject: Optional[str] = None,
                         include_performance: bool = True,
                         include_market_context: bool = True,
                         attachment_paths: Optional[List[str]] = None) -> bool:
        """
        Send trading signal email with predictions

        Args:
            predictions: Dictionary containing prediction data
            to_emails: Recipient email addresses (uses config default if None)
            subject: Email subject (auto-generated if None)
            include_performance: Whether to include model performance metrics
            include_market_context: Whether to include market context
            attachment_paths: Optional file attachments

        Returns:
            True if sent successfully, False otherwise
        """
        try:
            # Use default recipient if not specified
            recipients = to_emails or self.config.email_to

            # Generate subject if not provided
            if not subject:
                today = datetime.now().strftime("%B %d, %Y")
                subject = f"🔮 Trading Signals for {today}"

            # Prepare template data
            template_data = self._prepare_signal_template_data(
                predictions=predictions,
                include_performance=include_performance,
                include_market_context=include_market_context
            )

            # Render HTML content
            html_content = self._render_signal_template(template_data)

            # Generate plain text version
            text_content = self._generate_text_version(template_data)

            # Send email
            success = self.email_sender.send_email(
                to_emails=recipients,
                subject=subject,
                html_content=html_content,
                text_content=text_content,
                attachments=attachment_paths
            )

            if success:
                logger.info(f"Signal email sent successfully to {recipients}")
            else:
                logger.error("Failed to send signal email")

            return success

        except Exception as e:
            logger.error(f"Error sending signal email: {e}")
            return False

    def _prepare_signal_template_data(self,
                                    predictions: Dict[str, Any],
                                    include_performance: bool = True,
                                    include_market_context: bool = True) -> Dict[str, Any]:
        """
        Prepare data for signal email template

        Args:
            predictions: Prediction data
            include_performance: Include performance metrics
            include_market_context: Include market context

        Returns:
            Template data dictionary
        """
        now = datetime.now()

        # Base template data
        template_data = {
            'report_date': now.strftime("%A, %B %d, %Y"),
            'generation_timestamp': now.strftime("%Y-%m-%d %H:%M:%S UTC"),
            'support_email': self.config.email_from,
            'dashboard_url': '#',  # TODO: Add dashboard URL from config
        }

        # Process trader predictions
        traders_data = []
        if 'traders' in predictions:
            for trader_id, trader_pred in predictions['traders'].items():
                trader_data = self._process_trader_data(trader_id, trader_pred)
                traders_data.append(trader_data)

        template_data['traders'] = traders_data

        # Portfolio summary
        template_data.update(self._calculate_portfolio_summary(traders_data))

        # Model performance (if requested and available)
        if include_performance and 'model_performance' in predictions:
            template_data['model_performance'] = True
            template_data.update(self._process_performance_data(predictions['model_performance']))
        else:
            template_data['model_performance'] = False

        # Market context (if requested and available)
        if include_market_context and 'market_context' in predictions:
            template_data['market_context'] = self._process_market_context(predictions['market_context'])

        # Generate alerts based on predictions
        template_data['alerts'] = self._generate_alerts(predictions, traders_data)

        return template_data

    def _process_trader_data(self, trader_id: str, trader_pred: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process individual trader prediction data

        Args:
            trader_id: Trader identifier
            trader_pred: Trader prediction data

        Returns:
            Processed trader data for template
        """
        # Extract prediction values
        predicted_pnl = trader_pred.get('predicted_pnl', 0)
        confidence = trader_pred.get('confidence', 0) * 100  # Convert to percentage

        # Determine PnL class for styling
        if predicted_pnl > 50:
            pnl_class = 'positive'
        elif predicted_pnl < -50:
            pnl_class = 'negative'
        else:
            pnl_class = 'neutral'

        # Generate recommendation
        recommendation, action_class = self._generate_recommendation(predicted_pnl, confidence)

        # Format values
        trader_data = {
            'id': trader_id,
            'name': trader_pred.get('name', f'Trader {trader_id}'),
            'predicted_pnl': f"${predicted_pnl:,.2f}",
            'pnl_class': pnl_class,
            'confidence': f"{confidence:.1f}",
            'recommendation': recommendation,
            'action_class': action_class,
        }

        # Add optional fields if available
        if 'recent_performance' in trader_pred:
            recent_perf = trader_pred['recent_performance']
            trader_data['recent_performance'] = f"${recent_perf:,.2f}"
            trader_data['recent_performance_class'] = 'positive' if recent_perf > 0 else 'negative'

        if 'risk_notes' in trader_pred:
            trader_data['risk_notes'] = trader_pred['risk_notes']

        return trader_data

    def _generate_recommendation(self, predicted_pnl: float, confidence: float) -> tuple:
        """
        Generate trading recommendation based on prediction and confidence

        Args:
            predicted_pnl: Predicted PnL value
            confidence: Prediction confidence (0-100)

        Returns:
            Tuple of (recommendation_text, css_class)
        """
        if confidence < 30:
            return "🚫 Avoid Trading", "negative"
        elif predicted_pnl > 100 and confidence > 70:
            return "🚀 Strong Buy Signal", "positive"
        elif predicted_pnl > 50 and confidence > 50:
            return "📈 Moderate Buy", "positive"
        elif predicted_pnl > -50 and predicted_pnl <= 50:
            return "⚖️ Neutral", "neutral"
        elif predicted_pnl <= -50 and confidence > 50:
            return "📉 Consider Reducing", "negative"
        else:
            return "⏸️ Hold Position", "neutral"

    def _calculate_portfolio_summary(self, traders_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate portfolio-level summary metrics

        Args:
            traders_data: List of trader data dictionaries

        Returns:
            Portfolio summary data
        """
        if not traders_data:
            return {
                'total_traders': 0,
                'active_signals': 0,
                'avg_confidence': 0,
                'expected_pnl': '$0.00',
                'expected_pnl_class': 'neutral'
            }

        # Extract numeric values
        predicted_pnls = []
        confidences = []
        active_signals = 0

        for trader in traders_data:
            # Parse PnL (remove $ and commas)
            pnl_str = trader['predicted_pnl'].replace('$', '').replace(',', '')
            predicted_pnls.append(float(pnl_str))

            # Parse confidence
            confidences.append(float(trader['confidence']))

            # Count active signals (non-neutral recommendations)
            if trader['action_class'] in ['positive', 'negative']:
                active_signals += 1

        # Calculate aggregates
        total_expected_pnl = sum(predicted_pnls)
        avg_confidence = np.mean(confidences)

        # Determine expected PnL class
        if total_expected_pnl > 100:
            expected_pnl_class = 'positive'
        elif total_expected_pnl < -100:
            expected_pnl_class = 'negative'
        else:
            expected_pnl_class = 'neutral'

        return {
            'total_traders': len(traders_data),
            'active_signals': active_signals,
            'avg_confidence': f"{avg_confidence:.1f}",
            'expected_pnl': f"${total_expected_pnl:,.2f}",
            'expected_pnl_class': expected_pnl_class
        }

    def _process_performance_data(self, performance: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process model performance data for template

        Args:
            performance: Performance metrics dictionary

        Returns:
            Processed performance data
        """
        hit_rate = performance.get('hit_rate', 0) * 100
        sharpe_ratio = performance.get('sharpe_ratio', 0)
        r2_score = performance.get('r2_score', 0)

        return {
            'hit_rate': f"{hit_rate:.1f}",
            'hit_rate_class': 'positive' if hit_rate > 55 else 'negative' if hit_rate < 45 else 'neutral',
            'sharpe_ratio': f"{sharpe_ratio:.2f}",
            'sharpe_class': 'positive' if sharpe_ratio > 1 else 'negative' if sharpe_ratio < 0 else 'neutral',
            'r2_score': f"{r2_score:.3f}",
            'r2_class': 'positive' if r2_score > 0.1 else 'negative' if r2_score < 0 else 'neutral',
            'model_version': performance.get('model_version', 'v1.0')
        }

    def _process_market_context(self, market_context: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Process market context data for template

        Args:
            market_context: Market context dictionary

        Returns:
            List of market context items
        """
        context_items = []

        for key, value in market_context.items():
            # Format key as human-readable label
            label = key.replace('_', ' ').title()

            # Format value appropriately
            if isinstance(value, float):
                if key.endswith('_pct') or 'percentage' in key.lower():
                    formatted_value = f"{value:.2f}%"
                else:
                    formatted_value = f"{value:.2f}"
            else:
                formatted_value = str(value)

            context_items.append({
                'label': label,
                'value': formatted_value
            })

        return context_items

    def _generate_alerts(self, predictions: Dict[str, Any], traders_data: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Generate alerts based on prediction data

        Args:
            predictions: Full prediction data
            traders_data: Processed trader data

        Returns:
            List of alert dictionaries
        """
        alerts = []

        # Check for high-confidence signals
        high_confidence_traders = [t for t in traders_data if float(t['confidence']) > 80]
        if high_confidence_traders:
            alerts.append({
                'type': 'success',
                'title': 'High Confidence Signals',
                'message': f"{len(high_confidence_traders)} trader(s) have predictions with >80% confidence."
            })

        # Check for low confidence across portfolio
        avg_confidence = np.mean([float(t['confidence']) for t in traders_data]) if traders_data else 0
        if avg_confidence < 40:
            alerts.append({
                'type': 'warning',
                'title': 'Low Confidence Warning',
                'message': f"Portfolio average confidence is {avg_confidence:.1f}%. Consider reducing position sizes."
            })

        # Check for model performance issues
        if 'model_performance' in predictions:
            perf = predictions['model_performance']
            if perf.get('hit_rate', 1) < 0.45:  # Less than 45% hit rate
                alerts.append({
                    'type': 'warning',
                    'title': 'Model Performance Alert',
                    'message': "Recent model hit rate is below 45%. Review model predictions carefully."
                })

        # Check for large expected losses
        total_expected = sum([float(t['predicted_pnl'].replace('$', '').replace(',', '')) for t in traders_data])
        if total_expected < -500:
            alerts.append({
                'type': 'warning',
                'title': 'Large Expected Losses',
                'message': f"Portfolio expected PnL is ${total_expected:,.2f}. Consider risk management measures."
            })

        return alerts

    def _render_signal_template(self, template_data: Dict[str, Any]) -> str:
        """
        Render signal email HTML template

        Args:
            template_data: Data for template rendering

        Returns:
            Rendered HTML content
        """
        try:
            template = self.jinja_env.get_template('signal_email.html')
            return template.render(**template_data)
        except Exception as e:
            logger.error(f"Error rendering signal template: {e}")
            # Fallback to simple template
            return self._render_fallback_template(template_data)

    def _render_fallback_template(self, template_data: Dict[str, Any]) -> str:
        """
        Render a simple fallback template if main template fails

        Args:
            template_data: Template data

        Returns:
            Simple HTML content
        """
        html = f"""
        <html>
        <head><title>Trading Signals</title></head>
        <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
            <h1 style="color: #333;">🔮 Trading Signals - {template_data.get('report_date', 'Today')}</h1>

            <h2>Portfolio Summary</h2>
            <ul>
                <li>Total Traders: {template_data.get('total_traders', 0)}</li>
                <li>Active Signals: {template_data.get('active_signals', 0)}</li>
                <li>Expected PnL: {template_data.get('expected_pnl', '$0.00')}</li>
            </ul>

            <h2>Individual Predictions</h2>
        """

        for trader in template_data.get('traders', []):
            html += f"""
            <div style="border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px;">
                <h3>Trader {trader['id']}</h3>
                <p><strong>Expected PnL:</strong> {trader['predicted_pnl']}</p>
                <p><strong>Confidence:</strong> {trader['confidence']}%</p>
                <p><strong>Recommendation:</strong> {trader['recommendation']}</p>
            </div>
            """

        html += """
            <hr>
            <p style="color: #666; font-size: 12px;">
                Generated by Risk Tool AI Trading System<br>
                """ + template_data.get('generation_timestamp', '') + """
            </p>
        </body>
        </html>
        """

        return html

    def _generate_text_version(self, template_data: Dict[str, Any]) -> str:
        """
        Generate plain text version of signal email

        Args:
            template_data: Template data

        Returns:
            Plain text content
        """
        text = f"""
TRADING SIGNALS - {template_data.get('report_date', 'Today')}
{'=' * 50}

PORTFOLIO SUMMARY:
- Total Traders: {template_data.get('total_traders', 0)}
- Active Signals: {template_data.get('active_signals', 0)}
- Expected PnL: {template_data.get('expected_pnl', '$0.00')}
- Average Confidence: {template_data.get('avg_confidence', '0')}%

INDIVIDUAL PREDICTIONS:
"""

        for trader in template_data.get('traders', []):
            text += f"""
Trader {trader['id']} - {trader['name']}:
  Expected PnL: {trader['predicted_pnl']}
  Confidence: {trader['confidence']}%
  Recommendation: {trader['recommendation']}
"""

        if template_data.get('alerts'):
            text += "\nALERTS:\n"
            for alert in template_data['alerts']:
                text += f"- {alert['title']}: {alert['message']}\n"

        text += f"""
{'=' * 50}
Generated by Risk Tool AI Trading System
{template_data.get('generation_timestamp', '')}

DISCLAIMER: These predictions are generated by machine learning models
and should not be considered as investment advice. Always perform your
own analysis and risk assessment before making trading decisions.
"""

        return text

    def test_signal_email(self, to_email: Optional[str] = None) -> bool:
        """
        Send a test signal email with sample data

        Args:
            to_email: Recipient email (uses config default if None)

        Returns:
            True if sent successfully, False otherwise
        """
        # Generate sample prediction data
        sample_predictions = {
            'traders': {
                'TRADER001': {
                    'name': 'Alpha Trader',
                    'predicted_pnl': 150.75,
                    'confidence': 0.82,
                    'recent_performance': 320.50,
                    'risk_notes': 'High volatility in recent trading patterns'
                },
                'TRADER002': {
                    'name': 'Beta Trader',
                    'predicted_pnl': -75.25,
                    'confidence': 0.65,
                    'recent_performance': -45.30
                },
                'TRADER003': {
                    'name': 'Gamma Trader',
                    'predicted_pnl': 45.00,
                    'confidence': 0.71,
                    'recent_performance': 125.80
                }
            },
            'model_performance': {
                'hit_rate': 0.68,
                'sharpe_ratio': 1.25,
                'r2_score': 0.15,
                'model_version': 'v2.1-test'
            },
            'market_context': {
                'market_volatility': 18.5,
                'trading_volume_pct': 15.2,
                'market_trend': 'Bullish',
                'risk_level': 'Medium'
            }
        }

        return self.send_signal_email(
            predictions=sample_predictions,
            to_emails=to_email,
            subject="🧪 Test Trading Signal Report"
        )
