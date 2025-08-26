"""
Refactored SignalGenerator demonstrating clean architecture principles.

This is an example of how to refactor the existing monolithic SignalGenerator
using the new repository pattern, dependency injection, and service layer.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd

from .container import ServiceContainer
from .models.domain import RiskLevel, RiskAssessment
from .exceptions import RiskToolException

logger = logging.getLogger(__name__)


class SignalGenerator:
    """
    Refactored signal generator using clean architecture.

    This class now has a single responsibility: orchestrating
    signal generation using injected services.
    """

    def __init__(self, container: ServiceContainer):
        """
        Initialize with dependency injection container.

        Args:
            container: Service container with all dependencies
        """
        self.container = container
        self.risk_service = container.risk_service
        self.trader_service = container.trader_service
        self.prediction_service = container.prediction_service
        self.metrics_service = container.metrics_service
        self.config = container.config

    def generate_signals(self) -> List[Dict[str, Any]]:
        """
        Generate trading signals for all active traders.

        Returns:
            List of signal dictionaries
        """
        signals = []

        # Get active traders
        active_traders = self.trader_service.get_active_traders()
        logger.info(f"Generating signals for {len(active_traders)} active traders")

        for trader in active_traders:
            try:
                signal = self._generate_trader_signal(trader.id)
                if signal:
                    signals.append(signal)
            except RiskToolException as e:
                logger.warning(f"Failed to generate signal for trader {trader.id}: {e.message}")
            except Exception as e:
                logger.error(f"Unexpected error for trader {trader.id}: {str(e)}")

        # Sort signals by risk level
        signals.sort(key=lambda x: x['risk_score'], reverse=True)

        return signals

    def _generate_trader_signal(self, trader_id: int) -> Optional[Dict[str, Any]]:
        """
        Generate signal for a single trader.

        Args:
            trader_id: The trader ID

        Returns:
            Signal dictionary or None
        """
        # Get trader profile
        profile = self.trader_service.get_trader_profile(trader_id)

        # Generate prediction
        prediction = self.prediction_service.generate_prediction(trader_id)

        # Assess risk
        risk_assessment = self.risk_service.assess_trader_risk(trader_id)

        # Build signal
        signal = self._build_signal(profile, prediction, risk_assessment)

        return signal

    def _build_signal(self, profile, prediction, assessment) -> Dict[str, Any]:
        """
        Build signal dictionary from components.

        Args:
            profile: TraderProfile
            prediction: PredictionResult
            assessment: RiskAssessment

        Returns:
            Signal dictionary
        """
        return {
            'trader_id': profile.trader.id,
            'trader_name': profile.trader.display_name,
            'timestamp': datetime.now().isoformat(),

            # Risk metrics
            'risk_level': assessment.risk_level.value,
            'risk_score': assessment.risk_score,
            'var_95': assessment.var_95,
            'expected_shortfall': assessment.expected_shortfall,

            # Predictions
            'loss_probability': prediction.loss_probability,
            'expected_return': prediction.expected_return,
            'confidence_interval': prediction.confidence_interval,

            # Current metrics
            'sharpe_ratio': profile.current_metrics.sharpe,
            'win_rate': profile.current_metrics.bat_30d,
            'wl_ratio': profile.current_metrics.wl_ratio,
            'total_pnl': profile.current_metrics.total_pnl,

            # Recommendations
            'action': self._determine_action(assessment, prediction),
            'recommendation': assessment.get_recommendation(),
            'trend': profile.get_trend()
        }

    def _determine_action(self, assessment: RiskAssessment,
                         prediction) -> str:
        """
        Determine recommended action.

        Args:
            assessment: Risk assessment
            prediction: Prediction result

        Returns:
            Action string
        """
        if assessment.risk_level == RiskLevel.CRITICAL:
            return "SUSPEND"
        elif assessment.risk_level == RiskLevel.HIGH:
            return "REDUCE"
        elif prediction.loss_probability > 0.7:
            return "MONITOR"
        else:
            return "CONTINUE"

    def get_high_priority_signals(self,
                                 limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get high priority signals requiring attention.

        Args:
            limit: Maximum number of signals

        Returns:
            List of high priority signals
        """
        all_signals = self.generate_signals()

        # Filter for high priority
        high_priority = [
            s for s in all_signals
            if s['risk_level'] in ['high', 'critical'] or
               s['action'] in ['SUSPEND', 'REDUCE']
        ]

        return high_priority[:limit]

    def generate_summary_report(self) -> Dict[str, Any]:
        """
        Generate summary report of all signals.

        Returns:
            Summary dictionary
        """
        signals = self.generate_signals()

        if not signals:
            return {
                'status': 'NO_SIGNALS',
                'timestamp': datetime.now().isoformat()
            }

        # Aggregate metrics
        total_traders = len(signals)
        high_risk_count = len([s for s in signals if s['risk_level'] in ['high', 'critical']])

        avg_sharpe = sum(s['sharpe_ratio'] for s in signals) / total_traders
        avg_win_rate = sum(s['win_rate'] for s in signals) / total_traders
        total_pnl = sum(s['total_pnl'] for s in signals)

        return {
            'timestamp': datetime.now().isoformat(),
            'total_traders': total_traders,
            'high_risk_traders': high_risk_count,
            'avg_sharpe_ratio': avg_sharpe,
            'avg_win_rate': avg_win_rate,
            'total_pnl': total_pnl,
            'top_performers': self._get_top_performers(signals, 3),
            'worst_performers': self._get_worst_performers(signals, 3),
            'actions_required': self._get_required_actions(signals)
        }

    def _get_top_performers(self, signals: List[Dict],
                          limit: int) -> List[Dict]:
        """Get top performing traders."""
        sorted_signals = sorted(signals, key=lambda x: x['total_pnl'], reverse=True)
        return sorted_signals[:limit]

    def _get_worst_performers(self, signals: List[Dict],
                            limit: int) -> List[Dict]:
        """Get worst performing traders."""
        sorted_signals = sorted(signals, key=lambda x: x['total_pnl'])
        return sorted_signals[:limit]

    def _get_required_actions(self, signals: List[Dict]) -> Dict[str, int]:
        """Count required actions."""
        actions = {}
        for signal in signals:
            action = signal['action']
            actions[action] = actions.get(action, 0) + 1
        return actions


class SignalFormatter:
    """Separate class for formatting signals for different outputs."""

    @staticmethod
    def format_for_email(signals: List[Dict]) -> str:
        """Format signals for email notification."""
        if not signals:
            return "No signals generated."

        lines = ["Risk Signal Report", "=" * 50, ""]

        for signal in signals:
            lines.extend([
                f"Trader: {signal['trader_name']} ({signal['trader_id']})",
                f"Risk Level: {signal['risk_level'].upper()}",
                f"Action: {signal['action']}",
                f"Loss Probability: {signal['loss_probability']:.1%}",
                f"VaR (95%): ${signal['var_95']:,.2f}",
                f"Recommendation: {signal['recommendation']}",
                "-" * 30
            ])

        return "\n".join(lines)

    @staticmethod
    def format_for_dashboard(signals: List[Dict]) -> pd.DataFrame:
        """Format signals for dashboard display."""
        if not signals:
            return pd.DataFrame()

        df = pd.DataFrame(signals)

        # Select and rename columns for display
        display_columns = {
            'trader_id': 'Trader ID',
            'trader_name': 'Name',
            'risk_level': 'Risk',
            'risk_score': 'Score',
            'action': 'Action',
            'sharpe_ratio': 'Sharpe',
            'win_rate': 'Win %',
            'total_pnl': 'Total PnL'
        }

        df = df[list(display_columns.keys())]
        df = df.rename(columns=display_columns)

        # Format numeric columns
        df['Score'] = df['Score'].round(1)
        df['Sharpe'] = df['Sharpe'].round(2)
        df['Win %'] = df['Win %'].round(1)
        df['Total PnL'] = df['Total PnL'].round(2)

        return df

    @staticmethod
    def format_as_json(signals: List[Dict]) -> Dict[str, Any]:
        """Format signals as JSON for API response."""
        return {
            'timestamp': datetime.now().isoformat(),
            'count': len(signals),
            'signals': signals
        }


# Example usage
def main():
    """Example of using the refactored signal generator."""

    # Initialize container with dependency injection
    container = ServiceContainer(env='production')

    # Create signal generator
    generator = SignalGenerator(container)

    # Generate signals
    signals = generator.generate_signals()

    # Format for different outputs
    formatter = SignalFormatter()

    # Email format
    email_content = formatter.format_for_email(signals[:5])
    print("Email Format:")
    print(email_content)

    # Dashboard format
    dashboard_df = formatter.format_for_dashboard(signals)
    print("\nDashboard Format:")
    print(dashboard_df.head())

    # JSON format
    json_response = formatter.format_as_json(signals[:3])
    print("\nJSON Format:")
    print(json_response)

    # Generate summary
    summary = generator.generate_summary_report()
    print("\nSummary Report:")
    print(summary)


if __name__ == "__main__":
    main()
