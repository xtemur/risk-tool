"""
Dashboard Generator
Creates HTML dashboards and reports for trading risk management
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json

from src.core.constants import TradingConstants as TC
from src.backtesting.performance_metrics import PerformanceMetrics

logger = logging.getLogger(__name__)


class DashboardGenerator:
    """
    Generate interactive HTML dashboards for monitoring
    """

    def __init__(self,
                 output_dir: str = "reports",
                 template: str = "plotly_dark"):
        """
        Initialize dashboard generator

        Args:
            output_dir: Directory to save dashboards
            template: Plotly template for styling
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.template = template

        # Color schemes
        self.colors = {
            'profit': '#00cc96',
            'loss': '#ef553b',
            'neutral': '#636efa',
            'warning': '#ffa15a',
            'danger': '#ff0000',
            'info': '#00ccff'
        }

    def create_risk_dashboard(self,
                             predictions: pd.DataFrame,
                             historical_performance: pd.DataFrame,
                             feature_importance: Dict[str, float],
                             monitoring_metrics: Dict[str, Any],
                             drift_results: Optional[Dict[str, Any]] = None) -> str:
        """
        Create comprehensive risk management dashboard

        Args:
            predictions: Recent predictions with risk scores
            historical_performance: Historical P&L and metrics
            feature_importance: Feature importance scores
            monitoring_metrics: Model monitoring metrics
            drift_results: Data drift detection results

        Returns:
            Path to generated HTML dashboard
        """
        # Create figure with subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Risk Score Distribution',
                'P&L Performance',
                'Feature Importance',
                'Model Accuracy Trend',
                'Trading Activity',
                'Drift Detection'
            ),
            specs=[
                [{'type': 'bar'}, {'type': 'scatter'}],
                [{'type': 'bar'}, {'type': 'scatter'}],
                [{'type': 'scatter'}, {'type': 'bar'}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )

        # 1. Risk Score Distribution
        if 'risk_score' in predictions.columns:
            risk_dist = predictions['risk_score'].value_counts().sort_index()
            fig.add_trace(
                go.Bar(
                    x=risk_dist.index,
                    y=risk_dist.values,
                    name='Risk Distribution',
                    marker_color=self._get_risk_colors(risk_dist.index)
                ),
                row=1, col=1
            )

        # 2. P&L Performance
        if not historical_performance.empty and 'net_pnl' in historical_performance.columns:
            cumulative_pnl = historical_performance.groupby('date')['net_pnl'].sum().cumsum()
            fig.add_trace(
                go.Scatter(
                    x=cumulative_pnl.index,
                    y=cumulative_pnl.values,
                    mode='lines',
                    name='Cumulative P&L',
                    line=dict(color=self.colors['profit'] if cumulative_pnl.iloc[-1] > 0 else self.colors['loss'])
                ),
                row=1, col=2
            )

        # 3. Feature Importance
        if feature_importance:
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:15]
            fig.add_trace(
                go.Bar(
                    x=[f[1] for f in top_features],
                    y=[f[0] for f in top_features],
                    orientation='h',
                    name='Feature Importance',
                    marker_color=self.colors['info']
                ),
                row=2, col=1
            )

        # 4. Model Accuracy Trend
        if 'accuracy_history' in monitoring_metrics:
            accuracy_data = monitoring_metrics['accuracy_history']
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(accuracy_data))),
                    y=accuracy_data,
                    mode='lines+markers',
                    name='Model Accuracy',
                    line=dict(color=self.colors['neutral'])
                ),
                row=2, col=2
            )

        # 5. Trading Activity Heatmap
        if not historical_performance.empty and 'orders_count' in historical_performance.columns:
            # Aggregate by day of week and hour (if available)
            activity_data = self._create_activity_heatmap(historical_performance)
            if activity_data is not None:
                fig.add_trace(
                    go.Heatmap(
                        z=activity_data.values,
                        x=activity_data.columns,
                        y=activity_data.index,
                        colorscale='Blues',
                        name='Trading Activity'
                    ),
                    row=3, col=1
                )

        # 6. Drift Detection Results
        if drift_results:
            drift_scores = [r.drift_score for r in drift_results.values()]
            drift_labels = [r.feature_name for r in drift_results.values()]

            fig.add_trace(
                go.Bar(
                    x=drift_labels[:10],  # Top 10
                    y=drift_scores[:10],
                    name='Drift Scores',
                    marker_color=[self.colors['danger'] if s > 0.2 else self.colors['info']
                                 for s in drift_scores[:10]]
                ),
                row=3, col=2
            )

        # Update layout
        fig.update_layout(
            title={
                'text': f"Trading Risk Management Dashboard - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24}
            },
            template=self.template,
            height=1200,
            showlegend=False
        )

        # Update axes
        fig.update_xaxes(title_text="Risk Score", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)

        fig.update_xaxes(title_text="Date", row=1, col=2)
        fig.update_yaxes(title_text="Cumulative P&L", row=1, col=2)

        fig.update_xaxes(title_text="Importance", row=2, col=1)
        fig.update_yaxes(title_text="Feature", row=2, col=1)

        fig.update_xaxes(title_text="Time", row=2, col=2)
        fig.update_yaxes(title_text="Accuracy", row=2, col=2)

        # Save dashboard
        filename = f"risk_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        filepath = self.output_dir / filename

        fig.write_html(
            filepath,
            include_plotlyjs='cdn',
            config={'displayModeBar': True, 'displaylogo': False}
        )

        logger.info(f"Dashboard saved to {filepath}")
        return str(filepath)

    def create_performance_report(self,
                                 equity_curve: pd.Series,
                                 returns: pd.Series,
                                 metrics: Dict[str, float],
                                 trades: Optional[pd.DataFrame] = None) -> str:
        """
        Create detailed performance report

        Args:
            equity_curve: Equity curve time series
            returns: Returns time series
            metrics: Performance metrics dictionary
            trades: Optional trades DataFrame

        Returns:
            Path to generated HTML report
        """
        # Create figure
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Equity Curve',
                'Returns Distribution',
                'Drawdown',
                'Rolling Sharpe Ratio',
                'Monthly Returns',
                'Trade Analysis'
            ),
            specs=[
                [{'type': 'scatter'}, {'type': 'histogram'}],
                [{'type': 'scatter'}, {'type': 'scatter'}],
                [{'type': 'bar'}, {'type': 'scatter'}]
            ],
            vertical_spacing=0.12
        )

        # 1. Equity Curve
        fig.add_trace(
            go.Scatter(
                x=equity_curve.index,
                y=equity_curve.values,
                mode='lines',
                name='Equity',
                line=dict(color=self.colors['profit'])
            ),
            row=1, col=1
        )

        # 2. Returns Distribution
        fig.add_trace(
            go.Histogram(
                x=returns,
                nbinsx=50,
                name='Returns',
                marker_color=self.colors['neutral']
            ),
            row=1, col=2
        )

        # Add normal distribution overlay
        x_range = np.linspace(returns.min(), returns.max(), 100)
        normal_dist = stats.norm.pdf(x_range, returns.mean(), returns.std())
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=normal_dist * len(returns) * (returns.max() - returns.min()) / 50,
                mode='lines',
                name='Normal',
                line=dict(color='red', dash='dash')
            ),
            row=1, col=2
        )

        # 3. Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max

        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values * 100,
                mode='lines',
                fill='tozeroy',
                name='Drawdown',
                line=dict(color=self.colors['loss'])
            ),
            row=2, col=1
        )

        # 4. Rolling Sharpe Ratio
        rolling_sharpe = self._calculate_rolling_sharpe(returns, window=60)
        fig.add_trace(
            go.Scatter(
                x=rolling_sharpe.index,
                y=rolling_sharpe.values,
                mode='lines',
                name='Rolling Sharpe',
                line=dict(color=self.colors['info'])
            ),
            row=2, col=2
        )

        # Add target Sharpe line
        fig.add_hline(
            y=TC.TARGET_SHARPE,
            line_dash="dash",
            line_color="green",
            annotation_text="Target",
            row=2, col=2
        )

        # 5. Monthly Returns Heatmap
        monthly_returns = self._calculate_monthly_returns(returns)
        if monthly_returns is not None:
            fig.add_trace(
                go.Bar(
                    x=monthly_returns.index.strftime('%Y-%m'),
                    y=monthly_returns.values * 100,
                    name='Monthly Returns',
                    marker_color=[self.colors['profit'] if r > 0 else self.colors['loss']
                                 for r in monthly_returns.values]
                ),
                row=3, col=1
            )

        # 6. Trade Analysis
        if trades is not None and not trades.empty and 'pnl' in trades.columns:
            # Win/Loss distribution
            wins = trades[trades['pnl'] > 0]['pnl']
            losses = trades[trades['pnl'] < 0]['pnl']

            fig.add_trace(
                go.Scatter(
                    x=list(range(len(wins))),
                    y=sorted(wins.values),
                    mode='markers',
                    name='Wins',
                    marker=dict(color=self.colors['profit'])
                ),
                row=3, col=2
            )

            fig.add_trace(
                go.Scatter(
                    x=list(range(len(losses))),
                    y=sorted(losses.values),
                    mode='markers',
                    name='Losses',
                    marker=dict(color=self.colors['loss'])
                ),
                row=3, col=2
            )

        # Update layout
        fig.update_layout(
            title={
                'text': f"Performance Report - {metrics.get('total_return', 0):.1%} Total Return",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24}
            },
            template=self.template,
            height=1200,
            showlegend=False
        )

        # Add metrics annotations
        metrics_text = self._format_metrics_text(metrics)
        fig.add_annotation(
            text=metrics_text,
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="white",
            borderwidth=1,
            font=dict(size=10, family="monospace")
        )

        # Save report
        filename = f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        filepath = self.output_dir / filename

        fig.write_html(
            filepath,
            include_plotlyjs='cdn',
            config={'displayModeBar': True, 'displaylogo': False}
        )

        logger.info(f"Performance report saved to {filepath}")
        return str(filepath)

    def create_monitoring_dashboard(self,
                                   model_metrics: List[Dict[str, Any]],
                                   alert_summary: Dict[str, Any],
                                   system_health: Dict[str, Any]) -> str:
        """
        Create system monitoring dashboard

        Args:
            model_metrics: List of model performance metrics over time
            alert_summary: Alert system summary
            system_health: System health metrics

        Returns:
            Path to generated HTML dashboard
        """
        # Create figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Model Performance Trend',
                'Alert Status',
                'System Health',
                'Prediction Latency'
            ),
            specs=[
                [{'type': 'scatter'}, {'type': 'pie'}],
                [{'type': 'indicator'}, {'type': 'scatter'}]
            ],
            vertical_spacing=0.15
        )

        # 1. Model Performance Trend
        if model_metrics:
            timestamps = [m['timestamp'] for m in model_metrics]
            accuracy = [m.get('accuracy', 0) for m in model_metrics]

            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=accuracy,
                    mode='lines+markers',
                    name='Accuracy',
                    line=dict(color=self.colors['info'])
                ),
                row=1, col=1
            )

        # 2. Alert Status Pie Chart
        if alert_summary:
            alert_counts = alert_summary.get('by_severity', {})
            if alert_counts:
                fig.add_trace(
                    go.Pie(
                        labels=list(alert_counts.keys()),
                        values=list(alert_counts.values()),
                        marker_colors=[self._get_severity_color(s) for s in alert_counts.keys()],
                        name='Alerts'
                    ),
                    row=1, col=2
                )

        # 3. System Health Indicators
        if system_health:
            # Create gauge chart
            uptime_pct = system_health.get('uptime_percentage', 100)
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=uptime_pct,
                    title={'text': "System Uptime %"},
                    delta={'reference': 99.9},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': self._get_health_color(uptime_pct)},
                        'steps': [
                            {'range': [0, 95], 'color': "lightgray"},
                            {'range': [95, 99], 'color': "gray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 99.9
                        }
                    }
                ),
                row=2, col=1
            )

        # 4. Prediction Latency
        if model_metrics:
            latencies = [m.get('latency_ms', 0) for m in model_metrics]

            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=latencies,
                    mode='lines',
                    name='Latency',
                    line=dict(color=self.colors['warning'])
                ),
                row=2, col=2
            )

            # Add threshold line
            fig.add_hline(
                y=1000,  # 1 second threshold
                line_dash="dash",
                line_color="red",
                annotation_text="Threshold",
                row=2, col=2
            )

        # Update layout
        fig.update_layout(
            title={
                'text': f"System Monitoring Dashboard - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24}
            },
            template=self.template,
            height=800,
            showlegend=False
        )

        # Save dashboard
        filename = f"monitoring_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        filepath = self.output_dir / filename

        fig.write_html(
            filepath,
            include_plotlyjs='cdn',
            config={'displayModeBar': True, 'displaylogo': False}
        )

        logger.info(f"Monitoring dashboard saved to {filepath}")
        return str(filepath)

    def _get_risk_colors(self, risk_scores: pd.Series) -> List[str]:
        """Get colors based on risk scores"""
        colors = []
        for score in risk_scores:
            if score > TC.HIGH_RISK_SCORE:
                colors.append(self.colors['danger'])
            elif score > TC.MEDIUM_RISK_SCORE:
                colors.append(self.colors['warning'])
            else:
                colors.append(self.colors['info'])
        return colors

    def _create_activity_heatmap(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Create activity heatmap data"""
        try:
            if 'date' in data.columns:
                data['day_of_week'] = pd.to_datetime(data['date']).dt.day_name()
                data['week'] = pd.to_datetime(data['date']).dt.isocalendar().week

                # Aggregate activity
                activity = data.groupby(['week', 'day_of_week'])['orders_count'].sum()
                activity_pivot = activity.unstack(fill_value=0)

                # Reorder days
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
                activity_pivot = activity_pivot.reindex(columns=day_order, fill_value=0)

                return activity_pivot
        except Exception as e:
            logger.error(f"Error creating activity heatmap: {e}")
            return None

    def _calculate_rolling_sharpe(self, returns: pd.Series, window: int = 60) -> pd.Series:
        """Calculate rolling Sharpe ratio"""
        rolling_mean = returns.rolling(window).mean()
        rolling_std = returns.rolling(window).std()

        daily_rf = (1 + TC.RISK_FREE_RATE) ** (1/TC.TRADING_DAYS_PER_YEAR) - 1
        rolling_sharpe = (rolling_mean - daily_rf) / rolling_std * np.sqrt(TC.TRADING_DAYS_PER_YEAR)

        return rolling_sharpe

    def _calculate_monthly_returns(self, returns: pd.Series) -> Optional[pd.Series]:
        """Calculate monthly returns"""
        try:
            monthly = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
            return monthly
        except Exception as e:
            logger.error(f"Error calculating monthly returns: {e}")
            return None

    def _format_metrics_text(self, metrics: Dict[str, float]) -> str:
        """Format metrics for display"""
        text_lines = [
            f"Annual Return: {metrics.get('annual_return', 0):.1%}",
            f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}",
            f"Max Drawdown: {metrics.get('max_drawdown', 0):.1%}",
            f"Win Rate: {metrics.get('win_rate', 0):.1%}",
            f"Profit Factor: {metrics.get('profit_factor', 0):.2f}",
            f"Volatility: {metrics.get('annual_volatility', 0):.1%}"
        ]
        return "<br>".join(text_lines)

    def _get_severity_color(self, severity: str) -> str:
        """Get color for alert severity"""
        severity_colors = {
            'info': self.colors['info'],
            'warning': self.colors['warning'],
            'high': '#ff6600',
            'critical': self.colors['danger']
        }
        return severity_colors.get(severity, self.colors['neutral'])

    def _get_health_color(self, health_pct: float) -> str:
        """Get color based on health percentage"""
        if health_pct >= 99.9:
            return self.colors['profit']
        elif health_pct >= 99:
            return self.colors['warning']
        else:
            return self.colors['danger']
