"""
Dashboard generator for visualization and monitoring
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import datetime

class DashboardGenerator:
    """Dashboard generator for visualization and monitoring"""

    def __init__(self, title: str = "Risk Management Dashboard"):
        self.title = title
        self.app = dash.Dash(__name__)
        self.data: Dict[str, Any] = {}

    def add_data(self, key: str, data: Any) -> None:
        """Add data to dashboard"""
        self.data[key] = data

    def generate_layout(self) -> None:
        """Generate dashboard layout"""
        self.app.layout = html.Div([
            html.H1(self.title),

            # Performance Metrics
            html.Div([
                html.H2("Performance Metrics"),
                dcc.Graph(id='performance-metrics')
            ]),

            # Model Predictions
            html.Div([
                html.H2("Model Predictions"),
                dcc.Graph(id='model-predictions')
            ]),

            # Drift Monitoring
            html.Div([
                html.H2("Drift Monitoring"),
                dcc.Graph(id='drift-monitoring')
            ]),

            # Alert History
            html.Div([
                html.H2("Alert History"),
                dcc.Graph(id='alert-history')
            ]),

            # Date Range Selector
            html.Div([
                html.H3("Date Range"),
                dcc.DatePickerRange(
                    id='date-range',
                    start_date=datetime.datetime.now() - datetime.timedelta(days=30),
                    end_date=datetime.datetime.now()
                )
            ])
        ])

    def _create_performance_metrics(self) -> go.Figure:
        """Create performance metrics visualization"""
        if 'performance' not in self.data:
            return go.Figure()

        metrics = self.data['performance']

        fig = make_subplots(rows=2, cols=2,
                           subplot_titles=('Returns', 'Volatility',
                                         'Sharpe Ratio', 'Drawdown'))

        # Returns
        fig.add_trace(
            go.Scatter(x=metrics['dates'], y=metrics['returns'],
                      name='Returns'),
            row=1, col=1
        )

        # Volatility
        fig.add_trace(
            go.Scatter(x=metrics['dates'], y=metrics['volatility'],
                      name='Volatility'),
            row=1, col=2
        )

        # Sharpe Ratio
        fig.add_trace(
            go.Scatter(x=metrics['dates'], y=metrics['sharpe'],
                      name='Sharpe Ratio'),
            row=2, col=1
        )

        # Drawdown
        fig.add_trace(
            go.Scatter(x=metrics['dates'], y=metrics['drawdown'],
                      name='Drawdown'),
            row=2, col=2
        )

        fig.update_layout(height=800, showlegend=True)
        return fig

    def _create_model_predictions(self) -> go.Figure:
        """Create model predictions visualization"""
        if 'predictions' not in self.data:
            return go.Figure()

        preds = self.data['predictions']

        fig = make_subplots(rows=2, cols=1,
                           subplot_titles=('Predictions', 'Probabilities'))

        # Predictions
        fig.add_trace(
            go.Scatter(x=preds['dates'], y=preds['predictions'],
                      name='Predictions'),
            row=1, col=1
        )

        # Probabilities
        fig.add_trace(
            go.Scatter(x=preds['dates'], y=preds['probabilities'],
                      name='Probabilities'),
            row=2, col=1
        )

        fig.update_layout(height=800, showlegend=True)
        return fig

    def _create_drift_monitoring(self) -> go.Figure:
        """Create drift monitoring visualization"""
        if 'drift' not in self.data:
            return go.Figure()

        drift = self.data['drift']

        fig = make_subplots(rows=2, cols=1,
                           subplot_titles=('Drift Score', 'Feature Contributions'))

        # Drift Score
        fig.add_trace(
            go.Scatter(x=drift['dates'], y=drift['scores'],
                      name='Drift Score'),
            row=1, col=1
        )

        # Feature Contributions
        for feature, contribution in drift['contributions'].items():
            fig.add_trace(
                go.Scatter(x=drift['dates'], y=contribution,
                          name=feature),
                row=2, col=1
            )

        fig.update_layout(height=800, showlegend=True)
        return fig

    def _create_alert_history(self) -> go.Figure:
        """Create alert history visualization"""
        if 'alerts' not in self.data:
            return go.Figure()

        alerts = self.data['alerts']

        fig = go.Figure()

        # Plot alerts by type
        for alert_type in alerts['types'].unique():
            mask = alerts['types'] == alert_type
            fig.add_trace(
                go.Scatter(x=alerts[mask]['dates'],
                          y=[alert_type] * mask.sum(),
                          mode='markers',
                          name=alert_type)
            )

        fig.update_layout(
            title='Alert History',
            xaxis_title='Date',
            yaxis_title='Alert Type',
            height=400
        )

        return fig

    def register_callbacks(self) -> None:
        """Register dashboard callbacks"""
        @self.app.callback(
            [Output('performance-metrics', 'figure'),
             Output('model-predictions', 'figure'),
             Output('drift-monitoring', 'figure'),
             Output('alert-history', 'figure')],
            [Input('date-range', 'start_date'),
             Input('date-range', 'end_date')]
        )
        def update_graphs(start_date: str, end_date: str) -> Tuple[go.Figure, ...]:
            # Filter data by date range
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)

            # Update data with filtered range
            self._filter_data_by_date(start, end)

            # Create figures
            perf_fig = self._create_performance_metrics()
            pred_fig = self._create_model_predictions()
            drift_fig = self._create_drift_monitoring()
            alert_fig = self._create_alert_history()

            return perf_fig, pred_fig, drift_fig, alert_fig

    def _filter_data_by_date(self, start: pd.Timestamp,
                           end: pd.Timestamp) -> None:
        """Filter data by date range"""
        for key in self.data:
            if isinstance(self.data[key], pd.DataFrame):
                mask = (self.data[key].index >= start) & (self.data[key].index <= end)
                self.data[key] = self.data[key][mask]

    def run_server(self, debug: bool = True, port: int = 8050) -> None:
        """Run dashboard server"""
        self.app.run_server(debug=debug, port=port)
