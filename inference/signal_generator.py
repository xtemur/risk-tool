# inference/signal_generator.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config
from src.data_processing import create_trader_day_panel
from src.trader_data_processor import ImprovedTraderProcessor
from src.trader_metrics import TraderMetricsProvider
from src.risk_predictor import RiskPredictor
from .email_service import EmailService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SignalGenerator:
    """Generate risk signals for traders based on trained models."""

    def __init__(self, config_path: str = 'configs/main_config.yaml'):
        """Initialize signal generator with configuration."""
        self.config = load_config(config_path)
        self.trader_data = {}
        self.email_service = None
        self.trader_processor = ImprovedTraderProcessor(config_path)

        # Initialize new modular components
        self.metrics_provider = TraderMetricsProvider(self.config)
        self.risk_predictor = RiskPredictor(self.config)

        # Load all trader models
        self.risk_predictor.load_all_trader_models()

    def load_optimal_thresholds(self):
        """Load optimal thresholds for each trader (delegated to RiskPredictor)."""
        # This is now handled by the RiskPredictor class
        logger.info(f"Optimal thresholds loaded by RiskPredictor for {len(self.risk_predictor.optimal_thresholds)} traders")

    def load_trader_models(self):
        """Load trained models for each trader (delegated to RiskPredictor)."""
        # This is now handled by the RiskPredictor class
        self.risk_predictor.load_all_trader_models()
        logger.info(f"Successfully loaded {len(self.risk_predictor.trader_models)} trader models")

    def load_trader_data(self) -> Dict[int, pd.DataFrame]:
        """Load the most recent data for each trader."""
        trader_splits_dir = self.config['paths']['processed_features']

        logger.info("Loading trader data from splits...")

        for trader_id in self.config['active_traders']:
            trader_dir = os.path.join(trader_splits_dir, str(trader_id))

            # Try to load test data first (most recent), then train data
            for data_file in ['test_data.parquet', 'train_data.parquet']:
                data_path = os.path.join(trader_dir, data_file)

                if os.path.exists(data_path):
                    try:
                        df = pd.read_parquet(data_path)

                        # Convert date column if it exists
                        if 'date' in df.columns:
                            df['trade_date'] = pd.to_datetime(df['date'])
                        elif 'trade_date' not in df.columns:
                            logger.warning(f"No date column found for trader {trader_id}")
                            continue

                        # Store the latest records
                        if trader_id not in self.trader_data:
                            self.trader_data[trader_id] = df
                        else:
                            # Append and keep only recent data
                            combined = pd.concat([self.trader_data[trader_id], df])
                            self.trader_data[trader_id] = combined.drop_duplicates().sort_values('trade_date')

                        logger.info(f"Loaded {len(df)} records for trader {trader_id} from {data_file}")

                    except Exception as e:
                        logger.error(f"Error loading data for trader {trader_id} from {data_file}: {e}")

        logger.info(f"Loaded data for {len(self.trader_data)} traders")
        return self.trader_data

    def get_trader_names(self) -> Dict[int, str]:
        """Get trader account names from database (delegated to MetricsProvider)."""
        return self.metrics_provider.get_trader_names()

    def get_trader_metrics_from_db(self, lookback_days: int = 30) -> Dict[int, Dict]:
        """Get comprehensive trader metrics from database (delegated to MetricsProvider)."""
        return self.metrics_provider.get_trader_metrics_for_email(lookback_days)

    def calculate_heatmap_color(self, current_value, all_time_high, all_time_low, metric_type='higher_better') -> Tuple[str, str, str]:
        """
        Calculate heatmap color using normalized position between all-time high and low.

        Args:
            current_value: Current metric value (30-day period)
            all_time_high: All-time high value for this metric
            all_time_low: All-time low value for this metric
            metric_type: 'higher_better' for metrics like Sharpe, 'lower_better' for losses

        Returns:
            Tuple of (background_color, text_color, intensity_class)
        """
        # Handle invalid values (NaN, infinity, None)
        if (current_value is None or all_time_high is None or all_time_low is None or
            not np.isfinite(current_value) or not np.isfinite(all_time_high) or not np.isfinite(all_time_low)):
            return '#F5F5F5', '#000000', 'neutral'

        # Handle edge case where all-time high equals all-time low
        if all_time_high == all_time_low:
            return '#F5F5F5', '#000000', 'neutral'

        # Calculate normalized position (0 to 1) between all-time low and high
        try:
            if metric_type == 'higher_better':
                # For metrics where higher is better, normalize directly
                normalized_position = (current_value - all_time_low) / (all_time_high - all_time_low)
            else:  # lower_better (for losses)
                # For metrics where lower is better, invert the normalization
                normalized_position = (all_time_high - current_value) / (all_time_high - all_time_low)

            # Clamp to valid range
            normalized_position = max(0, min(1, normalized_position))

        except (ZeroDivisionError, TypeError, ValueError):
            return '#F5F5F5', '#000000', 'neutral'

        # Convert normalized position to percentage scale (-50 to +50)
        # 0.5 (middle) = 0%, 0 (worst) = -50%, 1 (best) = +50%
        percentage = (normalized_position - 0.5) * 100

        # Pleasant Trading Gradient: Soft Red → Cream → Soft Green
        def interpolate_color(pct):
            # Clamp percentage between -100 and 100
            pct = max(-100, min(100, pct))

            if pct < -10:
                # Poor performance: Soft red gradient
                # Light red to deeper red
                intensity = min(abs(pct + 10) / 90, 1.0)  # 0 to 1 for -10 to -100
                red = int(255 - 55 * intensity)    # 255 → 200 (softer red)
                green = int(220 - 120 * intensity) # 220 → 100 (pleasant transition)
                blue = int(220 - 120 * intensity)  # 220 → 100 (warm tone)
                intensity_class = 'poor' if pct < -50 else 'below_avg'
            elif pct > 10:
                # Good performance: Soft green gradient
                # Light green to deeper green
                intensity = min((pct - 10) / 90, 1.0)  # 0 to 1 for +10 to +100
                red = int(220 - 120 * intensity)   # 220 → 100 (warm undertone)
                green = int(255 - 35 * intensity)  # 255 → 220 (pleasant green)
                blue = int(220 - 70 * intensity)   # 220 → 150 (natural tone)
                intensity_class = 'excellent' if pct > 50 else 'above_avg'
            else:
                # Neutral range (-10 to +10): Soft cream/beige
                # Very subtle gradient in the neutral zone
                if pct >= 0:
                    # Slightly positive: very light green tint
                    intensity = pct / 10  # 0 to 1
                    red = int(248 - 28 * intensity)   # 248 → 220
                    green = int(252 - 7 * intensity)  # 252 → 245
                    blue = int(240 - 20 * intensity)  # 240 → 220
                else:
                    # Slightly negative: very light red tint
                    intensity = abs(pct) / 10  # 0 to 1
                    red = int(248 + 7 * intensity)    # 248 → 255
                    green = int(245 - 25 * intensity) # 245 → 220
                    blue = int(240 - 20 * intensity)  # 240 → 220
                intensity_class = 'neutral'

            return f'#{red:02X}{green:02X}{blue:02X}'

        bg_color = interpolate_color(percentage)

        # Text color based on background brightness
        # Convert hex to RGB for brightness calculation
        r = int(bg_color[1:3], 16)
        g = int(bg_color[3:5], 16)
        b = int(bg_color[5:7], 16)
        brightness = (r * 0.299 + g * 0.587 + b * 0.114)
        text_color = '#FFFFFF' if brightness < 128 else '#000000'

        # Determine intensity class based on percentage
        if percentage > 50:
            intensity_class = 'excellent'
        elif percentage > 20:
            intensity_class = 'very_good'
        elif percentage > 5:
            intensity_class = 'good'
        elif percentage >= -5:
            intensity_class = 'neutral'
        elif percentage >= -20:
            intensity_class = 'slightly_poor'
        elif percentage >= -50:
            intensity_class = 'poor'
        else:
            intensity_class = 'critical'

        return bg_color, text_color, intensity_class

    def get_latest_trader_data(self, lookback_days: int = 60) -> Dict[int, pd.Series]:
        """Get the most recent data for each active trader by fetching fresh data from database."""
        logger.info("Fetching fresh data from database for signal generation...")

        # Create fresh trader-day panel from database
        panel_df = create_trader_day_panel(self.config)

        latest_records = {}

        for trader_id in self.config['active_traders']:
            # Extract trader data
            trader_data = panel_df[panel_df['account_id'] == trader_id].copy()

            if trader_data.empty:
                logger.warning(f"No data found for trader {trader_id}")
                continue

            # Rename columns to match expected format
            trader_data = trader_data.rename(columns={'trade_date': 'date', 'account_id': 'trader_id'})

            # Create features using the existing processor
            try:
                trader_features = self.trader_processor.create_improved_features(trader_data)

                if not trader_features.empty:
                    # Get the latest record
                    latest_record = trader_features.sort_values('date').iloc[-1]
                    latest_records[trader_id] = latest_record
                    logger.info(f"Processed fresh data for trader {trader_id}, last date: {latest_record['date']}")
                else:
                    logger.warning(f"No features generated for trader {trader_id}")

            except Exception as e:
                logger.error(f"Error creating features for trader {trader_id}: {e}")
                continue

        logger.info(f"Generated fresh features for {len(latest_records)} traders")
        logger.info(latest_records)
        return latest_records

    def generate_predictions(self, trader_data_dict: Dict[int, pd.Series]) -> Dict[int, Dict]:
        """Generate predictions using the new RiskPredictor class."""
        return self.risk_predictor.generate_predictions_batch(trader_data_dict)

    def classify_risk_level(self, trader_id: int, var_pred: float, loss_prob: float,
                          use_weighted_formula: bool = True, alpha: float = 0.6,
                          beta: float = 0.4, thresholds: Dict[str, float] = None,
                          var_range: Tuple[float, float] = None) -> str:
        """
        Classify risk level using either binary thresholds or weighted formula.

        Args:
            trader_id: Trader account ID
            var_pred: VaR prediction value
            loss_prob: Loss probability prediction
            use_weighted_formula: Whether to use weighted formula (default True)
            alpha: Weight for VaR component
            beta: Weight for loss probability component
            thresholds: Risk level thresholds for weighted formula
            var_range: VaR normalization range

        Returns:
            Risk level classification
        """
        if use_weighted_formula:
            return self.risk_predictor.classify_risk_level_weighted(
                var_pred, loss_prob, alpha, beta, thresholds, var_range
            )
        else:
            return self.risk_predictor.classify_risk_level(trader_id, var_pred, loss_prob)

    def generate_warning_signals(self, trader_id: int, row: pd.Series) -> List[str]:
        """Generate warning signals based on trader metrics and optimal thresholds."""
        signals = []

        # Get trader-specific thresholds from RiskPredictor
        thresholds = self.risk_predictor.optimal_thresholds.get(trader_id, {})

        # Check for high volatility (if available in data)
        volatility_cols = [col for col in row.index if 'vol' in col.lower() or 'std' in col.lower()]
        if volatility_cols:
            volatility = row[volatility_cols[0]]
            if volatility > 2000:
                signals.append('HIGH_VOLATILITY')

        # Check for poor recent performance
        pnl_cols = [col for col in row.index if 'pnl' in col.lower() and ('mean' in col.lower() or 'avg' in col.lower())]
        if pnl_cols and row[pnl_cols[0]] < -1000:
            signals.append('POOR_PERFORMANCE')

        # Check for elevated risk based on model prediction
        if hasattr(row, 'loss_probability') and row.loss_probability > 0.3:
            signals.append('ELEVATED_RISK')

        return signals

    def generate_alerts(self, predictions_dict: Dict[int, Dict], trader_names: Dict[int, str] = None) -> List[Dict]:
        """Generate critical alerts based on intervention recommendations."""
        alerts = []

        if trader_names is None:
            trader_names = self.get_trader_names()

        for trader_id, pred_data in predictions_dict.items():
            var_prediction = pred_data['var_prediction']
            loss_prob = pred_data['loss_probability']

            # Get trader name
            trader_name = trader_names.get(trader_id, f"ID {trader_id}")
            trader_label = f"Trader {trader_id} ({trader_name})"

            # Use RiskPredictor to get intervention recommendation
            intervention = self.risk_predictor.generate_intervention_recommendation(
                trader_id, var_prediction, loss_prob
            )

            if intervention['should_intervene']:
                alerts.append({
                    'trader_id': str(trader_id),
                    'trader_label': trader_label,
                    'message': intervention['recommendation']
                })

        return alerts

    def generate_daily_signals(self, target_date: str = None, use_weighted_formula: bool = True,
                             alpha: float = 0.6, beta: float = 0.4,
                             thresholds: Dict[str, float] = None,
                             var_range: Tuple[float, float] = None) -> Dict:
        """
        Generate complete daily signal report using trader-specific models.

        Args:
            target_date: Date string (YYYY-MM-DD) or None for latest
            use_weighted_formula: Whether to use weighted risk formula (default True)
            alpha: Weight for VaR component (default 0.6)
            beta: Weight for loss probability component (default 0.4)
            thresholds: Risk level thresholds for weighted formula
            var_range: VaR normalization range for weighted formula

        Returns:
            Dictionary with signal data for email template
        """
        logger.info("Generating daily risk signals with trader-specific models...")

        # Load latest data and models
        latest_data = self.get_latest_trader_data()
        predictions = self.generate_predictions(latest_data)

        # Get trader names and database metrics
        trader_names = self.get_trader_names()
        trader_metrics = self.get_trader_metrics_from_db()

        # Prepare trader signals with enhanced data
        trader_signals = []
        for trader_id in self.config['active_traders']:
            if trader_id not in predictions:
                continue

            pred_data = predictions[trader_id]
            trader_name = trader_names.get(trader_id, f"ID {trader_id}")

            # Get database metrics for this trader
            db_metrics = trader_metrics.get(trader_id, {})

            # Get data series for warning signals
            data_series = latest_data.get(trader_id, pd.Series())

            # Calculate heatmap colors for each metric using normalized approach
            # For VaR, we don't have historical range, so use a fixed baseline
            var_color = self.calculate_heatmap_color(
                abs(pred_data['var_prediction']),
                20000,  # Reasonable high VaR value
                0,      # Best VaR is 0
                'lower_better'  # Lower VaR is better
            )

            # For loss probability, use 0-1 range
            loss_prob_color = self.calculate_heatmap_color(
                pred_data['loss_probability'],
                1.0,    # Maximum probability
                0.0,    # Minimum probability
                'lower_better'  # Lower probability is better
            )

            # For PnL metrics, use actual historical ranges
            last_day_pnl_color = self.calculate_heatmap_color(
                db_metrics.get('last_trading_day_pnl', 0),
                db_metrics.get('all_time_highest_pnl', 0),
                db_metrics.get('all_time_lowest_pnl', 0),
                'higher_better'
            )

            # For Sharpe ratio, estimate reasonable ranges if historical data is limited
            sharpe_high = max(db_metrics.get('all_time_sharpe', 0), 3.0)  # Good Sharpe is ~3
            sharpe_low = min(db_metrics.get('all_time_sharpe', 0), -2.0)  # Poor Sharpe is ~-2
            sharpe_color = self.calculate_heatmap_color(
                db_metrics.get('sharpe_30d', 0),
                sharpe_high,
                sharpe_low,
                'higher_better'
            )

            avg_daily_pnl_color = self.calculate_heatmap_color(
                db_metrics.get('avg_daily_pnl', 0),
                db_metrics.get('all_time_highest_pnl', 0),
                db_metrics.get('all_time_lowest_pnl', 0),
                'higher_better'
            )

            avg_winning_color = self.calculate_heatmap_color(
                db_metrics.get('avg_winning_trade', 0),
                db_metrics.get('all_time_highest_pnl', 0),
                0,  # Minimum winning trade is 0
                'higher_better'
            )

            avg_losing_color = self.calculate_heatmap_color(
                abs(db_metrics.get('avg_losing_trade', 0)),
                abs(db_metrics.get('all_time_lowest_pnl', 0)),
                0,  # Best losing trade is 0
                'lower_better'  # Lower loss is better
            )

            highest_pnl_color = self.calculate_heatmap_color(
                db_metrics.get('highest_pnl', 0),
                db_metrics.get('all_time_highest_pnl', 0),
                db_metrics.get('all_time_lowest_pnl', 0),
                'higher_better'
            )

            lowest_pnl_color = self.calculate_heatmap_color(
                db_metrics.get('lowest_pnl', 0),
                db_metrics.get('all_time_highest_pnl', 0),
                db_metrics.get('all_time_lowest_pnl', 0),
                'higher_better'  # For losses, less negative (higher) is better
            )

            # Calculate BAT and W/L heatmap colors
            bat_color = self.calculate_heatmap_color(
                db_metrics.get('bat_30d', 0),
                100,  # Maximum batting average is 100%
                0,    # Minimum batting average is 0%
                'higher_better'
            )

            # For W/L ratio, use reasonable ranges
            wl_ratio_color = self.calculate_heatmap_color(
                db_metrics.get('wl_ratio_30d', 0),
                5.0,  # Excellent W/L ratio
                0.1,  # Poor W/L ratio
                'higher_better'
            )

            signal = {
                'trader_id': str(trader_id),
                'trader_name': trader_name,
                'trader_label': f"{trader_id} ({trader_name})",
                'risk_level': self.classify_risk_level(
                    trader_id,
                    pred_data['var_prediction'],
                    pred_data['loss_probability'],
                    use_weighted_formula,
                    alpha,
                    beta,
                    thresholds,
                    var_range
                ),
                'var_5pct': pred_data['var_prediction'],
                'loss_probability': pred_data['loss_probability'],
                'model_confidence': pred_data.get('model_confidence', 0.5),
                'risk_score': self.risk_predictor.calculate_weighted_risk_score(
                    pred_data['var_prediction'],
                    pred_data['loss_probability'],
                    alpha, beta, var_range, 'sigmoid'
                ) if use_weighted_formula else None,
                'last_trade_date': str(db_metrics.get('last_trade_date', 'N/A')).replace('2025-', '') if db_metrics.get('last_trade_date', 'N/A') != 'N/A' else 'N/A',
                'last_trading_day_pnl': db_metrics.get('last_trading_day_pnl', 0),
                'sharpe_30d': db_metrics.get('sharpe_30d', 0),
                'avg_daily_pnl': db_metrics.get('avg_daily_pnl', 0),
                'avg_winning_trade': db_metrics.get('avg_winning_trade', 0),
                'avg_losing_trade': db_metrics.get('avg_losing_trade', 0),
                'highest_pnl': db_metrics.get('highest_pnl', 0),
                'lowest_pnl': db_metrics.get('lowest_pnl', 0),
                # New BAT and W/L metrics
                'bat_30d': db_metrics.get('bat_30d', 0),
                'bat_all_time': db_metrics.get('bat_all_time', 0),
                'wl_ratio_30d': db_metrics.get('wl_ratio_30d', 0),
                'wl_ratio_all_time': db_metrics.get('wl_ratio_all_time', 0),
                'volatility': data_series.get('pnl_std_7d', 0) if hasattr(data_series, 'get') else 0,
                'warning_signals': self.generate_warning_signals(trader_id, data_series),
                'optimal_thresholds': self.risk_predictor.optimal_thresholds.get(trader_id, {}),
                # Heatmap colors for all relevant metrics
                'var_heatmap': {'bg': var_color[0], 'text': var_color[1], 'class': var_color[2]},
                'loss_prob_heatmap': {'bg': loss_prob_color[0], 'text': loss_prob_color[1], 'class': loss_prob_color[2]},
                'last_day_pnl_heatmap': {'bg': last_day_pnl_color[0], 'text': last_day_pnl_color[1], 'class': last_day_pnl_color[2]},
                'sharpe_heatmap': {'bg': sharpe_color[0], 'text': sharpe_color[1], 'class': sharpe_color[2]},
                'avg_daily_pnl_heatmap': {'bg': avg_daily_pnl_color[0], 'text': avg_daily_pnl_color[1], 'class': avg_daily_pnl_color[2]},
                'avg_winning_heatmap': {'bg': avg_winning_color[0], 'text': avg_winning_color[1], 'class': avg_winning_color[2]},
                'avg_losing_heatmap': {'bg': avg_losing_color[0], 'text': avg_losing_color[1], 'class': avg_losing_color[2]},
                'highest_pnl_heatmap': {'bg': highest_pnl_color[0], 'text': highest_pnl_color[1], 'class': highest_pnl_color[2]},
                'lowest_pnl_heatmap': {'bg': lowest_pnl_color[0], 'text': lowest_pnl_color[1], 'class': lowest_pnl_color[2]},
                # New BAT and W/L heatmap colors
                'bat_heatmap': {'bg': bat_color[0], 'text': bat_color[1], 'class': bat_color[2]},
                'wl_ratio_heatmap': {'bg': wl_ratio_color[0], 'text': wl_ratio_color[1], 'class': wl_ratio_color[2]}
            }
            trader_signals.append(signal)

        # Sort by risk level (high risk first) and loss probability/risk score
        if use_weighted_formula:
            # 4-level classification sorting
            risk_order = {'High Risk': 0, 'Medium Risk': 1, 'Low Risk': 2, 'Neutral': 3}
            trader_signals.sort(key=lambda x: (
                risk_order.get(x['risk_level'], 3),
                -x.get('risk_score', 0) if x.get('risk_score') is not None else -x['loss_probability']
            ))
        else:
            # Binary classification sorting
            risk_order = {'high': 0, 'low': 1}
            trader_signals.sort(key=lambda x: (risk_order.get(x['risk_level'], 1), -x['loss_probability']))

        # Generate alerts
        alerts = self.generate_alerts(predictions, trader_names)

        # Calculate summary statistics
        if trader_signals:
            var_amounts = [abs(s['var_5pct']) for s in trader_signals]
            loss_probs = [s['loss_probability'] for s in trader_signals]

            summary_stats = {
                'avg_var': np.mean(var_amounts),
                'max_var': np.max(var_amounts),
                'avg_loss_prob': np.mean(loss_probs),
                'max_loss_prob': np.max(loss_probs),
                'total_warning_signals': sum(len(s['warning_signals']) for s in trader_signals),
                'using_optimal_thresholds': True,
                'intervention_based': True,
                'causal_impact_model': True,
                'weighted_formula_enabled': use_weighted_formula,
                'alpha': alpha if use_weighted_formula else None,
                'beta': beta if use_weighted_formula else None,
                'risk_classification_levels': 4 if use_weighted_formula else 2
            }
        else:
            summary_stats = {
                'avg_var': 0,
                'max_var': 0,
                'avg_loss_prob': 0,
                'max_loss_prob': 0,
                'total_warning_signals': 0,
                'using_optimal_thresholds': True,
                'intervention_based': True,
                'causal_impact_model': True,
                'weighted_formula_enabled': use_weighted_formula,
                'alpha': alpha if use_weighted_formula else None,
                'beta': beta if use_weighted_formula else None,
                'risk_classification_levels': 4 if use_weighted_formula else 2
            }

        # Prepare final signal data
        signal_data = {
            'date': target_date or datetime.now().strftime('%Y-%m-%d'),
            'trader_signals': trader_signals,
            'alerts': alerts,
            'summary_stats': summary_stats
        }

        logger.info(f"Generated signals for {len(trader_signals)} traders with {len(alerts)} alerts using optimal thresholds")

        return signal_data

    def send_email_signal(self, signal_data: Dict, to_emails: List[str] = None) -> bool:
        """
        Send the generated signals via email.

        Args:
            signal_data: Dictionary containing signal data
            to_emails: List of recipient email addresses (optional, uses default if not provided)

        Returns:
            True if email sent successfully, False otherwise
        """
        try:
            # Initialize email service if not already done
            if self.email_service is None:
                self.email_service = EmailService()

            # Use default recipients if none provided
            if to_emails is None:
                to_emails = self.email_service.default_recipients

            if not to_emails:
                logger.warning("No email recipients configured. Set EMAIL_RECIPIENTS in environment or pass to_emails parameter.")
                return False

            # Send the email
            success = self.email_service.send_daily_signals(signal_data, to_emails)

            if success:
                logger.info(f"Risk signals email sent successfully to {', '.join(to_emails)}")
            else:
                logger.error("Failed to send risk signals email")

            return success

        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return False


def test_signal_generator():
    """Test the signal generator."""
    generator = SignalGenerator()

    # Generate signals
    signals = generator.generate_daily_signals()

    # Print summary
    print(f"\nGenerated signals for {len(signals['trader_signals'])} traders")
    print(f"High risk: {sum(1 for s in signals['trader_signals'] if s['risk_level'] == 'high')}")
    print(f"Medium risk: {sum(1 for s in signals['trader_signals'] if s['risk_level'] == 'medium')}")
    print(f"Low risk: {sum(1 for s in signals['trader_signals'] if s['risk_level'] == 'low')}")
    print(f"\nAlerts: {len(signals['alerts'])}")
    print(f"Using optimal thresholds: {signals['summary_stats']['using_optimal_thresholds']}")
    print(f"Risk reduction level: {signals['summary_stats']['risk_reduction_level']}")

    # Send email
    email_success = generator.send_email_signal(signals)
    if email_success:
        print("Email sent successfully!")
    else:
        print("Failed to send email (check configuration)")

    return signals


if __name__ == '__main__':
    test_signal_generator()
