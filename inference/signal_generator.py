# inference/signal_generator.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from datetime import datetime, timedelta
import sys
import os
import pickle
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config
from .email_service import EmailService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SignalGenerator:
    """Generate risk signals for traders based on trained models."""

    def __init__(self, config_path: str = 'configs/main_config.yaml'):
        """Initialize signal generator with configuration."""
        self.config = load_config(config_path)
        self.trader_models = {}
        self.optimal_thresholds = {}
        self.trader_data = {}
        self.email_service = None

        # Load optimal thresholds
        self.load_optimal_thresholds()

    def load_optimal_thresholds(self):
        """Load optimal thresholds for each trader."""
        threshold_path = 'configs/optimal_thresholds/optimal_thresholds.json'

        try:
            with open(threshold_path, 'r') as f:
                threshold_data = json.load(f)

            for trader_threshold in threshold_data['thresholds']:
                trader_id = int(trader_threshold['trader_id'])
                self.optimal_thresholds[trader_id] = {
                    'var_threshold': trader_threshold['var_threshold'],
                    'loss_prob_threshold': trader_threshold['loss_prob_threshold']
                }

            logger.info(f"Loaded optimal thresholds for {len(self.optimal_thresholds)} traders")

        except Exception as e:
            logger.error(f"Error loading optimal thresholds: {e}")
            # Set default thresholds if loading fails
            for trader_id in self.config['active_traders']:
                self.optimal_thresholds[trader_id] = {
                    'var_threshold': -5000,
                    'loss_prob_threshold': 0.15
                }

    def load_trader_models(self):
        """Load trained models for each trader."""
        model_dir = self.config['paths']['model_dir']

        logger.info("Loading trader-specific models...")

        for trader_id in self.config['active_traders']:
            model_path = os.path.join(model_dir, f'{trader_id}_tuned_validated.pkl')

            try:
                with open(model_path, 'rb') as f:
                    self.trader_models[trader_id] = pickle.load(f)
                logger.info(f"Loaded model for trader {trader_id}")
            except Exception as e:
                logger.error(f"Error loading model for trader {trader_id}: {e}")

        logger.info(f"Successfully loaded {len(self.trader_models)} trader models")

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
        """Get trader account names from database."""
        import sqlite3
        db_path = self.config['paths']['db_path']

        trader_names = {}
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT account_id, account_name FROM accounts WHERE is_active = 1')
            for row in cursor.fetchall():
                trader_names[row[0]] = row[1]
            conn.close()
        except Exception as e:
            logger.warning(f"Could not load trader names: {e}")

        return trader_names

    def get_trader_metrics_from_db(self, lookback_days: int = 30) -> Dict[int, Dict]:
        """Get comprehensive trader metrics directly from database."""
        import sqlite3
        from datetime import datetime, timedelta

        db_path = self.config['paths']['db_path']
        metrics = {}

        try:
            conn = sqlite3.connect(db_path)

            # Get cutoff date
            cutoff_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')

            # Get comprehensive metrics for each trader
            query = """
            WITH daily_pnl AS (
                SELECT
                    account_id,
                    trade_date,
                    SUM(net) as daily_pnl,
                    COUNT(*) as trade_count,
                    SUM(CASE WHEN net > 0 THEN net ELSE 0 END) as winning_pnl,
                    SUM(CASE WHEN net < 0 THEN net ELSE 0 END) as losing_pnl,
                    COUNT(CASE WHEN net > 0 THEN 1 END) as winning_trades,
                    COUNT(CASE WHEN net < 0 THEN 1 END) as losing_trades,
                    MAX(net) as highest_trade_pnl,
                    MIN(net) as lowest_trade_pnl
                FROM trades
                WHERE trade_date >= ?
                GROUP BY account_id, trade_date
            ),
            trader_stats AS (
                SELECT
                    account_id,
                    AVG(daily_pnl) as avg_daily_pnl,
                    SUM(daily_pnl) as total_pnl,
                    COUNT(*) as trading_days,
                    CASE
                        WHEN COUNT(*) > 1 THEN
                            COALESCE((AVG(daily_pnl) / NULLIF(
                                SQRT(ABS(SUM((daily_pnl - (SELECT AVG(daily_pnl) FROM daily_pnl d2 WHERE d2.account_id = daily_pnl.account_id)) *
                                        (daily_pnl - (SELECT AVG(daily_pnl) FROM daily_pnl d2 WHERE d2.account_id = daily_pnl.account_id))) /
                                     (COUNT(*) - 1))), 0)) * SQRT(30), 0)
                        ELSE 0
                    END as sharpe_30d,
                    AVG(CASE WHEN winning_trades > 0 THEN winning_pnl / winning_trades END) as avg_winning_trade,
                    AVG(CASE WHEN losing_trades > 0 THEN losing_pnl / losing_trades END) as avg_losing_trade,
                    MAX(highest_trade_pnl) as highest_pnl,
                    MIN(lowest_trade_pnl) as lowest_pnl
                FROM daily_pnl
                GROUP BY account_id
            ),
            last_trade_dates AS (
                SELECT
                    account_id,
                    MAX(trade_date) as last_trade_date
                FROM trades
                GROUP BY account_id
            ),
            all_time_daily_pnl AS (
                SELECT
                    account_id,
                    trade_date,
                    SUM(net) as daily_pnl,
                    COUNT(*) as trade_count,
                    SUM(CASE WHEN net > 0 THEN net ELSE 0 END) as winning_pnl,
                    SUM(CASE WHEN net < 0 THEN net ELSE 0 END) as losing_pnl,
                    COUNT(CASE WHEN net > 0 THEN 1 END) as winning_trades,
                    COUNT(CASE WHEN net < 0 THEN 1 END) as losing_trades,
                    MAX(net) as highest_trade_pnl,
                    MIN(net) as lowest_trade_pnl
                FROM trades
                GROUP BY account_id, trade_date
            ),
            all_time_stats AS (
                SELECT
                    account_id,
                    AVG(daily_pnl) as all_time_avg_daily_pnl,
                    CASE
                        WHEN COUNT(*) > 1 THEN
                            COALESCE(AVG(daily_pnl) / NULLIF(
                                SQRT(ABS((SUM(daily_pnl * daily_pnl) - SUM(daily_pnl) * SUM(daily_pnl) / COUNT(*)) / (COUNT(*) - 1))), 0) * SQRT(30), 0)
                        ELSE 0
                    END as all_time_sharpe,
                    AVG(CASE WHEN winning_trades > 0 THEN winning_pnl / winning_trades END) as all_time_avg_winning_trade,
                    AVG(CASE WHEN losing_trades > 0 THEN losing_pnl / losing_trades END) as all_time_avg_losing_trade,
                    MAX(highest_trade_pnl) as all_time_highest_pnl,
                    MIN(lowest_trade_pnl) as all_time_lowest_pnl
                FROM all_time_daily_pnl
                GROUP BY account_id
            ),
            latest_pnl AS (
                SELECT
                    account_id,
                    SUM(net) as last_trading_day_pnl
                FROM trades
                WHERE trade_date = (SELECT MAX(trade_date) FROM trades WHERE account_id = trades.account_id)
                GROUP BY account_id
            )
            SELECT
                ts.account_id,
                ltd.last_trade_date,
                COALESCE(lp.last_trading_day_pnl, 0) as last_trading_day_pnl,
                COALESCE(ts.sharpe_30d, 0) as sharpe_30d,
                COALESCE(ts.avg_daily_pnl, 0) as avg_daily_pnl,
                COALESCE(ts.avg_winning_trade, 0) as avg_winning_trade,
                COALESCE(ts.avg_losing_trade, 0) as avg_losing_trade,
                COALESCE(ts.highest_pnl, 0) as highest_pnl,
                COALESCE(ts.lowest_pnl, 0) as lowest_pnl,
                COALESCE(ts.total_pnl, 0) as total_pnl,
                COALESCE(ts.trading_days, 0) as trading_days,
                COALESCE(ats.all_time_sharpe, 0) as all_time_sharpe,
                COALESCE(ats.all_time_avg_daily_pnl, 0) as all_time_avg_daily_pnl,
                COALESCE(ats.all_time_avg_winning_trade, 0) as all_time_avg_winning_trade,
                COALESCE(ats.all_time_avg_losing_trade, 0) as all_time_avg_losing_trade,
                COALESCE(ats.all_time_highest_pnl, 0) as all_time_highest_pnl,
                COALESCE(ats.all_time_lowest_pnl, 0) as all_time_lowest_pnl
            FROM trader_stats ts
            LEFT JOIN last_trade_dates ltd ON ts.account_id = ltd.account_id
            LEFT JOIN latest_pnl lp ON ts.account_id = lp.account_id
            LEFT JOIN all_time_stats ats ON ts.account_id = ats.account_id
            """

            cursor = conn.cursor()
            cursor.execute(query, (cutoff_date,))

            for row in cursor.fetchall():
                account_id = row[0]
                metrics[account_id] = {
                    'last_trade_date': row[1],
                    'last_trading_day_pnl': row[2],
                    'sharpe_30d': row[3],
                    'avg_daily_pnl': row[4],
                    'avg_winning_trade': row[5],
                    'avg_losing_trade': row[6],
                    'highest_pnl': row[7],
                    'lowest_pnl': row[8],
                    'total_pnl': row[9],
                    'trading_days': row[10],
                    'all_time_sharpe': row[11],
                    'all_time_avg_daily_pnl': row[12],
                    'all_time_avg_winning_trade': row[13],
                    'all_time_avg_losing_trade': row[14],
                    'all_time_highest_pnl': row[15],
                    'all_time_lowest_pnl': row[16]
                }

            conn.close()
            logger.info(f"Retrieved metrics for {len(metrics)} traders from database")

        except Exception as e:
            logger.error(f"Error getting trader metrics from database: {e}")

        return metrics

    def calculate_heatmap_color(self, current_value, all_time_value, metric_type='higher_better') -> Tuple[str, str, str]:
        """
        Calculate heatmap color using standard trading gradient (smooth red-to-green).

        Args:
            current_value: 30-day metric value
            all_time_value: All-time metric value
            metric_type: 'higher_better' for metrics like Sharpe, 'lower_better' for losses

        Returns:
            Tuple of (background_color, text_color, intensity_class)
        """
        # Handle invalid values (NaN, infinity, None, zero)
        if (all_time_value == 0 or current_value is None or all_time_value is None or
            not np.isfinite(current_value) or not np.isfinite(all_time_value)):
            return '#F5F5F5', '#000000', 'neutral'

        # Calculate performance ratio with additional safety checks
        try:
            if metric_type == 'higher_better':
                ratio = current_value / all_time_value if all_time_value != 0 else 1
            else:  # lower_better (for losses)
                ratio = all_time_value / current_value if current_value != 0 else 1

            # Additional safety check for the ratio itself
            if not np.isfinite(ratio):
                return '#F5F5F5', '#000000', 'neutral'

        except (ZeroDivisionError, TypeError, ValueError):
            return '#F5F5F5', '#000000', 'neutral'

        # Convert ratio to percentage for smooth gradient
        # Map ratio to a scale from -100 to +100
        if ratio >= 1:
            # Positive performance (green side)
            percentage = min((ratio - 1) * 100, 100)  # Cap at 100%
        else:
            # Negative performance (red side)
            percentage = max((ratio - 1) * 100, -100)  # Cap at -100%

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

    def get_latest_trader_data(self, lookback_days: int = 1) -> Dict[int, pd.Series]:
        """Get the most recent data for each active trader."""
        if not self.trader_data:
            self.load_trader_data()

        latest_records = {}

        for trader_id, df in self.trader_data.items():
            if df.empty:
                continue

            # Get the latest record for this trader
            latest_record = df.sort_values('trade_date').iloc[-1]
            latest_records[trader_id] = latest_record

        return latest_records

    def generate_predictions(self, trader_data_dict: Dict[int, pd.Series]) -> Dict[int, Dict]:
        """Generate predictions for each trader using their specific model."""
        if not self.trader_models:
            self.load_trader_models()

        predictions = {}

        for trader_id, data_series in trader_data_dict.items():
            if trader_id not in self.trader_models:
                logger.warning(f"No model found for trader {trader_id}")
                continue

            model_data = self.trader_models[trader_id]

            try:
                # Extract the actual model from the dictionary
                if isinstance(model_data, dict):
                    model = model_data.get('classification_model')
                    feature_names = model_data.get('feature_names', [])
                else:
                    model = model_data
                    feature_names = []

                if model is None:
                    logger.error(f"No classification model found for trader {trader_id}")
                    continue

                # Convert series to dataframe for prediction
                data_df = pd.DataFrame([data_series])

                # Use feature names from model if available, otherwise filter columns
                if feature_names:
                    # Check if all required features are available
                    missing_features = [f for f in feature_names if f not in data_df.columns]
                    if missing_features:
                        logger.warning(f"Missing features for trader {trader_id}: {missing_features}")
                        # Use available features only
                        available_features = [f for f in feature_names if f in data_df.columns]
                        if not available_features:
                            logger.error(f"No features available for trader {trader_id}")
                            continue
                        feature_cols = available_features
                    else:
                        feature_cols = feature_names
                else:
                    # Fallback to removing known non-feature columns
                    feature_cols = [col for col in data_df.columns if col not in [
                        'trader_id', 'date', 'target_pnl', 'target_large_loss', 'trade_date'
                    ]]

                if not feature_cols:
                    logger.warning(f"No feature columns found for trader {trader_id}")
                    continue

                X = data_df[feature_cols]

                # Make prediction using the trader's model
                prediction = model.predict(X)[0]
                prediction_proba = model.predict_proba(X)[0, 1] if hasattr(model, 'predict_proba') else 0.5

                # Convert prediction to VaR-like interpretation
                # Since these models predict large loss probability, we'll use it directly
                var_prediction = prediction_proba * -5000  # Scale to dollar amount using probability
                loss_probability = prediction_proba

                predictions[trader_id] = {
                    'var_prediction': var_prediction,
                    'loss_probability': loss_probability,
                    'model_confidence': max(prediction_proba, 1 - prediction_proba),
                    'feature_count': len(feature_cols)
                }

                logger.info(f"Generated prediction for trader {trader_id}: VaR=${var_prediction:.2f}, P(Loss)={loss_probability:.3f}")

            except Exception as e:
                logger.error(f"Error generating prediction for trader {trader_id}: {e}")
                continue

        return predictions

    def classify_risk_level(self, trader_id: int, var_pred: float, loss_prob: float) -> str:
        """Classify risk level based on predictions and optimal thresholds."""
        # Get trader-specific thresholds
        thresholds = self.optimal_thresholds.get(trader_id, {
            'var_threshold': -5000,
            'loss_prob_threshold': 0.15
        })

        var_threshold = thresholds['var_threshold']
        loss_prob_threshold = thresholds['loss_prob_threshold']

        # Apply 70% risk reduction logic (optimal configuration from analysis)
        adjusted_var_threshold = var_threshold * 0.7  # More conservative
        adjusted_loss_prob_threshold = loss_prob_threshold * 0.7  # More conservative

        # High risk conditions (using optimal thresholds)
        if (loss_prob >= adjusted_loss_prob_threshold or var_pred <= adjusted_var_threshold):
            return 'high'

        # Medium risk conditions (using standard thresholds)
        elif (loss_prob >= loss_prob_threshold * 0.5 or var_pred <= var_threshold * 0.5):
            return 'medium'

        # Low risk
        else:
            return 'low'

    def generate_warning_signals(self, trader_id: int, row: pd.Series) -> List[str]:
        """Generate warning signals based on trader metrics and optimal thresholds."""
        signals = []

        # Get trader-specific thresholds
        thresholds = self.optimal_thresholds.get(trader_id, {})

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
        """Generate critical alerts for high-risk situations."""
        alerts = []

        if trader_names is None:
            trader_names = self.get_trader_names()

        for trader_id, pred_data in predictions_dict.items():
            var_amount = abs(pred_data['var_prediction'])
            loss_prob = pred_data['loss_probability']

            # Get trader name
            trader_name = trader_names.get(trader_id, f"ID {trader_id}")
            trader_label = f"Trader {trader_id} ({trader_name})"

            # Get trader-specific thresholds
            thresholds = self.optimal_thresholds.get(trader_id, {})
            var_threshold = abs(thresholds.get('var_threshold', -5000))
            loss_prob_threshold = thresholds.get('loss_prob_threshold', 0.15)

            # Critical risk: Exceeds optimal thresholds significantly
            if var_amount >= var_threshold * 1.5 or loss_prob >= loss_prob_threshold * 1.5:
                alerts.append({
                    'trader_id': str(trader_id),
                    'trader_label': trader_label,
                    'message': f"CRITICAL: Exceeds optimal risk thresholds. VaR: ${var_amount:,.0f} (threshold: ${var_threshold:,.0f}), Loss Prob: {loss_prob:.1%} (threshold: {loss_prob_threshold:.1%})"
                })

            # High risk: Exceeds optimal thresholds
            elif var_amount >= var_threshold or loss_prob >= loss_prob_threshold:
                alerts.append({
                    'trader_id': str(trader_id),
                    'trader_label': trader_label,
                    'message': f"WARNING: Risk approaching limits. Consider position reduction per optimal strategy."
                })

        return alerts

    def generate_daily_signals(self, target_date: str = None) -> Dict:
        """
        Generate complete daily signal report using trader-specific models.

        Args:
            target_date: Date string (YYYY-MM-DD) or None for latest

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

            # Calculate heatmap colors for each metric
            var_baseline = 2000
            var_color = self.calculate_heatmap_color(
                var_baseline,
                abs(pred_data['var_prediction']),
                'higher_better'
            )

            loss_prob_baseline = 0.15
            loss_prob_color = self.calculate_heatmap_color(
                loss_prob_baseline,
                pred_data['loss_probability'],
                'higher_better'
            )

            last_day_pnl_color = self.calculate_heatmap_color(
                db_metrics.get('last_trading_day_pnl', 0),
                db_metrics.get('avg_daily_pnl', 0),
                'higher_better'
            )

            sharpe_color = self.calculate_heatmap_color(
                db_metrics.get('sharpe_30d', 0),
                db_metrics.get('all_time_sharpe', 0),
                'higher_better'
            )

            avg_daily_pnl_color = self.calculate_heatmap_color(
                db_metrics.get('avg_daily_pnl', 0),
                db_metrics.get('all_time_avg_daily_pnl', 0),
                'higher_better'
            )

            avg_winning_color = self.calculate_heatmap_color(
                db_metrics.get('avg_winning_trade', 0),
                db_metrics.get('all_time_avg_winning_trade', 0),
                'higher_better'
            )

            avg_losing_color = self.calculate_heatmap_color(
                abs(db_metrics.get('avg_losing_trade', 0)),
                abs(db_metrics.get('all_time_avg_losing_trade', 0)),
                'lower_better'
            )

            highest_pnl_color = self.calculate_heatmap_color(
                db_metrics.get('highest_pnl', 0),
                db_metrics.get('all_time_highest_pnl', 0),
                'higher_better'
            )

            lowest_pnl_color = self.calculate_heatmap_color(
                abs(db_metrics.get('lowest_pnl', 0)),
                abs(db_metrics.get('all_time_lowest_pnl', 0)),
                'lower_better'
            )

            signal = {
                'trader_id': str(trader_id),
                'trader_name': trader_name,
                'trader_label': f"{trader_id} ({trader_name})",
                'risk_level': self.classify_risk_level(
                    trader_id,
                    pred_data['var_prediction'],
                    pred_data['loss_probability']
                ),
                'var_5pct': pred_data['var_prediction'],
                'loss_probability': pred_data['loss_probability'],
                'model_confidence': pred_data.get('model_confidence', 0.5),
                'last_trade_date': db_metrics.get('last_trade_date', 'N/A').replace('2025-', '') if db_metrics.get('last_trade_date', 'N/A') != 'N/A' else 'N/A',
                'last_trading_day_pnl': db_metrics.get('last_trading_day_pnl', 0),
                'sharpe_30d': db_metrics.get('sharpe_30d', 0),
                'avg_daily_pnl': db_metrics.get('avg_daily_pnl', 0),
                'avg_winning_trade': db_metrics.get('avg_winning_trade', 0),
                'avg_losing_trade': db_metrics.get('avg_losing_trade', 0),
                'highest_pnl': db_metrics.get('highest_pnl', 0),
                'lowest_pnl': db_metrics.get('lowest_pnl', 0),
                'volatility': data_series.get('pnl_std_7d', 0) if hasattr(data_series, 'get') else 0,
                'warning_signals': self.generate_warning_signals(trader_id, data_series),
                'optimal_thresholds': self.optimal_thresholds.get(trader_id, {}),
                # Heatmap colors for all relevant metrics
                'var_heatmap': {'bg': var_color[0], 'text': var_color[1], 'class': var_color[2]},
                'loss_prob_heatmap': {'bg': loss_prob_color[0], 'text': loss_prob_color[1], 'class': loss_prob_color[2]},
                'last_day_pnl_heatmap': {'bg': last_day_pnl_color[0], 'text': last_day_pnl_color[1], 'class': last_day_pnl_color[2]},
                'sharpe_heatmap': {'bg': sharpe_color[0], 'text': sharpe_color[1], 'class': sharpe_color[2]},
                'avg_daily_pnl_heatmap': {'bg': avg_daily_pnl_color[0], 'text': avg_daily_pnl_color[1], 'class': avg_daily_pnl_color[2]},
                'avg_winning_heatmap': {'bg': avg_winning_color[0], 'text': avg_winning_color[1], 'class': avg_winning_color[2]},
                'avg_losing_heatmap': {'bg': avg_losing_color[0], 'text': avg_losing_color[1], 'class': avg_losing_color[2]},
                'highest_pnl_heatmap': {'bg': highest_pnl_color[0], 'text': highest_pnl_color[1], 'class': highest_pnl_color[2]},
                'lowest_pnl_heatmap': {'bg': lowest_pnl_color[0], 'text': lowest_pnl_color[1], 'class': lowest_pnl_color[2]}
            }
            trader_signals.append(signal)

        # Sort by risk level and loss probability
        risk_order = {'high': 0, 'medium': 1, 'low': 2}
        trader_signals.sort(key=lambda x: (risk_order[x['risk_level']], -x['loss_probability']))

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
                'risk_reduction_level': '70%'  # From our analysis
            }
        else:
            summary_stats = {
                'avg_var': 0,
                'max_var': 0,
                'avg_loss_prob': 0,
                'max_loss_prob': 0,
                'total_warning_signals': 0,
                'using_optimal_thresholds': True,
                'risk_reduction_level': '70%'
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
