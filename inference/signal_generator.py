# inference/signal_generator.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from datetime import datetime, timedelta
import sys
import os
import pickle

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, load_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SignalGenerator:
    """Generate risk signals for traders based on trained models."""

    def __init__(self, config_path: str = 'configs/main_config.yaml'):
        """Initialize signal generator with configuration."""
        self.config = load_config(config_path)
        self.var_model = None
        self.loss_model = None
        self.feature_data = None

    def load_models(self):
        """Load trained VaR and loss models."""
        model_dir = self.config['paths']['model_dir']

        logger.info("Loading trained models...")
        self.var_model = load_model(os.path.join(model_dir, 'lgbm_var_model.joblib'))
        self.loss_model = load_model(os.path.join(model_dir, 'lgbm_loss_model.joblib'))
        logger.info("Models loaded successfully")

    def load_latest_features(self) -> pd.DataFrame:
        """Load the most recent feature data."""
        # Try pickle first, then parquet
        pickle_path = self.config['paths']['processed_features'].replace('.parquet', '.pkl')

        if os.path.exists(pickle_path):
            logger.info(f"Loading features from {pickle_path}")
            with open(pickle_path, 'rb') as f:
                self.feature_data = pickle.load(f)
        else:
            self.feature_data = pd.read_parquet(self.config['paths']['processed_features'])

        # Add date_idx if not present
        if 'date_idx' not in self.feature_data.columns:
            unique_dates = self.feature_data['trade_date'].unique()
            date_to_idx = {date: idx for idx, date in enumerate(unique_dates)}
            self.feature_data['date_idx'] = self.feature_data['trade_date'].map(date_to_idx)

        return self.feature_data

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
                    MAX(trade_date) as last_trade_date,
                    AVG(daily_pnl) as avg_daily_pnl,
                    SUM(daily_pnl) as total_pnl,
                    COUNT(*) as trading_days,
                    CASE
                        WHEN COUNT(*) > 1 THEN
                            (AVG(daily_pnl) / NULLIF(
                                SQRT(SUM((daily_pnl - (SELECT AVG(daily_pnl) FROM daily_pnl d2 WHERE d2.account_id = daily_pnl.account_id)) *
                                        (daily_pnl - (SELECT AVG(daily_pnl) FROM daily_pnl d2 WHERE d2.account_id = daily_pnl.account_id))) /
                                     (COUNT(*) - 1)), 0)) * SQRT(30)
                        ELSE 0
                    END as sharpe_30d,
                    AVG(CASE WHEN winning_trades > 0 THEN winning_pnl / winning_trades END) as avg_winning_trade,
                    AVG(CASE WHEN losing_trades > 0 THEN losing_pnl / losing_trades END) as avg_losing_trade,
                    MAX(highest_trade_pnl) as highest_pnl,
                    MIN(lowest_trade_pnl) as lowest_pnl
                FROM daily_pnl
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
                            AVG(daily_pnl) / NULLIF(
                                SQRT((SUM(daily_pnl * daily_pnl) - SUM(daily_pnl) * SUM(daily_pnl) / COUNT(*)) / (COUNT(*) - 1)), 0) * SQRT(30)
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
                ts.last_trade_date,
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
        if all_time_value == 0 or current_value is None or all_time_value is None:
            return '#F5F5F5', '#000000', 'neutral'

        # Calculate performance ratio
        if metric_type == 'higher_better':
            ratio = current_value / all_time_value if all_time_value != 0 else 1
        else:  # lower_better (for losses)
            ratio = all_time_value / current_value if current_value != 0 else 1

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

    def get_latest_trader_data(self, lookback_days: int = 1) -> pd.DataFrame:
        """Get the most recent data for each active trader."""
        if self.feature_data is None:
            self.load_latest_features()

        # Get the latest date in the data
        latest_date = self.feature_data['trade_date'].max()
        cutoff_date = latest_date - timedelta(days=lookback_days)

        # Get recent data
        recent_data = self.feature_data[self.feature_data['trade_date'] >= cutoff_date]

        # Get the latest record for each trader
        latest_records = recent_data.sort_values('trade_date').groupby('account_id').last()

        return latest_records

    def generate_predictions(self, trader_data: pd.DataFrame) -> pd.DataFrame:
        """Generate VaR and loss predictions for traders."""
        if self.var_model is None or self.loss_model is None:
            self.load_models()

        # Load selected features from model metadata
        model_dir = self.config['paths']['model_dir']
        metadata_path = os.path.join(model_dir, 'model_metadata.json')

        if os.path.exists(metadata_path):
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            selected_features = metadata.get('selected_features', [])
            logger.info(f"Using {len(selected_features)} selected features from model training")
        else:
            # Fallback to all features if metadata not available
            logger.warning("Model metadata not found, using all available features")
            selected_features = [col for col in trader_data.columns if col not in [
                'account_id', 'trade_date', 'target_pnl', 'target_large_loss',
                'daily_pnl', 'large_loss_threshold'
            ]]

        # Check if all selected features are available
        missing_features = [f for f in selected_features if f not in trader_data.columns]
        if missing_features:
            logger.error(f"Missing required features: {missing_features}")
            raise ValueError(f"Missing features required for prediction: {missing_features}")

        # Use only the selected features for prediction
        X = trader_data[selected_features]
        logger.info(f"Making predictions with {len(selected_features)} features: {selected_features}")

        # Generate predictions
        var_predictions = self.var_model.predict(X)
        loss_probabilities = self.loss_model.predict_proba(X)[:, 1]

        # Add predictions to dataframe
        trader_data = trader_data.copy()
        trader_data['var_prediction'] = var_predictions
        trader_data['loss_probability'] = loss_probabilities

        return trader_data

    def classify_risk_level(self, var_pred: float, loss_prob: float, volatility: float) -> str:
        """Classify risk level based on predictions with improved logic."""
        # Normalize VaR to positive number for easier comparison
        var_amount = abs(var_pred)

        # Dynamic thresholds based on actual data patterns
        high_loss_prob = 0.15  # Lowered based on actual data
        medium_loss_prob = 0.06

        # VaR thresholds (absolute values)
        extreme_var = 10000  # $10K+ is extreme risk
        high_var = 4000      # $4K+ is high risk

        # Volatility threshold
        high_vol_threshold = 2000

        # High risk conditions
        if (loss_prob >= high_loss_prob or
            var_amount >= extreme_var or
            (loss_prob >= medium_loss_prob and var_amount >= high_var)):
            return 'high'

        # Medium risk conditions
        elif (loss_prob >= medium_loss_prob or
              var_amount >= high_var or
              volatility > high_vol_threshold):
            return 'medium'

        # Low risk
        else:
            return 'low'

    def generate_warning_signals(self, row: pd.Series) -> List[str]:
        """Generate warning signals based on trader metrics."""
        signals = []

        # Check for revenge trading
        if row.get('revenge_trading_proxy', 0) == 1:
            signals.append('REVENGE_TRADING')

        # Check for high volatility
        if row.get('rolling_vol_7', 0) > 2000:  # Adjust threshold
            signals.append('HIGH_VOLATILITY')

        # Check for poor recent performance
        if row.get('win_rate_21d', 1) < 0.3:
            signals.append('LOW_WIN_RATE')

        # Check for large drawdown
        if row.get('rolling_max_drawdown', 0) < -5000:  # Adjust threshold
            signals.append('LARGE_DRAWDOWN')

        # Check for elevated risk
        if row.get('loss_probability', 0) > 0.4:
            signals.append('ELEVATED_RISK')

        return signals

    def generate_alerts(self, predictions_df: pd.DataFrame, trader_names: Dict[int, str] = None) -> List[Dict]:
        """Generate critical alerts for high-risk situations."""
        alerts = []

        if trader_names is None:
            trader_names = self.get_trader_names()

        for idx, row in predictions_df.iterrows():
            trader_id = idx if isinstance(idx, (int, str)) else row.get('account_id', idx)
            var_amount = abs(row['var_prediction'])

            # Get trader name
            trader_name = trader_names.get(int(trader_id), f"ID {trader_id}")
            trader_label = f"Trader {trader_id} ({trader_name})"

            # Extreme VaR levels (>$10K)
            if var_amount >= 10000:
                alerts.append({
                    'trader_id': str(trader_id),
                    'trader_label': trader_label,
                    'message': f"Extreme VaR level (${var_amount:,.0f}). Consider immediate position size reduction."
                })

            # High loss probability (>15%)
            elif row['loss_probability'] >= 0.15:
                alerts.append({
                    'trader_id': str(trader_id),
                    'trader_label': trader_label,
                    'message': f"High loss probability ({row['loss_probability']:.1%}). Monitor closely."
                })

            # Multiple warning signals combined with medium-high risk
            warning_count = len(self.generate_warning_signals(row))
            if warning_count >= 3 and var_amount >= 4000:
                alerts.append({
                    'trader_id': str(trader_id),
                    'trader_label': trader_label,
                    'message': f"Multiple risk factors detected ({warning_count} warnings) with significant VaR exposure."
                })

            # Revenge trading with any elevated risk
            if row.get('revenge_trading_proxy', 0) == 1 and row['loss_probability'] > 0.05:
                alerts.append({
                    'trader_id': str(trader_id),
                    'trader_label': trader_label,
                    'message': "Revenge trading pattern detected. Behavioral intervention recommended."
                })

        return alerts

    def generate_daily_signals(self, target_date: str = None) -> Dict:
        """
        Generate complete daily signal report.

        Args:
            target_date: Date string (YYYY-MM-DD) or None for latest

        Returns:
            Dictionary with signal data for email template
        """
        logger.info("Generating daily risk signals...")

        # Load latest data
        latest_data = self.get_latest_trader_data()

        # Generate predictions
        predictions = self.generate_predictions(latest_data)

        # Get trader names and database metrics
        trader_names = self.get_trader_names()
        trader_metrics = self.get_trader_metrics_from_db()

        # Prepare trader signals with enhanced data
        trader_signals = []
        for idx, row in predictions.iterrows():
            trader_id = idx if isinstance(idx, (int, str)) else row.get('account_id', idx)
            trader_name = trader_names.get(int(trader_id), f"ID {trader_id}")

            # Get database metrics for this trader
            db_metrics = trader_metrics.get(int(trader_id), {})

            # Calculate heatmap colors for each metric
            # For VaR: Compare current VaR to a baseline of $2000 (typical acceptable level)
            var_baseline = 2000
            var_color = self.calculate_heatmap_color(
                var_baseline,  # Baseline is "good"
                abs(row['var_prediction']),  # Current VaR (higher is worse)
                'higher_better'  # In this flipped comparison, higher baseline vs current = better
            )

            # For Loss Probability: Compare to 15% baseline (acceptable loss probability)
            loss_prob_baseline = 0.15
            loss_prob_color = self.calculate_heatmap_color(
                loss_prob_baseline,  # 15% is acceptable baseline
                row['loss_probability'],  # Current probability
                'higher_better'  # Higher baseline vs current = better (lower current prob is better)
            )
            last_day_pnl_color = self.calculate_heatmap_color(
                db_metrics.get('last_trading_day_pnl', 0),
                db_metrics.get('avg_daily_pnl', 0),  # Compare to 30-day average
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
                'lower_better'  # Lower absolute loss is better
            )
            highest_pnl_color = self.calculate_heatmap_color(
                db_metrics.get('highest_pnl', 0),
                db_metrics.get('all_time_highest_pnl', 0),
                'higher_better'
            )
            lowest_pnl_color = self.calculate_heatmap_color(
                abs(db_metrics.get('lowest_pnl', 0)),
                abs(db_metrics.get('all_time_lowest_pnl', 0)),
                'lower_better'  # Lower absolute loss is better
            )

            signal = {
                'trader_id': str(trader_id),
                'trader_name': trader_name,
                'trader_label': f"{trader_id} ({trader_name})",
                'risk_level': self.classify_risk_level(
                    row['var_prediction'],
                    row['loss_probability'],
                    row.get('rolling_vol_7', 0)
                ),
                'var_5pct': row['var_prediction'],
                'loss_probability': row['loss_probability'],
                'last_trade_date': db_metrics.get('last_trade_date', 'N/A').replace('2025-', '') if db_metrics.get('last_trade_date', 'N/A') != 'N/A' else 'N/A',
                'last_trading_day_pnl': db_metrics.get('last_trading_day_pnl', 0),
                'sharpe_30d': db_metrics.get('sharpe_30d', 0),
                'avg_daily_pnl': db_metrics.get('avg_daily_pnl', 0),
                'avg_winning_trade': db_metrics.get('avg_winning_trade', 0),
                'avg_losing_trade': db_metrics.get('avg_losing_trade', 0),
                'highest_pnl': db_metrics.get('highest_pnl', 0),
                'lowest_pnl': db_metrics.get('lowest_pnl', 0),
                'volatility': row.get('rolling_vol_7', 0),
                'warning_signals': self.generate_warning_signals(row),
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
        var_amounts = [abs(s['var_5pct']) for s in trader_signals]
        loss_probs = [s['loss_probability'] for s in trader_signals]

        summary_stats = {
            'avg_var': np.mean(var_amounts),
            'max_var': np.max(var_amounts),
            'avg_loss_prob': np.mean(loss_probs),
            'max_loss_prob': np.max(loss_probs),
            'total_warning_signals': sum(len(s['warning_signals']) for s in trader_signals)
        }

        # Prepare final signal data
        signal_data = {
            'date': target_date or datetime.now().strftime('%Y-%m-%d'),
            'trader_signals': trader_signals,
            'alerts': alerts,
            'summary_stats': summary_stats
        }

        logger.info(f"Generated signals for {len(trader_signals)} traders with {len(alerts)} alerts")

        return signal_data


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

    return signals


if __name__ == '__main__':
    test_signal_generator()
