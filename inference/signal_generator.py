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

    def generate_alerts(self, predictions_df: pd.DataFrame) -> List[Dict]:
        """Generate critical alerts for high-risk situations."""
        alerts = []

        for idx, row in predictions_df.iterrows():
            trader_id = idx if isinstance(idx, (int, str)) else row.get('account_id', idx)
            var_amount = abs(row['var_prediction'])

            # Extreme VaR levels (>$10K)
            if var_amount >= 10000:
                alerts.append({
                    'trader_id': str(trader_id),
                    'message': f"Extreme VaR level (${var_amount:,.0f}). Consider immediate position size reduction."
                })

            # High loss probability (>15%)
            elif row['loss_probability'] >= 0.15:
                alerts.append({
                    'trader_id': str(trader_id),
                    'message': f"High loss probability ({row['loss_probability']:.1%}). Monitor closely."
                })

            # Multiple warning signals combined with medium-high risk
            warning_count = len(self.generate_warning_signals(row))
            if warning_count >= 3 and var_amount >= 4000:
                alerts.append({
                    'trader_id': str(trader_id),
                    'message': f"Multiple risk factors detected ({warning_count} warnings) with significant VaR exposure."
                })

            # Revenge trading with any elevated risk
            if row.get('revenge_trading_proxy', 0) == 1 and row['loss_probability'] > 0.05:
                alerts.append({
                    'trader_id': str(trader_id),
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

        # Prepare trader signals
        trader_signals = []
        for idx, row in predictions.iterrows():
            trader_id = idx if isinstance(idx, (int, str)) else row.get('account_id', idx)

            signal = {
                'trader_id': str(trader_id),
                'risk_level': self.classify_risk_level(
                    row['var_prediction'],
                    row['loss_probability'],
                    row.get('rolling_vol_7', 0)
                ),
                'var_5pct': row['var_prediction'],
                'loss_probability': row['loss_probability'],
                'current_pnl': row.get('daily_pnl', 0),
                'volatility': row.get('rolling_vol_7', 0),
                'warning_signals': self.generate_warning_signals(row)
            }
            trader_signals.append(signal)

        # Sort by risk level and loss probability
        risk_order = {'high': 0, 'medium': 1, 'low': 2}
        trader_signals.sort(key=lambda x: (risk_order[x['risk_level']], -x['loss_probability']))

        # Generate alerts
        alerts = self.generate_alerts(predictions)

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
