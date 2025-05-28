import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')


class PersonalizedARIMAModel:
    """
    ARIMA model with external regressors for each trader
    Predicts probability of positive P&L tomorrow
    """

    def __init__(self, trader_id: str):
        self.trader_id = trader_id
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.best_order = None
        self.threshold = 0.5  # Decision threshold

    def check_stationarity(self, series):
        """Check if series is stationary using ADF test"""
        result = adfuller(series.dropna())
        return result[1] < 0.05  # p-value < 0.05 means stationary

    def find_best_arima_order(self, y, exog, max_p=3, max_d=2, max_q=3):
        """Find best ARIMA order using AIC"""
        best_aic = np.inf
        best_order = None

        # Try different combinations
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        model = ARIMA(y, order=(p, d, q), exog=exog)
                        fitted = model.fit()
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_order = (p, d, q)
                    except:
                        continue

        return best_order if best_order else (1, 1, 1)  # Default order

    def fit(self, features_df: pd.DataFrame, feature_columns: list):
        """Fit ARIMA model with external features"""

        self.feature_columns = feature_columns

        # Prepare data
        X = features_df[feature_columns].values
        y = features_df['target'].values

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Find best ARIMA order (simplified search for MVP)
        self.best_order = self.find_best_arima_order(
            y, X_scaled, max_p=2, max_d=1, max_q=2
        )

        # Fit final model
        self.model = ARIMA(y, order=self.best_order, exog=X_scaled)
        self.fitted_model = self.model.fit()

        # Calculate in-sample predictions for threshold optimization
        in_sample_pred = self.fitted_model.fittedvalues

        # Find optimal threshold using in-sample data
        thresholds = np.arange(0.3, 0.7, 0.05)
        best_threshold = 0.5
        best_profit = -np.inf

        for thresh in thresholds:
            signals = (in_sample_pred > thresh).astype(int)
            # Calculate profit if we only traded on positive signals
            profit = np.sum(features_df['target_value'].values * signals)
            if profit > best_profit:
                best_profit = profit
                best_threshold = thresh

        self.threshold = best_threshold

        return self

    def predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Predict trading signals for test data"""

        X = features_df[self.feature_columns].values
        X_scaled = self.scaler.transform(X)

        # Get predictions
        predictions = self.fitted_model.forecast(
            steps=len(features_df),
            exog=X_scaled
        )

        # Create output dataframe
        results = features_df[['date', 'account_id', 'net_pnl', 'target_value']].copy()
        results['risk_score'] = predictions
        results['trade_signal'] = (predictions > self.threshold).astype(int)

        return results

    def evaluate(self, predictions_df: pd.DataFrame) -> dict:
        """Evaluate model performance"""

        # Calculate metrics
        actual_pnl = predictions_df['net_pnl'].sum()

        # P&L if following signals (only trade when signal = 1)
        signal_pnl = (predictions_df['net_pnl'] * predictions_df['trade_signal']).sum()

        # Days traded
        total_days = len(predictions_df)
        traded_days = predictions_df['trade_signal'].sum()

        # Win rate when trading
        trades = predictions_df[predictions_df['trade_signal'] == 1]
        win_rate = (trades['net_pnl'] > 0).mean() if len(trades) > 0 else 0

        # Risk-adjusted metrics
        avoided_losses = predictions_df[
            (predictions_df['trade_signal'] == 0) & (predictions_df['net_pnl'] < 0)
        ]['net_pnl'].sum()

        missed_gains = predictions_df[
            (predictions_df['trade_signal'] == 0) & (predictions_df['net_pnl'] > 0)
        ]['net_pnl'].sum()

        return {
            'trader_id': self.trader_id,
            'actual_total_pnl': actual_pnl,
            'signal_total_pnl': signal_pnl,
            'pnl_improvement': signal_pnl - actual_pnl,
            'pnl_improvement_pct': ((signal_pnl - actual_pnl) / abs(actual_pnl)) * 100 if actual_pnl != 0 else 0,
            'total_days': total_days,
            'traded_days': traded_days,
            'trade_reduction_pct': ((total_days - traded_days) / total_days) * 100,
            'win_rate_when_trading': win_rate * 100,
            'avoided_losses': abs(avoided_losses),
            'missed_gains': missed_gains,
            'threshold': self.threshold,
            'arima_order': self.best_order
        }
