import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')


class PersonalizedLGBMModel:
    """
    LightGBM model for each trader with rolling window cross-validation
    Predicts probability of positive P&L tomorrow
    """

    def __init__(self, trader_id: str):
        self.trader_id = trader_id
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.best_params = None
        self.threshold = 0.5
        self.feature_importance = None

    def optimize_hyperparameters(self, X_train, y_train, n_splits=5):
        """Optimize hyperparameters using TimeSeriesSplit cross-validation"""

        # Define parameter grid
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [15, 31, 50],
            'min_child_samples': [20, 30, 40],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }

        # Use TimeSeriesSplit for rolling window validation
        tscv = TimeSeriesSplit(n_splits=n_splits)

        best_score = -np.inf
        best_params = {}

        # Simplified grid search (for MVP speed)
        for n_est in [50, 100]:
            for depth in [3, 5]:
                for lr in [0.05, 0.1]:
                    params = {
                        'objective': 'binary',
                        'metric': 'binary_logloss',
                        'boosting_type': 'gbdt',
                        'n_estimators': n_est,
                        'max_depth': depth,
                        'learning_rate': lr,
                        'num_leaves': 31,
                        'min_child_samples': 30,
                        'subsample': 0.9,
                        'colsample_bytree': 0.9,
                        'random_state': 42,
                        'verbose': -1
                    }

                    # Cross-validation scores
                    cv_scores = []

                    for train_idx, val_idx in tscv.split(X_train):
                        X_cv_train, X_cv_val = X_train[train_idx], X_train[val_idx]
                        y_cv_train, y_cv_val = y_train[train_idx], y_train[val_idx]

                        model = lgb.LGBMClassifier(**params)
                        model.fit(X_cv_train, y_cv_train)

                        # Predict probabilities
                        y_pred_proba = model.predict_proba(X_cv_val)[:, 1]

                        # Calculate AUC score
                        try:
                            score = roc_auc_score(y_cv_val, y_pred_proba)
                            cv_scores.append(score)
                        except:
                            cv_scores.append(0.5)

                    avg_score = np.mean(cv_scores)

                    if avg_score > best_score:
                        best_score = avg_score
                        best_params = params.copy()

        return best_params

    def fit(self, features_df: pd.DataFrame, feature_columns: list):
        """Fit LightGBM model with optimized hyperparameters"""

        self.feature_columns = feature_columns

        # Prepare data
        X = features_df[feature_columns].values
        y = features_df['target'].values

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Optimize hyperparameters
        print(f"Optimizing hyperparameters for {self.trader_id}...")
        self.best_params = self.optimize_hyperparameters(X_scaled, y)

        # Train final model on all training data
        self.model = lgb.LGBMClassifier(**self.best_params)
        self.model.fit(X_scaled, y)

        # Get feature importance
        self.feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        # Calculate in-sample predictions for threshold optimization
        in_sample_pred = self.model.predict_proba(X_scaled)[:, 1]

        # Find optimal threshold
        thresholds = np.arange(0.3, 0.7, 0.05)
        best_threshold = 0.5
        best_profit = -np.inf

        for thresh in thresholds:
            signals = (in_sample_pred > thresh).astype(int)
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

        # Get probability predictions
        predictions = self.model.predict_proba(X_scaled)[:, 1]

        # Create output dataframe
        results = features_df[['date', 'account_id', 'net_pnl', 'target_value']].copy()
        results['risk_score'] = predictions
        results['trade_signal'] = (predictions > self.threshold).astype(int)

        return results

    def evaluate(self, predictions_df: pd.DataFrame) -> dict:
        """Evaluate model performance"""

        # Calculate metrics
        actual_pnl = predictions_df['net_pnl'].sum()

        # P&L if following signals
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

        # Get top 5 important features
        top_features = self.feature_importance.head(5)['feature'].tolist()

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
            'best_params': self.best_params,
            'top_features': top_features
        }
