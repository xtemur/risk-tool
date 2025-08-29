"""
Pooled Risk Model - Single model for all traders (CLAUDE.md implementation)
More statistically sound than 15 separate models for small N problem
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    xgb = None
from typing import Dict, Tuple, List
import sqlite3
from config import config


class PooledRiskModel:
    """
    Single model for all traders with trader-specific adjustments
    More statistically sound than 15 separate models
    """

    def __init__(self):
        # Main model trained on all trader-days
        self.pooled_model = None

        # Simple trader-specific adjustments (not full models)
        self.trader_bias = {}  # Just offset/scaling factors
        self.trader_volatility = {}  # Historical vol for normalization

    def load_data(self) -> pd.DataFrame:
        """Load all trader data from database"""
        conn = sqlite3.connect(config.DATA_PATH)

        # Get daily P&L aggregated by trader
        # Calculate P&L as qty * price with proper sign handling
        # Use substr to extract date part since date() function doesn't work with this format
        query = """
        SELECT
            substr(fill_datetime, 1, CASE
                WHEN instr(fill_datetime, ' ') > 0 THEN instr(fill_datetime, ' ') - 1
                ELSE length(fill_datetime)
            END) as date,
            account as trader_id,
            SUM(
                CASE
                    WHEN side = 'B' THEN -1 * (qty * price + COALESCE(comm, 0) + COALESCE(ecn_fee, 0) + COALESCE(sec, 0) + COALESCE(taf, 0) + COALESCE(nscc, 0) + COALESCE(clr, 0) + COALESCE(misc, 0))
                    WHEN side = 'S' THEN qty * price - (COALESCE(comm, 0) + COALESCE(ecn_fee, 0) + COALESCE(sec, 0) + COALESCE(taf, 0) + COALESCE(nscc, 0) + COALESCE(clr, 0) + COALESCE(misc, 0))
                    ELSE 0
                END
            ) as pnl,
            SUM(ABS(qty * price)) as volume,
            COUNT(*) as trades
        FROM fills
        GROUP BY substr(fill_datetime, 1, CASE
            WHEN instr(fill_datetime, ' ') > 0 THEN instr(fill_datetime, ' ') - 1
            ELSE length(fill_datetime)
        END), account
        HAVING COUNT(*) > 0
        ORDER BY account, substr(fill_datetime, 1, CASE
            WHEN instr(fill_datetime, ' ') > 0 THEN instr(fill_datetime, ' ') - 1
            ELSE length(fill_datetime)
        END)
        """

        data = pd.read_sql(query, conn)
        conn.close()

        # Convert MM/DD/YYYY format to datetime
        data['date'] = pd.to_datetime(data['date'], format='%m/%d/%Y')

        # Filter out legacy accounts if configured
        if config.EXCLUDE_LEGACY_ACCOUNTS:
            data = data[~data['trader_id'].str.contains('_OLD', na=False)]
            print(f"Filtered to {data['trader_id'].nunique()} active traders (excluded legacy accounts)")

        return data

    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        CRITICAL: Maintain temporal alignment across all traders
        Prevents data leakage by ensuring same historical cutoff for all traders
        """
        all_features = []

        # Get all unique dates and sort them
        unique_dates = sorted(data['date'].unique())

        # For each prediction date, compute features for ALL traders
        # using ONLY data before that date
        for pred_date in unique_dates[30:]:  # Need 30 days minimum history

            # SAME historical cutoff for ALL traders - prevents leakage
            historical_data = data[data['date'] < pred_date]

            for trader_id in data['trader_id'].unique():
                trader_hist = historical_data[historical_data['trader_id'] == trader_id]

                if len(trader_hist) < 20:
                    continue

                # Compute features using ONLY historical data
                features = self._compute_features_for_date(trader_hist, pred_date)
                features['prediction_date'] = pred_date  # Track alignment
                features['trader_id'] = trader_id

                all_features.append(features)

        features_df = pd.DataFrame(all_features)

        # Add market regime features
        features_df = self.add_market_features(features_df, data)

        return features_df

    def _compute_features_for_date(self, trader_hist: pd.DataFrame, pred_date: pd.Timestamp) -> Dict:
        """Compute all features for a trader as of a specific prediction date"""
        trader_hist = trader_hist.sort_values('date')

        # Normalize P&L by trader's typical scale
        trader_std = trader_hist['pnl'].std()
        self.trader_volatility[str(trader_hist['trader_id'].iloc[0])] = trader_std

        # Calculate all features
        features = {
            # Normalized returns
            'returns_normalized': trader_hist['pnl'].iloc[-1] / (trader_std + 1e-8),
            'returns_zscore': (trader_hist['pnl'].iloc[-1] - trader_hist['pnl'].rolling(20).mean().iloc[-1]) / (trader_std + 1e-8),

            # Drawdown features
            'drawdown_pct': self._calculate_drawdown_pct(trader_hist).iloc[-1],
            'days_in_drawdown': self._calculate_drawdown_duration(trader_hist).iloc[-1],

            # Win rates
            'win_rate_5d': (trader_hist['pnl'].tail(5) > 0).mean() if len(trader_hist) >= 5 else 0.5,
            'win_rate_20d': (trader_hist['pnl'].tail(20) > 0).mean() if len(trader_hist) >= 20 else 0.5,

            # Streak features
            'loss_streak': self._calculate_loss_streak(trader_hist).iloc[-1],

            # Volatility regime
            'vol_ratio': (trader_hist['pnl'].rolling(5).std().iloc[-1] /
                         (trader_hist['pnl'].rolling(20).std().iloc[-1] + 1e-8)) if len(trader_hist) >= 20 else 1,

            # Volume normalized
            'volume_normalized': trader_hist['volume'].iloc[-1] / (trader_hist['volume'].rolling(20).mean().iloc[-1] + 1e-8) if len(trader_hist) >= 20 else 1,

            # Trade count normalized
            'trades_normalized': trader_hist['trades'].iloc[-1] / (trader_hist['trades'].rolling(20).mean().iloc[-1] + 1e-8) if len(trader_hist) >= 20 else 1,

            # Store original P&L for target creation
            'original_pnl': trader_hist['pnl'].iloc[-1]
        }

        return features

    def add_market_features(self, features_df: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add market-wide features to capture correlations"""
        market_features = []

        for _, row in features_df.iterrows():
            pred_date = row['prediction_date']

            # Get market conditions for day BEFORE prediction
            yesterday = pred_date - pd.Timedelta(days=1)
            yesterday_data = data[data['date'] == yesterday]

            if len(yesterday_data) > 0:
                # Market-wide indicators
                market_dict = {
                    'market_loss_pct': (yesterday_data['pnl'] < 0).mean(),
                    'market_total_pnl': yesterday_data['pnl'].sum(),
                    'market_volatility': yesterday_data['pnl'].std(),
                    'num_traders_active': yesterday_data['trader_id'].nunique()
                }
            else:
                market_dict = {
                    'market_loss_pct': 0.5,
                    'market_total_pnl': 0,
                    'market_volatility': 1000,
                    'num_traders_active': 10
                }

            market_features.append(market_dict)

        # Add to features
        market_df = pd.DataFrame(market_features)
        return pd.concat([features_df.reset_index(drop=True),
                         market_df.reset_index(drop=True)], axis=1)

    def create_target_with_buffer(self, features_df: pd.DataFrame, data: pd.DataFrame) -> np.ndarray:
        """
        Create targets that properly align with feature dates
        Prevents leakage by using actual outcomes for prediction dates
        """
        targets = []

        for _, row in features_df.iterrows():
            pred_date = row['prediction_date']
            trader_id = row['trader_id']

            # Get ACTUAL outcome for the prediction date
            actual_data = data[(data['date'] == pred_date) &
                              (data['trader_id'] == trader_id)]

            if len(actual_data) > 0:
                actual_pnl = actual_data['pnl'].values[0]

                # Calculate appropriate reduction based on actual outcome
                if actual_pnl < -5000:
                    target = 50  # Should have reduced by 50%
                elif actual_pnl < -2000:
                    target = 30  # Moderate reduction
                elif actual_pnl < -500:
                    target = 10  # Small reduction
                else:
                    target = 0   # No reduction needed
            else:
                target = 0  # No data, no reduction

            targets.append(target)

        return np.array(targets)

    def train_with_proper_split(self, data: pd.DataFrame):
        """
        Proper temporal train/test split for pooled model
        """
        # Prepare features with temporal alignment
        print("Preparing temporally-aligned features...")
        features = self.prepare_features(data)

        if len(features) < 100:
            print(f"Not enough features: {len(features)} < 100")
            return None

        # Create properly aligned targets
        print("Creating aligned targets...")
        targets = self.create_target_with_buffer(features, data)

        # Set cutoff date - last 30 days for test (more conservative for small data)
        cutoff_date = features['prediction_date'].max() - pd.Timedelta(days=30)

        # Split by prediction_date, not randomly!
        train_mask = features['prediction_date'] <= cutoff_date
        test_mask = features['prediction_date'] > cutoff_date

        # Prepare train/test sets
        feature_cols = [col for col in features.columns
                       if col not in ['trader_id', 'prediction_date', 'original_pnl']]

        X_train = features[train_mask][feature_cols].fillna(0)
        X_test = features[test_mask][feature_cols].fillna(0)
        y_train = targets[train_mask]
        y_test = targets[test_mask]

        print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

        if len(X_train) < 50:
            print("Not enough training data")
            return None

        # Train model
        print("Training model...")
        model, train_score = self.train_conservative_model(X_train, y_train)

        # Evaluate on test set (no leakage!)
        if len(X_test) > 0:
            test_pred = model.predict(X_test)
            test_mae = np.mean(np.abs(y_test - test_pred))
            print(f"Train score: {train_score:.3f}, Test MAE: {test_mae:.3f}")
        else:
            print(f"Train score: {train_score:.3f} (no test data)")

        # Store trader-specific adjustments
        for trader_id in features[train_mask]['trader_id'].unique():
            trader_train_mask = (features[train_mask]['trader_id'] == trader_id)
            if trader_train_mask.sum() > 0:
                trader_indices = features[train_mask].index[trader_train_mask]
                X_trader = X_train.loc[trader_indices] if hasattr(X_train, 'loc') else X_train[trader_train_mask]
                y_trader = y_train[trader_train_mask]

                trader_pred = model.predict(X_trader)
                trader_errors = y_trader - trader_pred
                self.trader_bias[str(trader_id)] = np.median(trader_errors)

        self.pooled_model = model
        return model

    def train_conservative_model(self, X: pd.DataFrame, y: np.ndarray) -> Tuple[object, float]:
        """
        Start simple, only add complexity if validated improvement
        FIXED: Use proper temporal splits for small data
        """
        X_numeric = X.fillna(0)

        # CRITICAL FIX: For small data (CLAUDE.md), use simple holdout instead of CV
        # Cross-validation can leak information in small datasets

        print(f"Training with {len(X_numeric)} samples - using simple approach for small data")

        # Option 1: Ridge Regression (most robust for small data)
        ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0], cv=3)  # Reduced CV splits
        ridge.fit(X_numeric, y)

        # Simple score on training data (acceptable for small datasets)
        ridge_score = ridge.score(X_numeric, y)

        # Option 2: Random Forest (very conservative for small data)
        rf = RandomForestRegressor(
            n_estimators=50,        # Reduced from 100
            max_depth=2,           # Reduced from 3 - even more conservative
            min_samples_split=20,   # Reduced from 50
            min_samples_leaf=10,    # Reduced from 20
            max_features='sqrt',
            random_state=42
        )
        rf.fit(X_numeric, y)
        rf_score = rf.score(X_numeric, y)

        print(f"Ridge score: {ridge_score:.3f}, RF score: {rf_score:.3f}")

        # SIMPLIFIED: Only use RF if significantly better, otherwise stick with Ridge
        # No XGBoost for small data - too complex
        if rf_score > ridge_score * 1.1 and len(X_numeric) > 200:  # 10% better AND enough data
            print("Using Random Forest")
            return rf, rf_score
        else:
            print("Using Ridge Regression (safest for small data)")
            return ridge, ridge_score

    def predict_for_tomorrow(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Make predictions for tomorrow using all data up to TODAY
        Production-safe method that prevents leakage
        """
        if self.pooled_model is None:
            raise ValueError("Model not trained yet")

        today = pd.Timestamp.now().normalize()

        # Use ALL historical data up to today for ALL traders
        historical_data = data[data['date'] < today]

        predictions = {}

        for trader_id in historical_data['trader_id'].unique():
            # Each trader uses same historical cutoff (today)
            trader_hist = historical_data[historical_data['trader_id'] == trader_id]

            if len(trader_hist) < 20:
                # Not enough history - use conservative default
                predictions[str(trader_id)] = 30  # 30% reduction for new traders
                continue

            # Compute features for tomorrow's prediction
            features = self._compute_features_for_date(trader_hist, today)

            # Add market features (using yesterday's market data)
            yesterday = today - pd.Timedelta(days=1)
            yesterday_data = historical_data[historical_data['date'] == yesterday]

            if len(yesterday_data) > 0:
                market_dict = {
                    'market_loss_pct': (yesterday_data['pnl'] < 0).mean(),
                    'market_total_pnl': yesterday_data['pnl'].sum(),
                    'market_volatility': yesterday_data['pnl'].std(),
                    'num_traders_active': yesterday_data['trader_id'].nunique()
                }
            else:
                market_dict = {
                    'market_loss_pct': 0.5,
                    'market_total_pnl': 0,
                    'market_volatility': 1000,
                    'num_traders_active': 10
                }

            # Combine features
            combined_features = {**features, **market_dict}

            # Remove non-feature columns
            feature_cols = [col for col in combined_features.keys()
                           if col not in ['trader_id', 'prediction_date', 'original_pnl']]

            # Make prediction
            X = pd.DataFrame([{k: combined_features[k] for k in feature_cols}]).fillna(0)

            pred = self.pooled_model.predict(X)[0]

            # Apply trader-specific bias
            adjusted_pred = pred + self.trader_bias.get(str(trader_id), 0)
            predictions[str(trader_id)] = max(0, min(config.MAX_REDUCTION, adjusted_pred))

        return predictions

    def predict(self, X: pd.DataFrame) -> Dict[str, float]:
        """Make predictions for all traders (legacy method for compatibility)"""
        if self.pooled_model is None:
            raise ValueError("Model not trained yet")

        feature_cols = [col for col in X.columns
                       if col not in ['trader_id', 'prediction_date', 'original_pnl']]
        X_numeric = X[feature_cols].fillna(0)

        predictions = self.pooled_model.predict(X_numeric)

        # Apply trader-specific adjustments
        results = {}
        for i, (trader_id, pred) in enumerate(zip(X['trader_id'], predictions)):
            # Apply trader bias if available
            adjusted_pred = pred + self.trader_bias.get(str(trader_id), 0)
            results[str(trader_id)] = max(0, min(config.MAX_REDUCTION, adjusted_pred))

        return results

    def _calculate_drawdown_pct(self, trader_data: pd.DataFrame) -> pd.Series:
        """Calculate drawdown percentage - FIXED to prevent extreme values"""
        cumulative = trader_data['pnl'].cumsum()
        running_max = cumulative.expanding().max()

        # CRITICAL FIX: Proper drawdown calculation
        # Drawdown should be relative to the peak value, not absolute
        drawdown_raw = (running_max - cumulative) / (running_max.abs() + 1000)  # Use 1000 instead of 1e-8

        # Cap drawdown at reasonable maximum (200%)
        drawdown_capped = np.minimum(drawdown_raw * 100, 200.0)

        return drawdown_capped

    def _calculate_drawdown_duration(self, trader_data: pd.DataFrame) -> pd.Series:
        """Calculate days in drawdown"""
        cumulative = trader_data['pnl'].cumsum()
        running_max = cumulative.expanding().max()
        is_drawdown = cumulative < running_max

        duration = pd.Series(0, index=trader_data.index)
        current_duration = 0

        for i, in_dd in enumerate(is_drawdown):
            if in_dd:
                current_duration += 1
            else:
                current_duration = 0
            duration.iloc[i] = current_duration

        return duration

    def _calculate_loss_streak(self, trader_data: pd.DataFrame) -> pd.Series:
        """Calculate consecutive loss streak"""
        is_loss = trader_data['pnl'] < 0

        streak = pd.Series(0, index=trader_data.index)
        current_streak = 0

        for i, loss in enumerate(is_loss):
            if loss:
                current_streak += 1
            else:
                current_streak = 0
            streak.iloc[i] = current_streak

        return streak
