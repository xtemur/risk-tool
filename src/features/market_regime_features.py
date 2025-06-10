"""
Market Regime Features
Captures market conditions and regime characteristics
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
import logging
from scipy import stats
from sklearn.mixture import GaussianMixture

from src.features.base_features import BaseFeatures
from src.core.constants import TradingConstants as TC

logger = logging.getLogger(__name__)


class MarketRegimeFeatures(BaseFeatures):
    """
    Market regime detection and characterization
    Identifies different market conditions that affect trading performance
    """

    def __init__(self):
        super().__init__(feature_prefix='regime', lookback_days=60)
        self.n_regimes = 3  # Bull, Bear, Sideways
        self.regime_model = None

    def create_features(self,
                       totals_df: pd.DataFrame,
                       fills_df: Optional[pd.DataFrame] = None,
                       as_of_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """
        Create market regime features

        Features include:
        - Volatility regimes
        - Trend strength indicators
        - Market stress metrics
        - Cross-sectional dispersion
        - Correlation breakdowns
        """
        # Prepare data
        df = self._ensure_datetime_index(totals_df.copy())
        df = self._apply_point_in_time(df, as_of_date)

        if df.empty:
            return pd.DataFrame()

        # First, calculate market-wide metrics
        market_features = self._calculate_market_metrics(df)

        # Initialize feature container
        all_features = []

        # Process each account with market context
        for account_id in df['account_id'].unique():
            acc_df = df[df['account_id'] == account_id].sort_index()

            if len(acc_df) < 10:
                continue

            features = pd.DataFrame(index=acc_df.index)
            features['account_id'] = account_id

            # 1. Volatility regime features
            features = self._add_volatility_regime_features(features, acc_df, market_features)

            # 2. Trend regime features
            features = self._add_trend_regime_features(features, acc_df, market_features)

            # 3. Market stress indicators
            features = self._add_stress_features(features, acc_df, market_features)

            # 4. Correlation features
            features = self._add_correlation_features(features, acc_df, market_features)

            # 5. Market microstructure regime
            if fills_df is not None and not fills_df.empty:
                features = self._add_microstructure_regime_features(
                    features, acc_df, fills_df[fills_df['account_id'] == account_id]
                )

            all_features.append(features)

        # Combine all accounts
        if not all_features:
            return pd.DataFrame()

        result = pd.concat(all_features, axis=0).sort_index()

        # Handle missing values
        result = self._handle_missing_data(result, method='forward_fill')

        # Add feature prefix
        result = self._add_feature_prefix(result)

        # Validate
        result = self._validate_features(result)

        return result

    def _calculate_market_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate aggregate market metrics across all traders"""

        # Group by date for market-wide calculations
        daily_market = df.groupby(df.index).agg({
            'net_pnl': ['sum', 'mean', 'std'],
            'quantity': 'sum',
            'orders_count': 'sum',
            'account_id': 'count'  # Number of active traders
        })

        daily_market.columns = ['_'.join(col).strip() for col in daily_market.columns]
        daily_market = daily_market.rename(columns={'account_id_count': 'active_traders'})

        # Market returns
        daily_market['market_return'] = daily_market['net_pnl_sum'].pct_change()

        # Market volatility (20-day rolling)
        daily_market['market_volatility'] = (
            daily_market['market_return'].rolling(20).std() * np.sqrt(TC.TRADING_DAYS_PER_YEAR)
        )

        # Cross-sectional dispersion
        daily_market['cross_sectional_std'] = daily_market['net_pnl_std']
        daily_market['dispersion_ratio'] = (
            daily_market['cross_sectional_std'] /
            (daily_market['net_pnl_mean'].abs() + TC.MIN_VARIANCE)
        )

        # Market activity
        daily_market['market_volume'] = daily_market['quantity_sum']
        daily_market['market_orders'] = daily_market['orders_count_sum']

        # Volatility regimes using EWMA
        daily_market['volatility_short'] = (
            daily_market['market_return'].ewm(span=5, adjust=False).std()
        )
        daily_market['volatility_long'] = (
            daily_market['market_return'].ewm(span=20, adjust=False).std()
        )
        daily_market['vol_regime_ratio'] = (
            daily_market['volatility_short'] /
            (daily_market['volatility_long'] + TC.MIN_VARIANCE)
        )

        return daily_market

    def _add_volatility_regime_features(self,
                                       features: pd.DataFrame,
                                       acc_df: pd.DataFrame,
                                       market_features: pd.DataFrame) -> pd.DataFrame:
        """Add volatility regime features"""

        # Align market features with account data
        market_aligned = market_features.reindex(acc_df.index, method='ffill')

        # 1. Volatility level classification
        vol_percentiles = market_aligned['market_volatility'].rolling(252).rank(pct=True)
        features['vol_percentile'] = vol_percentiles

        # Volatility regime (Low, Medium, High)
        features['vol_regime_low'] = (vol_percentiles < 0.33).astype(int)
        features['vol_regime_medium'] = ((vol_percentiles >= 0.33) & (vol_percentiles < 0.67)).astype(int)
        features['vol_regime_high'] = (vol_percentiles >= 0.67).astype(int)

        # 2. Volatility trend
        features['vol_trend'] = market_aligned['vol_regime_ratio']
        features['vol_increasing'] = (features['vol_trend'] > 1.2).astype(int)
        features['vol_decreasing'] = (features['vol_trend'] < 0.8).astype(int)

        # 3. Volatility of volatility (vol clustering)
        features['vol_of_vol'] = market_aligned['market_volatility'].rolling(20).std()

        # 4. Realized vs implied volatility proxy
        # Using high-low range as proxy for implied vol
        if 'quantity' in acc_df.columns:
            daily_range = acc_df['net_pnl'].rolling(1).apply(
                lambda x: (x.max() - x.min()) / (x.mean() + TC.MIN_VARIANCE) if len(x) > 0 else 0
            )
            features['range_volatility'] = daily_range.rolling(20).mean()
            features['vol_premium'] = (
                features['range_volatility'] -
                market_aligned['market_volatility']
            )

        # 5. Personal volatility vs market
        personal_vol = acc_df['net_pnl'].rolling(20).std()
        features['personal_vs_market_vol'] = (
            personal_vol / (market_aligned['market_volatility'] + TC.MIN_VARIANCE)
        )

        # 6. Volatility regime persistence
        features['vol_regime_days'] = self._calculate_regime_duration(vol_percentiles > 0.5)

        return features

    def _add_trend_regime_features(self,
                                  features: pd.DataFrame,
                                  acc_df: pd.DataFrame,
                                  market_features: pd.DataFrame) -> pd.DataFrame:
        """Add trend regime features"""

        market_aligned = market_features.reindex(acc_df.index, method='ffill')

        # 1. Market trend strength
        market_returns = market_aligned['market_return'].fillna(0)

        # Simple moving averages
        features['market_sma_5'] = market_returns.rolling(5).mean()
        features['market_sma_20'] = market_returns.rolling(20).mean()
        features['market_sma_60'] = market_returns.rolling(60).mean()

        # Trend direction
        features['trend_up'] = (features['market_sma_5'] > features['market_sma_20']).astype(int)
        features['trend_strong_up'] = (
            (features['market_sma_5'] > features['market_sma_20']) &
            (features['market_sma_20'] > features['market_sma_60'])
        ).astype(int)

        # 2. Trend strength (ADX-like)
        dm_plus = market_returns.clip(lower=0).rolling(14).sum()
        dm_minus = (-market_returns.clip(upper=0)).rolling(14).sum()
        tr = market_returns.abs().rolling(14).sum()

        di_plus = dm_plus / (tr + TC.MIN_VARIANCE)
        di_minus = dm_minus / (tr + TC.MIN_VARIANCE)

        dx = np.abs(di_plus - di_minus) / (di_plus + di_minus + TC.MIN_VARIANCE)
        features['trend_strength'] = dx.rolling(14).mean()

        # 3. Market momentum
        features['market_momentum_5d'] = market_returns.rolling(5).sum()
        features['market_momentum_20d'] = market_returns.rolling(20).sum()

        # 4. Trend consistency
        positive_days = (market_returns > 0).rolling(20).sum()
        features['trend_consistency'] = np.abs(positive_days - 10) / 10  # 0 = perfectly mixed

        # 5. Personal performance vs market trend
        personal_returns = acc_df['net_pnl'].pct_change()
        features['personal_vs_market_return'] = (
            personal_returns - market_returns
        ).rolling(20).mean()

        # Beta to market
        if len(personal_returns.dropna()) > 20:
            features['market_beta'] = personal_returns.rolling(60).apply(
                lambda x: self._calculate_beta(x, market_returns.loc[x.index])
            )

        return features

    def _add_stress_features(self,
                           features: pd.DataFrame,
                           acc_df: pd.DataFrame,
                           market_features: pd.DataFrame) -> pd.DataFrame:
        """Add market stress indicators"""

        market_aligned = market_features.reindex(acc_df.index, method='ffill')

        # 1. Drawdown-based stress
        market_cumsum = market_aligned['net_pnl_sum'].cumsum()
        market_running_max = market_cumsum.expanding().max()
        market_drawdown = (market_cumsum - market_running_max) / (market_running_max + TC.MIN_VARIANCE)

        features['market_drawdown'] = market_drawdown
        features['market_stress_level'] = market_drawdown.abs()
        features['high_stress'] = (features['market_stress_level'] > 0.1).astype(int)

        # 2. Correlation breakdown indicator
        features['correlation_stress'] = market_aligned['dispersion_ratio'].rolling(20).mean()

        # 3. Liquidity stress (volume-based)
        volume_ma = market_aligned['market_volume'].rolling(20).mean()
        features['volume_ratio'] = market_aligned['market_volume'] / (volume_ma + TC.MIN_VARIANCE)
        features['liquidity_stress'] = (features['volume_ratio'] < 0.5).astype(int)

        # 4. Participation rate
        features['participation_rate'] = (
            market_aligned['active_traders'] /
            market_aligned['active_traders'].rolling(60).mean()
        )

        # 5. Tail risk indicators
        market_returns = market_aligned['market_return'].fillna(0)

        # Value at Risk (VaR)
        features['market_var_95'] = market_returns.rolling(60).quantile(0.05)
        features['market_cvar_95'] = market_returns.rolling(60).apply(
            lambda x: x[x <= x.quantile(0.05)].mean() if len(x[x <= x.quantile(0.05)]) > 0 else np.nan
        )

        # Tail ratio
        left_tail = market_returns.rolling(60).apply(lambda x: x[x < x.median()].std())
        right_tail = market_returns.rolling(60).apply(lambda x: x[x > x.median()].std())
        features['tail_ratio'] = left_tail / (right_tail + TC.MIN_VARIANCE)

        # 6. Stress persistence
        features['stress_duration'] = self._calculate_regime_duration(
            features['market_stress_level'] > 0.05
        )

        return features

    def _add_correlation_features(self,
                                 features: pd.DataFrame,
                                 acc_df: pd.DataFrame,
                                 market_features: pd.DataFrame) -> pd.DataFrame:
        """Add correlation-based features"""

        market_aligned = market_features.reindex(acc_df.index, method='ffill')

        # 1. Rolling correlation with market
        personal_returns = acc_df['net_pnl'].pct_change()
        market_returns = market_aligned['market_return']

        features['correlation_20d'] = personal_returns.rolling(20).corr(market_returns)
        features['correlation_60d'] = personal_returns.rolling(60).corr(market_returns)

        # Correlation stability
        features['correlation_stability'] = (
            features['correlation_20d'] - features['correlation_60d']
        ).abs()

        # 2. Correlation regime
        features['high_correlation'] = (features['correlation_60d'] > 0.7).astype(int)
        features['negative_correlation'] = (features['correlation_60d'] < -0.3).astype(int)
        features['decorrelated'] = (features['correlation_60d'].abs() < 0.3).astype(int)

        # 3. Beta stability
        if 'market_beta' in features.columns:
            features['beta_stability'] = features['market_beta'].rolling(20).std()

        # 4. Diversification benefit
        # Lower correlation during stress = better diversification
        stress_correlation = personal_returns.where(
            market_drawdown < -0.05
        ).rolling(20).corr(market_returns)

        features['stress_correlation'] = stress_correlation
        features['diversification_benefit'] = (
            features['correlation_60d'] - stress_correlation
        )

        return features

    def _add_microstructure_regime_features(self,
                                           features: pd.DataFrame,
                                           acc_df: pd.DataFrame,
                                           fills_df: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure regime features"""

        if fills_df.empty:
            return features

        fills_df = fills_df.copy()
        if 'datetime' in fills_df.columns:
            fills_df['date'] = pd.to_datetime(fills_df['datetime']).dt.date
            fills_df['hour'] = pd.to_datetime(fills_df['datetime']).dt.hour
            fills_df['time_diff'] = pd.to_datetime(fills_df['datetime']).diff()

        # Daily microstructure metrics
        for date in features.index:
            date_fills = fills_df[fills_df['date'] == date.date()]

            if date_fills.empty:
                continue

            # 1. Execution speed (time between trades)
            if 'time_diff' in date_fills.columns:
                avg_time_between = date_fills['time_diff'].mean().total_seconds()
                features.loc[date, 'avg_execution_speed'] = 1 / (avg_time_between + 1)

            # 2. Price impact proxy
            if len(date_fills) > 1:
                # Serial correlation of trades (momentum vs reversion)
                price_changes = date_fills['price'].pct_change()
                if len(price_changes.dropna()) > 1:
                    features.loc[date, 'price_serial_corr'] = price_changes.autocorr(lag=1)

            # 3. Trade clustering
            if 'hour' in date_fills.columns:
                # Measure how clustered trades are in time
                time_std = date_fills['hour'].std()
                features.loc[date, 'trade_time_clustering'] = 1 / (time_std + 1)

            # 4. Symbol concentration changes
            symbol_entropy = -(
                date_fills['symbol'].value_counts(normalize=True).apply(lambda x: x * np.log(x + 1e-8))
            ).sum()
            features.loc[date, 'symbol_diversity'] = symbol_entropy

        # Fill forward microstructure features
        micro_cols = ['avg_execution_speed', 'price_serial_corr',
                     'trade_time_clustering', 'symbol_diversity']
        for col in micro_cols:
            if col in features.columns:
                features[col] = features[col].fillna(method='ffill', limit=5)

        # 5. Microstructure regime classification
        if all(col in features.columns for col in ['avg_execution_speed', 'trade_time_clustering']):
            # High speed + high clustering = momentum regime
            features['momentum_regime'] = (
                (features['avg_execution_speed'] > features['avg_execution_speed'].median()) &
                (features['trade_time_clustering'] > features['trade_time_clustering'].median())
            ).astype(int)

            # Low speed + low clustering = mean reversion regime
            features['mean_reversion_regime'] = (
                (features['avg_execution_speed'] < features['avg_execution_speed'].median()) &
                (features['trade_time_clustering'] < features['trade_time_clustering'].median())
            ).astype(int)

        return features

    def _calculate_regime_duration(self, regime_indicator: pd.Series) -> pd.Series:
        """Calculate how long current regime has persisted"""
        # Create regime change indicator
        regime_change = regime_indicator != regime_indicator.shift(1)

        # Create regime groups
        regime_groups = regime_change.cumsum()

        # Count duration within each regime
        duration = regime_indicator.groupby(regime_groups).cumcount() + 1

        # Only count duration when in the regime (indicator is True)
        return duration.where(regime_indicator, 0)

    def _calculate_beta(self, returns1: pd.Series, returns2: pd.Series) -> float:
        """Calculate beta between two return series"""
        try:
            # Remove NaN values
            mask = ~(returns1.isna() | returns2.isna())
            r1 = returns1[mask]
            r2 = returns2[mask]

            if len(r1) < 10:  # Need minimum data
                return np.nan

            # Calculate covariance and variance
            covariance = np.cov(r1, r2)[0, 1]
            variance = np.var(r2)

            if variance > TC.MIN_VARIANCE:
                return covariance / variance
            else:
                return 0.0

        except Exception as e:
            logger.debug(f"Beta calculation failed: {e}")
            return np.nan

    def fit_regime_model(self, features: pd.DataFrame) -> 'MarketRegimeFeatures':
        """
        Fit a regime classification model
        Can be used for more sophisticated regime detection
        """
        # Extract key features for regime detection
        regime_features = []
        feature_cols = ['market_volatility', 'market_return', 'dispersion_ratio']

        for col in feature_cols:
            if col in features.columns:
                regime_features.append(features[col].values.reshape(-1, 1))

        if regime_features:
            X = np.hstack(regime_features)

            # Fit Gaussian Mixture Model for regime detection
            self.regime_model = GaussianMixture(
                n_components=self.n_regimes,
                covariance_type='full',
                random_state=42
            )

            # Remove NaN values for fitting
            X_clean = X[~np.isnan(X).any(axis=1)]
            if len(X_clean) > self.n_regimes * 10:
                self.regime_model.fit(X_clean)
                logger.info(f"Fitted regime model with {self.n_regimes} regimes")

        return self
