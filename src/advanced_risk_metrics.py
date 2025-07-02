# src/advanced_risk_metrics.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from scipy import stats
from sklearn.metrics import mean_squared_error
import warnings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_conditional_var(returns: pd.Series, alpha: float = 0.05) -> float:
    """
    Calculate Conditional Value-at-Risk (Expected Shortfall).

    Args:
        returns: Series of returns/PnL
        alpha: Confidence level (e.g., 0.05 for 5%)

    Returns:
        CVaR value
    """
    var_threshold = returns.quantile(alpha)
    tail_losses = returns[returns <= var_threshold]

    if len(tail_losses) == 0:
        return var_threshold

    cvar = tail_losses.mean()
    return cvar


def calculate_expected_shortfall_backtest(backtest_results: pd.DataFrame, alpha: float = 0.05) -> Dict:
    """
    Backtest Expected Shortfall predictions.

    Args:
        backtest_results: DataFrame with true_pnl and pred_var columns
        alpha: Confidence level

    Returns:
        Dict with ES backtest metrics
    """
    logger.info(f"Calculating Expected Shortfall backtest at {alpha*100}% level...")

    # Calculate actual CVaR for each period
    window_size = 21  # 1-month rolling window
    actual_cvar = []
    predicted_cvar = []

    for i in range(window_size, len(backtest_results)):
        window_data = backtest_results.iloc[i-window_size:i]

        # Actual CVaR from realized returns
        actual_es = calculate_conditional_var(window_data['true_pnl'], alpha)
        actual_cvar.append(actual_es)

        # Predicted CVaR (average of VaR predictions in tail)
        var_threshold = window_data['pred_var'].quantile(alpha)
        tail_predictions = window_data['pred_var'][window_data['pred_var'] <= var_threshold]
        if len(tail_predictions) > 0:
            predicted_es = tail_predictions.mean()
        else:
            predicted_es = var_threshold
        predicted_cvar.append(predicted_es)

    # Calculate backtest metrics
    if len(actual_cvar) > 0:
        correlation = np.corrcoef(actual_cvar, predicted_cvar)[0, 1] if len(actual_cvar) > 1 else 0
        rmse = np.sqrt(mean_squared_error(actual_cvar, predicted_cvar))
        mae = np.mean(np.abs(np.array(actual_cvar) - np.array(predicted_cvar)))

        # Direction accuracy (did ES increase/decrease correctly?)
        if len(actual_cvar) > 1:
            actual_changes = np.diff(actual_cvar)
            predicted_changes = np.diff(predicted_cvar)
            direction_accuracy = np.mean(np.sign(actual_changes) == np.sign(predicted_changes))
        else:
            direction_accuracy = 0
    else:
        correlation = rmse = mae = direction_accuracy = 0

    es_metrics = {
        'cvar_correlation': correlation,
        'cvar_rmse': rmse,
        'cvar_mae': mae,
        'cvar_direction_accuracy': direction_accuracy,
        'n_periods': len(actual_cvar)
    }

    logger.info(f"Expected Shortfall Backtest Results:")
    logger.info(f"  Correlation: {correlation:.3f}")
    logger.info(f"  RMSE: {rmse:.2f}")
    logger.info(f"  Direction Accuracy: {direction_accuracy:.1%}")

    return es_metrics


def implement_multi_horizon_predictions(df: pd.DataFrame, config: Dict, model_var, model_loss) -> Dict:
    """
    Generate predictions for multiple time horizons.

    Args:
        df: Feature DataFrame
        config: Configuration dictionary
        model_var: Trained VaR model
        model_loss: Trained loss model

    Returns:
        Dict with multi-horizon predictions
    """
    logger.info("Generating multi-horizon risk predictions...")

    horizons = config.get('advanced_metrics', {}).get('multi_horizon', [1, 5, 21])

    # Prepare features
    feature_cols = [col for col in df.columns if col not in [
        'account_id', 'trade_date', 'target_pnl', 'target_large_loss',
        'daily_pnl', 'large_loss_threshold'
    ]]

    X = df[feature_cols]

    # Base predictions (1-day horizon)
    base_var_pred = model_var.predict(X)
    base_loss_pred = model_loss.predict_proba(X)[:, 1]

    multi_horizon_results = {
        'horizons': horizons,
        'predictions': {}
    }

    for horizon in horizons:
        logger.info(f"Calculating {horizon}-day horizon predictions...")

        if horizon == 1:
            # Use base predictions
            var_pred = base_var_pred
            loss_pred = base_loss_pred
        else:
            # Scale predictions for longer horizons
            # VaR scaling: assumes square-root scaling for variance
            scaling_factor = np.sqrt(horizon)
            var_pred = base_var_pred * scaling_factor

            # Loss probability: convert to per-period probability
            # P(no loss in H days) = (1 - p_daily)^H
            # P(loss in H days) = 1 - (1 - p_daily)^H
            daily_prob = base_loss_pred
            # Avoid numerical issues with probabilities close to 1
            daily_prob = np.clip(daily_prob, 1e-8, 1-1e-8)
            horizon_prob = 1 - (1 - daily_prob) ** horizon
            loss_pred = horizon_prob

        # Calculate realized targets for validation (if available)
        realized_targets = None
        if horizon <= 21:  # Only calculate if we have enough future data
            try:
                # Calculate realized multi-period returns
                groupby_account = df.groupby('account_id')
                realized_pnl = groupby_account['daily_pnl'].transform(
                    lambda x: x.shift(-horizon, fill_value=np.nan).rolling(horizon, min_periods=1).sum()
                )

                # Calculate realized large loss indicator
                realized_large_loss = (realized_pnl < df['large_loss_threshold']).astype(int)

                realized_targets = {
                    'realized_pnl': realized_pnl,
                    'realized_large_loss': realized_large_loss
                }

                # Validation metrics
                valid_mask = ~realized_pnl.isnull()
                if valid_mask.sum() > 0:
                    # VaR violation rate
                    violation_rate = (realized_pnl[valid_mask] < var_pred[valid_mask]).mean()

                    # Loss prediction AUC
                    from sklearn.metrics import roc_auc_score
                    try:
                        auc = roc_auc_score(realized_large_loss[valid_mask], loss_pred[valid_mask])
                    except:
                        auc = 0.5  # Default for edge cases

                    logger.info(f"  {horizon}-day horizon validation:")
                    logger.info(f"    VaR violation rate: {violation_rate:.1%}")
                    logger.info(f"    Loss prediction AUC: {auc:.3f}")

                    validation_metrics = {
                        'violation_rate': violation_rate,
                        'auc': auc,
                        'valid_samples': valid_mask.sum()
                    }
                else:
                    validation_metrics = None
            except Exception as e:
                logger.warning(f"Could not calculate realized targets for {horizon}-day horizon: {str(e)}")
                validation_metrics = None
        else:
            validation_metrics = None

        multi_horizon_results['predictions'][f'{horizon}d'] = {
            'var_predictions': var_pred,
            'loss_predictions': loss_pred,
            'realized_targets': realized_targets,
            'validation_metrics': validation_metrics,
            'scaling_factor': scaling_factor if horizon > 1 else 1.0
        }

    logger.info(f"Multi-horizon predictions completed for {len(horizons)} horizons")
    return multi_horizon_results


def detect_market_regimes(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Simple market regime detection using volatility and return characteristics.

    Args:
        df: Feature DataFrame
        config: Configuration dictionary

    Returns:
        DataFrame with regime indicators
    """
    logger.info("Detecting market regimes...")

    df_regime = df.copy()

    # Calculate market-wide metrics (aggregate across all traders)
    daily_market_pnl = df.groupby('trade_date')['daily_pnl'].agg(['sum', 'std', 'count']).reset_index()
    daily_market_pnl.columns = ['trade_date', 'market_pnl', 'market_volatility', 'active_traders']

    # Calculate rolling regime indicators
    window = 21  # 1-month window

    # Volatility regime (high/low based on rolling percentiles)
    vol_rolling = daily_market_pnl['market_volatility'].rolling(window).mean()
    vol_threshold_high = vol_rolling.quantile(0.75)
    vol_threshold_low = vol_rolling.quantile(0.25)

    daily_market_pnl['volatility_regime'] = 'medium'
    daily_market_pnl.loc[vol_rolling > vol_threshold_high, 'volatility_regime'] = 'high'
    daily_market_pnl.loc[vol_rolling < vol_threshold_low, 'volatility_regime'] = 'low'

    # Return regime (bull/bear based on rolling returns)
    return_rolling = daily_market_pnl['market_pnl'].rolling(window).mean()
    return_threshold = return_rolling.quantile([0.33, 0.67])

    daily_market_pnl['return_regime'] = 'neutral'
    daily_market_pnl.loc[return_rolling > return_threshold.iloc[1], 'return_regime'] = 'bull'
    daily_market_pnl.loc[return_rolling < return_threshold.iloc[0], 'return_regime'] = 'bear'

    # Stress regime (high volatility + negative returns)
    stress_condition = (
        (daily_market_pnl['volatility_regime'] == 'high') &
        (daily_market_pnl['return_regime'] == 'bear')
    )
    daily_market_pnl['stress_regime'] = stress_condition.astype(int)

    # Merge back to main dataframe
    df_regime = df_regime.merge(
        daily_market_pnl[['trade_date', 'volatility_regime', 'return_regime', 'stress_regime']],
        on='trade_date',
        how='left'
    )

    # Log regime statistics
    regime_stats = df_regime.groupby(['volatility_regime', 'return_regime']).size().unstack(fill_value=0)
    stress_days = df_regime['stress_regime'].sum()
    total_days = len(df_regime['trade_date'].unique())

    logger.info(f"Market regime detection completed:")
    logger.info(f"  Stress regime days: {stress_days}/{total_days} ({stress_days/total_days:.1%})")
    logger.info(f"  Regime distribution:\n{regime_stats}")

    return df_regime


def calculate_regime_specific_metrics(backtest_results: pd.DataFrame, df_with_regimes: pd.DataFrame) -> Dict:
    """
    Calculate performance metrics specific to different market regimes.

    Args:
        backtest_results: Backtest results DataFrame
        df_with_regimes: DataFrame with regime indicators

    Returns:
        Dict with regime-specific metrics
    """
    logger.info("Calculating regime-specific performance metrics...")

    # Merge regime information with backtest results
    merged_data = backtest_results.merge(
        df_with_regimes[['trade_date', 'account_id', 'volatility_regime', 'return_regime', 'stress_regime']],
        on=['trade_date', 'account_id'],
        how='left'
    )

    regime_metrics = {}

    # Calculate metrics for each volatility regime
    for regime in ['low', 'medium', 'high']:
        regime_data = merged_data[merged_data['volatility_regime'] == regime]

        if len(regime_data) > 10:  # Minimum samples for meaningful metrics
            violation_rate = (regime_data['true_pnl'] < regime_data['pred_var']).mean()

            try:
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(regime_data['true_large_loss'], regime_data['pred_loss_proba'])
            except:
                auc = 0.5

            avg_loss_size = regime_data.loc[
                regime_data['true_pnl'] < regime_data['pred_var'], 'true_pnl'
            ].mean()

            regime_metrics[f'volatility_{regime}'] = {
                'violation_rate': violation_rate,
                'auc': auc,
                'avg_violation_size': avg_loss_size,
                'n_samples': len(regime_data)
            }

    # Calculate metrics for stress periods
    stress_data = merged_data[merged_data['stress_regime'] == 1]
    if len(stress_data) > 5:
        stress_violation_rate = (stress_data['true_pnl'] < stress_data['pred_var']).mean()
        try:
            stress_auc = roc_auc_score(stress_data['true_large_loss'], stress_data['pred_loss_proba'])
        except:
            stress_auc = 0.5

        regime_metrics['stress_periods'] = {
            'violation_rate': stress_violation_rate,
            'auc': stress_auc,
            'n_samples': len(stress_data)
        }

    # Calculate overall regime stability
    regime_violation_rates = []
    for regime in ['low', 'medium', 'high']:
        if f'volatility_{regime}' in regime_metrics:
            regime_violation_rates.append(regime_metrics[f'volatility_{regime}']['violation_rate'])

    if len(regime_violation_rates) > 1:
        regime_stability = 1 - (np.std(regime_violation_rates) / np.mean(regime_violation_rates))
    else:
        regime_stability = 1.0

    regime_metrics['regime_stability'] = regime_stability

    # Log results
    logger.info("Regime-specific performance:")
    for regime_name, metrics in regime_metrics.items():
        if isinstance(metrics, dict) and 'violation_rate' in metrics:
            logger.info(f"  {regime_name}: VaR violations {metrics['violation_rate']:.1%}, "
                       f"AUC {metrics['auc']:.3f} ({metrics['n_samples']} samples)")

    logger.info(f"Overall regime stability: {regime_stability:.3f}")

    return regime_metrics


def generate_advanced_risk_report(backtest_results: pd.DataFrame, df: pd.DataFrame,
                                config: Dict, model_var, model_loss) -> Dict:
    """
    Generate comprehensive advanced risk metrics report.

    Args:
        backtest_results: Backtest results DataFrame
        df: Feature DataFrame
        config: Configuration dictionary
        model_var: Trained VaR model
        model_loss: Trained loss model

    Returns:
        Dict with all advanced risk metrics
    """
    logger.info("Generating comprehensive advanced risk metrics report...")

    advanced_report = {
        'generation_timestamp': pd.Timestamp.now().isoformat(),
        'config_used': config.get('advanced_metrics', {})
    }

    # 1. Expected Shortfall Analysis
    if config.get('advanced_metrics', {}).get('enable_cvar', True):
        cvar_alpha = config.get('advanced_metrics', {}).get('cvar_alpha', 0.05)
        es_metrics = calculate_expected_shortfall_backtest(backtest_results, cvar_alpha)
        advanced_report['expected_shortfall'] = es_metrics

    # 2. Multi-horizon Predictions
    multi_horizon_config = config.get('advanced_metrics', {}).get('multi_horizon', [1, 5, 21])
    if len(multi_horizon_config) > 1:
        multi_horizon_results = implement_multi_horizon_predictions(df, config, model_var, model_loss)
        advanced_report['multi_horizon'] = multi_horizon_results

    # 3. Market Regime Analysis
    if config.get('advanced_metrics', {}).get('regime_detection', True):
        df_with_regimes = detect_market_regimes(df, config)
        regime_metrics = calculate_regime_specific_metrics(backtest_results, df_with_regimes)
        advanced_report['regime_analysis'] = regime_metrics

        # Save regime-enhanced dataset for future use
        regime_path = config['paths']['processed_features'].replace('.parquet', '_with_regimes.parquet').replace('.pkl', '_with_regimes.pkl')
        try:
            df_with_regimes.to_parquet(regime_path, index=False)
            logger.info(f"Regime-enhanced dataset saved to {regime_path}")
        except:
            import pickle
            with open(regime_path, 'wb') as f:
                pickle.dump(df_with_regimes, f)

    # 4. Risk Concentration Analysis
    trader_risk_concentration = calculate_risk_concentration(backtest_results)
    advanced_report['risk_concentration'] = trader_risk_concentration

    # 5. Tail Risk Metrics
    tail_risk_metrics = calculate_tail_risk_metrics(backtest_results)
    advanced_report['tail_risk'] = tail_risk_metrics

    logger.info("Advanced risk metrics report generation completed")

    return advanced_report


def calculate_risk_concentration(backtest_results: pd.DataFrame) -> Dict:
    """Calculate risk concentration metrics across traders."""

    # Risk concentration by trader
    trader_var = backtest_results.groupby('account_id')['pred_var'].mean().abs()
    trader_loss_prob = backtest_results.groupby('account_id')['pred_loss_proba'].mean()

    # Herfindahl-Hirschman Index for risk concentration
    var_shares = trader_var / trader_var.sum()
    risk_hhi = (var_shares ** 2).sum()

    # Top trader concentration
    top_3_var_share = var_shares.nlargest(3).sum()
    top_1_var_share = var_shares.max()

    concentration_metrics = {
        'risk_hhi': risk_hhi,
        'top_1_var_share': top_1_var_share,
        'top_3_var_share': top_3_var_share,
        'n_traders': len(trader_var),
        'var_gini_coefficient': calculate_gini_coefficient(trader_var.values)
    }

    return concentration_metrics


def calculate_gini_coefficient(values: np.ndarray) -> float:
    """Calculate Gini coefficient for inequality measurement."""
    values = np.sort(values)
    n = len(values)
    cumsum = np.cumsum(values)
    return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n


def calculate_tail_risk_metrics(backtest_results: pd.DataFrame) -> Dict:
    """Calculate various tail risk metrics."""

    returns = backtest_results['true_pnl']

    # Basic tail metrics
    var_95 = returns.quantile(0.05)
    var_99 = returns.quantile(0.01)
    cvar_95 = calculate_conditional_var(returns, 0.05)
    cvar_99 = calculate_conditional_var(returns, 0.01)

    # Extreme value statistics
    tail_losses = returns[returns < returns.quantile(0.1)]
    if len(tail_losses) > 10:
        # Fit generalized extreme value distribution
        try:
            from scipy.stats import genextreme
            params = genextreme.fit(tail_losses)
            tail_shape = params[0]  # Shape parameter
        except:
            tail_shape = np.nan
    else:
        tail_shape = np.nan

    # Maximum drawdown simulation
    cumulative_returns = returns.cumsum()
    running_max = cumulative_returns.expanding().max()
    drawdowns = cumulative_returns - running_max
    max_drawdown = drawdowns.min()

    tail_metrics = {
        'var_95': var_95,
        'var_99': var_99,
        'cvar_95': cvar_95,
        'cvar_99': cvar_99,
        'max_drawdown': max_drawdown,
        'tail_shape_parameter': tail_shape,
        'extreme_loss_frequency': (returns < var_99).mean()
    }

    return tail_metrics
