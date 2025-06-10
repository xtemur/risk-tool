# src/evaluation/model_evaluator.py
"""
Model Evaluator
Comprehensive evaluation framework for trading models
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime
import json
from pathlib import Path

from src.models.base_model import BaseModel
from src.pipeline.feature_pipeline import FeaturePipeline
from src.backtesting.performance_metrics import PerformanceMetrics
from src.core.constants import TradingConstants as TC

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluation with multiple metrics
    """

    def __init__(self,
                 evaluation_dir: str = "evaluations",
                 backtesting_enabled: bool = True):
        self.evaluation_dir = Path(evaluation_dir)
        self.evaluation_dir.mkdir(exist_ok=True, parents=True)
        self.backtesting_enabled = backtesting_enabled

        # Performance calculator
        self.perf_calc = PerformanceMetrics()

    def evaluate_model(self,
                      model: BaseModel,
                      X_test: pd.DataFrame,
                      y_test: pd.Series,
                      trader_ids: Optional[pd.Series] = None,
                      prices: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Comprehensive model evaluation

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            trader_ids: Optional trader IDs for grouped analysis
            prices: Optional price data for backtesting

        Returns:
            Dictionary with all evaluation metrics
        """
        eval_start = datetime.now()

        # 1. Basic predictions
        predictions = model.predict(X_test)

        # 2. Statistical metrics
        stat_metrics = self._calculate_statistical_metrics(y_test, predictions)

        # 3. Trading metrics
        trading_metrics = self._calculate_trading_metrics(y_test, predictions)

        # 4. Risk metrics
        risk_metrics = self._calculate_risk_metrics(y_test, predictions)

        # 5. Per-trader analysis (if available)
        trader_metrics = {}
        if trader_ids is not None:
            trader_metrics = self._calculate_trader_metrics(
                y_test, predictions, trader_ids
            )

        # 6. Feature analysis
        feature_metrics = self._analyze_features(model, X_test)

        # 7. Temporal analysis
        temporal_metrics = self._calculate_temporal_metrics(
            y_test, predictions, X_test.index
        )

        # 8. Backtesting (if prices available)
        backtest_results = {}
        if self.backtesting_enabled and prices is not None:
            backtest_results = self._run_backtest(predictions, prices)

        # Combine all results
        evaluation = {
            'model_name': model.model_name,
            'evaluation_date': datetime.now().isoformat(),
            'evaluation_time': (datetime.now() - eval_start).total_seconds(),
            'n_samples': len(y_test),
            'statistical_metrics': stat_metrics,
            'trading_metrics': trading_metrics,
            'risk_metrics': risk_metrics,
            'trader_metrics': trader_metrics,
            'feature_metrics': feature_metrics,
            'temporal_metrics': temporal_metrics,
            'backtest_results': backtest_results
        }

        return evaluation

    def _calculate_statistical_metrics(self,
                                     y_true: pd.Series,
                                     y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate statistical performance metrics"""

        # Ensure alignment
        y_true = y_true.values if isinstance(y_true, pd.Series) else y_true

        # Basic metrics
        mae = np.mean(np.abs(y_true - y_pred))
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)

        # Relative metrics
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + TC.MIN_VARIANCE))) * 100

        # Directional accuracy
        direction_correct = np.mean((y_true > 0) == (y_pred > 0))

        # R-squared
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        r2 = 1 - (ss_res / (ss_tot + TC.MIN_VARIANCE))

        # Correlation
        correlation = np.corrcoef(y_true, y_pred)[0, 1]

        return {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'mape': float(mape),
            'r2': float(r2),
            'correlation': float(correlation),
            'directional_accuracy': float(direction_correct)
        }

    def _calculate_trading_metrics(self,
                                 y_true: pd.Series,
                                 y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate trading-specific metrics"""

        # Convert predictions to signals
        signal = np.sign(y_pred)

        # Calculate returns when following signals
        strategy_returns = signal * y_true

        # Win rate
        winning_trades = strategy_returns > 0
        win_rate = np.mean(winning_trades)

        # Average win/loss
        avg_win = strategy_returns[winning_trades].mean() if winning_trades.any() else 0
        avg_loss = strategy_returns[~winning_trades].mean() if (~winning_trades).any() else 0

        # Profit factor
        total_wins = strategy_returns[winning_trades].sum() if winning_trades.any() else 0
        total_losses = abs(strategy_returns[~winning_trades].sum()) if (~winning_trades).any() else 0
        profit_factor = total_wins / (total_losses + TC.MIN_VARIANCE)

        # Expectancy
        expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

        # Hit rate for different thresholds
        hit_rate_high = np.mean((y_pred > np.percentile(y_pred, 80)) & (y_true > 0))
        hit_rate_low = np.mean((y_pred < np.percentile(y_pred, 20)) & (y_true < 0))

        return {
            'win_rate': float(win_rate),
            'avg_win': float(avg_win),
            'avg_loss': float(avg_loss),
            'profit_factor': float(profit_factor),
            'expectancy': float(expectancy),
            'hit_rate_high_confidence': float(hit_rate_high),
            'hit_rate_low_confidence': float(hit_rate_low),
            'total_return': float(strategy_returns.sum())
        }

    def _calculate_risk_metrics(self,
                              y_true: pd.Series,
                              y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate risk-adjusted performance metrics"""

        # Strategy returns
        strategy_returns = np.sign(y_pred) * y_true

        # Sharpe ratio
        sharpe = (
            strategy_returns.mean() / (strategy_returns.std() + TC.MIN_VARIANCE) *
            np.sqrt(TC.TRADING_DAYS_PER_YEAR)
        )

        # Sortino ratio
        downside_returns = strategy_returns[strategy_returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else TC.MIN_VARIANCE
        sortino = (
            strategy_returns.mean() / downside_std *
            np.sqrt(TC.TRADING_DAYS_PER_YEAR)
        )

        # Maximum drawdown
        cumulative = (1 + pd.Series(strategy_returns)).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Calmar ratio
        annual_return = strategy_returns.mean() * TC.TRADING_DAYS_PER_YEAR
        calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0

        # Value at Risk
        var_95 = np.percentile(strategy_returns, 5)
        cvar_95 = strategy_returns[strategy_returns <= var_95].mean()

        # Tail ratio
        right_tail = np.percentile(strategy_returns, 95)
        left_tail = abs(np.percentile(strategy_returns, 5))
        tail_ratio = right_tail / (left_tail + TC.MIN_VARIANCE)

        return {
            'sharpe_ratio': float(sharpe),
            'sortino_ratio': float(sortino),
            'max_drawdown': float(max_drawdown),
            'calmar_ratio': float(calmar),
            'var_95': float(var_95),
            'cvar_95': float(cvar_95),
            'tail_ratio': float(tail_ratio),
            'volatility': float(strategy_returns.std() * np.sqrt(TC.TRADING_DAYS_PER_YEAR))
        }

    def _calculate_trader_metrics(self,
                                y_true: pd.Series,
                                y_pred: np.ndarray,
                                trader_ids: pd.Series) -> Dict[str, Any]:
        """Calculate per-trader performance metrics"""

        trader_results = {}

        for trader_id in trader_ids.unique():
            mask = trader_ids == trader_id

            if mask.sum() < 10:  # Need minimum samples
                continue

            trader_true = y_true[mask]
            trader_pred = y_pred[mask]

            # Calculate metrics for this trader
            trader_metrics = {
                'n_samples': int(mask.sum()),
                'rmse': float(np.sqrt(np.mean((trader_true - trader_pred) ** 2))),
                'directional_accuracy': float(np.mean((trader_true > 0) == (trader_pred > 0))),
                'total_return': float((np.sign(trader_pred) * trader_true).sum()),
                'correlation': float(np.corrcoef(trader_true, trader_pred)[0, 1])
            }

            trader_results[str(trader_id)] = trader_metrics

        # Summary statistics
        summary = {
            'n_traders': len(trader_results),
            'avg_rmse': np.mean([t['rmse'] for t in trader_results.values()]),
            'avg_accuracy': np.mean([t['directional_accuracy'] for t in trader_results.values()]),
            'best_trader': max(trader_results.items(),
                             key=lambda x: x[1]['directional_accuracy'])[0],
            'worst_trader': min(trader_results.items(),
                              key=lambda x: x[1]['directional_accuracy'])[0]
        }

        return {
            'summary': summary,
            'individual': trader_results
        }

    def _analyze_features(self, model: BaseModel, X_test: pd.DataFrame) -> Dict[str, Any]:
        """Analyze feature importance and usage"""

        feature_importance = model.get_feature_importance()

        if feature_importance.empty:
            return {}

        # Top features
        top_10 = feature_importance.head(10)

        # Feature categories
        categories = {}
        for _, row in feature_importance.iterrows():
            category = row['feature'].split('_')[0]
            if category not in categories:
                categories[category] = 0
            categories[category] += row['importance']

        # Feature stability (how consistent are values)
        feature_stability = {}
        for col in X_test.columns[:20]:  # Top 20 features
            if col in feature_importance['feature'].values:
                stability = 1 - (X_test[col].std() / (X_test[col].mean() + TC.MIN_VARIANCE))
                feature_stability[col] = float(stability)

        return {
            'top_10_features': top_10.to_dict('records'),
            'category_importance': categories,
            'n_features_used': len(feature_importance),
            'feature_stability': feature_stability,
            'importance_concentration': float(
                top_10['importance'].sum() / feature_importance['importance'].sum()
            )
        }

    def _calculate_temporal_metrics(self,
                                  y_true: pd.Series,
                                  y_pred: np.ndarray,
                                  dates: pd.DatetimeIndex) -> Dict[str, Any]:
        """Analyze performance over time"""

        # Create DataFrame for easier analysis
        results_df = pd.DataFrame({
            'date': dates,
            'y_true': y_true.values,
            'y_pred': y_pred,
            'error': y_true.values - y_pred,
            'abs_error': np.abs(y_true.values - y_pred)
        })

        # Monthly performance
        monthly = results_df.set_index('date').resample('M').agg({
            'abs_error': 'mean',
            'error': ['mean', 'std']
        })

        # Performance trend (is model getting better or worse?)
        first_half = results_df.iloc[:len(results_df)//2]['abs_error'].mean()
        second_half = results_df.iloc[len(results_df)//2:]['abs_error'].mean()
        performance_trend = (first_half - second_half) / first_half

        # Day of week analysis
        results_df['dow'] = results_df['date'].dt.dayofweek
        dow_performance = results_df.groupby('dow')['abs_error'].mean()

        # Error autocorrelation (are errors clustered?)
        error_autocorr = results_df['error'].autocorr(lag=1)

        return {
            'performance_trend': float(performance_trend),
            'error_autocorrelation': float(error_autocorr),
            'best_dow': int(dow_performance.idxmin()),
            'worst_dow': int(dow_performance.idxmax()),
            'monthly_mae_std': float(monthly['abs_error']['mean'].std()),
            'error_volatility_trend': float(
                np.polyfit(range(len(monthly)), monthly['error']['std'].values, 1)[0]
            )
        }

    def compare_models(self,
                      evaluations: List[Dict[str, Any]],
                      output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Compare multiple model evaluations

        Args:
            evaluations: List of evaluation results
            output_path: Optional path to save comparison

        Returns:
            DataFrame with model comparison
        """
        comparison_data = []

        for eval_result in evaluations:
            row = {
                'model_name': eval_result['model_name'],
                'n_samples': eval_result['n_samples']
            }

            # Add key metrics
            for metric_group in ['statistical_metrics', 'trading_metrics', 'risk_metrics']:
                if metric_group in eval_result:
                    for metric, value in eval_result[metric_group].items():
                        row[f"{metric_group.split('_')[0]}_{metric}"] = value

            comparison_data.append(row)

        comparison_df = pd.DataFrame(comparison_data)

        # Sort by Sharpe ratio
        if 'risk_sharpe_ratio' in comparison_df.columns:
            comparison_df = comparison_df.sort_values('risk_sharpe_ratio', ascending=False)

        if output_path:
            comparison_df.to_csv(output_path, index=False)

        return comparison_df
