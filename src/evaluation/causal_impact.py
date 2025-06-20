import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple
from datetime import datetime

class CausalImpactAnalyzer:
    """
    Analyze the causal impact of using the risk model vs baseline trading.
    Shows what would have happened if traders followed the model's signals.
    """

    def __init__(self):
        pass

    def analyze_trading_strategy_impact(self,
                                      predictions_df: pd.DataFrame,
                                      strategy_type: str = 'position_sizing') -> Dict:
        """
        Analyze the causal impact of different trading strategies based on risk signals.

        Args:
            predictions_df: DataFrame with predictions and actual results
            strategy_type: 'position_sizing', 'trade_filtering', or 'combined'

        Returns:
            Dictionary with causal impact metrics
        """

        # Baseline: Original trading performance
        baseline_pnl = predictions_df['actual_pnl'].sum()
        baseline_returns = predictions_df['actual_pnl'].values

        # Strategy 1: Position Sizing Based on Risk Signals
        strategy_returns = self._apply_position_sizing_strategy(predictions_df)

        # Strategy 2: Trade Filtering (avoid high-risk days)
        filtered_returns = self._apply_trade_filtering_strategy(predictions_df)

        # Strategy 3: Combined approach
        combined_returns = self._apply_combined_strategy(predictions_df)

        # Calculate impact metrics
        results = {
            'baseline': self._calculate_strategy_metrics(baseline_returns, 'Baseline Trading'),
            'position_sizing': self._calculate_strategy_metrics(strategy_returns, 'Risk-Adjusted Position Sizing'),
            'trade_filtering': self._calculate_strategy_metrics(filtered_returns, 'High-Risk Day Filtering'),
            'combined': self._calculate_strategy_metrics(combined_returns, 'Combined Strategy')
        }

        # Add comparative metrics
        results['causal_impact'] = self._calculate_causal_impact(results)

        return results

    def _apply_position_sizing_strategy(self, df: pd.DataFrame) -> np.ndarray:
        """
        Apply position sizing based on risk signals:
        - High Risk: 50% position size
        - Neutral: 100% position size
        - Low Risk: 150% position size
        """
        position_multipliers = {0: 0.5, 1: 1.0, 2: 1.5}  # High, Neutral, Low risk

        strategy_returns = []
        for _, row in df.iterrows():
            multiplier = position_multipliers[row['risk_signal']]
            adjusted_return = row['actual_pnl'] * multiplier
            strategy_returns.append(adjusted_return)

        return np.array(strategy_returns)

    def _apply_trade_filtering_strategy(self, df: pd.DataFrame) -> np.ndarray:
        """
        Filter out high-risk trading days (set returns to 0).
        """
        filtered_returns = []
        for _, row in df.iterrows():
            if row['risk_signal'] == 0:  # High risk - avoid trading
                filtered_returns.append(0.0)
            else:
                filtered_returns.append(row['actual_pnl'])

        return np.array(filtered_returns)

    def _apply_combined_strategy(self, df: pd.DataFrame) -> np.ndarray:
        """
        Combined strategy: Filter high-risk days AND adjust position sizing.
        """
        strategy_returns = []
        for _, row in df.iterrows():
            if row['risk_signal'] == 0:  # High risk - avoid trading
                strategy_returns.append(0.0)
            elif row['risk_signal'] == 1:  # Neutral - normal position
                strategy_returns.append(row['actual_pnl'])
            else:  # Low risk - increase position
                strategy_returns.append(row['actual_pnl'] * 1.3)

        return np.array(strategy_returns)

    def _calculate_strategy_metrics(self, returns: np.ndarray, strategy_name: str) -> Dict:
        """Calculate comprehensive metrics for a trading strategy."""

        if len(returns) == 0:
            return {'error': 'No returns data'}

        # Remove zero returns for some calculations
        non_zero_returns = returns[returns != 0]

        # Basic metrics
        total_pnl = np.sum(returns)
        avg_daily_pnl = np.mean(returns)
        volatility = np.std(returns)

        # Risk metrics
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        max_drawdown = self._calculate_max_drawdown(np.cumsum(returns))
        var_95 = np.percentile(returns, 5) if len(returns) > 0 else 0

        # Trading metrics
        win_rate = np.sum(returns > 0) / len(returns) if len(returns) > 0 else 0
        profit_factor = self._calculate_profit_factor(returns)

        # Days with trading activity
        active_days = len(non_zero_returns)
        total_days = len(returns)

        return {
            'strategy_name': strategy_name,
            'total_pnl': total_pnl,
            'avg_daily_pnl': avg_daily_pnl,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'active_days': active_days,
            'total_days': total_days,
            'activity_rate': active_days / total_days if total_days > 0 else 0
        }

    def _calculate_causal_impact(self, results: Dict) -> Dict:
        """Calculate the causal impact of each strategy vs baseline."""

        baseline = results['baseline']
        impact_metrics = {}

        for strategy_name, strategy_results in results.items():
            if strategy_name in ['baseline', 'causal_impact']:
                continue

            # PnL impact
            pnl_improvement = strategy_results['total_pnl'] - baseline['total_pnl']
            pnl_improvement_pct = (pnl_improvement / abs(baseline['total_pnl'])) * 100 if baseline['total_pnl'] != 0 else 0

            # Risk impact
            sharpe_improvement = strategy_results['sharpe_ratio'] - baseline['sharpe_ratio']
            drawdown_improvement = baseline['max_drawdown'] - strategy_results['max_drawdown']  # Positive = better

            # Win rate impact
            win_rate_improvement = strategy_results['win_rate'] - baseline['win_rate']

            impact_metrics[strategy_name] = {
                'pnl_improvement': pnl_improvement,
                'pnl_improvement_pct': pnl_improvement_pct,
                'sharpe_improvement': sharpe_improvement,
                'drawdown_improvement': drawdown_improvement,
                'win_rate_improvement': win_rate_improvement,
                'is_profitable': pnl_improvement > 0,
                'is_less_risky': sharpe_improvement > 0
            }

        return impact_metrics

    def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0

        excess_returns = returns - risk_free_rate
        return np.mean(excess_returns) / np.std(returns) * np.sqrt(252)  # Annualized

    def _calculate_max_drawdown(self, cumulative_returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        if len(cumulative_returns) == 0:
            return 0.0

        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / np.maximum(np.abs(peak), 1e-10)
        return np.min(drawdown)

    def _calculate_profit_factor(self, returns: np.ndarray) -> float:
        """Calculate profit factor (total wins / total losses)."""
        wins = returns[returns > 0]
        losses = returns[returns < 0]

        total_wins = np.sum(wins) if len(wins) > 0 else 0
        total_losses = abs(np.sum(losses)) if len(losses) > 0 else 1e-10

        return total_wins / total_losses

    def generate_daily_impact_analysis(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate day-by-day impact analysis showing what would have happened.
        """

        daily_analysis = predictions_df.copy()

        # Calculate strategy returns
        daily_analysis['baseline_pnl'] = daily_analysis['actual_pnl']
        daily_analysis['position_sizing_pnl'] = self._apply_position_sizing_strategy(daily_analysis)
        daily_analysis['filtered_pnl'] = self._apply_trade_filtering_strategy(daily_analysis)
        daily_analysis['combined_pnl'] = self._apply_combined_strategy(daily_analysis)

        # Calculate cumulative performance
        daily_analysis['baseline_cumulative'] = daily_analysis['baseline_pnl'].cumsum()
        daily_analysis['position_sizing_cumulative'] = daily_analysis['position_sizing_pnl'].cumsum()
        daily_analysis['filtered_cumulative'] = daily_analysis['filtered_pnl'].cumsum()
        daily_analysis['combined_cumulative'] = daily_analysis['combined_pnl'].cumsum()

        # Calculate daily improvements
        daily_analysis['position_sizing_improvement'] = daily_analysis['position_sizing_pnl'] - daily_analysis['baseline_pnl']
        daily_analysis['filtered_improvement'] = daily_analysis['filtered_pnl'] - daily_analysis['baseline_pnl']
        daily_analysis['combined_improvement'] = daily_analysis['combined_pnl'] - daily_analysis['baseline_pnl']

        return daily_analysis

    def print_causal_impact_report(self, impact_results: Dict) -> None:
        """Print a comprehensive causal impact report."""

        print("="*80)
        print("CAUSAL IMPACT ANALYSIS - RISK MODEL VS BASELINE TRADING")
        print("="*80)

        # Baseline performance
        baseline = impact_results['baseline']
        print(f"\n📊 BASELINE PERFORMANCE (Original Trading):")
        print(f"   Total PnL: ${baseline['total_pnl']:,.2f}")
        print(f"   Sharpe Ratio: {baseline['sharpe_ratio']:.4f}")
        print(f"   Max Drawdown: {baseline['max_drawdown']:.4f}")
        print(f"   Win Rate: {baseline['win_rate']:.2%}")
        print(f"   Active Days: {baseline['active_days']}/{baseline['total_days']}")

        # Strategy comparisons
        strategies = ['position_sizing', 'trade_filtering', 'combined']
        strategy_names = ['Position Sizing', 'Trade Filtering', 'Combined Strategy']

        for strategy, display_name in zip(strategies, strategy_names):
            if strategy not in impact_results:
                continue

            print(f"\n🎯 {display_name.upper()} STRATEGY:")

            # Strategy performance
            strat_results = impact_results[strategy]
            print(f"   Total PnL: ${strat_results['total_pnl']:,.2f}")
            print(f"   Sharpe Ratio: {strat_results['sharpe_ratio']:.4f}")
            print(f"   Max Drawdown: {strat_results['max_drawdown']:.4f}")
            print(f"   Win Rate: {strat_results['win_rate']:.2%}")
            print(f"   Active Days: {strat_results['active_days']}/{strat_results['total_days']}")

            # Causal impact
            if 'causal_impact' in impact_results and strategy in impact_results['causal_impact']:
                impact = impact_results['causal_impact'][strategy]
                print(f"\n   📈 CAUSAL IMPACT:")
                print(f"   PnL Improvement: ${impact['pnl_improvement']:,.2f} ({impact['pnl_improvement_pct']:+.1f}%)")
                print(f"   Sharpe Improvement: {impact['sharpe_improvement']:+.4f}")
                print(f"   Drawdown Improvement: {impact['drawdown_improvement']:+.4f}")
                print(f"   Win Rate Improvement: {impact['win_rate_improvement']:+.2%}")

                # Summary verdict
                if impact['is_profitable'] and impact['is_less_risky']:
                    verdict = "✅ SUPERIOR PERFORMANCE (Higher PnL + Lower Risk)"
                elif impact['is_profitable']:
                    verdict = "⚡ HIGHER RETURNS (but potentially higher risk)"
                elif impact['is_less_risky']:
                    verdict = "🛡️ LOWER RISK (but potentially lower returns)"
                else:
                    verdict = "❌ UNDERPERFORMED (Lower PnL + Higher Risk)"

                print(f"   Verdict: {verdict}")

        print("\n" + "="*80)
        print("CONCLUSION:")

        # Find best strategy
        best_strategy = None
        best_improvement = float('-inf')

        if 'causal_impact' in impact_results:
            for strategy, impact in impact_results['causal_impact'].items():
                # Score based on PnL improvement and risk reduction
                score = impact['pnl_improvement'] + (impact['sharpe_improvement'] * 10000)  # Weight Sharpe heavily
                if score > best_improvement:
                    best_improvement = score
                    best_strategy = strategy

        if best_strategy:
            impact = impact_results['causal_impact'][best_strategy]
            print(f"🏆 Best Strategy: {best_strategy.replace('_', ' ').title()}")
            print(f"   Would have generated ${impact['pnl_improvement']:,.2f} additional profit")
            print(f"   That's a {impact['pnl_improvement_pct']:+.1f}% improvement over baseline trading")
        else:
            print("📉 No strategy outperformed baseline trading in the test period")

        print("="*80)
