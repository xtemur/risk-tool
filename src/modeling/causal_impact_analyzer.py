"""
Causal Impact Analyzer

Analyzes the causal impact of model predictions on trader PnL.
Calculates how much PnL would change if traders had listened to model recommendations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class CausalImpactAnalyzer:
    """
    Analyzes causal impact of model predictions on trading outcomes
    """

    def __init__(self):
        """
        Initialize the causal impact analyzer
        """
        pass

    def calculate_trading_impact(self,
                               actual_pnl: np.ndarray,
                               predicted_pnl: np.ndarray,
                               dates: Optional[pd.DatetimeIndex] = None,
                               trader_id: str = 'trader') -> Dict[str, Any]:
        """
        Calculate the causal impact of following model predictions

        Args:
            actual_pnl: Actual PnL achieved by trader
            predicted_pnl: Model predicted PnL
            dates: Optional dates for time series analysis
            trader_id: Trader identifier

        Returns:
            Comprehensive causal impact analysis
        """
        if len(actual_pnl) != len(predicted_pnl):
            raise ValueError("Actual and predicted PnL arrays must have same length")

        logger.info(f"Calculating causal impact for {trader_id}, {len(actual_pnl)} samples")

        # Basic statistics
        actual_total = np.sum(actual_pnl)
        predicted_total = np.sum(predicted_pnl)

        # Scenario 1: Perfect model following (replace actual with predicted)
        perfect_following_impact = predicted_total - actual_total

        # Scenario 2: Directional trading (only trade when model suggests profitable direction)
        directional_impact = self._calculate_directional_impact(actual_pnl, predicted_pnl)

        # Scenario 3: Risk-adjusted trading (scale position size by prediction confidence)
        risk_adjusted_impact = self._calculate_risk_adjusted_impact(actual_pnl, predicted_pnl)

        # Scenario 4: Selective trading (only trade on high-confidence predictions)
        selective_impact = self._calculate_selective_impact(actual_pnl, predicted_pnl)

        # Daily impact analysis
        daily_analysis = self._analyze_daily_impact(actual_pnl, predicted_pnl, dates)

        # Statistical significance
        significance = self._calculate_statistical_significance(actual_pnl, predicted_pnl)

        return {
            'trader_id': trader_id,
            'analysis_period': {
                'start_date': dates.min() if dates is not None else None,
                'end_date': dates.max() if dates is not None else None,
                'trading_days': len(actual_pnl)
            },
            'baseline_performance': {
                'actual_total_pnl': actual_total,
                'actual_avg_daily_pnl': np.mean(actual_pnl),
                'actual_volatility': np.std(actual_pnl),
                'actual_sharpe': np.mean(actual_pnl) / (np.std(actual_pnl) + 1e-6) * np.sqrt(252),
                'winning_days': np.sum(actual_pnl > 0),
                'losing_days': np.sum(actual_pnl < 0),
                'win_rate': np.mean(actual_pnl > 0)
            },
            'model_performance': {
                'predicted_total_pnl': predicted_total,
                'predicted_avg_daily_pnl': np.mean(predicted_pnl),
                'predicted_volatility': np.std(predicted_pnl),
                'predicted_sharpe': np.mean(predicted_pnl) / (np.std(predicted_pnl) + 1e-6) * np.sqrt(252)
            },
            'causal_impact_scenarios': {
                'perfect_following': {
                    'pnl_improvement': perfect_following_impact,
                    'pnl_improvement_pct': (perfect_following_impact / abs(actual_total) * 100) if actual_total != 0 else 0,
                    'description': 'PnL if trader perfectly followed model predictions'
                },
                'directional_trading': directional_impact,
                'risk_adjusted_trading': risk_adjusted_impact,
                'selective_trading': selective_impact
            },
            'daily_analysis': daily_analysis,
            'statistical_significance': significance
        }

    def _calculate_directional_impact(self, actual_pnl: np.ndarray,
                                    predicted_pnl: np.ndarray) -> Dict[str, Any]:
        """
        Calculate impact of following model's directional signals
        Only trade when model predicts positive PnL
        """
        # Days when model predicted positive PnL
        positive_prediction_mask = predicted_pnl > 0

        # Days when model predicted negative PnL (don't trade)
        negative_prediction_mask = predicted_pnl <= 0

        # Scenario: Trade only on positive predictions, skip negative days
        trade_days = np.sum(positive_prediction_mask)
        skip_days = np.sum(negative_prediction_mask)

        # PnL from trading only on positive predictions
        directional_pnl = np.sum(actual_pnl[positive_prediction_mask])

        # PnL saved by not trading on negative predictions
        saved_pnl = -np.sum(actual_pnl[negative_prediction_mask])  # Negative because we avoid losses

        # Total impact
        total_directional_impact = directional_pnl + saved_pnl - np.sum(actual_pnl)

        return {
            'pnl_improvement': total_directional_impact,
            'pnl_improvement_pct': (total_directional_impact / abs(np.sum(actual_pnl)) * 100) if np.sum(actual_pnl) != 0 else 0,
            'trade_days': trade_days,
            'skip_days': skip_days,
            'trading_frequency': trade_days / len(actual_pnl),
            'pnl_from_trades': directional_pnl,
            'pnl_saved': saved_pnl,
            'avg_pnl_per_trade': directional_pnl / trade_days if trade_days > 0 else 0,
            'description': 'PnL if trader only traded when model predicted profit'
        }

    def _calculate_risk_adjusted_impact(self, actual_pnl: np.ndarray,
                                      predicted_pnl: np.ndarray) -> Dict[str, Any]:
        """
        Calculate impact of scaling position size by prediction confidence
        Scale position size proportional to prediction magnitude
        """
        # Normalize predictions to create position sizing weights
        pred_abs = np.abs(predicted_pnl)
        max_pred = np.max(pred_abs) if np.max(pred_abs) > 0 else 1
        position_weights = pred_abs / max_pred

        # Apply position weights to actual PnL
        risk_adjusted_pnl = actual_pnl * position_weights

        # Calculate impact
        total_risk_adjusted = np.sum(risk_adjusted_pnl)
        total_actual = np.sum(actual_pnl)
        impact = total_risk_adjusted - total_actual

        return {
            'pnl_improvement': impact,
            'pnl_improvement_pct': (impact / abs(total_actual) * 100) if total_actual != 0 else 0,
            'avg_position_weight': np.mean(position_weights),
            'max_position_weight': np.max(position_weights),
            'min_position_weight': np.min(position_weights),
            'total_risk_adjusted_pnl': total_risk_adjusted,
            'description': 'PnL if position size scaled by model confidence'
        }

    def _calculate_selective_impact(self, actual_pnl: np.ndarray,
                                  predicted_pnl: np.ndarray,
                                  confidence_threshold: float = 0.7) -> Dict[str, Any]:
        """
        Calculate impact of trading only on high-confidence predictions
        """
        # Calculate confidence scores (normalized absolute predictions)
        pred_abs = np.abs(predicted_pnl)
        max_pred = np.max(pred_abs) if np.max(pred_abs) > 0 else 1
        confidence_scores = pred_abs / max_pred

        # High confidence mask
        high_confidence_mask = confidence_scores >= confidence_threshold

        # Trade only on high confidence days
        selective_pnl = np.sum(actual_pnl[high_confidence_mask])

        # Days traded vs skipped
        trade_days = np.sum(high_confidence_mask)
        skip_days = len(actual_pnl) - trade_days

        # Impact calculation
        total_actual = np.sum(actual_pnl)
        # Assuming trader makes 0 on skipped days vs their actual performance
        skipped_pnl = np.sum(actual_pnl[~high_confidence_mask])
        impact = selective_pnl + 0 - total_actual  # 0 PnL on skipped days

        return {
            'pnl_improvement': impact,
            'pnl_improvement_pct': (impact / abs(total_actual) * 100) if total_actual != 0 else 0,
            'confidence_threshold': confidence_threshold,
            'trade_days': trade_days,
            'skip_days': skip_days,
            'trading_frequency': trade_days / len(actual_pnl),
            'high_confidence_pnl': selective_pnl,
            'skipped_pnl': skipped_pnl,
            'avg_pnl_per_high_confidence_trade': selective_pnl / trade_days if trade_days > 0 else 0,
            'description': f'PnL if trader only traded on predictions with >{confidence_threshold:.0%} confidence'
        }

    def _analyze_daily_impact(self, actual_pnl: np.ndarray,
                            predicted_pnl: np.ndarray,
                            dates: Optional[pd.DatetimeIndex] = None) -> Dict[str, Any]:
        """
        Analyze day-by-day impact of following model
        """
        # Daily differences
        daily_differences = predicted_pnl - actual_pnl

        # Days where model would have improved/hurt performance
        improvement_days = np.sum(daily_differences > 0)
        deterioration_days = np.sum(daily_differences < 0)
        neutral_days = np.sum(daily_differences == 0)

        # Magnitude of improvements/deteriorations
        improvements = daily_differences[daily_differences > 0]
        deteriorations = daily_differences[daily_differences < 0]

        return {
            'improvement_days': improvement_days,
            'deterioration_days': deterioration_days,
            'neutral_days': neutral_days,
            'improvement_rate': improvement_days / len(daily_differences),
            'avg_daily_improvement': np.mean(improvements) if len(improvements) > 0 else 0,
            'avg_daily_deterioration': np.mean(deteriorations) if len(deteriorations) > 0 else 0,
            'max_daily_improvement': np.max(improvements) if len(improvements) > 0 else 0,
            'max_daily_deterioration': np.min(deteriorations) if len(deteriorations) > 0 else 0,
            'total_improvement': np.sum(improvements) if len(improvements) > 0 else 0,
            'total_deterioration': np.sum(deteriorations) if len(deteriorations) > 0 else 0
        }

    def _calculate_statistical_significance(self, actual_pnl: np.ndarray,
                                          predicted_pnl: np.ndarray) -> Dict[str, Any]:
        """
        Calculate statistical significance of model impact
        """
        from scipy import stats

        # Paired t-test for difference in means
        differences = predicted_pnl - actual_pnl

        if len(differences) > 1:
            t_stat, p_value = stats.ttest_1samp(differences, 0)

            # Effect size (Cohen's d)
            effect_size = np.mean(differences) / (np.std(differences) + 1e-6)
        else:
            t_stat, p_value, effect_size = 0, 1, 0

        # Confidence interval for mean difference
        if len(differences) > 1:
            sem = stats.sem(differences)
            ci_95 = stats.t.interval(0.95, len(differences)-1,
                                   loc=np.mean(differences), scale=sem)
        else:
            ci_95 = (0, 0)

        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'is_significant': p_value < 0.05,
            'effect_size': effect_size,
            'confidence_interval_95': ci_95,
            'mean_difference': np.mean(differences),
            'std_difference': np.std(differences),
            'interpretation': self._interpret_significance(p_value, effect_size)
        }

    def _interpret_significance(self, p_value: float, effect_size: float) -> str:
        """
        Interpret statistical significance results
        """
        significance = "significant" if p_value < 0.05 else "not significant"

        if abs(effect_size) < 0.2:
            magnitude = "negligible"
        elif abs(effect_size) < 0.5:
            magnitude = "small"
        elif abs(effect_size) < 0.8:
            magnitude = "medium"
        else:
            magnitude = "large"

        direction = "positive" if effect_size > 0 else "negative"

        return f"The model impact is statistically {significance} with a {magnitude} {direction} effect size"

    def analyze_multiple_traders(self, trader_results: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Analyze causal impact across multiple traders

        Args:
            trader_results: Dict mapping trader_id to individual causal impact results

        Returns:
            Aggregate analysis across all traders
        """
        if not trader_results:
            return {}

        logger.info(f"Analyzing aggregate impact across {len(trader_results)} traders")

        # Collect metrics across traders
        trader_impacts = []
        baseline_totals = []
        improvement_amounts = []
        improvement_pcts = []
        trading_frequencies = []
        win_rates = []

        for trader_id, results in trader_results.items():
            if 'causal_impact_scenarios' in results:
                baseline = results['baseline_performance']['actual_total_pnl']
                baseline_totals.append(baseline)

                # Perfect following scenario
                perfect_impact = results['causal_impact_scenarios']['perfect_following']
                improvement_amounts.append(perfect_impact['pnl_improvement'])
                improvement_pcts.append(perfect_impact['pnl_improvement_pct'])

                # Directional trading
                directional = results['causal_impact_scenarios']['directional_trading']
                trading_frequencies.append(directional['trading_frequency'])

                # Baseline metrics
                win_rates.append(results['baseline_performance']['win_rate'])

                trader_impacts.append({
                    'trader_id': trader_id,
                    'baseline_pnl': baseline,
                    'improvement': perfect_impact['pnl_improvement'],
                    'improvement_pct': perfect_impact['pnl_improvement_pct']
                })

        # Aggregate statistics
        total_baseline_pnl = np.sum(baseline_totals)
        total_improvement = np.sum(improvement_amounts)

        return {
            'aggregate_metrics': {
                'total_traders': len(trader_results),
                'total_baseline_pnl': total_baseline_pnl,
                'total_improvement': total_improvement,
                'total_improvement_pct': (total_improvement / abs(total_baseline_pnl) * 100) if total_baseline_pnl != 0 else 0,
                'avg_improvement_per_trader': np.mean(improvement_amounts),
                'median_improvement_per_trader': np.median(improvement_amounts),
                'avg_improvement_pct': np.mean(improvement_pcts),
                'traders_improved': np.sum(np.array(improvement_amounts) > 0),
                'traders_deteriorated': np.sum(np.array(improvement_amounts) < 0),
                'improvement_success_rate': np.mean(np.array(improvement_amounts) > 0)
            },
            'distribution_stats': {
                'improvement_std': np.std(improvement_amounts),
                'improvement_range': (np.min(improvement_amounts), np.max(improvement_amounts)),
                'improvement_percentiles': {
                    '25th': np.percentile(improvement_amounts, 25),
                    '50th': np.percentile(improvement_amounts, 50),
                    '75th': np.percentile(improvement_amounts, 75),
                    '90th': np.percentile(improvement_amounts, 90)
                }
            },
            'trader_level_results': trader_impacts,
            'trading_behavior': {
                'avg_trading_frequency': np.mean(trading_frequencies),
                'avg_baseline_win_rate': np.mean(win_rates)
            }
        }

    def generate_impact_report(self, causal_results: Dict[str, Any]) -> str:
        """
        Generate a comprehensive causal impact report

        Args:
            causal_results: Results from calculate_trading_impact

        Returns:
            Formatted report string
        """
        trader_id = causal_results.get('trader_id', 'Unknown')
        baseline = causal_results['baseline_performance']
        scenarios = causal_results['causal_impact_scenarios']

        report = f"""
CAUSAL IMPACT ANALYSIS REPORT
=============================

Trader: {trader_id}
Analysis Period: {causal_results['analysis_period']['trading_days']} trading days
Date Range: {causal_results['analysis_period']['start_date']} to {causal_results['analysis_period']['end_date']}

BASELINE PERFORMANCE
--------------------
Actual Total PnL: ${baseline['actual_total_pnl']:,.2f}
Average Daily PnL: ${baseline['actual_avg_daily_pnl']:,.2f}
Volatility: ${baseline['actual_volatility']:,.2f}
Sharpe Ratio: {baseline['actual_sharpe']:.3f}
Win Rate: {baseline['win_rate']:.1%}
Winning Days: {baseline['winning_days']}
Losing Days: {baseline['losing_days']}

CAUSAL IMPACT SCENARIOS
-----------------------

1. PERFECT MODEL FOLLOWING
   PnL Improvement: ${scenarios['perfect_following']['pnl_improvement']:,.2f}
   Improvement %: {scenarios['perfect_following']['pnl_improvement_pct']:,.1f}%
   Description: {scenarios['perfect_following']['description']}

2. DIRECTIONAL TRADING
   PnL Improvement: ${scenarios['directional_trading']['pnl_improvement']:,.2f}
   Improvement %: {scenarios['directional_trading']['pnl_improvement_pct']:,.1f}%
   Trading Frequency: {scenarios['directional_trading']['trading_frequency']:.1%}
   Trade Days: {scenarios['directional_trading']['trade_days']}
   Skip Days: {scenarios['directional_trading']['skip_days']}
   PnL from Trades: ${scenarios['directional_trading']['pnl_from_trades']:,.2f}
   PnL Saved: ${scenarios['directional_trading']['pnl_saved']:,.2f}
   Description: {scenarios['directional_trading']['description']}

3. RISK-ADJUSTED TRADING
   PnL Improvement: ${scenarios['risk_adjusted_trading']['pnl_improvement']:,.2f}
   Improvement %: {scenarios['risk_adjusted_trading']['pnl_improvement_pct']:,.1f}%
   Average Position Weight: {scenarios['risk_adjusted_trading']['avg_position_weight']:.2f}
   Description: {scenarios['risk_adjusted_trading']['description']}

4. SELECTIVE TRADING
   PnL Improvement: ${scenarios['selective_trading']['pnl_improvement']:,.2f}
   Improvement %: {scenarios['selective_trading']['pnl_improvement_pct']:,.1f}%
   Trading Frequency: {scenarios['selective_trading']['trading_frequency']:.1%}
   Confidence Threshold: {scenarios['selective_trading']['confidence_threshold']:.0%}
   Description: {scenarios['selective_trading']['description']}

STATISTICAL SIGNIFICANCE
-------------------------
Test Statistic: {causal_results['statistical_significance']['t_statistic']:.3f}
P-value: {causal_results['statistical_significance']['p_value']:.6f}
Is Significant: {causal_results['statistical_significance']['is_significant']}
Effect Size: {causal_results['statistical_significance']['effect_size']:.3f}
95% Confidence Interval: [{causal_results['statistical_significance']['confidence_interval_95'][0]:.2f}, {causal_results['statistical_significance']['confidence_interval_95'][1]:.2f}]
Interpretation: {causal_results['statistical_significance']['interpretation']}

DAILY IMPACT ANALYSIS
----------------------
Days Model Would Improve: {causal_results['daily_analysis']['improvement_days']}
Days Model Would Hurt: {causal_results['daily_analysis']['deterioration_days']}
Improvement Rate: {causal_results['daily_analysis']['improvement_rate']:.1%}
Average Daily Improvement: ${causal_results['daily_analysis']['avg_daily_improvement']:,.2f}
Average Daily Deterioration: ${causal_results['daily_analysis']['avg_daily_deterioration']:,.2f}
Max Daily Improvement: ${causal_results['daily_analysis']['max_daily_improvement']:,.2f}
Max Daily Deterioration: ${causal_results['daily_analysis']['max_daily_deterioration']:,.2f}

CONCLUSION
----------
The model shows potential for {'positive' if scenarios['perfect_following']['pnl_improvement'] > 0 else 'negative'} impact on trader performance.
Best strategy appears to be: {self._recommend_best_strategy(scenarios)}
"""

        return report

    def _recommend_best_strategy(self, scenarios: Dict[str, Any]) -> str:
        """
        Recommend the best strategy based on causal impact analysis
        """
        strategies = {
            'Perfect Following': scenarios['perfect_following']['pnl_improvement'],
            'Directional Trading': scenarios['directional_trading']['pnl_improvement'],
            'Risk-Adjusted Trading': scenarios['risk_adjusted_trading']['pnl_improvement'],
            'Selective Trading': scenarios['selective_trading']['pnl_improvement']
        }

        best_strategy = max(strategies.keys(), key=lambda k: strategies[k])
        best_improvement = strategies[best_strategy]

        return f"{best_strategy} with ${best_improvement:,.2f} improvement"
