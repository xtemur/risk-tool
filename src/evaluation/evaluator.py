import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from sklearn.metrics import classification_report, confusion_matrix

class ModelEvaluator:
    """
    Comprehensive evaluation of risk model performance.
    Focus on financial metrics and actionable insights.
    """

    def __init__(self):
        pass

    def evaluate_regression_performance(self,
                                      y_true: np.ndarray,
                                      y_pred: np.ndarray,
                                      dates: pd.Series = None) -> Dict:
        """
        Evaluate regression performance with financial focus.

        Args:
            y_true: True target values (volatility-normalized PNL)
            y_pred: Predicted values
            dates: Dates for time-based analysis

        Returns:
            Dictionary of evaluation metrics
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        metrics = {
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
            'correlation': np.corrcoef(y_true, y_pred)[0, 1],
            'mean_true': np.mean(y_true),
            'mean_pred': np.mean(y_pred),
            'std_true': np.std(y_true),
            'std_pred': np.std(y_pred)
        }

        return metrics

    def evaluate_risk_signals(self,
                             y_true: np.ndarray,
                             risk_signals: np.ndarray,
                             trader_ids: pd.Series = None) -> Dict:
        """
        Evaluate risk signal classification performance.

        Args:
            y_true: True target values
            risk_signals: Risk signals (0=High Risk, 1=Neutral, 2=Low Risk)
            trader_ids: Trader IDs for per-trader analysis

        Returns:
            Classification performance metrics
        """
        # Convert continuous targets to risk categories for evaluation
        true_signals = self._convert_to_risk_categories(y_true)

        # Classification metrics
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support

        accuracy = accuracy_score(true_signals, risk_signals)
        precision, recall, f1, support = precision_recall_fscore_support(
            true_signals, risk_signals, average='weighted'
        )

        # Per-class performance
        class_report = classification_report(
            true_signals, risk_signals,
            target_names=['High Risk', 'Neutral', 'Low Risk'],
            output_dict=True
        )

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'class_report': class_report,
            'confusion_matrix': confusion_matrix(true_signals, risk_signals)
        }

        return metrics

    def calculate_financial_metrics(self,
                                  predictions: np.ndarray,
                                  actual_pnl: np.ndarray,
                                  dates: pd.Series = None) -> Dict:
        """
        Calculate financial performance metrics.

        Args:
            predictions: Model predictions (volatility-normalized)
            actual_pnl: Actual realized PNL (raw values)
            dates: Trading dates

        Returns:
            Financial performance metrics
        """
        # Simple strategy: go long when prediction > 0, short when < 0
        position_sizes = np.sign(predictions)
        strategy_returns = position_sizes * actual_pnl

        # Remove zero returns for meaningful calculations
        non_zero_returns = strategy_returns[strategy_returns != 0]
        actual_non_zero = actual_pnl[actual_pnl != 0]

        if len(non_zero_returns) == 0:
            return {'error': 'No non-zero returns to evaluate'}

        # Financial metrics
        total_pnl = np.sum(strategy_returns)
        total_baseline_pnl = np.sum(actual_pnl)

        sharpe_ratio = self._calculate_sharpe_ratio(strategy_returns)
        baseline_sharpe = self._calculate_sharpe_ratio(actual_pnl)

        max_drawdown = self._calculate_max_drawdown(np.cumsum(strategy_returns))
        baseline_drawdown = self._calculate_max_drawdown(np.cumsum(actual_pnl))

        win_rate = np.sum(strategy_returns > 0) / len(non_zero_returns)
        baseline_win_rate = np.sum(actual_pnl > 0) / len(actual_non_zero)

        metrics = {
            'total_pnl': total_pnl,
            'baseline_pnl': total_baseline_pnl,
            'pnl_improvement': total_pnl - total_baseline_pnl,
            'sharpe_ratio': sharpe_ratio,
            'baseline_sharpe': baseline_sharpe,
            'max_drawdown': max_drawdown,
            'baseline_drawdown': baseline_drawdown,
            'win_rate': win_rate,
            'baseline_win_rate': baseline_win_rate,
            'num_trades': len(non_zero_returns),
            'avg_trade_pnl': np.mean(strategy_returns)
        }

        return metrics

    def _convert_to_risk_categories(self, y_true: np.ndarray) -> np.ndarray:
        """Convert continuous targets to risk categories."""
        low_threshold = np.percentile(y_true, 10)
        high_threshold = np.percentile(y_true, 75)

        categories = np.ones(len(y_true))  # Default neutral
        categories[y_true < low_threshold] = 0  # High risk
        categories[y_true > high_threshold] = 2  # Low risk

        return categories.astype(int)

    def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0

        excess_returns = returns - risk_free_rate
        return np.mean(excess_returns) / np.std(returns) * np.sqrt(252)  # Annualized

    def _calculate_max_drawdown(self, cumulative_returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        if len(cumulative_returns) == 0:
            return 0.0

        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / np.maximum(peak, 1e-10)
        return np.min(drawdown)

    def generate_evaluation_report(self,
                                 y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 actual_pnl: np.ndarray,
                                 risk_signals: np.ndarray,
                                 dates: pd.Series = None) -> Dict:
        """Generate comprehensive evaluation report."""

        report = {
            'regression_metrics': self.evaluate_regression_performance(y_true, y_pred, dates),
            'signal_metrics': self.evaluate_risk_signals(y_true, risk_signals),
            'financial_metrics': self.calculate_financial_metrics(y_pred, actual_pnl, dates)
        }

        return report

    def print_evaluation_summary(self, report: Dict) -> None:
        """Print formatted evaluation summary."""
        print("="*60)
        print("MODEL EVALUATION SUMMARY")
        print("="*60)

        # Regression Performance
        reg_metrics = report['regression_metrics']
        print(f"\nREGRESSION PERFORMANCE:")
        print(f"  MAE: {reg_metrics['mae']:.4f}")
        print(f"  RMSE: {reg_metrics['rmse']:.4f}")
        print(f"  R²: {reg_metrics['r2']:.4f}")
        print(f"  Correlation: {reg_metrics['correlation']:.4f}")

        # Signal Performance
        signal_metrics = report['signal_metrics']
        print(f"\nRISK SIGNAL PERFORMANCE:")
        print(f"  Accuracy: {signal_metrics['accuracy']:.4f}")
        print(f"  Precision: {signal_metrics['precision']:.4f}")
        print(f"  Recall: {signal_metrics['recall']:.4f}")
        print(f"  F1-Score: {signal_metrics['f1_score']:.4f}")

        # Financial Performance
        if 'error' not in report['financial_metrics']:
            fin_metrics = report['financial_metrics']
            print(f"\nFINANCIAL PERFORMANCE:")
            print(f"  Total PNL: ${fin_metrics['total_pnl']:,.2f}")
            print(f"  Baseline PNL: ${fin_metrics['baseline_pnl']:,.2f}")
            print(f"  PNL Improvement: ${fin_metrics['pnl_improvement']:,.2f}")
            print(f"  Sharpe Ratio: {fin_metrics['sharpe_ratio']:.4f}")
            print(f"  Baseline Sharpe: {fin_metrics['baseline_sharpe']:.4f}")
            print(f"  Max Drawdown: {fin_metrics['max_drawdown']:.4f}")
            print(f"  Win Rate: {fin_metrics['win_rate']:.4f}")
        else:
            print(f"\nFINANCIAL PERFORMANCE: {report['financial_metrics']['error']}")

        print("="*60)
