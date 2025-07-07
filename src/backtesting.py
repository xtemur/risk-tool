import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, average_precision_score,
    classification_report, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class BacktestAnalyzer:
    def __init__(self):
        self.results = {}
        self.metrics = {}

    def analyze_validation_performance(self, predictions: pd.DataFrame) -> Dict:
        """
        Analyze expanding window validation performance
        """
        analysis = {}

        for trader_id in predictions['trader_id'].unique():
            trader_preds = predictions[predictions['trader_id'] == trader_id]

            if len(trader_preds['actual'].unique()) == 1:
                logger.warning(f"Trader {trader_id} has no class variation in validation")
                continue

            trader_analysis = self._analyze_trader_performance(trader_preds)
            analysis[trader_id] = trader_analysis

        return analysis

    def _analyze_trader_performance(self, trader_preds: pd.DataFrame) -> Dict:
        """
        Detailed performance analysis for a single trader
        """
        trader_preds = trader_preds.sort_values('date')

        overall_auc = roc_auc_score(trader_preds['actual'], trader_preds['prediction'])
        overall_ap = average_precision_score(trader_preds['actual'], trader_preds['prediction'])

        # Time-based performance analysis
        trader_preds['period'] = pd.cut(
            range(len(trader_preds)),
            bins=3,
            labels=['early', 'middle', 'late']
        )

        period_performance = {}
        for period in ['early', 'middle', 'late']:
            period_data = trader_preds[trader_preds['period'] == period]
            if len(period_data['actual'].unique()) > 1:
                period_auc = roc_auc_score(period_data['actual'], period_data['prediction'])
                period_ap = average_precision_score(period_data['actual'], period_data['prediction'])
            else:
                period_auc = period_ap = None

            period_performance[period] = {
                'auc': period_auc,
                'average_precision': period_ap,
                'n_samples': len(period_data),
                'positive_rate': period_data['actual'].mean()
            }

        # Feature importance stability (if available)
        stability_score = self._calculate_prediction_stability(trader_preds)

        return {
            'overall_auc': overall_auc,
            'overall_ap': overall_ap,
            'period_performance': period_performance,
            'stability_score': stability_score,
            'total_predictions': len(trader_preds),
            'positive_rate': trader_preds['actual'].mean(),
            'prediction_range': (trader_preds['prediction'].min(), trader_preds['prediction'].max())
        }

    def _calculate_prediction_stability(self, trader_preds: pd.DataFrame) -> float:
        """
        Calculate prediction stability using rolling standard deviation
        """
        if len(trader_preds) < 20:
            return None

        rolling_std = trader_preds['prediction'].rolling(window=10).std()
        return rolling_std.mean()

    def compare_test_vs_validation(self, val_predictions: pd.DataFrame,
                                 test_predictions: pd.DataFrame) -> Dict:
        """
        Compare validation and test performance to detect overfitting
        """
        comparison = {}

        for trader_id in val_predictions['trader_id'].unique():
            val_data = val_predictions[val_predictions['trader_id'] == trader_id]
            test_data = test_predictions[test_predictions['trader_id'] == trader_id]

            if len(val_data['actual'].unique()) == 1 or len(test_data['actual'].unique()) == 1:
                continue

            val_auc = roc_auc_score(val_data['actual'], val_data['prediction'])
            test_auc = roc_auc_score(test_data['actual'], test_data['prediction'])

            val_ap = average_precision_score(val_data['actual'], val_data['prediction'])
            test_ap = average_precision_score(test_data['actual'], test_data['prediction'])

            comparison[trader_id] = {
                'validation_auc': val_auc,
                'test_auc': test_auc,
                'auc_difference': val_auc - test_auc,
                'validation_ap': val_ap,
                'test_ap': test_ap,
                'ap_difference': val_ap - test_ap,
                'overfitting_flag': (val_auc - test_auc) > 0.1
            }

        return comparison

    def analyze_model_adaptation(self, predictions: pd.DataFrame) -> Dict:
        """
        Analyze how quickly models adapt to regime changes
        """
        adaptation_analysis = {}

        for trader_id in predictions['trader_id'].unique():
            trader_preds = predictions[predictions['trader_id'] == trader_id].sort_values('date')

            if len(trader_preds) < 50:
                continue

            # Detect regime changes using rolling statistics
            rolling_pos_rate = trader_preds['actual'].rolling(window=20).mean()
            regime_changes = self._detect_regime_changes(rolling_pos_rate)

            # Measure adaptation speed after regime changes
            adaptation_speeds = []
            for change_date in regime_changes:
                speed = self._measure_adaptation_speed(trader_preds, change_date)
                if speed is not None:
                    adaptation_speeds.append(speed)

            adaptation_analysis[trader_id] = {
                'regime_changes': len(regime_changes),
                'avg_adaptation_speed': np.mean(adaptation_speeds) if adaptation_speeds else None,
                'adaptation_scores': adaptation_speeds
            }

        return adaptation_analysis

    def _detect_regime_changes(self, rolling_stats: pd.Series) -> List[pd.Timestamp]:
        """
        Detect significant changes in rolling statistics
        """
        changes = []

        for i in range(1, len(rolling_stats)):
            if abs(rolling_stats.iloc[i] - rolling_stats.iloc[i-1]) > 0.2:  # 20% change threshold
                changes.append(rolling_stats.index[i])

        return changes

    def _measure_adaptation_speed(self, trader_preds: pd.DataFrame,
                                change_date: pd.Timestamp) -> Optional[int]:
        """
        Measure how many days it takes for model to adapt after regime change
        """
        pre_change = trader_preds[trader_preds['date'] < change_date]
        post_change = trader_preds[trader_preds['date'] >= change_date]

        if len(pre_change) < 10 or len(post_change) < 10:
            return None

        pre_auc = roc_auc_score(pre_change['actual'], pre_change['prediction'])

        # Find how many days post-change to reach pre-change performance
        for days in range(10, min(len(post_change), 50)):
            recent_data = post_change.iloc[:days]
            if len(recent_data['actual'].unique()) > 1:
                recent_auc = roc_auc_score(recent_data['actual'], recent_data['prediction'])
                if recent_auc >= pre_auc * 0.9:  # 90% of pre-change performance
                    return days

        return None

    def generate_performance_report(self, val_predictions: pd.DataFrame,
                                  test_predictions: pd.DataFrame) -> pd.DataFrame:
        """
        Generate comprehensive performance report
        """
        val_analysis = self.analyze_validation_performance(val_predictions)
        test_vs_val = self.compare_test_vs_validation(val_predictions, test_predictions)
        adaptation_analysis = self.analyze_model_adaptation(val_predictions)

        report_data = []

        for trader_id in val_predictions['trader_id'].unique():
            if trader_id not in val_analysis:
                continue

            val_metrics = val_analysis[trader_id]
            comparison = test_vs_val.get(trader_id, {})
            adaptation = adaptation_analysis.get(trader_id, {})

            report_data.append({
                'trader_id': trader_id,
                'validation_auc': val_metrics['overall_auc'],
                'validation_ap': val_metrics['overall_ap'],
                'test_auc': comparison.get('test_auc'),
                'test_ap': comparison.get('test_ap'),
                'auc_difference': comparison.get('auc_difference'),
                'overfitting_flag': comparison.get('overfitting_flag', False),
                'prediction_stability': val_metrics['stability_score'],
                'regime_changes': adaptation.get('regime_changes', 0),
                'adaptation_speed': adaptation.get('avg_adaptation_speed'),
                'early_period_auc': val_metrics['period_performance']['early']['auc'],
                'late_period_auc': val_metrics['period_performance']['late']['auc'],
                'performance_trend': self._classify_performance_trend(val_metrics),
                'total_predictions': val_metrics['total_predictions'],
                'positive_rate': val_metrics['positive_rate']
            })

        return pd.DataFrame(report_data)

    def _classify_performance_trend(self, val_metrics: Dict) -> str:
        """
        Classify performance trend based on early vs late period comparison
        """
        early_auc = val_metrics['period_performance']['early']['auc']
        late_auc = val_metrics['period_performance']['late']['auc']

        if early_auc is None or late_auc is None:
            return 'insufficient_data'

        if late_auc < early_auc * 0.9:
            return 'degrading'
        elif late_auc > early_auc * 1.1:
            return 'improving'
        else:
            return 'stable'

    def create_performance_visualizations(self, val_predictions: pd.DataFrame,
                                        test_predictions: pd.DataFrame,
                                        output_dir: str = 'models/expanding_window/plots'):
        """
        Create performance visualization plots
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        # Performance over time plot
        self._plot_performance_over_time(val_predictions, output_dir)

        # AUC comparison plot
        self._plot_auc_comparison(val_predictions, test_predictions, output_dir)

        # Prediction distribution plot
        self._plot_prediction_distributions(val_predictions, test_predictions, output_dir)

        # Regime change detection plot
        self._plot_regime_analysis(val_predictions, output_dir)

        logger.info(f"Performance visualizations saved to: {output_dir}")

    def _plot_performance_over_time(self, predictions: pd.DataFrame, output_dir: str):
        """
        Plot AUC performance over time for each trader
        """
        plt.figure(figsize=(15, 10))

        for i, trader_id in enumerate(predictions['trader_id'].unique()):
            trader_preds = predictions[predictions['trader_id'] == trader_id].sort_values('date')

            if len(trader_preds) < 20:
                continue

            # Calculate rolling AUC
            rolling_auc = []
            dates = []

            for j in range(20, len(trader_preds)):
                window_data = trader_preds.iloc[j-20:j]
                if len(window_data['actual'].unique()) > 1:
                    auc = roc_auc_score(window_data['actual'], window_data['prediction'])
                    rolling_auc.append(auc)
                    dates.append(window_data['date'].iloc[-1])

            plt.subplot(3, 4, i+1)
            plt.plot(dates, rolling_auc, label=f'Trader {trader_id}')
            plt.title(f'Trader {trader_id} - Rolling AUC')
            plt.xlabel('Date')
            plt.ylabel('AUC')
            plt.xticks(rotation=45)
            plt.ylim(0.3, 1.0)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/performance_over_time.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_auc_comparison(self, val_predictions: pd.DataFrame,
                           test_predictions: pd.DataFrame, output_dir: str):
        """
        Plot validation vs test AUC comparison
        """
        comparison = self.compare_test_vs_validation(val_predictions, test_predictions)

        traders = list(comparison.keys())
        val_aucs = [comparison[t]['validation_auc'] for t in traders]
        test_aucs = [comparison[t]['test_auc'] for t in traders]

        plt.figure(figsize=(10, 8))
        plt.scatter(val_aucs, test_aucs, alpha=0.7, s=100)
        plt.plot([0.5, 1.0], [0.5, 1.0], 'r--', label='Perfect Agreement')
        plt.xlabel('Validation AUC')
        plt.ylabel('Test AUC')
        plt.title('Validation vs Test AUC Comparison')
        plt.legend()

        for i, trader in enumerate(traders):
            plt.annotate(f'T{trader}', (val_aucs[i], test_aucs[i]),
                        xytext=(5, 5), textcoords='offset points')

        plt.grid(True, alpha=0.3)
        plt.savefig(f'{output_dir}/auc_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_prediction_distributions(self, val_predictions: pd.DataFrame,
                                     test_predictions: pd.DataFrame, output_dir: str):
        """
        Plot prediction score distributions
        """
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.hist(val_predictions['prediction'], bins=50, alpha=0.7, label='Validation')
        plt.xlabel('Prediction Score')
        plt.ylabel('Frequency')
        plt.title('Validation Prediction Distribution')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.hist(test_predictions['prediction'], bins=50, alpha=0.7, label='Test', color='orange')
        plt.xlabel('Prediction Score')
        plt.ylabel('Frequency')
        plt.title('Test Prediction Distribution')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f'{output_dir}/prediction_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_regime_analysis(self, predictions: pd.DataFrame, output_dir: str):
        """
        Plot regime change analysis
        """
        adaptation_analysis = self.analyze_model_adaptation(predictions)

        traders = list(adaptation_analysis.keys())
        regime_changes = [adaptation_analysis[t]['regime_changes'] for t in traders]
        adaptation_speeds = [adaptation_analysis[t]['avg_adaptation_speed'] or 0 for t in traders]

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.bar(range(len(traders)), regime_changes)
        plt.xlabel('Trader')
        plt.ylabel('Number of Regime Changes')
        plt.title('Detected Regime Changes by Trader')
        plt.xticks(range(len(traders)), [f'T{t}' for t in traders])

        plt.subplot(1, 2, 2)
        plt.bar(range(len(traders)), adaptation_speeds)
        plt.xlabel('Trader')
        plt.ylabel('Average Adaptation Speed (days)')
        plt.title('Model Adaptation Speed by Trader')
        plt.xticks(range(len(traders)), [f'T{t}' for t in traders])

        plt.tight_layout()
        plt.savefig(f'{output_dir}/regime_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()


def run_comprehensive_backtest(val_predictions_path: str, test_predictions_path: str):
    """
    Run comprehensive backtest analysis
    """
    logger.info("Starting comprehensive backtest analysis")

    val_predictions = pd.read_csv(val_predictions_path)
    test_predictions = pd.read_csv(test_predictions_path)

    analyzer = BacktestAnalyzer()

    # Generate performance report
    performance_report = analyzer.generate_performance_report(val_predictions, test_predictions)

    # Create visualizations
    analyzer.create_performance_visualizations(val_predictions, test_predictions)

    # Save report
    report_path = 'models/expanding_window/comprehensive_backtest_report.csv'
    performance_report.to_csv(report_path, index=False)

    logger.info(f"Comprehensive backtest report saved to: {report_path}")

    # Print summary
    logger.info("\n=== BACKTEST SUMMARY ===")
    logger.info(f"Total traders analyzed: {len(performance_report)}")
    logger.info(f"Average validation AUC: {performance_report['validation_auc'].mean():.4f}")
    logger.info(f"Average test AUC: {performance_report['test_auc'].mean():.4f}")
    logger.info(f"Traders with overfitting: {performance_report['overfitting_flag'].sum()}")
    logger.info(f"Traders with degrading performance: {sum(performance_report['performance_trend'] == 'degrading')}")
    logger.info(f"Traders with stable performance: {sum(performance_report['performance_trend'] == 'stable')}")
    logger.info(f"Traders with improving performance: {sum(performance_report['performance_trend'] == 'improving')}")

    return performance_report


if __name__ == "__main__":
    run_comprehensive_backtest(
        'models/expanding_window/validation_predictions.csv',
        'models/expanding_window/test_predictions.csv'
    )
