#!/usr/bin/env python3
"""
Performance Comparison Report Generator

Compares traditional models vs enhanced models with fills-based features.
Generates comprehensive performance analysis and recommendations.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceComparator:
    """Generate comprehensive performance comparison between traditional and enhanced models."""

    def __init__(self):
        self.reports_dir = Path('reports')
        self.reports_dir.mkdir(exist_ok=True)

    def load_evaluation_results(self):
        """Load evaluation results from both traditional and enhanced models."""
        results = {}

        # Load enhanced evaluation results
        enhanced_path = self.reports_dir / 'enhanced_evaluation_results.json'
        if enhanced_path.exists():
            with open(enhanced_path, 'r') as f:
                results['enhanced'] = json.load(f)
            logger.info("Loaded enhanced evaluation results")
        else:
            logger.warning("Enhanced evaluation results not found")
            results['enhanced'] = None

        # Try to load traditional results if they exist
        traditional_path = self.reports_dir / 'traditional_evaluation_results.json'
        if traditional_path.exists():
            with open(traditional_path, 'r') as f:
                results['traditional'] = json.load(f)
            logger.info("Loaded traditional evaluation results")
        else:
            logger.info("Traditional evaluation results not available - enhanced comparison only")
            results['traditional'] = None

        return results

    def extract_model_metrics(self, results, model_type):
        """Extract key metrics from evaluation results."""
        if not results:
            return {}

        metrics = {}
        individual_results = results.get('individual_results', {})

        for trader_id, trader_results in individual_results.items():
            if 'error' in trader_results:
                continue

            trader_metrics = {}
            eval_metrics = trader_results.get('evaluation_metrics', {})

            # Classification metrics
            if 'classification' in eval_metrics:
                cls_metrics = eval_metrics['classification']
                if isinstance(cls_metrics, dict) and 'auc' in cls_metrics:
                    trader_metrics['auc'] = cls_metrics['auc']
                    trader_metrics['accuracy'] = cls_metrics.get('accuracy', None)
                    trader_metrics['precision'] = cls_metrics.get('precision', None)
                    trader_metrics['recall'] = cls_metrics.get('recall', None)

            # Regression metrics
            if 'regression' in eval_metrics:
                reg_metrics = eval_metrics['regression']
                trader_metrics['rmse'] = reg_metrics.get('rmse', None)
                trader_metrics['mae'] = reg_metrics.get('mae', None)
                trader_metrics['r2'] = reg_metrics.get('r2', None)

            # Risk metrics
            if 'risk_metrics' in eval_metrics:
                risk_metrics = eval_metrics['risk_metrics']
                trader_metrics['large_loss_rate'] = risk_metrics.get('large_loss_rate', None)
                trader_metrics['avg_loss_prob_on_loss_days'] = risk_metrics.get('avg_loss_probability_on_loss_days', None)
                trader_metrics['avg_loss_prob_on_normal_days'] = risk_metrics.get('avg_loss_probability_on_normal_days', None)

            # Enhanced model specific metrics
            if model_type == 'enhanced':
                # Feature analysis
                feature_analysis = trader_results.get('feature_analysis', {})
                if 'features_by_category' in feature_analysis:
                    features_by_cat = feature_analysis['features_by_category']
                    trader_metrics['traditional_features'] = features_by_cat.get('traditional', 0)
                    trader_metrics['fills_features'] = features_by_cat.get('fills_based', 0)
                    trader_metrics['execution_features'] = features_by_cat.get('execution_quality', 0)
                    trader_metrics['cross_features'] = features_by_cat.get('cross_features', 0)

                # Model insights
                model_insights = trader_results.get('model_insights', {})
                enhanced_impact = model_insights.get('enhanced_features_impact', {})
                trader_metrics['fills_importance'] = enhanced_impact.get('fills_based_importance', 0)
                trader_metrics['execution_importance'] = enhanced_impact.get('execution_quality_importance', 0)

            metrics[trader_id] = trader_metrics

        return metrics

    def compare_performance(self, traditional_metrics, enhanced_metrics):
        """Compare performance between traditional and enhanced models."""
        comparison = {
            'summary': {},
            'trader_comparisons': {},
            'improvements': {},
            'regressions': {}
        }

        # Get common traders
        if traditional_metrics:
            common_traders = set(traditional_metrics.keys()) & set(enhanced_metrics.keys())
        else:
            common_traders = set(enhanced_metrics.keys())

        comparison['summary']['total_traders'] = len(common_traders)
        logger.info(f"Comparing performance for {len(common_traders)} traders")

        if not traditional_metrics:
            # Enhanced-only analysis
            comparison['summary']['comparison_type'] = 'enhanced_only'
            comparison['summary']['enhanced_model_performance'] = self._analyze_enhanced_only(enhanced_metrics)
            return comparison

        # Full comparison
        comparison['summary']['comparison_type'] = 'traditional_vs_enhanced'

        improvements = {'auc': [], 'rmse': [], 'r2': []}
        regressions = {'auc': [], 'rmse': [], 'r2': []}

        for trader_id in common_traders:
            trad_metrics = traditional_metrics[trader_id]
            enh_metrics = enhanced_metrics[trader_id]

            trader_comparison = {}

            # AUC comparison
            if 'auc' in trad_metrics and 'auc' in enh_metrics:
                trad_auc = trad_metrics['auc']
                enh_auc = enh_metrics['auc']
                improvement = enh_auc - trad_auc
                improvement_pct = (improvement / trad_auc) * 100 if trad_auc > 0 else 0

                trader_comparison['auc'] = {
                    'traditional': trad_auc,
                    'enhanced': enh_auc,
                    'improvement': improvement,
                    'improvement_pct': improvement_pct
                }

                if improvement > 0:
                    improvements['auc'].append(improvement)
                elif improvement < 0:
                    regressions['auc'].append(improvement)

            # RMSE comparison (lower is better)
            if 'rmse' in trad_metrics and 'rmse' in enh_metrics:
                trad_rmse = trad_metrics['rmse']
                enh_rmse = enh_metrics['rmse']
                improvement = trad_rmse - enh_rmse  # Positive means enhanced is better
                improvement_pct = (improvement / trad_rmse) * 100 if trad_rmse > 0 else 0

                trader_comparison['rmse'] = {
                    'traditional': trad_rmse,
                    'enhanced': enh_rmse,
                    'improvement': improvement,
                    'improvement_pct': improvement_pct
                }

                if improvement > 0:
                    improvements['rmse'].append(improvement)
                elif improvement < 0:
                    regressions['rmse'].append(improvement)

            # R² comparison
            if 'r2' in trad_metrics and 'r2' in enh_metrics:
                trad_r2 = trad_metrics['r2']
                enh_r2 = enh_metrics['r2']
                improvement = enh_r2 - trad_r2

                trader_comparison['r2'] = {
                    'traditional': trad_r2,
                    'enhanced': enh_r2,
                    'improvement': improvement
                }

                if improvement > 0:
                    improvements['r2'].append(improvement)
                elif improvement < 0:
                    regressions['r2'].append(improvement)

            # Enhanced features impact
            trader_comparison['enhanced_features'] = {
                'fills_importance': enh_metrics.get('fills_importance', 0),
                'execution_importance': enh_metrics.get('execution_importance', 0),
                'total_fills_features': enh_metrics.get('fills_features', 0),
                'total_execution_features': enh_metrics.get('execution_features', 0)
            }

            comparison['trader_comparisons'][trader_id] = trader_comparison

        # Aggregate improvements/regressions
        for metric in ['auc', 'rmse', 'r2']:
            if improvements[metric]:
                comparison['improvements'][metric] = {
                    'count': len(improvements[metric]),
                    'mean': np.mean(improvements[metric]),
                    'std': np.std(improvements[metric]),
                    'max': max(improvements[metric])
                }

            if regressions[metric]:
                comparison['regressions'][metric] = {
                    'count': len(regressions[metric]),
                    'mean': np.mean(regressions[metric]),
                    'std': np.std(regressions[metric]),
                    'min': min(regressions[metric])
                }

        # Overall summary statistics
        all_auc_improvements = improvements['auc']
        all_rmse_improvements = improvements['rmse']
        all_r2_improvements = improvements['r2']

        comparison['summary']['overall'] = {
            'traders_with_auc_improvement': len(all_auc_improvements),
            'traders_with_rmse_improvement': len(all_rmse_improvements),
            'traders_with_r2_improvement': len(all_r2_improvements),
            'mean_auc_improvement': np.mean(all_auc_improvements) if all_auc_improvements else 0,
            'mean_rmse_improvement_pct': np.mean([comp['trader_comparisons'][tid]['rmse']['improvement_pct']
                                               for tid in comp['trader_comparisons']
                                               if 'rmse' in comp['trader_comparisons'][tid]]) if comparison['trader_comparisons'] else 0
        }

        return comparison

    def _analyze_enhanced_only(self, enhanced_metrics):
        """Analyze enhanced model performance when traditional models are not available."""
        analysis = {}

        traders = list(enhanced_metrics.keys())
        analysis['total_traders'] = len(traders)

        # AUC analysis
        aucs = [metrics.get('auc') for metrics in enhanced_metrics.values() if metrics.get('auc') is not None]
        if aucs:
            analysis['auc'] = {
                'mean': np.mean(aucs),
                'std': np.std(aucs),
                'min': min(aucs),
                'max': max(aucs),
                'count': len(aucs)
            }

        # RMSE analysis
        rmses = [metrics.get('rmse') for metrics in enhanced_metrics.values() if metrics.get('rmse') is not None]
        if rmses:
            analysis['rmse'] = {
                'mean': np.mean(rmses),
                'std': np.std(rmses),
                'min': min(rmses),
                'max': max(rmses),
                'count': len(rmses)
            }

        # Enhanced features analysis
        fills_features = [metrics.get('fills_features', 0) for metrics in enhanced_metrics.values()]
        execution_features = [metrics.get('execution_features', 0) for metrics in enhanced_metrics.values()]
        fills_importance = [metrics.get('fills_importance', 0) for metrics in enhanced_metrics.values()]

        analysis['enhanced_features'] = {
            'avg_fills_features_per_trader': np.mean(fills_features),
            'avg_execution_features_per_trader': np.mean(execution_features),
            'avg_fills_importance': np.mean(fills_importance),
            'traders_with_fills_features': sum(1 for f in fills_features if f > 0),
            'traders_with_execution_features': sum(1 for f in execution_features if f > 0)
        }

        return analysis

    def generate_report(self):
        """Generate comprehensive performance comparison report."""
        logger.info("Generating performance comparison report...")

        # Load evaluation results
        results = self.load_evaluation_results()

        # Extract metrics
        enhanced_metrics = self.extract_model_metrics(results['enhanced'], 'enhanced')
        traditional_metrics = self.extract_model_metrics(results['traditional'], 'traditional') if results['traditional'] else {}

        # Compare performance
        comparison = self.compare_performance(traditional_metrics, enhanced_metrics)

        # Generate report
        report = {
            'report_metadata': {
                'generation_timestamp': datetime.now().isoformat(),
                'report_type': 'performance_comparison',
                'enhanced_models_available': bool(enhanced_metrics),
                'traditional_models_available': bool(traditional_metrics)
            },
            'executive_summary': self._generate_executive_summary(comparison),
            'detailed_comparison': comparison,
            'recommendations': self._generate_recommendations(comparison),
            'model_metrics': {
                'enhanced': enhanced_metrics,
                'traditional': traditional_metrics
            }
        }

        # Save report
        report_path = self.reports_dir / f'performance_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Performance comparison report saved to {report_path}")

        # Generate summary markdown
        self._generate_markdown_summary(report, report_path.with_suffix('.md'))

        return report

    def _generate_executive_summary(self, comparison):
        """Generate executive summary of performance comparison."""
        summary = comparison['summary']

        exec_summary = {
            'total_traders_evaluated': summary.get('total_traders', 0),
            'comparison_type': summary.get('comparison_type', 'enhanced_only')
        }

        if summary.get('comparison_type') == 'enhanced_only':
            enh_perf = summary.get('enhanced_model_performance', {})
            exec_summary['enhanced_model_summary'] = {
                'mean_auc': enh_perf.get('auc', {}).get('mean', 0),
                'mean_rmse': enh_perf.get('rmse', {}).get('mean', 0),
                'avg_fills_features': enh_perf.get('enhanced_features', {}).get('avg_fills_features_per_trader', 0),
                'traders_using_fills_features': enh_perf.get('enhanced_features', {}).get('traders_with_fills_features', 0)
            }
        else:
            overall = summary.get('overall', {})
            exec_summary['performance_improvements'] = {
                'traders_with_auc_improvement': overall.get('traders_with_auc_improvement', 0),
                'traders_with_rmse_improvement': overall.get('traders_with_rmse_improvement', 0),
                'mean_auc_improvement': overall.get('mean_auc_improvement', 0),
                'mean_rmse_improvement_pct': overall.get('mean_rmse_improvement_pct', 0)
            }

        return exec_summary

    def _generate_recommendations(self, comparison):
        """Generate actionable recommendations based on performance comparison."""
        recommendations = []

        if comparison['summary']['comparison_type'] == 'enhanced_only':
            enh_perf = comparison['summary'].get('enhanced_model_performance', {})

            recommendations.append({
                'category': 'Model Performance',
                'priority': 'High',
                'recommendation': f"Enhanced models achieve mean AUC of {enh_perf.get('auc', {}).get('mean', 0):.3f}. Monitor performance and consider retraining if AUC drops below 0.600.",
                'action_items': [
                    'Set up automated performance monitoring',
                    'Define AUC threshold alerts',
                    'Schedule quarterly model retraining'
                ]
            })

            fills_usage = enh_perf.get('enhanced_features', {})
            if fills_usage.get('traders_with_fills_features', 0) > 0:
                recommendations.append({
                    'category': 'Feature Engineering',
                    'priority': 'Medium',
                    'recommendation': f"Fills-based features are being used by {fills_usage.get('traders_with_fills_features', 0)} traders with average importance of {fills_usage.get('avg_fills_importance', 0):.1f}%. Consider expanding fills feature engineering.",
                    'action_items': [
                        'Analyze top-performing fills features',
                        'Develop additional execution quality metrics',
                        'Test cross-trader feature sharing'
                    ]
                })

        else:
            # Traditional vs Enhanced comparison
            overall = comparison['summary'].get('overall', {})

            if overall.get('traders_with_auc_improvement', 0) > overall.get('total_traders', 1) * 0.6:
                recommendations.append({
                    'category': 'Model Migration',
                    'priority': 'High',
                    'recommendation': f"Enhanced models show AUC improvements for {overall.get('traders_with_auc_improvement', 0)} out of {comparison['summary']['total_traders']} traders. Recommend full migration to enhanced models.",
                    'action_items': [
                        'Deploy enhanced models to production',
                        'Archive traditional models as backup',
                        'Update monitoring and alerting systems'
                    ]
                })

            if overall.get('mean_rmse_improvement_pct', 0) > 5:
                recommendations.append({
                    'category': 'Risk Prediction',
                    'priority': 'High',
                    'recommendation': f"Enhanced models show {overall.get('mean_rmse_improvement_pct', 0):.1f}% RMSE improvement on average. This indicates better risk prediction accuracy.",
                    'action_items': [
                        'Update risk limits based on improved predictions',
                        'Recalibrate position sizing algorithms',
                        'Review and update risk management policies'
                    ]
                })

        # General recommendations
        recommendations.append({
            'category': 'Data Quality',
            'priority': 'Medium',
            'recommendation': 'Enhanced models rely on fills data with ~48% coverage. Improving fills data quality could further enhance model performance.',
            'action_items': [
                'Investigate fills data gaps',
                'Implement data quality checks',
                'Consider alternative data sources for missing periods'
            ]
        })

        return recommendations

    def _generate_markdown_summary(self, report, output_path):
        """Generate human-readable markdown summary."""
        exec_summary = report['executive_summary']
        recommendations = report['recommendations']

        markdown = f"""# Risk Model Performance Comparison Report

**Generated:** {report['report_metadata']['generation_timestamp']}

## Executive Summary

"""

        if exec_summary['comparison_type'] == 'enhanced_only':
            enh_summary = exec_summary['enhanced_model_summary']
            markdown += f"""
- **Traders Evaluated:** {exec_summary['total_traders_evaluated']}
- **Model Type:** Enhanced models with fills-based features only
- **Mean AUC:** {enh_summary['mean_auc']:.4f}
- **Mean RMSE:** {enh_summary['mean_rmse']:.1f}
- **Traders Using Fills Features:** {enh_summary['traders_using_fills_features']} / {exec_summary['total_traders_evaluated']}
- **Average Fills Features per Trader:** {enh_summary['avg_fills_features']:.1f}
"""
        else:
            perf_improvements = exec_summary['performance_improvements']
            markdown += f"""
- **Traders Evaluated:** {exec_summary['total_traders_evaluated']}
- **Model Type:** Traditional vs Enhanced comparison
- **Traders with AUC Improvement:** {perf_improvements['traders_with_auc_improvement']}
- **Traders with RMSE Improvement:** {perf_improvements['traders_with_rmse_improvement']}
- **Mean AUC Improvement:** {perf_improvements['mean_auc_improvement']:.4f}
- **Mean RMSE Improvement:** {perf_improvements['mean_rmse_improvement_pct']:.1f}%
"""

        markdown += "\n## Key Recommendations\n\n"

        for i, rec in enumerate(recommendations, 1):
            markdown += f"""### {i}. {rec['category']} ({rec['priority']} Priority)

**Recommendation:** {rec['recommendation']}

**Action Items:**
"""
            for action in rec['action_items']:
                markdown += f"- {action}\n"
            markdown += "\n"

        markdown += """
## Next Steps

1. Review detailed comparison results in the JSON report
2. Implement high-priority recommendations
3. Schedule follow-up performance evaluation
4. Update production deployment plan

---
*This report was generated automatically by the Risk Tool Performance Comparison system.*
"""

        with open(output_path, 'w') as f:
            f.write(markdown)

        logger.info(f"Markdown summary saved to {output_path}")


def main():
    """Generate performance comparison report."""
    comparator = PerformanceComparator()
    report = comparator.generate_report()

    exec_summary = report['executive_summary']
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("="*60)

    print(f"Traders Evaluated: {exec_summary['total_traders_evaluated']}")
    print(f"Comparison Type: {exec_summary['comparison_type']}")

    if exec_summary['comparison_type'] == 'enhanced_only':
        enh = exec_summary['enhanced_model_summary']
        print(f"Mean AUC: {enh['mean_auc']:.4f}")
        print(f"Mean RMSE: {enh['mean_rmse']:.1f}")
        print(f"Traders using fills features: {enh['traders_using_fills_features']}")
    else:
        perf = exec_summary['performance_improvements']
        print(f"Traders with AUC improvement: {perf['traders_with_auc_improvement']}")
        print(f"Traders with RMSE improvement: {perf['traders_with_rmse_improvement']}")
        print(f"Mean AUC improvement: {perf['mean_auc_improvement']:.4f}")

    print(f"\nRecommendations: {len(report['recommendations'])}")
    print("="*60)

    return report


if __name__ == "__main__":
    main()
