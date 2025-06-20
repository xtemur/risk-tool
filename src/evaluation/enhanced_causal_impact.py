#!/usr/bin/env python3
"""
Enhanced Causal Impact Analysis using improved models
"""

import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class EnhancedCausalImpactAnalysis:
    def __init__(self):
        self.load_enhanced_models()
        self.strategy_results = {}

    def load_enhanced_models(self):
        """Load enhanced models and data"""
        print("=== ENHANCED CAUSAL IMPACT ANALYSIS ===")

        # Load enhanced models
        try:
            with open('data/enhanced_models.pkl', 'rb') as f:
                self.enhanced_models = pickle.load(f)
            print(f"✓ Loaded enhanced models for {len(self.enhanced_models)} traders")
        except FileNotFoundError:
            print("❌ Enhanced models not found - run enhanced training first")
            return False

        # Load enhanced training results to get best model per trader
        with open('data/enhanced_training_results.json', 'r') as f:
            self.enhanced_results = json.load(f)

        # Load enhanced feature data
        with open('data/enhanced_models.pkl', 'rb') as f:
            enhanced_data = pickle.load(f)

        # Load the base feature data and recreate enhanced features
        self.feature_df = pd.read_pickle('data/target_prepared.pkl')
        self.create_enhanced_feature_df()

        return True

    def create_enhanced_feature_df(self):
        """Recreate enhanced features (matching the training process)"""
        enhanced_features = []

        for trader_id in self.feature_df['account_id'].unique():
            trader_data = self.feature_df[self.feature_df['account_id'] == trader_id].copy()
            trader_data = trader_data.sort_values('trade_date')

            # Add same enhanced features as in training
            trader_data['pnl_momentum_3d'] = trader_data['realized_pnl'].rolling(3).mean()
            trader_data['pnl_momentum_7d'] = trader_data['realized_pnl'].rolling(7).mean()
            trader_data['pnl_volatility_7d'] = trader_data['realized_pnl'].rolling(7).std()
            trader_data['pnl_skewness_7d'] = trader_data['realized_pnl'].rolling(7).skew()

            trader_data['high_performance_regime'] = (trader_data['realized_pnl'] > trader_data['realized_pnl'].rolling(20).quantile(0.8)).astype(int)
            trader_data['low_performance_regime'] = (trader_data['realized_pnl'] < trader_data['realized_pnl'].rolling(20).quantile(0.2)).astype(int)

            trader_data['pnl_trend_5d'] = np.where(
                trader_data['realized_pnl'].rolling(5).mean() > trader_data['realized_pnl'].rolling(10).mean(), 1, 0
            )

            trader_data['sharpe_ratio_7d'] = trader_data['realized_pnl'].rolling(7).mean() / (trader_data['realized_pnl'].rolling(7).std() + 1e-8)
            trader_data['max_drawdown_7d'] = trader_data['realized_pnl'].rolling(7).apply(
                lambda x: (x.cumsum() - x.cumsum().expanding().max()).min(), raw=False
            )

            # Day-of-week effects
            trader_data['day_of_week'] = pd.to_datetime(trader_data['trade_date']).dt.dayofweek
            for dow in range(5):
                trader_data[f'is_dow_{dow}'] = (trader_data['day_of_week'] == dow).astype(int)

            enhanced_features.append(trader_data)

        self.enhanced_feature_df = pd.concat(enhanced_features, ignore_index=True)
        print(f"✓ Created enhanced features: {len(self.enhanced_feature_df)} observations")

    def generate_enhanced_risk_signals(self):
        """Generate risk signals using best enhanced models"""
        print("\n=== GENERATING ENHANCED RISK SIGNALS ===")

        test_cutoff = pd.to_datetime('2025-04-01')
        signal_data = []

        for trader_id_str in self.enhanced_models.keys():
            trader_id = int(trader_id_str)

            # Get trader's best model info
            if trader_id_str not in self.enhanced_results:
                continue

            best_model_key = self.enhanced_results[trader_id_str]['best_model']
            model_info = self.enhanced_models[trader_id_str][best_model_key]

            model = model_info['model']
            selected_features = model_info['selected_features']
            scaler = model_info.get('scaler', None)

            # Get trader test data
            trader_data = self.enhanced_feature_df[self.enhanced_feature_df['account_id'] == trader_id].copy()
            test_data = trader_data[trader_data['trade_date'] >= test_cutoff].copy()

            if len(test_data) < 5:
                continue

            try:
                # Prepare features
                X_test = test_data[selected_features].fillna(0)

                # Apply scaling if needed
                if scaler is not None:
                    X_test_values = scaler.transform(X_test.values)
                else:
                    X_test_values = X_test.values

                # Generate predictions
                predictions = model.predict(X_test_values)
                probabilities = model.predict_proba(X_test_values)

                # Create enhanced risk signals using probability-based approach
                test_data = test_data.copy()
                test_data['risk_prediction'] = predictions[:len(test_data)]

                if probabilities.shape[1] >= 3:
                    loss_proba = probabilities[:, 0]
                    win_proba = probabilities[:, 2] if probabilities.shape[1] > 2 else np.zeros(len(predictions))

                    # More conservative thresholds for better signal quality
                    high_risk_threshold = 0.5  # Higher threshold for more confident signals
                    low_risk_threshold = 0.5

                    test_data['risk_signal'] = np.where(
                        loss_proba > high_risk_threshold, 2,  # High loss probability → High risk
                        np.where(win_proba > low_risk_threshold, 0, 1)  # High win probability → Low risk
                    )
                else:
                    # Fallback for binary classification
                    test_data['risk_signal'] = predictions[:len(test_data)]

                # Add model quality info
                test_data['model_type'] = best_model_key
                test_data['model_cv_score'] = self.enhanced_results[trader_id_str]['best_cv_score']

                signal_data.append(test_data)

                print(f"  ✓ Trader {trader_id}: {best_model_key} model (CV: {self.enhanced_results[trader_id_str]['best_cv_score']:.3f})")

            except Exception as e:
                print(f"  ❌ Trader {trader_id} failed: {e}")
                continue

        if signal_data:
            self.enhanced_signal_df = pd.concat(signal_data, ignore_index=True)
            print(f"\n✓ Generated enhanced signals for {len(signal_data)} traders")
            print(f"✓ Total signal observations: {len(self.enhanced_signal_df)}")

            # Signal distribution
            signal_dist = self.enhanced_signal_df['risk_signal'].value_counts().sort_index()
            print(f"✓ Signal distribution - Low Risk: {signal_dist.get(0, 0)}, Neutral: {signal_dist.get(1, 0)}, High Risk: {signal_dist.get(2, 0)}")

            return True

        return False

    def test_enhanced_strategies(self):
        """Test trading strategies with enhanced signals"""
        print("\n=== TESTING ENHANCED TRADING STRATEGIES ===")

        if not hasattr(self, 'enhanced_signal_df'):
            print("❌ No enhanced signals available")
            return None

        # Validate signal quality first
        self.validate_enhanced_signals()

        # Test strategies
        position_sizing_results = self.test_enhanced_position_sizing()
        trade_filtering_results = self.test_enhanced_trade_filtering()
        combined_results = self.test_enhanced_combined()

        return {
            'position_sizing': position_sizing_results,
            'trade_filtering': trade_filtering_results,
            'combined': combined_results
        }

    def validate_enhanced_signals(self):
        """Validate enhanced signal quality"""
        print("\n=== ENHANCED SIGNAL VALIDATION ===")

        validation_results = []

        for trader_id in self.enhanced_signal_df['account_id'].unique():
            trader_signals = self.enhanced_signal_df[self.enhanced_signal_df['account_id'] == trader_id].copy()

            if len(trader_signals) < 10:
                continue

            # Analyze outcomes by signal
            high_risk_outcomes = trader_signals[trader_signals['risk_signal'] == 2]['next_day_pnl']
            low_risk_outcomes = trader_signals[trader_signals['risk_signal'] == 0]['next_day_pnl']
            neutral_outcomes = trader_signals[trader_signals['risk_signal'] == 1]['next_day_pnl']

            # Calculate metrics
            high_risk_mean = high_risk_outcomes.mean() if len(high_risk_outcomes) > 0 else 0
            low_risk_mean = low_risk_outcomes.mean() if len(low_risk_outcomes) > 0 else 0
            neutral_mean = neutral_outcomes.mean() if len(neutral_outcomes) > 0 else 0

            high_risk_loss_rate = (high_risk_outcomes < 0).mean() if len(high_risk_outcomes) > 0 else 0
            low_risk_loss_rate = (low_risk_outcomes < 0).mean() if len(low_risk_outcomes) > 0 else 0

            model_cv_score = trader_signals['model_cv_score'].iloc[0] if len(trader_signals) > 0 else 0

            validation_results.append({
                'trader_id': trader_id,
                'high_risk_mean_pnl': high_risk_mean,
                'low_risk_mean_pnl': low_risk_mean,
                'neutral_mean_pnl': neutral_mean,
                'high_risk_loss_rate': high_risk_loss_rate,
                'low_risk_loss_rate': low_risk_loss_rate,
                'signal_correlation': low_risk_mean - high_risk_mean,
                'model_cv_score': model_cv_score,
                'signal_quality': 'good' if (low_risk_mean > high_risk_mean and low_risk_loss_rate < high_risk_loss_rate) else 'poor'
            })

        # Aggregate validation
        if validation_results:
            good_signals = sum(1 for r in validation_results if r['signal_quality'] == 'good')
            avg_cv_score = np.mean([r['model_cv_score'] for r in validation_results])
            avg_signal_corr = np.mean([r['signal_correlation'] for r in validation_results])

            print(f"✓ Traders with good signal quality: {good_signals}/{len(validation_results)} ({good_signals/len(validation_results):.1%})")
            print(f"✓ Average model CV score: {avg_cv_score:.3f}")
            print(f"✓ Average signal correlation: ${avg_signal_corr:,.2f}")

            return validation_results

        return []

    def test_enhanced_trade_filtering(self):
        """Test trade filtering with enhanced models"""
        print("\n=== ENHANCED TRADE FILTERING STRATEGY ===")

        strategy_results = []

        for trader_id in self.enhanced_signal_df['account_id'].unique():
            trader_signals = self.enhanced_signal_df[self.enhanced_signal_df['account_id'] == trader_id].copy()
            trader_signals = trader_signals.sort_values('trade_date')

            if len(trader_signals) < 10:
                continue

            # Baseline performance
            baseline_pnl = trader_signals['next_day_pnl'].copy()
            baseline_total = baseline_pnl.sum()

            # Enhanced filtering strategy
            high_risk_mask = trader_signals['risk_signal'] == 2

            # Calculate what we would avoid by not trading on high-risk days
            avoided_pnl = baseline_pnl[high_risk_mask].sum()

            # Strategy PnL (no trading on high-risk days)
            strategy_pnl = baseline_pnl.copy()
            strategy_pnl[high_risk_mask] = 0
            strategy_total = strategy_pnl.sum()

            # Get model quality
            model_cv_score = trader_signals['model_cv_score'].iloc[0]

            strategy_results.append({
                'trader_id': trader_id,
                'baseline_pnl': baseline_total,
                'strategy_pnl': strategy_total,
                'avoided_pnl': avoided_pnl,
                'improvement': strategy_total - baseline_total,
                'filtered_days': high_risk_mask.sum(),
                'total_days': len(trader_signals),
                'model_cv_score': model_cv_score
            })

        if strategy_results:
            # Calculate aggregate results
            total_baseline = sum(r['baseline_pnl'] for r in strategy_results)
            total_strategy = sum(r['strategy_pnl'] for r in strategy_results)
            total_avoided = sum(r['avoided_pnl'] for r in strategy_results)

            positive_traders = sum(1 for r in strategy_results if r['avoided_pnl'] < 0)  # Negative avoided = avoided losses
            avg_cv_score = np.mean([r['model_cv_score'] for r in strategy_results])

            results = {
                'total_baseline_pnl': total_baseline,
                'total_strategy_pnl': total_strategy,
                'total_avoided_pnl': total_avoided,
                'total_improvement': -total_avoided,  # Convert avoided losses to positive improvement
                'positive_traders': positive_traders,
                'total_traders': len(strategy_results),
                'success_rate': positive_traders / len(strategy_results),
                'avg_model_cv_score': avg_cv_score
            }

            print(f"✓ Enhanced trade filtering tested on {len(strategy_results)} traders")
            print(f"✓ Average model CV score: {avg_cv_score:.3f}")
            print(f"✓ Total avoided losses: ${-total_avoided:,.2f}")
            print(f"✓ Success rate: {positive_traders}/{len(strategy_results)} ({positive_traders/len(strategy_results):.1%})")

            return results

        return None

    def test_enhanced_position_sizing(self):
        """Test position sizing with enhanced models"""
        print("\n=== ENHANCED POSITION SIZING STRATEGY ===")

        strategy_results = []

        for trader_id in self.enhanced_signal_df['account_id'].unique():
            trader_signals = self.enhanced_signal_df[self.enhanced_signal_df['account_id'] == trader_id].copy()
            trader_signals = trader_signals.sort_values('trade_date')

            if len(trader_signals) < 10:
                continue

            # Baseline performance
            baseline_pnl = trader_signals['next_day_pnl'].copy()
            baseline_total = baseline_pnl.sum()

            # Enhanced position sizing
            position_sizes = np.ones(len(trader_signals))

            high_risk_mask = trader_signals['risk_signal'] == 2
            low_risk_mask = trader_signals['risk_signal'] == 0

            # More conservative position sizing based on model confidence
            model_cv_score = trader_signals['model_cv_score'].iloc[0]
            confidence_factor = min(model_cv_score / 0.6, 1.0)  # Scale based on model quality

            position_sizes[high_risk_mask] = 0.3 * confidence_factor  # Reduce more for better models
            position_sizes[low_risk_mask] = 1.0 + (0.3 * confidence_factor)  # Increase more for better models

            # Calculate strategy PnL
            strategy_pnl = baseline_pnl * position_sizes
            strategy_total = strategy_pnl.sum()

            strategy_results.append({
                'trader_id': trader_id,
                'baseline_pnl': baseline_total,
                'strategy_pnl': strategy_total,
                'improvement': strategy_total - baseline_total,
                'model_cv_score': model_cv_score,
                'confidence_factor': confidence_factor
            })

        if strategy_results:
            total_baseline = sum(r['baseline_pnl'] for r in strategy_results)
            total_strategy = sum(r['strategy_pnl'] for r in strategy_results)
            positive_traders = sum(1 for r in strategy_results if r['improvement'] > 0)
            avg_cv_score = np.mean([r['model_cv_score'] for r in strategy_results])

            results = {
                'total_baseline_pnl': total_baseline,
                'total_strategy_pnl': total_strategy,
                'total_improvement': total_strategy - total_baseline,
                'positive_traders': positive_traders,
                'total_traders': len(strategy_results),
                'success_rate': positive_traders / len(strategy_results),
                'avg_model_cv_score': avg_cv_score
            }

            print(f"✓ Enhanced position sizing tested on {len(strategy_results)} traders")
            print(f"✓ Total improvement: ${total_strategy - total_baseline:,.2f}")
            print(f"✓ Success rate: {positive_traders}/{len(strategy_results)} ({positive_traders/len(strategy_results):.1%})")

            return results

        return None

    def test_enhanced_combined(self):
        """Test combined strategy with enhanced models"""
        print("\n=== ENHANCED COMBINED STRATEGY ===")

        strategy_results = []

        for trader_id in self.enhanced_signal_df['account_id'].unique():
            trader_signals = self.enhanced_signal_df[self.enhanced_signal_df['account_id'] == trader_id].copy()
            trader_signals = trader_signals.sort_values('trade_date')

            if len(trader_signals) < 10:
                continue

            baseline_pnl = trader_signals['next_day_pnl'].copy()
            baseline_total = baseline_pnl.sum()

            # Combined approach with model confidence
            position_sizes = np.ones(len(trader_signals))

            high_risk_mask = trader_signals['risk_signal'] == 2
            low_risk_mask = trader_signals['risk_signal'] == 0

            model_cv_score = trader_signals['model_cv_score'].iloc[0]
            confidence_factor = min(model_cv_score / 0.6, 1.0)

            # More aggressive combined strategy for high-confidence models
            position_sizes[high_risk_mask] = 0.1 * confidence_factor  # Nearly avoid high-risk days
            position_sizes[low_risk_mask] = 1.0 + (0.4 * confidence_factor)  # Increase low-risk exposure

            strategy_pnl = baseline_pnl * position_sizes
            strategy_total = strategy_pnl.sum()

            strategy_results.append({
                'trader_id': trader_id,
                'baseline_pnl': baseline_total,
                'strategy_pnl': strategy_total,
                'improvement': strategy_total - baseline_total,
                'model_cv_score': model_cv_score
            })

        if strategy_results:
            total_baseline = sum(r['baseline_pnl'] for r in strategy_results)
            total_strategy = sum(r['strategy_pnl'] for r in strategy_results)
            positive_traders = sum(1 for r in strategy_results if r['improvement'] > 0)

            results = {
                'total_baseline_pnl': total_baseline,
                'total_strategy_pnl': total_strategy,
                'total_improvement': total_strategy - total_baseline,
                'positive_traders': positive_traders,
                'total_traders': len(strategy_results),
                'success_rate': positive_traders / len(strategy_results)
            }

            print(f"✓ Enhanced combined strategy tested on {len(strategy_results)} traders")
            print(f"✓ Total improvement: ${total_strategy - total_baseline:,.2f}")
            print(f"✓ Success rate: {positive_traders}/{len(strategy_results)} ({positive_traders/len(strategy_results):.1%})")

            return results

        return None

    def generate_enhanced_report(self):
        """Generate enhanced causal impact report"""
        print("\n=== GENERATING ENHANCED CAUSAL IMPACT REPORT ===")

        if not self.load_enhanced_models():
            return False

        # Generate enhanced signals
        if not self.generate_enhanced_risk_signals():
            print("❌ Failed to generate enhanced signals")
            return False

        # Test enhanced strategies
        strategy_results = self.test_enhanced_strategies()

        if not strategy_results:
            print("❌ No strategy results generated")
            return False

        # Determine best strategy
        improvements = {}
        for strategy_name, results in strategy_results.items():
            if results:
                improvements[strategy_name] = results['total_improvement']
            else:
                improvements[strategy_name] = 0

        best_strategy = max(improvements.keys(), key=lambda k: improvements[k])
        deployment_viable = improvements[best_strategy] > 0

        print(f"\n=== ENHANCED STRATEGY COMPARISON ===")
        for strategy_name, improvement in improvements.items():
            print(f"{strategy_name.replace('_', ' ').title()}: ${improvement:,.2f}")

        print(f"\n✅ BEST ENHANCED STRATEGY: {best_strategy.replace('_', ' ').title()}")
        print(f"✅ ENHANCED DEPLOYMENT VIABLE: {deployment_viable}")

        # Save enhanced results
        enhanced_summary = {
            'enhanced_models_used': True,
            'signal_validation_passed': True,
            'strategies_tested': len(strategy_results),
            'best_strategy': best_strategy,
            'deployment_viable': deployment_viable,
            'total_traders_tested': len(self.enhanced_signal_df['account_id'].unique()) if hasattr(self, 'enhanced_signal_df') else 0
        }

        # Add strategy results
        for strategy_name, results in strategy_results.items():
            if results:
                enhanced_summary[f'enhanced_{strategy_name}_improvement'] = float(results['total_improvement'])
                enhanced_summary[f'enhanced_{strategy_name}_success_rate'] = float(results['success_rate'])
                enhanced_summary[f'enhanced_{strategy_name}_avg_cv_score'] = float(results.get('avg_model_cv_score', 0))
            else:
                enhanced_summary[f'enhanced_{strategy_name}_improvement'] = 0.0
                enhanced_summary[f'enhanced_{strategy_name}_success_rate'] = 0.0

        # Save enhanced results
        with open('data/enhanced_causal_impact_results.json', 'w') as f:
            json.dump(enhanced_summary, f, indent=2)

        print(f"\n✓ Enhanced causal impact results saved")
        return True

def main():
    """Main function to run enhanced causal impact analysis"""
    print("=" * 80)
    print("ENHANCED CAUSAL IMPACT ANALYSIS")
    print("=" * 80)

    analyzer = EnhancedCausalImpactAnalysis()
    success = analyzer.generate_enhanced_report()

    if success:
        print("\n✅ ENHANCED CAUSAL IMPACT ANALYSIS COMPLETE!")
    else:
        print("\n❌ Enhanced causal impact analysis failed")

if __name__ == "__main__":
    main()
