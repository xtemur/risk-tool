#!/usr/bin/env python3
"""
Main Pipeline for Trader Risk Management System
Orchestrates the entire 7-step process
"""

import sys
import os
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.data_validator import DataValidator
from features.feature_engineer import FeatureEngineer
from models.target_strategy import TargetVariableStrategy
from models.trader_models import TraderModelTraining
from evaluation.backtesting import RigorousBacktesting
from evaluation.causal_impact import CausalImpactAnalysis
from models.signal_generator import DeploymentReadySignals

class RiskManagementPipeline:
    def __init__(self):
        self.checkpoint_results = {}
        self.start_time = datetime.now()

    def run_complete_pipeline(self):
        """Run the complete 7-step risk management pipeline"""
        print("="*80)
        print("TRADER RISK MANAGEMENT SYSTEM - COMPLETE PIPELINE")
        print("Following CLAUDE.md methodology with separate models per trader")
        print("="*80)

        try:
            # Step 1: Data Validation & Exploration
            step1_pass = self.run_step1_data_validation()
            if not step1_pass:
                print("❌ Pipeline failed at Step 1")
                return False

            # Step 2: Feature Engineering
            step2_pass = self.run_step2_feature_engineering()
            if not step2_pass:
                print("❌ Pipeline failed at Step 2")
                return False

            # Step 3: Target Variable Strategy
            step3_pass = self.run_step3_target_strategy()
            if not step3_pass:
                print("❌ Pipeline failed at Step 3")
                return False

            # Step 4: Model Training
            step4_pass = self.run_step4_model_training()
            if not step4_pass:
                print("❌ Pipeline failed at Step 4")
                return False

            # Step 5: Rigorous Backtesting
            step5_pass = self.run_step5_backtesting()
            if not step5_pass:
                print("❌ Pipeline failed at Step 5")
                return False

            # Step 6: Causal Impact Analysis
            step6_pass = self.run_step6_causal_impact()
            if not step6_pass:
                print("❌ Pipeline failed at Step 6")
                return False

            # Step 7: Signal Generation & Deployment
            step7_pass = self.run_step7_deployment_ready()
            if not step7_pass:
                print("❌ Pipeline failed at Step 7")
                return False

            # Final summary
            self.generate_final_summary()
            return True

        except Exception as e:
            print(f"❌ Pipeline failed with error: {e}")
            return False

    def run_step1_data_validation(self):
        """Step 1: Data Validation & Exploration"""
        print("\\n" + "="*60)
        print("STEP 1: DATA VALIDATION & EXPLORATION")
        print("="*60)

        try:
            validator = DataValidator()

            # Load and validate data
            validator.load_and_validate_data()

            # Create daily aggregations
            validator.create_daily_aggregations()

            # Validate data quality
            validator.validate_data_quality()

            # Analyze predictability
            validator.analyze_predictability()

            # Generate summary
            checkpoint_pass = validator.generate_summary_report()

            # Save processed data
            validator.daily_df.to_pickle('data/daily_aggregated.pkl')
            print(f"✓ Saved daily aggregated data to data/daily_aggregated.pkl")

            self.checkpoint_results['step1'] = checkpoint_pass
            return checkpoint_pass

        except Exception as e:
            print(f"❌ Step 1 failed: {e}")
            return False

    def run_step2_feature_engineering(self):
        """Step 2: Feature Engineering"""
        print("\\n" + "="*60)
        print("STEP 2: FEATURE ENGINEERING")
        print("="*60)

        try:
            engineer = FeatureEngineer()

            # Create features in sequence
            engineer.create_basic_features()
            engineer.create_rolling_features_ewma()
            engineer.create_lagged_features()
            engineer.create_advanced_features()

            # Validate features
            engineer.validate_features()

            # Finalize feature set
            available_features = engineer.finalize_features()

            # Generate checkpoint report
            checkpoint_pass = all([
                len(available_features) >= 20,
                engineer.feature_df.groupby('account_id').size().min() >= 60
            ])

            # Save feature data
            engineer.feature_df.to_pickle('data/features_engineered.pkl')
            print(f"✓ Saved engineered features to data/features_engineered.pkl")

            # Save feature list
            with open('data/feature_list.txt', 'w') as f:
                for feature in available_features:
                    f.write(f"{feature}\\n")

            self.checkpoint_results['step2'] = checkpoint_pass
            return checkpoint_pass

        except Exception as e:
            print(f"❌ Step 2 failed: {e}")
            return False

    def run_step3_target_strategy(self):
        """Step 3: Target Variable Strategy"""
        print("\\n" + "="*60)
        print("STEP 3: TARGET VARIABLE STRATEGY")
        print("="*60)

        try:
            strategy = TargetVariableStrategy()

            # Prepare data
            feature_cols = strategy.prepare_data()

            # Compare target options
            best_strategy = strategy.compare_target_options()

            # Generate checkpoint report
            checkpoint_pass = best_strategy is not None

            if checkpoint_pass and best_strategy:
                # Prepare final target data
                final_df, target_col = strategy.prepare_final_target_data()

                # Save results
                final_df.to_pickle('data/target_prepared.pkl')

                # Save target strategy info
                import json
                strategy_info = {
                    'best_strategy': best_strategy['option_name'],
                    'target_column': target_col,
                    'model_performance': best_strategy['overall_score'],
                    'predictability_score': best_strategy['predictability']['predictability_score']
                }

                with open('data/target_strategy.json', 'w') as f:
                    json.dump(strategy_info, f, indent=2)

                print(f"✓ Saved target data to data/target_prepared.pkl")
                print(f"✓ Saved strategy info to data/target_strategy.json")

            self.checkpoint_results['step3'] = checkpoint_pass
            return checkpoint_pass

        except Exception as e:
            print(f"❌ Step 3 failed: {e}")
            return False

    def run_step4_model_training(self):
        """Step 4: Model Selection and Training"""
        print("\\n" + "="*60)
        print("STEP 4: MODEL SELECTION AND TRAINING")
        print("="*60)

        try:
            trainer = TraderModelTraining()

            # Train all models
            training_success = trainer.train_all_models()

            # Validate model features
            feature_validation = trainer.validate_model_features()

            # Save models and results
            save_success = trainer.save_models_and_results()

            # Generate checkpoint report
            checkpoint_pass = trainer.generate_checkpoint_report()

            self.checkpoint_results['step4'] = checkpoint_pass
            return checkpoint_pass

        except Exception as e:
            print(f"❌ Step 4 failed: {e}")
            return False

    def run_step5_backtesting(self):
        """Step 5: Rigorous Backtesting & Validation"""
        print("\\n" + "="*60)
        print("STEP 5: RIGOROUS BACKTESTING & VALIDATION")
        print("="*60)

        try:
            backtester = RigorousBacktesting()

            # Perform walk-forward validation
            validation_success = backtester.perform_walk_forward_validation()

            # Validate signal directions
            signal_validation = backtester.validate_signal_directions()

            # Test model stability
            stability_check = backtester.test_model_stability()

            # Analyze feature correlations
            feature_check = backtester.analyze_feature_correlations()

            # Save backtest results
            save_success = backtester.save_backtest_results()

            # Generate checkpoint report
            checkpoint_pass = backtester.generate_checkpoint_report()

            self.checkpoint_results['step5'] = checkpoint_pass
            return checkpoint_pass

        except Exception as e:
            print(f"❌ Step 5 failed: {e}")
            return False

    def run_step6_causal_impact(self):
        """Step 6: Causal Impact Analysis"""
        print("\\n" + "="*60)
        print("STEP 6: CAUSAL IMPACT ANALYSIS")
        print("="*60)

        try:
            analyzer = CausalImpactAnalysis()

            # Generate comprehensive causal impact report
            deployment_viable = analyzer.generate_causal_impact_report()

            # Generate checkpoint report
            checkpoint_pass = analyzer.generate_checkpoint_report()

            self.checkpoint_results['step6'] = checkpoint_pass
            return checkpoint_pass

        except Exception as e:
            print(f"❌ Step 6 failed: {e}")
            return False

    def run_step7_deployment_ready(self):
        """Step 7: Signal Generation & Deployment Readiness"""
        print("\\n" + "="*60)
        print("STEP 7: SIGNAL GENERATION & DEPLOYMENT READINESS")
        print("="*60)

        try:
            deployment = DeploymentReadySignals()

            # Create 3-tier signal system
            signal_system = deployment.create_three_tier_signal_system()

            # Generate real-time signals
            signals = deployment.generate_real_time_signals()

            # Create deployment interface
            config, api = deployment.create_deployment_interface()

            # Implement safeguards
            safeguards = deployment.implement_safeguards()

            # Create documentation
            docs = deployment.create_documentation()

            # Generate final deployment checklist
            deployment_status = deployment.generate_final_deployment_checklist()

            # Generate checkpoint report
            checkpoint_pass = deployment.generate_checkpoint_report()

            self.checkpoint_results['step7'] = checkpoint_pass
            return checkpoint_pass

        except Exception as e:
            print(f"❌ Step 7 failed: {e}")
            return False

    def generate_final_summary(self):
        """Generate final pipeline summary"""
        print("\\n" + "="*80)
        print("PIPELINE EXECUTION SUMMARY")
        print("="*80)

        execution_time = datetime.now() - self.start_time

        print(f"\\n📊 EXECUTION RESULTS:")
        print(f"   Total execution time: {execution_time}")
        print(f"   Pipeline start: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Pipeline end: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        print(f"\\n✅ CHECKPOINT RESULTS:")
        all_passed = True
        for step, passed in self.checkpoint_results.items():
            status = "PASSED" if passed else "FAILED"
            symbol = "✅" if passed else "❌"
            print(f"   {symbol} {step.upper()}: {status}")
            if not passed:
                all_passed = False

        if all_passed:
            print(f"\\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            print(f"✅ All 7 steps passed validation")
            print(f"✅ System ready for deployment")

            # Load final results
            try:
                import json
                with open('data/causal_impact_results.json', 'r') as f:
                    causal_results = json.load(f)

                print(f"\\n📈 CAUSAL IMPACT RESULTS:")
                print(f"   Best strategy: {causal_results.get('best_strategy', 'N/A')}")
                print(f"   Deployment viable: {causal_results.get('deployment_viable', False)}")
                print(f"   Traders tested: {causal_results.get('total_traders_tested', 0)}")

                best_improvement = 0
                for key, value in causal_results.items():
                    if key.endswith('_improvement'):
                        if value > best_improvement:
                            best_improvement = value

                if best_improvement > 0:
                    print(f"   Best improvement: ${best_improvement:,.2f}")

            except Exception as e:
                print(f"   (Could not load final results: {e})")

        else:
            print(f"\\n❌ PIPELINE FAILED")
            print(f"❌ Some steps did not pass validation")
            print(f"❌ Review failed steps before deployment")

        print("\\n" + "="*80)

def main():
    """Main entry point for the pipeline"""
    pipeline = RiskManagementPipeline()
    success = pipeline.run_complete_pipeline()

    if success:
        print("\\n🎯 Pipeline completed successfully!")
        return 0
    else:
        print("\\n💥 Pipeline failed!")
        return 1

if __name__ == "__main__":
    exit(main())
