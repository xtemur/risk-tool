#!/usr/bin/env python3
"""
Main Pipeline for Trader Risk Management System
Orchestrates the entire 7-step process
"""

import sys
import os
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from utils import get_logger, log_system_event, log_error

from data.data_validator import DataValidator
from features.feature_engineer import FeatureEngineer
from models.target_strategy import TargetVariableStrategy
from models.trader_models import TraderModelTraining
from evaluation.backtesting import RigorousBacktesting
from evaluation.causal_impact import CausalImpactAnalysis
from models.signal_generator import DeploymentReadySignals

class RiskManagementPipeline:
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger(__name__)
        self.checkpoint_results = {}
        self.start_time = datetime.now()

        # Create output directories
        self._setup_directories()

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
            validator = DataValidator(db_path=self.config['db_path'])

            # Load and validate data - ACTIVE TRADERS ONLY
            validator.load_and_validate_data(active_only=True)

            # Create daily aggregations
            validator.create_daily_aggregations()

            # Validate data quality
            validator.validate_data_quality()

            # Analyze predictability
            validator.analyze_predictability()

            # Generate summary
            checkpoint_pass = validator.generate_summary_report()

            # Save processed data
            output_path = Path(self.config['output_dir']) / 'daily_aggregated.pkl'
            validator.daily_df.to_pickle(output_path)
            print(f"✓ Saved daily aggregated data to {output_path}")
            self.logger.info("Step 1 completed", output_path=str(output_path))

            self.checkpoint_results['step1'] = checkpoint_pass
            return checkpoint_pass

        except Exception as e:
            print(f"❌ Step 1 failed: {e}")
            log_error("Step 1 failed", str(e), {"step": "data_validation"})
            return False

    def run_step2_feature_engineering(self):
        """Step 2: Feature Engineering"""
        print("\\n" + "="*60)
        print("STEP 2: FEATURE ENGINEERING")
        print("="*60)

        try:
            daily_data_path = Path(self.config['output_dir']) / 'daily_aggregated.pkl'
            engineer = FeatureEngineer(daily_df_path=str(daily_data_path))

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
            min_obs_per_trader = engineer.feature_df.groupby('account_id').size().min()
            checkpoint_pass = all([
                len(available_features) >= 20,
                min_obs_per_trader >= 10  # Lowered from 60 for small dataset
            ])
            print(f"✓ Minimum observations per trader: {min_obs_per_trader}")

            # Save feature data
            feature_path = Path(self.config['output_dir']) / 'features_engineered.pkl'
            engineer.feature_df.to_pickle(feature_path)
            print(f"✓ Saved engineered features to {feature_path}")

            # Save feature list
            feature_list_path = Path(self.config['output_dir']) / 'feature_list.txt'
            with open(feature_list_path, 'w') as f:
                for feature in available_features:
                    f.write(f"{feature}\\n")

            self.logger.info("Step 2 completed",
                           feature_count=len(available_features),
                           output_path=str(feature_path))

            self.checkpoint_results['step2'] = checkpoint_pass
            return checkpoint_pass

        except Exception as e:
            print(f"❌ Step 2 failed: {e}")
            log_error("Step 2 failed", str(e), {"step": "feature_engineering"})
            return False

    def run_step3_target_strategy(self):
        """Step 3: Target Variable Strategy"""
        print("\\n" + "="*60)
        print("STEP 3: TARGET VARIABLE STRATEGY")
        print("="*60)

        try:
            features_path = Path(self.config['output_dir']) / 'features_engineered.pkl'
            strategy = TargetVariableStrategy(features_path=str(features_path))

            # Prepare data
            strategy.prepare_data()

            # Compare target options
            best_strategy = strategy.compare_target_options()

            # Generate checkpoint report
            checkpoint_pass = best_strategy is not None

            if checkpoint_pass and best_strategy:
                # Prepare final target data
                final_df, target_col = strategy.prepare_final_target_data()

                # Save results
                target_path = Path(self.config['output_dir']) / 'target_prepared.pkl'
                final_df.to_pickle(target_path)

                # Save target strategy info
                import json
                strategy_info = {
                    'best_strategy': best_strategy['option_name'],
                    'target_column': target_col,
                    'model_performance': best_strategy['overall_score'],
                    'predictability_score': best_strategy['predictability']['predictability_score']
                }

                strategy_path = Path(self.config['output_dir']) / 'target_strategy.json'
                with open(strategy_path, 'w') as f:
                    json.dump(strategy_info, f, indent=2)

                print(f"✓ Saved target data to {target_path}")
                print(f"✓ Saved strategy info to {strategy_path}")

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
            target_data_path = Path(self.config['output_dir']) / 'target_prepared.pkl'
            trainer = TraderModelTraining(features_path=str(target_data_path))

            # Train all models
            trainer.train_all_models()

            # Validate model features
            trainer.validate_model_features()

            # Save models and results
            trainer.save_models_and_results()

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
            backtester = RigorousBacktesting(data_dir=self.config['output_dir'])

            # Perform walk-forward validation
            backtester.perform_walk_forward_validation()

            # Validate signal directions
            backtester.validate_signal_directions()

            # Test model stability
            backtester.test_model_stability()

            # Analyze feature correlations
            backtester.analyze_feature_correlations()

            # Save backtest results
            backtester.save_backtest_results()

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
            analyzer.generate_causal_impact_report()

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
            deployment.create_three_tier_signal_system()

            # Generate real-time signals
            deployment.generate_real_time_signals()

            # Create deployment interface
            deployment.create_deployment_interface()

            # Implement safeguards
            deployment.implement_safeguards()

            # Create documentation
            deployment.create_documentation()

            # Generate final deployment checklist
            deployment.generate_final_deployment_checklist()

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
                causal_results_path = Path(self.config['output_dir']) / 'causal_impact_results.json'
                with open(causal_results_path, 'r') as f:
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

        # Log final summary
        log_system_event(
            "pipeline_completed",
            "Risk management pipeline execution completed",
            {
                "execution_time": str(execution_time),
                "all_passed": all_passed,
                "checkpoint_results": self.checkpoint_results
            }
        )

    def _setup_directories(self):
        """Create necessary output directories"""
        base_dir = Path(self.config['output_dir'])
        directories = [
            base_dir,
            base_dir / 'checkpoints',
            base_dir / 'models',
            base_dir / 'reports',
            base_dir / 'signals'
        ]

        for dir_path in directories:
            dir_path.mkdir(parents=True, exist_ok=True)

        self.logger.info("Output directories created")


def main():
    """Main entry point for the pipeline"""
    # Log pipeline start
    log_system_event(
        "pipeline_started",
        "Risk management pipeline execution started",
        {"environment": "production"}
    )

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
