#!/usr/bin/env python
"""
Enhanced Daily Prediction Pipeline
- Downloads data until yesterday
- Retrains all models from scratch
- Generates predictions for today
- Sends email report
"""

import sys
import logging
from pathlib import Path
from datetime import date, timedelta
import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.database import Database
from src.data_downloader import DataDownloader
from src.feature_engineer import FeatureEngineer
from src.model_trainer import ModelTrainer
from src.predictor import RiskPredictor
from src.email_service import EmailService


def setup_logging():
    """Configure logging"""
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/daily_prediction.log')
        ]
    )


class DailyPredictionPipeline:
    """Complete daily prediction pipeline with retraining"""

    def __init__(self):
        self.db = Database()
        self.downloader = DataDownloader()
        self.feature_engineer = FeatureEngineer()
        self.model_trainer = ModelTrainer()
        self.email_service = EmailService()

        self.yesterday = (date.today() - timedelta(days=1)).strftime('%Y-%m-%d')
        self.today = date.today().strftime('%Y-%m-%d')

        self.logger = logging.getLogger(__name__)

    def step1_download_recent_data(self, days_back: int = 7) -> bool:
        """Download recent data to ensure we have latest information"""
        self.logger.info(f"Step 1: Downloading recent data ({days_back} days)")

        try:
            results = self.downloader.download_recent(days_back=days_back)
            success_count = sum(results.values())
            total_count = len(results)

            self.logger.info(f"Downloaded data for {success_count}/{total_count} traders")

            if success_count == 0:
                self.logger.error("Failed to download any recent data")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Data download failed: {str(e)}")
            return False

    def step2_retrain_all_models(self) -> dict:
        """Retrain all models from scratch using data up to yesterday"""
        self.logger.info(f"Step 2: Retraining all models with data until {self.yesterday}")

        # Get all active traders
        traders_df = self.db.get_all_traders()

        training_results = {}
        successful_models = 0

        for _, trader in traders_df.iterrows():
            account_id = str(trader['account_id'])
            trader_name = trader['trader_name']

            self.logger.info(f"Retraining model for {trader_name} ({account_id})")

            try:
                # Get all data up to yesterday
                totals_df, fills_df = self.db.get_trader_data(
                    account_id,
                    end_date=self.yesterday
                )

                if totals_df.empty:
                    self.logger.warning(f"No data for {trader_name}")
                    training_results[account_id] = {'success': False, 'reason': 'No data'}
                    continue

                # Create features
                features_df = self.feature_engineer.create_features(totals_df, fills_df)

                if len(features_df) < 30:
                    self.logger.warning(f"Insufficient data for {trader_name}: {len(features_df)} days")
                    training_results[account_id] = {'success': False, 'reason': 'Insufficient data'}
                    continue

                # Train model
                result = self.model_trainer.train_personal_model(
                    features_df,
                    account_id,
                    self.feature_engineer.get_feature_columns()
                )

                if result:
                    successful_models += 1
                    training_results[account_id] = {
                        'success': True,
                        'rmse': result['rmse'],
                        'mae': result['mae'],
                        'n_samples': result['n_samples'],
                        'trader_name': trader_name
                    }
                    self.logger.info(f"✓ Model trained for {trader_name}: RMSE={result['rmse']:.2f}")
                else:
                    training_results[account_id] = {'success': False, 'reason': 'Training failed'}
                    self.logger.warning(f"✗ Model training failed for {trader_name}")

            except Exception as e:
                self.logger.error(f"Error training model for {trader_name}: {str(e)}")
                training_results[account_id] = {'success': False, 'reason': str(e)}

        self.logger.info(f"Training complete: {successful_models}/{len(traders_df)} models trained")

        return training_results

    def step3_generate_predictions(self) -> tuple:
        """Generate predictions for today using retrained models"""
        self.logger.info(f"Step 3: Generating predictions for {self.today}")

        try:
            # Create enhanced predictor
            predictor = EnhancedRiskPredictor(self.db, self.feature_engineer, self.model_trainer)

            # Generate predictions
            predictions = predictor.predict_all_traders_for_date(self.today)
            summary = predictor.get_risk_summary(predictions)

            self.logger.info(f"Generated predictions for {len(predictions)} traders")
            self.logger.info(f"Risk distribution: High={summary['high_risk_count']}, "
                           f"Medium={summary['medium_risk_count']}, Low={summary['low_risk_count']}")

            return predictions, summary

        except Exception as e:
            self.logger.error(f"Prediction generation failed: {str(e)}")
            return [], {}

    def step4_send_email_report(self, predictions: list, summary: dict, training_results: dict) -> bool:
        """Send enhanced email report with training status"""
        self.logger.info("Step 4: Sending email report")

        try:
            # Add training status to summary
            successful_models = sum(1 for r in training_results.values() if r.get('success', False))
            total_traders = len(training_results)

            summary['models_retrained'] = successful_models
            summary['total_traders_processed'] = total_traders
            summary['retrain_success_rate'] = successful_models / total_traders if total_traders > 0 else 0
            summary['pipeline_date'] = self.today

            # Enhance predictions with training status
            enhanced_predictions = []
            for pred in predictions:
                account_id = pred['account_id']
                training_status = training_results.get(account_id, {})

                pred['model_freshness'] = 'Fresh' if training_status.get('success', False) else 'Stale'
                pred['training_rmse'] = training_status.get('rmse', 'N/A')
                enhanced_predictions.append(pred)

            # Send email
            success = self.email_service.send_enhanced_daily_report(
                enhanced_predictions, summary, training_results
            )

            if success:
                self.logger.info("✓ Email report sent successfully")
            else:
                self.logger.error("✗ Failed to send email report")

            return success

        except Exception as e:
            self.logger.error(f"Email sending failed: {str(e)}")
            return False

    def step5_save_results(self, predictions: list, training_results: dict) -> bool:
        """Save results for analysis and backup"""
        self.logger.info("Step 5: Saving results")

        try:
            # Create results directory
            results_dir = Path('data/daily_results')
            results_dir.mkdir(exist_ok=True)

            # Save predictions
            if predictions:
                predictions_df = pd.DataFrame(predictions)
                predictions_file = results_dir / f"predictions_{self.today}.csv"
                predictions_df.to_csv(predictions_file, index=False)
                self.logger.info(f"Predictions saved to {predictions_file}")

            # Save training results
            training_df = pd.DataFrame(training_results).T
            training_file = results_dir / f"training_{self.today}.csv"
            training_df.to_csv(training_file, index=False)
            self.logger.info(f"Training results saved to {training_file}")

            # Save to database
            if predictions:
                self.db.save_predictions(predictions)
                self.logger.info("Predictions saved to database")

            return True

        except Exception as e:
            self.logger.error(f"Failed to save results: {str(e)}")
            return False

    def run_full_pipeline(self) -> bool:
        """Run the complete daily prediction pipeline"""
        self.logger.info("="*60)
        self.logger.info(f"STARTING DAILY PREDICTION PIPELINE for {self.today}")
        self.logger.info("="*60)

        pipeline_success = True

        try:
            # Step 1: Download recent data
            if not self.step1_download_recent_data():
                self.logger.error("Pipeline failed at data download step")
                return False

            # Step 2: Retrain all models
            training_results = self.step2_retrain_all_models()
            successful_models = sum(1 for r in training_results.values() if r.get('success', False))

            if successful_models == 0:
                self.logger.error("Pipeline failed: No models were successfully trained")
                return False

            # Step 3: Generate predictions
            predictions, summary = self.step3_generate_predictions()

            if not predictions:
                self.logger.error("Pipeline failed: No predictions generated")
                return False

            # Step 4: Send email report
            email_success = self.step4_send_email_report(predictions, summary, training_results)
            if not email_success:
                self.logger.warning("Email sending failed, but continuing pipeline")
                pipeline_success = False

            # Step 5: Save results
            save_success = self.step5_save_results(predictions, training_results)
            if not save_success:
                self.logger.warning("Failed to save some results")
                pipeline_success = False

            # Final summary
            self.logger.info("="*60)
            self.logger.info("PIPELINE SUMMARY")
            self.logger.info("="*60)
            self.logger.info(f"Models retrained: {successful_models}/{len(training_results)}")
            self.logger.info(f"Predictions generated: {len(predictions)}")
            self.logger.info(f"High risk traders: {summary.get('high_risk_count', 0)}")
            self.logger.info(f"Email sent: {'Yes' if email_success else 'No'}")
            self.logger.info(f"Results saved: {'Yes' if save_success else 'No'}")
            self.logger.info(f"Overall success: {'Yes' if pipeline_success else 'Partial'}")

            return pipeline_success

        except Exception as e:
            self.logger.error(f"Pipeline failed with exception: {str(e)}")
            return False


class EnhancedRiskPredictor:
    """Enhanced predictor for daily pipeline"""

    def __init__(self, db, feature_engineer, model_trainer):
        self.db = db
        self.feature_engineer = feature_engineer
        self.model_trainer = model_trainer
        self.logger = logging.getLogger(__name__)

    def predict_all_traders_for_date(self, target_date: str) -> list:
        """Generate predictions for all traders for a specific date"""

        predictions = []
        models = self.model_trainer.get_all_models()

        self.logger.info(f"Loaded {len(models)} models for prediction")

        # Get all traders
        traders_df = self.db.get_all_traders()

        for _, trader in traders_df.iterrows():
            account_id = str(trader['account_id'])
            trader_name = trader['trader_name']

            try:
                prediction = self._predict_trader_for_date(account_id, trader_name, target_date, models)
                if prediction:
                    predictions.append(prediction)

            except Exception as e:
                self.logger.error(f"Prediction failed for {trader_name}: {str(e)}")
                # Add placeholder prediction
                predictions.append({
                    'account_id': account_id,
                    'trader_name': trader_name,
                    'predicted_pnl': 0,
                    'risk_level': 'Unknown',
                    'risk_score': 0.5,
                    'confidence': 'None',
                    'recent_pnl_5d': 0,
                    'recommendation': 'Model unavailable',
                    'error': str(e)
                })

        return predictions

    def _predict_trader_for_date(self, account_id: str, trader_name: str,
                                target_date: str, models: dict) -> dict:
        """Generate prediction for a single trader"""

        if account_id not in models:
            self.logger.warning(f"No model available for {trader_name}")
            return None

        model_data = models[account_id]
        model = model_data['model']
        feature_columns = model_data['feature_columns']
        threshold = model_data.get('threshold', 0)

        # Get recent data (last 60 days for feature calculation)
        end_date = (pd.to_datetime(target_date) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        start_date = (pd.to_datetime(end_date) - pd.Timedelta(days=60)).strftime('%Y-%m-%d')

        totals_df, fills_df = self.db.get_trader_data(account_id, start_date, end_date)

        if totals_df.empty:
            self.logger.warning(f"No recent data for {trader_name}")
            return None

        # Create features
        features_df = self.feature_engineer.create_features(totals_df, fills_df)

        if features_df.empty or len(features_df) < 5:
            self.logger.warning(f"Insufficient feature data for {trader_name}")
            return None

        # Get latest features (most recent day)
        latest_features = features_df[feature_columns].iloc[-1:].values

        # Make prediction
        predicted_pnl = model.predict(latest_features, num_iteration=model.best_iteration)[0]

        # Calculate additional metrics
        recent_pnl = totals_df['net_pnl'].tail(5).sum()
        recent_volatility = totals_df['net_pnl'].tail(20).std()

        # Determine risk level and recommendation
        if predicted_pnl < -1000:
            risk_level = "High"
            risk_score = 0.9
            recommendation = "Consider reducing position sizes by 50% or avoiding trading"
        elif predicted_pnl < 0:
            risk_level = "Medium"
            risk_score = 0.6
            recommendation = "Monitor closely, implement tighter stops"
        else:
            risk_level = "Low"
            risk_score = 0.3
            recommendation = "Normal trading conditions"

        # Adjust risk based on recent performance
        if recent_pnl < -2000:
            risk_score = min(1.0, risk_score + 0.2)
        elif recent_pnl > 2000:
            risk_score = max(0.1, risk_score - 0.1)

        return {
            'account_id': account_id,
            'trader_name': trader_name,
            'predicted_pnl': predicted_pnl,
            'risk_level': risk_level,
            'risk_score': risk_score,
            'confidence': 'High',
            'recent_pnl_5d': recent_pnl,
            'recent_volatility': recent_volatility,
            'recommendation': recommendation,
            'threshold': threshold,
            'prediction_date': target_date,
            'last_update': pd.Timestamp.now()
        }

    def get_risk_summary(self, predictions: list) -> dict:
        """Generate summary statistics from predictions"""

        if not predictions:
            return {}

        df = pd.DataFrame(predictions)

        summary = {
            'date': pd.Timestamp.now().strftime('%Y-%m-%d'),
            'total_traders': len(predictions),
            'high_risk_count': len(df[df['risk_level'] == 'High']),
            'medium_risk_count': len(df[df['risk_level'] == 'Medium']),
            'low_risk_count': len(df[df['risk_level'] == 'Low']),
            'unknown_risk_count': len(df[df['risk_level'] == 'Unknown']),
            'total_predicted_pnl': df['predicted_pnl'].sum(),
            'total_recent_pnl': df['recent_pnl_5d'].sum(),
            'models_available': len(df[df['confidence'] == 'High']),
            'avg_risk_score': df['risk_score'].mean(),
            'timestamp': pd.Timestamp.now()
        }

        # Top risk traders
        summary['top_risk_traders'] = df.nlargest(5, 'risk_score')[
            ['trader_name', 'risk_level', 'predicted_pnl', 'recommendation']
        ].to_dict('records')

        return summary


def main():
    """Main function"""
    setup_logging()

    pipeline = DailyPredictionPipeline()
    success = pipeline.run_full_pipeline()

    if success:
        print("✅ Daily prediction pipeline completed successfully")
        exit(0)
    else:
        print("❌ Daily prediction pipeline completed with errors")
        exit(1)


if __name__ == "__main__":
    main()
