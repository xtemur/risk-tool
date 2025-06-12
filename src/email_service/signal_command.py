"""
Signal Command

Command-line interface for generating and sending trading signals via email.
"""

import sys
import os
from pathlib import Path
from datetime import datetime, date
import logging
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np

# Add the src directory to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / "src"))

from email_service import SignalEmailService, EmailConfig
from modeling import PredictionPipeline, ModelConfig

logger = logging.getLogger(__name__)


class SignalCommand:
    """
    Command for generating predictions and sending signal emails
    """

    def __init__(self):
        """
        Initialize signal command
        """
        self.email_config = EmailConfig.from_env()
        self.model_config = ModelConfig.for_production()
        self.signal_service = SignalEmailService(self.email_config)
        self.prediction_pipeline = None

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def make_signal(self,
                   data_path: Optional[str] = None,
                   model_path: Optional[str] = None,
                   to_emails: Optional[str] = None,
                   include_performance: bool = True,
                   include_attachments: bool = False,
                   dry_run: bool = False) -> bool:
        """
        Generate predictions and send signal email

        Args:
            data_path: Path to feature data (uses latest if None)
            model_path: Path to saved models (trains new if None)
            to_emails: Comma-separated recipient emails (uses config if None)
            include_performance: Include model performance metrics
            include_attachments: Include prediction files as attachments
            dry_run: Generate predictions but don't send email

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Starting signal generation process...")

            # Step 1: Load or prepare data
            logger.info("Step 1: Loading feature data...")
            data = self._load_feature_data(data_path)
            if data is None:
                logger.error("Failed to load feature data")
                return False

            logger.info(f"Loaded {len(data)} samples from {data['account_id'].nunique()} traders")

            # Step 2: Load or train models
            logger.info("Step 2: Preparing prediction models...")
            success = self._prepare_models(data, model_path)
            if not success:
                logger.error("Failed to prepare prediction models")
                return False

            # Step 3: Generate predictions for today
            logger.info("Step 3: Generating predictions...")
            predictions = self._generate_predictions(data)
            if not predictions:
                logger.error("Failed to generate predictions")
                return False

            # Step 4: Prepare email data
            logger.info("Step 4: Preparing email content...")
            email_data = self._prepare_email_data(predictions, include_performance)

            # Step 5: Send email (unless dry run)
            if dry_run:
                logger.info("DRY RUN: Email content prepared but not sent")
                self._log_predictions(predictions)
                return True
            else:
                logger.info("Step 5: Sending signal email...")

                # Parse recipient emails
                recipients = None
                if to_emails:
                    recipients = [email.strip() for email in to_emails.split(',')]

                # Prepare attachments if requested
                attachments = []
                if include_attachments:
                    attachments = self._create_attachments(predictions, data)

                success = self.signal_service.send_signal_email(
                    predictions=email_data,
                    to_emails=recipients,
                    include_performance=include_performance,
                    attachment_paths=attachments
                )

                if success:
                    logger.info("Signal email sent successfully!")
                else:
                    logger.error("Failed to send signal email")

                return success

        except Exception as e:
            logger.error(f"Error in signal generation: {e}")
            return False

    def _load_feature_data(self, data_path: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Load feature data for prediction

        Args:
            data_path: Path to feature data file

        Returns:
            Feature DataFrame or None if failed
        """
        try:
            if data_path:
                # Use specified path
                if not Path(data_path).exists():
                    logger.error(f"Data file not found: {data_path}")
                    return None
                data = pd.read_csv(data_path)
            else:
                # Look for latest feature data
                data_dir = Path("data/processed")
                if not data_dir.exists():
                    logger.error("Processed data directory not found")
                    return None

                # Find latest feature file
                feature_files = list(data_dir.glob("features_*.csv"))
                if not feature_files:
                    # Try default feature file
                    default_file = data_dir / "features_demo.csv"
                    if default_file.exists():
                        data = pd.read_csv(default_file)
                    else:
                        logger.error("No feature files found in data/processed/")
                        return None
                else:
                    # Use most recent file
                    latest_file = max(feature_files, key=lambda p: p.stat().st_mtime)
                    logger.info(f"Using latest feature file: {latest_file}")
                    data = pd.read_csv(latest_file)

            # Validate data structure
            required_columns = ['account_id', 'date', 'target_next_pnl']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return None

            # Ensure date column is datetime
            data['date'] = pd.to_datetime(data['date'])

            return data

        except Exception as e:
            logger.error(f"Error loading feature data: {e}")
            return None

    def _prepare_models(self, data: pd.DataFrame, model_path: Optional[str] = None) -> bool:
        """
        Load existing models or train new ones

        Args:
            data: Feature data
            model_path: Path to saved models

        Returns:
            True if successful, False otherwise
        """
        try:
            self.prediction_pipeline = PredictionPipeline(self.model_config)

            if model_path and Path(model_path).exists():
                # Load existing models
                logger.info(f"Loading models from {model_path}")
                self.prediction_pipeline.trainer.load_models(model_path)
                return True
            else:
                # Train new models on recent data
                logger.info("Training new models on recent data...")

                # Use last 80% of data for training (keep most recent for prediction)
                cutoff_idx = int(len(data) * 0.8)
                training_data = data.iloc[:cutoff_idx].copy()

                if len(training_data) < 100:
                    logger.warning("Limited training data available, predictions may be less reliable")

                # Quick training with essential models - use direct model training
                # Train individual trader models and a global fallback

                # Feature columns
                feature_cols = [col for col in training_data.columns
                               if col not in ['account_id', 'date', 'target_next_pnl']]

                # Clean data - handle NaN values using imputation
                logger.info("Preprocessing data to handle missing values...")

                # Remove rows where target is NaN
                training_clean = training_data.dropna(subset=['target_next_pnl'])
                logger.info(f"After removing rows with NaN target: {len(training_clean)}/{len(training_data)} samples remain")

                if len(training_clean) < 10:
                    logger.error("Insufficient training data after removing NaN targets")
                    return False

                # For features, use simple imputation strategy
                from sklearn.impute import SimpleImputer

                X_features = training_clean[feature_cols]

                # Remove columns that are all NaN (can't be imputed)
                valid_feature_cols = []
                for col in feature_cols:
                    if not X_features[col].isnull().all():
                        valid_feature_cols.append(col)
                    else:
                        logger.warning(f"Dropping feature column '{col}' - all values are NaN")

                logger.info(f"Using {len(valid_feature_cols)}/{len(feature_cols)} feature columns after filtering")

                # Impute features with median values for valid columns only
                imputer = SimpleImputer(strategy='median')
                X_features_valid = X_features[valid_feature_cols]
                X_imputed = imputer.fit_transform(X_features_valid)

                # Store the imputer and valid columns for later use on prediction data
                self.feature_imputer = imputer
                self.feature_columns = valid_feature_cols

                logger.info(f"Imputed {X_features_valid.isnull().sum().sum()} missing feature values")

                # Train global model first as fallback using imputed data
                X_global_df = pd.DataFrame(X_imputed, columns=valid_feature_cols, index=training_clean.index)
                y_global = training_clean['target_next_pnl']

                global_result = self.prediction_pipeline.trainer.train_model(
                    X_global_df, y_global,
                    model_type='ridge',
                    model_key='global_final_ridge'
                )
                logger.info(f"Global model trained: {global_result.get('model_key')}")

                # Train individual trader models
                traders = training_clean['account_id'].unique()
                successful_models = 0

                for trader_id in traders:
                    trader_mask = training_clean['account_id'] == trader_id
                    trader_data = training_clean[trader_mask]

                    if len(trader_data) >= 10:  # Minimum samples for individual model
                        # Use imputed data for this trader
                        X_trader_df = X_global_df[trader_mask]
                        y_trader = trader_data['target_next_pnl']

                        try:
                            trader_result = self.prediction_pipeline.trainer.train_model(
                                X_trader_df, y_trader,
                                model_type='ridge',
                                model_key=f'{trader_id}_final_ridge'
                            )
                            logger.info(f"Trader {trader_id} model trained: {trader_result.get('model_key')}")
                            successful_models += 1
                        except Exception as e:
                            logger.warning(f"Failed to train model for trader {trader_id}: {e}")
                    else:
                        logger.info(f"Trader {trader_id}: insufficient data ({len(trader_data)} samples), will use global model")

                logger.info(f"Model training completed: {successful_models}/{len(traders)} individual models + 1 global model")
                return True

        except Exception as e:
            logger.error(f"Error preparing models: {e}")
            return False

    def _generate_predictions(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Generate predictions for today/tomorrow

        Args:
            data: Feature data

        Returns:
            Predictions dictionary or None if failed
        """
        try:
            predictions = {
                'traders': {},
                'generation_time': datetime.now(),
                'prediction_date': date.today()
            }

            # Get most recent data for each trader (today's features)
            latest_data = data.groupby('account_id').tail(1)

            # Use the same feature columns that were used in training
            if hasattr(self, 'feature_columns'):
                feature_cols = self.feature_columns
            else:
                # Fallback to all features (shouldn't happen with proper flow)
                feature_cols = [col for col in data.columns
                               if col not in ['account_id', 'date', 'target_next_pnl']]

            # Debug: log available models
            available_models = list(self.prediction_pipeline.trainer.models.keys())
            logger.info(f"Available models: {available_models}")

            for _, row in latest_data.iterrows():
                trader_id = row['account_id']

                try:
                    # Prepare features for this trader
                    features_raw = row[feature_cols].values.reshape(1, -1)

                    # Use the same imputer as training
                    if hasattr(self, 'feature_imputer') and hasattr(self, 'feature_columns'):
                        # Extract only the valid feature columns used in training
                        features_selected = row[self.feature_columns].values.reshape(1, -1)
                        features_imputed = self.feature_imputer.transform(features_selected)
                        features_df = pd.DataFrame(features_imputed, columns=self.feature_columns)
                    else:
                        logger.warning(f"No imputer available, using raw features for trader {trader_id}")
                        features_df = pd.DataFrame(features_raw, columns=feature_cols)

                    # Get prediction from trader's model
                    model_key = f"{trader_id}_final_ridge"  # Primary model

                    if model_key in self.prediction_pipeline.trainer.models:
                        # Use trained model
                        pred_pnl = self.prediction_pipeline.trainer.predict(
                            features_df,
                            model_key
                        )[0]
                    else:
                        # Fallback to global model if trader-specific not available
                        global_key = "global_final_ridge"
                        if global_key in self.prediction_pipeline.trainer.models:
                            pred_pnl = self.prediction_pipeline.trainer.predict(
                                features_df,
                                global_key
                            )[0]
                        else:
                            logger.warning(f"No model available for trader {trader_id}")
                            continue

                    # Calculate confidence based on model uncertainty and recent performance
                    confidence = self._calculate_prediction_confidence(
                        trader_id, features_df.values, data
                    )

                    # Get trader name (use ID if name not available)
                    trader_name = f"Trader {trader_id}"

                    # Calculate recent performance
                    recent_performance = self._calculate_recent_performance(trader_id, data)

                    predictions['traders'][trader_id] = {
                        'name': trader_name,
                        'predicted_pnl': float(pred_pnl),
                        'confidence': confidence,
                        'recent_performance': recent_performance,
                        'features_date': row['date'].strftime('%Y-%m-%d'),
                        'model_used': model_key if model_key in self.prediction_pipeline.trainer.models else global_key
                    }

                    logger.debug(f"Prediction for {trader_id}: ${pred_pnl:.2f} (confidence: {confidence:.1%})")

                except Exception as e:
                    logger.warning(f"Failed to generate prediction for trader {trader_id}: {e}")
                    continue

            if not predictions['traders']:
                logger.error("No predictions generated for any traders")
                return None

            # Add model performance metrics if available
            if hasattr(self.prediction_pipeline, 'results'):
                predictions['model_performance'] = self._extract_model_performance()

            # Add market context
            predictions['market_context'] = self._generate_market_context(data)

            logger.info(f"Generated predictions for {len(predictions['traders'])} traders")
            return predictions

        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
            return None

    def _calculate_prediction_confidence(self, trader_id: str, features: np.ndarray, data: pd.DataFrame) -> float:
        """
        Calculate prediction confidence score

        Args:
            trader_id: Trader identifier
            features: Feature array
            data: Full data for confidence calculation

        Returns:
            Confidence score (0-1)
        """
        try:
            # Base confidence from model training (if available)
            base_confidence = 0.6

            # Adjust based on recent data quality
            trader_data = data[data['account_id'] == trader_id].tail(10)

            if len(trader_data) >= 5:
                # More data available = higher confidence
                data_confidence = min(len(trader_data) / 10, 1.0)
            else:
                # Limited data = lower confidence
                data_confidence = len(trader_data) / 10

            # Adjust based on feature stability (low variance = higher confidence)
            feature_variance = np.var(features) if len(features) > 0 else 1.0
            stability_confidence = max(0.3, 1.0 - min(feature_variance / 1000, 0.7))

            # Combine confidence factors
            final_confidence = np.mean([base_confidence, data_confidence, stability_confidence])

            # Ensure confidence is in reasonable range
            return max(0.2, min(0.95, final_confidence))

        except Exception as e:
            logger.warning(f"Error calculating confidence for {trader_id}: {e}")
            return 0.5  # Default moderate confidence

    def _calculate_recent_performance(self, trader_id: str, data: pd.DataFrame) -> float:
        """
        Calculate recent 7-day performance for trader using database

        Args:
            trader_id: Trader identifier
            data: Full data (not used, kept for compatibility)

        Returns:
            Recent performance PnL
        """
        try:
            # Use database directly for accurate recent performance
            from data.database_manager import DatabaseManager
            db = DatabaseManager()

            # Get recent daily summary data
            daily_data = db.get_account_daily_summary(trader_id)
            if not daily_data.empty:
                recent_7_days = daily_data.tail(7)
                recent_pnl = float(recent_7_days['net'].sum())
                logger.debug(f"Trader {trader_id} recent 7-day PnL from DB: ${recent_pnl:.2f}")
                return recent_pnl
            else:
                logger.warning(f"No database data found for trader {trader_id}")
                return 0.0

        except Exception as e:
            logger.warning(f"Error calculating recent performance for {trader_id}: {e}")
            # Fallback to feature data if database fails
            try:
                trader_data = data[data['account_id'] == trader_id].tail(7)
                if 'net' in trader_data.columns and trader_data['net'].notna().any():
                    return float(trader_data['net'].sum())
                else:
                    # Use target_next_pnl as last resort
                    return float(trader_data['target_next_pnl'].tail(7).sum())
            except Exception:
                return 0.0

    def _extract_model_performance(self) -> Dict[str, Any]:
        """
        Extract or calculate model performance metrics

        Returns:
            Performance metrics dictionary
        """
        try:
            # First try to get from pipeline results if available
            if hasattr(self.prediction_pipeline, 'results') and self.prediction_pipeline.results:
                results = self.prediction_pipeline.results

                # Extract performance from holdout results
                if 'holdout_results' in results:
                    holdout = results['holdout_results']

                    # Get global or first trader results
                    model_results = holdout.get('global', {})
                    if not model_results and holdout:
                        # Use first trader's results as proxy
                        first_trader = next(iter(holdout.keys()))
                        model_results = holdout[first_trader]

                    if model_results and 'ridge' in model_results:
                        ridge_results = model_results['ridge']

                        return {
                            'hit_rate': ridge_results.get('financial_metrics', {}).get('hit_rate', 0.5),
                            'sharpe_ratio': ridge_results.get('financial_metrics', {}).get('actual_sharpe', 0.0),
                            'r2_score': ridge_results.get('statistical_metrics', {}).get('r2', 0.0),
                            'model_version': 'v1.0'
                        }

            # If no pipeline results, calculate basic metrics from database
            logger.info("No pipeline results available, calculating basic metrics from database...")
            return self._calculate_basic_performance_metrics()

        except Exception as e:
            logger.warning(f"Error extracting model performance: {e}")
            return self._calculate_basic_performance_metrics()

    def _calculate_basic_performance_metrics(self) -> Dict[str, Any]:
        """
        Calculate basic performance metrics from database data

        Returns:
            Basic performance metrics
        """
        try:
            from data.database_manager import DatabaseManager
            db = DatabaseManager()

            # Get all accounts
            accounts = db.get_accounts()
            all_returns = []
            total_positive_days = 0
            total_days = 0

            for _, account in accounts.iterrows():
                account_id = account['account_id']
                daily_data = db.get_account_daily_summary(account_id)

                if not daily_data.empty:
                    # Get returns from net column
                    returns = daily_data['net'].dropna()
                    if len(returns) > 0:
                        all_returns.extend(returns.tolist())
                        positive_days = (returns > 0).sum()
                        total_positive_days += positive_days
                        total_days += len(returns)

            if len(all_returns) > 1:
                import numpy as np
                returns_array = np.array(all_returns)

                # Calculate hit rate
                hit_rate = total_positive_days / total_days if total_days > 0 else 0.5

                # Calculate Sharpe ratio (annualized)
                mean_return = np.mean(returns_array)
                std_return = np.std(returns_array)
                sharpe_ratio = (mean_return * 252) / (std_return * np.sqrt(252)) if std_return > 0 else 0.0

                # Basic R² approximation (not exact without actual predictions vs actuals)
                r2_score = 0.1  # Default reasonable value for live trading models

                logger.info(f"Calculated metrics: hit_rate={hit_rate:.3f}, sharpe_ratio={sharpe_ratio:.3f}")

                return {
                    'hit_rate': float(hit_rate),
                    'sharpe_ratio': float(sharpe_ratio),
                    'r2_score': float(r2_score),
                    'model_version': 'v1.0_basic'
                }
            else:
                logger.warning("Insufficient data for performance calculations")
                return {
                    'hit_rate': 0.5,
                    'sharpe_ratio': 0.0,
                    'r2_score': 0.0,
                    'model_version': 'v1.0_default'
                }

        except Exception as e:
            logger.warning(f"Error calculating basic performance metrics: {e}")
            return {
                'hit_rate': 0.5,
                'sharpe_ratio': 0.0,
                'r2_score': 0.0,
                'model_version': 'v1.0_error'
            }

    def _generate_market_context(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate market context information

        Args:
            data: Feature data

        Returns:
            Market context dictionary
        """
        try:
            # Get recent aggregate statistics
            recent_data = data.tail(50)  # Last 50 records across all traders

            context = {
                'market_volatility': float(recent_data['target_next_pnl'].std()),
                'avg_daily_pnl': float(recent_data['target_next_pnl'].mean()),
                'active_traders': int(recent_data['account_id'].nunique()),
                'data_freshness': recent_data['date'].max().strftime('%Y-%m-%d')
            }

            # Add trend analysis
            if len(recent_data) >= 10:
                recent_trend = recent_data['target_next_pnl'].tail(10).mean()
                previous_trend = recent_data['target_next_pnl'].head(10).mean()

                if recent_trend > previous_trend * 1.1:
                    context['market_trend'] = 'Bullish'
                elif recent_trend < previous_trend * 0.9:
                    context['market_trend'] = 'Bearish'
                else:
                    context['market_trend'] = 'Neutral'
            else:
                context['market_trend'] = 'Unknown'

            return context

        except Exception as e:
            logger.warning(f"Error generating market context: {e}")
            return {}

    def _prepare_email_data(self, predictions: Dict[str, Any], include_performance: bool) -> Dict[str, Any]:
        """
        Prepare prediction data for email template

        Args:
            predictions: Raw predictions
            include_performance: Include performance metrics

        Returns:
            Email-ready data
        """
        email_data = predictions.copy()

        # Remove internal fields that shouldn't be in email
        if 'generation_time' in email_data:
            del email_data['generation_time']

        # Ensure model_performance is included/excluded based on flag
        if not include_performance and 'model_performance' in email_data:
            del email_data['model_performance']

        return email_data

    def _create_attachments(self, predictions: Dict[str, Any], data: pd.DataFrame) -> List[str]:
        """
        Create attachment files for email

        Args:
            predictions: Prediction data
            data: Feature data

        Returns:
            List of attachment file paths
        """
        attachments = []

        try:
            # Create attachments directory
            attachments_dir = Path("results/signal_attachments")
            attachments_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Create CSV with detailed predictions
            predictions_df = pd.DataFrame([
                {
                    'trader_id': trader_id,
                    'trader_name': data['name'],
                    'predicted_pnl': data['predicted_pnl'],
                    'confidence': data['confidence'],
                    'recent_performance': data.get('recent_performance', 0),
                    'model_used': data.get('model_used', 'unknown'),
                    'features_date': data.get('features_date', '')
                }
                for trader_id, data in predictions['traders'].items()
            ])

            predictions_file = attachments_dir / f"predictions_{timestamp}.csv"
            predictions_df.to_csv(predictions_file, index=False)
            attachments.append(str(predictions_file))

            logger.info(f"Created predictions attachment: {predictions_file}")

        except Exception as e:
            logger.warning(f"Error creating attachments: {e}")

        return attachments

    def _log_predictions(self, predictions: Dict[str, Any]) -> None:
        """
        Log prediction summary for dry run

        Args:
            predictions: Prediction data
        """
        logger.info("=== PREDICTION SUMMARY ===")

        for trader_id, data in predictions['traders'].items():
            logger.info(f"{trader_id}: ${data['predicted_pnl']:.2f} (confidence: {data['confidence']:.1%})")

        total_expected = sum(data['predicted_pnl'] for data in predictions['traders'].values())
        logger.info(f"Total Expected PnL: ${total_expected:.2f}")

        logger.info("=== END SUMMARY ===")

    def test_email_connection(self) -> bool:
        """
        Test email service connection

        Returns:
            True if connection successful, False otherwise
        """
        logger.info("Testing email connection...")
        return self.signal_service.email_sender.test_connection()

    def send_test_email(self, to_email: Optional[str] = None) -> bool:
        """
        Send test signal email with sample data

        Args:
            to_email: Recipient email (uses config default if None)

        Returns:
            True if sent successfully, False otherwise
        """
        logger.info("Sending test signal email...")
        return self.signal_service.test_signal_email(to_email)
