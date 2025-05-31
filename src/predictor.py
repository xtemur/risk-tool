"""
Simplified Risk Predictor for Risk Management MVP
Uses only personal models for predictions
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path

from src.database import Database
from src.feature_engineer import FeatureEngineer
from src.model_trainer import ModelTrainer

logger = logging.getLogger(__name__)


class RiskPredictor:
    """Generate risk predictions using personal models"""

    def __init__(self):
        self.db = Database()
        self.feature_engineer = FeatureEngineer()
        self.model_trainer = ModelTrainer()
        self.models = self.model_trainer.get_all_models()

        logger.info(f"Loaded {len(self.models)} personal models")

    def predict_trader(self, account_id: str, lookback_days: int = 60) -> Optional[Dict]:
        """Generate prediction for a single trader"""

        # Check if model exists
        if account_id not in self.models:
            logger.warning(f"No model available for {account_id}")
            return None

        model_data = self.models[account_id]
        model = model_data['model']
        feature_columns = model_data['feature_columns']
        threshold = model_data['threshold']

        # Get recent data
        end_date = pd.Timestamp.now().date()
        start_date = end_date - pd.Timedelta(days=lookback_days)

        totals_df, fills_df = self.db.get_trader_data(
            account_id,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )

        if totals_df.empty:
            logger.warning(f"No recent data for {account_id}")
            return None

        # Create features
        features_df = self.feature_engineer.create_features(totals_df, fills_df)

        if features_df.empty or len(features_df) < 5:
            logger.warning(f"Insufficient feature data for {account_id}")
            return None

        # Get latest features
        latest_features = features_df[feature_columns].iloc[-1:].values

        # Make prediction
        predicted_pnl = model.predict(latest_features, num_iteration=model.best_iteration)[0]

        # Calculate risk metrics
        recent_pnl = totals_df['net_pnl'].tail(5).sum()
        recent_volatility = totals_df['net_pnl'].tail(20).std()

        # Determine risk level
        if predicted_pnl < -1000:
            risk_level = "High"
            risk_score = 0.9
        elif predicted_pnl < 0:
            risk_level = "Medium"
            risk_score = 0.6
        else:
            risk_level = "Low"
            risk_score = 0.3

        # Adjust risk score based on recent performance
        if recent_pnl < -2000:
            risk_score = min(1.0, risk_score + 0.2)
        elif recent_pnl > 2000:
            risk_score = max(0.1, risk_score - 0.1)

        # Trading recommendation
        if predicted_pnl < threshold:
            recommendation = "Reduce position sizes or skip trading"
        else:
            recommendation = "Normal trading"

        return {
            'account_id': account_id,
            'predicted_pnl': predicted_pnl,
            'risk_level': risk_level,
            'risk_score': risk_score,
            'confidence': 'High',  # Personal model available
            'recent_pnl_5d': recent_pnl,
            'recent_volatility': recent_volatility,
            'recommendation': recommendation,
            'threshold': threshold,
            'last_update': pd.Timestamp.now()
        }

    def predict_all_traders(self) -> List[Dict]:
        """Generate predictions for all traders with models"""
        predictions = []

        # Get all traders
        traders_df = self.db.get_all_traders()

        for _, trader in traders_df.iterrows():
            account_id = trader['account_id']

            prediction = self.predict_trader(account_id)

            if prediction:
                prediction['trader_name'] = trader['trader_name']
                predictions.append(prediction)
            else:
                # Add placeholder for traders without predictions
                predictions.append({
                    'account_id': account_id,
                    'trader_name': trader['trader_name'],
                    'predicted_pnl': 0,
                    'risk_level': 'Unknown',
                    'risk_score': 0.5,
                    'confidence': 'Low',
                    'recent_pnl_5d': 0,
                    'recommendation': 'No model available',
                    'last_update': pd.Timestamp.now()
                })

        # Sort by risk score (highest risk first)
        predictions.sort(key=lambda x: x['risk_score'], reverse=True)

        # Save predictions to database
        self.db.save_predictions(predictions)

        logger.info(f"Generated predictions for {len(predictions)} traders")

        return predictions

    def get_risk_summary(self, predictions: List[Dict]) -> Dict:
        """Generate summary statistics from predictions"""

        df = pd.DataFrame(predictions)

        summary = {
            'total_traders': len(predictions),
            'high_risk_count': len(df[df['risk_level'] == 'High']),
            'medium_risk_count': len(df[df['risk_level'] == 'Medium']),
            'low_risk_count': len(df[df['risk_level'] == 'Low']),
            'unknown_risk_count': len(df[df['risk_level'] == 'Unknown']),
            'total_predicted_pnl': df['predicted_pnl'].sum(),
            'total_recent_pnl': df['recent_pnl_5d'].sum(),
            'models_available': len(df[df['confidence'] == 'High']),
            'timestamp': pd.Timestamp.now()
        }

        # Identify top risk traders
        summary['top_risk_traders'] = df.nlargest(5, 'risk_score')[
            ['trader_name', 'risk_level', 'predicted_pnl', 'recommendation']
        ].to_dict('records')

        return summary
