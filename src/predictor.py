import logging
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml


class RiskPredictor:
    def __init__(self):
        self.models_path = Path("data/models")
        self.logger = logging.getLogger(__name__)

        # Load models and metadata
        self.load_models()

    def load_models(self):
        """Load trained models and metadata"""
        try:
            # Load global model
            self.global_model = joblib.load(self.models_path / "global_model.pkl")

            # Load personal models
            self.personal_models = {}
            for model_file in self.models_path.glob("personal_*.pkl"):
                account_id = model_file.stem.replace("personal_", "")
                self.personal_models[account_id] = joblib.load(model_file)

            # Load ARIMA models
            self.arima_models = {}
            for model_file in self.models_path.glob("arima_*.pkl"):
                account_id = model_file.stem.replace("arima_", "")
                self.arima_models[account_id] = joblib.load(model_file)

            # Load metadata
            with open(self.models_path / "model_metadata.yaml", "r") as f:
                self.metadata = yaml.safe_load(f)

            self.feature_cols = self.metadata["feature_columns"]

            # Load config for ensemble weights
            with open("config/config.yaml", "r") as f:
                config = yaml.safe_load(f)

            self.global_weight = 0.6  # Default weights for ensemble
            self.personal_weight = 0.4

            self.logger.info(
                f"Loaded global model, {len(self.personal_models)} personal models, and {len(self.arima_models)} ARIMA models"
            )

        except Exception as e:
            self.logger.error(f"Error loading models: {str(e)}")
            raise

    def predict_single_trader(self, trader_data: pd.DataFrame, account_id: str) -> Dict:
        """Predict next day's total_delta for a single trader"""
        try:
            # Get latest features
            latest_data = trader_data.iloc[-1:][self.feature_cols]

            # Global model prediction
            global_pred = self.global_model.predict(latest_data)[0]

            # Personal model prediction (if available)
            if account_id in self.personal_models:
                personal_pred = self.personal_models[account_id].predict(latest_data)[0]

                # Ensemble prediction
                ensemble_pred = (
                    self.global_weight * global_pred
                    + self.personal_weight * personal_pred
                )

                confidence = "High"  # Has personal model
            else:
                # Use only global model
                ensemble_pred = global_pred
                personal_pred = None
                confidence = "Medium"  # No personal model

            # ARIMA prediction (if available)
            arima_pred = None
            if account_id in self.arima_models:
                try:
                    arima_pred = self.arima_models[account_id].predict(n_periods=1)[0]
                except:
                    pass

            # Risk categorization based on predicted P&L
            if ensemble_pred < -1000:
                risk_level = "High"
            elif ensemble_pred < 0:
                risk_level = "Medium"
            else:
                risk_level = "Low"

            return {
                "account_id": account_id,
                "predicted_pnl": ensemble_pred,
                "risk_level": risk_level,
                "confidence": confidence,
                "global_prediction": global_pred,
                "personal_prediction": personal_pred,
                "arima_prediction": arima_pred,
                "recent_pnl": trader_data["net_pnl"].tail(5).sum(),
                "recent_performance": trader_data["total_delta"].tail(3).sum(),
            }

        except Exception as e:
            self.logger.error(f"Error predicting for {account_id}: {str(e)}")
            return {
                "account_id": account_id,
                "predicted_pnl": 0,
                "risk_level": "Unknown",
                "confidence": "Low",
                "error": str(e),
            }

    def predict_all_traders(self, all_data: Dict) -> List[Dict]:
        """Generate predictions for all traders"""
        predictions = []

        for account_id, data in all_data.items():
            trader_prediction = self.predict_single_trader(data["features"], account_id)
            trader_prediction["trader_name"] = data.get("name", account_id)
            predictions.append(trader_prediction)

        # Sort by predicted P&L (lowest first - highest risk)
        predictions.sort(key=lambda x: x["predicted_pnl"])

        self.logger.info(f"Generated predictions for {len(predictions)} traders")
        return predictions
