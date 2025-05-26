import logging
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima


class ModelTrainer:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.models_path = Path("data/models")
        self.models_path.mkdir(exist_ok=True)

        self.logger = logging.getLogger(__name__)

        # Model parameters for regression
        self.global_params = {
            "objective": "regression",
            "boosting_type": "gbdt",
            "metric": "rmse",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "n_estimators": 1000,
            "random_state": 42
        }

        self.personal_params = {
            "objective": "regression",
            "boosting_type": "gbdt",
            "metric": "rmse",
            "num_leaves": 15,
            "learning_rate": 0.1,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.7,
            "bagging_freq": 5,
            "verbose": -1,
            "n_estimators": 500,
            "random_state": 42
        }

    def create_time_splits(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create time-based train/validation/test splits"""
        df = df.sort_values("date")

        total_days = len(df["date"].unique())
        holdout_days = int(total_days * 0.15)  # 15% for holdout
        val_days = int(total_days * 0.15)  # 15% for validation

        unique_dates = sorted(df["date"].unique())

        train_end_date = unique_dates[-(holdout_days + val_days)]
        val_end_date = unique_dates[-holdout_days]

        train_df = df[df["date"] <= train_end_date]
        val_df = df[(df["date"] > train_end_date) & (df["date"] <= val_end_date)]
        test_df = df[df["date"] > val_end_date]

        self.logger.info(
            f"Train: {len(train_df)} samples, Val: {len(val_df)} samples, Test: {len(test_df)} samples"
        )

        return train_df, val_df, test_df

    def train_arima_baseline(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict:
        """Train ARIMA models as baseline for each trader"""
        self.logger.info("Training ARIMA baseline models...")
        arima_models = {}

        for account_id in train_df["account_id"].unique():
            trader_train = train_df[train_df["account_id"] == account_id].sort_values("date")

            if len(trader_train) < 30:
                self.logger.warning(f"Insufficient data for ARIMA {account_id}")
                continue

            try:
                # Use total_delta as time series
                ts = trader_train.set_index("date")["total_delta"].asfreq('D').fillna(0)

                # Auto ARIMA for best parameters
                auto_model = auto_arima(
                    ts,
                    start_p=0, start_q=0,
                    max_p=5, max_q=5,
                    seasonal=False,
                    stepwise=True,
                    suppress_warnings=True,
                    error_action='ignore',
                    n_fits=20
                )

                arima_models[account_id] = auto_model

                # Save model
                joblib.dump(auto_model, self.models_path / f"arima_{account_id}.pkl")

                self.logger.info(f"Trained ARIMA{auto_model.order} for {account_id}")

            except Exception as e:
                self.logger.error(f"Error training ARIMA for {account_id}: {str(e)}")

        return arima_models

    def train_global_model(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: List[str]
    ) -> lgb.LGBMRegressor:
        """Train global regression model on all traders"""
        self.logger.info("Training global regression model...")

        X_train = train_df[feature_cols]
        y_train = train_df["target"]  # This is already next day's total_delta
        X_val = val_df[feature_cols]
        y_val = val_df["target"]

        # Initialize regression model
        model = lgb.LGBMRegressor(**self.global_params)

        # Train with early stopping
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )

        # Evaluate
        val_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        mae = mean_absolute_error(y_val, val_pred)
        r2 = r2_score(y_val, val_pred)

        self.logger.info(f"Global model - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

        # Save model
        joblib.dump(model, self.models_path / "global_model.pkl")

        return model

    def train_personal_models(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: List[str]
    ) -> Dict[str, lgb.LGBMRegressor]:
        """Train individual regression models for each trader"""
        personal_models = {}
        min_days = 30

        for account_id in train_df["account_id"].unique():
            trader_train = train_df[train_df["account_id"] == account_id]
            trader_val = val_df[val_df["account_id"] == account_id]

            # Skip if insufficient data
            if len(trader_train) < min_days:
                self.logger.warning(
                    f"Insufficient data for {account_id}: {len(trader_train)} days"
                )
                continue

            try:
                X_train = trader_train[feature_cols]
                y_train = trader_train["target"]

                model = lgb.LGBMRegressor(**self.personal_params)

                if len(trader_val) > 0:
                    X_val = trader_val[feature_cols]
                    y_val = trader_val["target"]
                    model.fit(
                        X_train,
                        y_train,
                        eval_set=[(X_val, y_val)],
                        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)],
                    )
                else:
                    model.fit(X_train, y_train)

                personal_models[account_id] = model

                # Save individual model
                joblib.dump(model, self.models_path / f"personal_{account_id}.pkl")

                self.logger.info(f"Trained personal model for {account_id}")

            except Exception as e:
                self.logger.error(f"Error training model for {account_id}: {str(e)}")

        self.logger.info(f"Trained {len(personal_models)} personal models")
        return personal_models

    def evaluate_models(
        self,
        test_df: pd.DataFrame,
        global_model: lgb.LGBMRegressor,
        personal_models: Dict,
        arima_models: Dict,
        feature_cols: List[str],
    ) -> Dict:
        """Evaluate all models' performance"""
        results = {}

        # Global model evaluation
        X_test = test_df[feature_cols]
        y_test = test_df["target"]

        global_pred = global_model.predict(X_test)
        global_rmse = np.sqrt(mean_squared_error(y_test, global_pred))
        global_mae = mean_absolute_error(y_test, global_pred)
        global_r2 = r2_score(y_test, global_pred)

        results["global"] = {
            "rmse": global_rmse,
            "mae": global_mae,
            "r2": global_r2
        }

        # Personal models evaluation
        personal_metrics = {"rmse": [], "mae": [], "r2": []}

        for account_id in test_df["account_id"].unique():
            trader_test = test_df[test_df["account_id"] == account_id]

            if account_id in personal_models and len(trader_test) > 0:
                X_trader = trader_test[feature_cols]
                y_trader = trader_test["target"]

                pred = personal_models[account_id].predict(X_trader)
                personal_metrics["rmse"].append(np.sqrt(mean_squared_error(y_trader, pred)))
                personal_metrics["mae"].append(mean_absolute_error(y_trader, pred))
                personal_metrics["r2"].append(r2_score(y_trader, pred))

        results["personal"] = {
            "rmse_mean": np.mean(personal_metrics["rmse"]) if personal_metrics["rmse"] else 0,
            "mae_mean": np.mean(personal_metrics["mae"]) if personal_metrics["mae"] else 0,
            "r2_mean": np.mean(personal_metrics["r2"]) if personal_metrics["r2"] else 0,
        }

        # ARIMA baseline evaluation
        arima_metrics = {"rmse": [], "mae": [], "r2": []}

        for account_id in test_df["account_id"].unique():
            if account_id in arima_models:
                trader_test = test_df[test_df["account_id"] == account_id].sort_values("date")

                try:
                    # Forecast for test period
                    forecast = arima_models[account_id].predict(n_periods=len(trader_test))
                    y_true = trader_test["target"].values

                    arima_metrics["rmse"].append(np.sqrt(mean_squared_error(y_true, forecast)))
                    arima_metrics["mae"].append(mean_absolute_error(y_true, forecast))
                    arima_metrics["r2"].append(r2_score(y_true, forecast))
                except:
                    pass

        results["arima"] = {
            "rmse_mean": np.mean(arima_metrics["rmse"]) if arima_metrics["rmse"] else 0,
            "mae_mean": np.mean(arima_metrics["mae"]) if arima_metrics["mae"] else 0,
            "r2_mean": np.mean(arima_metrics["r2"]) if arima_metrics["r2"] else 0,
        }

        self.logger.info(f"Global Model - RMSE: {global_rmse:.4f}, MAE: {global_mae:.4f}, R2: {global_r2:.4f}")
        self.logger.info(f"Personal Models - RMSE: {results['personal']['rmse_mean']:.4f}, MAE: {results['personal']['mae_mean']:.4f}, R2: {results['personal']['r2_mean']:.4f}")
        self.logger.info(f"ARIMA Baseline - RMSE: {results['arima']['rmse_mean']:.4f}, MAE: {results['arima']['mae_mean']:.4f}, R2: {results['arima']['r2_mean']:.4f}")

        # Save results
        with open(self.models_path / "evaluation_results.yaml", "w") as f:
            yaml.dump(results, f)

        return results

    # ────────────────────────── helpers ──────────────────────────
    @staticmethod
    def _to_builtin(obj):
        """Recursively convert NumPy / pandas scalars & arrays to pure-Python types."""
        import numpy as np

        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: ModelTrainer._to_builtin(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [ModelTrainer._to_builtin(v) for v in obj]
        return obj

    # ────────────────────────── metadata ──────────────────────────
    def save_model_metadata(self, feature_cols: List[str], results: Dict):
        """Persist training metadata in YAML; output contains only built-in types."""
        metadata = {
            "feature_columns": feature_cols,
            "model_params": {
                "global": self.global_params,
                "personal": self.personal_params,
            },
            "evaluation_results": results,
            "training_date": pd.Timestamp.now().isoformat(),
        }

        clean = ModelTrainer._to_builtin(metadata)

        self.models_path.mkdir(parents=True, exist_ok=True)
        meta_file = self.models_path / "model_metadata.yaml"
        with meta_file.open("w") as f:
            yaml.safe_dump(clean, f, sort_keys=False)
