import logging
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import classification_report, precision_recall_curve, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit


class ModelTrainer:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.models_path = Path("data/models")
        self.models_path.mkdir(exist_ok=True)

        self.logger = logging.getLogger(__name__)

        # Model parameters
        self.global_params = self.config["model"]["global_model"]
        self.personal_params = self.config["model"]["personal_model"]

    def create_time_splits(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create time-based train/validation/test splits"""
        df = df.sort_values("Date")

        total_days = len(df["Date"].unique())
        holdout_days = int(total_days * 0.15)  # 15% for holdout
        val_days = int(total_days * 0.15)  # 15% for validation

        unique_dates = sorted(df["Date"].unique())

        train_end_date = unique_dates[-(holdout_days + val_days)]
        val_end_date = unique_dates[-holdout_days]

        train_df = df[df["Date"] <= train_end_date]
        val_df = df[(df["Date"] > train_end_date) & (df["Date"] <= val_end_date)]
        test_df = df[df["Date"] > val_end_date]

        self.logger.info(
            f"Train: {len(train_df)} samples, Val: {len(val_df)} samples, Test: {len(test_df)} samples"
        )

        return train_df, val_df, test_df

    def train_global_model(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: List[str]
    ) -> lgb.LGBMClassifier:
        """Train global model on all traders"""
        self.logger.info("Training global model...")

        X_train = train_df[feature_cols]
        y_train = train_df["target"]
        X_val = val_df[feature_cols]
        y_val = val_df["target"]

        # Initialize model
        model = lgb.LGBMClassifier(**self.global_params)

        # Train with early stopping
        print(X_train.head())
        print(y_train.head())
        print(X_train.dtypes)
        print(y_train.dtypes)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )

        # Evaluate
        val_pred = model.predict_proba(X_val)[:, 1]
        auc_score = roc_auc_score(y_val, val_pred)

        self.logger.info(f"Global model AUC: {auc_score:.4f}")

        # Save model
        joblib.dump(model, self.models_path / "global_model.pkl")

        return model

    def train_personal_models(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: List[str]
    ) -> Dict[str, lgb.LGBMClassifier]:
        """Train individual models for each trader"""
        personal_models = {}
        min_days = self.personal_params.pop("min_trading_days", 30)

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

                # Check if we have both classes
                if len(y_train.unique()) < 2:
                    self.logger.warning(
                        f"Only one class for {account_id}, skipping personal model"
                    )
                    continue

                model = lgb.LGBMClassifier(**self.personal_params)

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
        global_model: lgb.LGBMClassifier,
        personal_models: Dict,
        feature_cols: List[str],
    ) -> Dict:
        """Evaluate model performance"""
        results = {}

        # Global model evaluation
        X_test = test_df[feature_cols]
        y_test = test_df["target"]

        global_pred = global_model.predict_proba(X_test)[:, 1]
        global_auc = roc_auc_score(y_test, global_pred)

        results["global_auc"] = global_auc

        # Personal models evaluation
        personal_aucs = []

        for account_id in test_df["account_id"].unique():
            trader_test = test_df[test_df["account_id"] == account_id]

            if account_id in personal_models and len(trader_test) > 0:
                X_trader = trader_test[feature_cols]
                y_trader = trader_test["target"]

                if len(y_trader.unique()) > 1:  # Need both classes for AUC
                    pred = personal_models[account_id].predict_proba(X_trader)[:, 1]
                    auc = roc_auc_score(y_trader, pred)
                    personal_aucs.append(auc)

        results["personal_auc_mean"] = np.mean(personal_aucs) if personal_aucs else 0
        results["personal_auc_std"] = np.std(personal_aucs) if personal_aucs else 0

        self.logger.info(f"Global AUC: {global_auc:.4f}")
        self.logger.info(
            f"Personal AUC: {results['personal_auc_mean']:.4f} ± {results['personal_auc_std']:.4f}"
        )

        # Save results
        with open(self.models_path / "evaluation_results.yaml", "w") as f:
            yaml.dump(results, f)

        return results

    def save_model_metadata(self, feature_cols: List[str], results: Dict):
        """Save model metadata"""
        metadata = {
            "feature_columns": feature_cols,
            "model_params": {
                "global": self.global_params,
                "personal": self.personal_params,
            },
            "evaluation_results": results,
            "training_date": pd.Timestamp.now().isoformat(),
        }

        with open(self.models_path / "model_metadata.yaml", "w") as f:
            yaml.dump(metadata, f)
