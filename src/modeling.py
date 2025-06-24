# src/modeling.py

import pandas as pd
import numpy as np
from typing import Dict, Tuple
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import joblib
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_walk_forward_backtest(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Implement rigorous walk-forward backtesting procedure.

    Args:
        df: Feature DataFrame
        config: Configuration dictionary

    Returns:
        pd.DataFrame: Backtest results with predictions
    """
    logger.info("Starting walk-forward backtesting...")

    # Prepare features and targets
    feature_cols = [col for col in df.columns if col not in [
        'account_id', 'trade_date', 'target_pnl', 'target_large_loss',
        'daily_pnl', 'large_loss_threshold'
    ]]

    X = df[feature_cols]
    y_var = df['target_pnl']
    y_loss = df['target_large_loss']

    # Initialize TimeSeriesSplit
    tscv = TimeSeriesSplit(
        n_splits=config['backtesting']['n_splits'],
        test_size=config['backtesting']['test_days'],
        gap=config['backtesting']['gap_days']
    )

    # Storage for out-of-sample predictions
    oos_predictions = []

    # Get unique dates for proper splitting
    unique_dates = df['trade_date'].unique()
    date_to_idx = {date: idx for idx, date in enumerate(unique_dates)}
    df['date_idx'] = df['trade_date'].map(date_to_idx)

    # Perform walk-forward validation
    fold = 0
    for train_idx, test_idx in tscv.split(unique_dates):
        fold += 1
        logger.info(f"Processing fold {fold}/{config['backtesting']['n_splits']}")

        # Get train and test dates
        train_dates = unique_dates[train_idx]
        test_dates = unique_dates[test_idx]

        # Create train and test masks
        train_mask = df['trade_date'].isin(train_dates)
        test_mask = df['trade_date'].isin(test_dates)

        # Split data
        X_train, X_test = X[train_mask], X[test_mask]
        y_var_train, y_var_test = y_var[train_mask], y_var[test_mask]
        y_loss_train, y_loss_test = y_loss[train_mask], y_loss[test_mask]

        # Train VaR model (quantile regression)
        var_params = config['production_model']['var_model'].copy()
        var_model = lgb.LGBMRegressor(**var_params)
        var_model.fit(
            X_train, y_var_train,
            eval_set=[(X_test, y_var_test)],
            callbacks=[lgb.early_stopping(10), lgb.log_evaluation(50)]
        )

        # Train loss model (binary classification)
        loss_params = config['production_model']['loss_model'].copy()
        loss_model = lgb.LGBMClassifier(**loss_params)
        loss_model.fit(
            X_train, y_loss_train,
            eval_set=[(X_test, y_loss_test)],
            callbacks=[lgb.early_stopping(10), lgb.log_evaluation(50)]
        )

        # Make predictions
        var_pred = var_model.predict(X_test)
        loss_pred_proba = loss_model.predict_proba(X_test)[:, 1]

        # Store results
        fold_results = pd.DataFrame({
            'fold': fold,
            'account_id': df.loc[test_mask, 'account_id'].values,
            'trade_date': df.loc[test_mask, 'trade_date'].values,
            'true_pnl': y_var_test.values,
            'pred_var': var_pred,
            'true_large_loss': y_loss_test.values,
            'pred_loss_proba': loss_pred_proba
        })

        oos_predictions.append(fold_results)

    # Combine all out-of-sample predictions
    backtest_results = pd.concat(oos_predictions, ignore_index=True)

    # Calculate performance metrics
    logger.info("Calculating backtest performance metrics...")

    # VaR violation rate
    alpha = config['production_model']['var_model']['alpha']
    violations = (backtest_results['true_pnl'] < backtest_results['pred_var']).mean()
    logger.info(f"VaR violation rate: {violations:.4f} (expected: {alpha})")

    # Loss prediction AUC
    auc = roc_auc_score(backtest_results['true_large_loss'], backtest_results['pred_loss_proba'])
    logger.info(f"Large loss prediction AUC: {auc:.4f}")

    # Save backtest results
    results_path = os.path.join(config['paths']['model_dir'], 'backtest_results.csv')
    backtest_results.to_csv(results_path, index=False)
    logger.info(f"Backtest results saved to {results_path}")

    return backtest_results


def train_production_model(df: pd.DataFrame, config: Dict) -> Tuple[lgb.LGBMRegressor, lgb.LGBMClassifier]:
    """
    Train final production models on entire dataset.

    Args:
        df: Feature DataFrame
        config: Configuration dictionary

    Returns:
        Tuple of trained VaR and loss models
    """
    logger.info("Training production models on full dataset...")

    # Prepare features and targets
    feature_cols = [col for col in df.columns if col not in [
        'account_id', 'trade_date', 'target_pnl', 'target_large_loss',
        'daily_pnl', 'large_loss_threshold'
    ]]

    X = df[feature_cols]
    y_var = df['target_pnl']
    y_loss = df['target_large_loss']

    # Train VaR model
    logger.info("Training VaR model...")
    var_params = config['production_model']['var_model'].copy()
    var_model = lgb.LGBMRegressor(**var_params)
    var_model.fit(X, y_var)

    # Train loss model
    logger.info("Training loss model...")
    loss_params = config['production_model']['loss_model'].copy()
    loss_model = lgb.LGBMClassifier(**loss_params)
    loss_model.fit(X, y_loss)

    # Save models
    var_model_path = os.path.join(config['paths']['model_dir'], 'lgbm_var_model.joblib')
    loss_model_path = os.path.join(config['paths']['model_dir'], 'lgbm_loss_model.joblib')

    joblib.dump(var_model, var_model_path)
    joblib.dump(loss_model, loss_model_path)

    logger.info(f"Models saved to {config['paths']['model_dir']}")

    # Save feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'var_importance': var_model.feature_importances_,
        'loss_importance': loss_model.feature_importances_
    }).sort_values('var_importance', ascending=False)

    importance_path = os.path.join(config['paths']['model_dir'], 'feature_importance.csv')
    feature_importance.to_csv(importance_path, index=False)

    logger.info("Top 10 most important features for VaR model:")
    logger.info(feature_importance[['feature', 'var_importance']].head(10))

    return var_model, loss_model
