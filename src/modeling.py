# src/modeling.py

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.feature_selection import RFE
from scipy import stats
import joblib
import logging
import os
import warnings
from datetime import datetime
from .advanced_risk_metrics import generate_advanced_risk_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_strict_walk_forward_backtest(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Implement STRICT walk-forward backtesting with temporal ordering and feature selection inside CV.
    This prevents all forms of temporal leakage by strictly respecting time boundaries.

    Args:
        df: Feature DataFrame
        config: Configuration dictionary

    Returns:
        pd.DataFrame: Backtest results with predictions
    """
    logger.info("Starting STRICT walk-forward backtesting with in-fold feature selection...")

    # Sort by date to ensure temporal ordering
    df = df.sort_values('trade_date').reset_index(drop=True)

    # Create time-based splits
    unique_dates = sorted(df['trade_date'].unique())
    n_splits = config['backtesting']['n_splits']
    min_train_months = 6  # Minimum 6 months training data
    gap_days = 7  # 1 week gap between train/test
    test_months = 3  # 3 months test period

    logger.info(f"Creating {n_splits} time-based splits with {test_months}-month test periods")

    results = []

    for fold in range(n_splits):
        logger.info(f"Processing fold {fold + 1}/{n_splits}")

        # Calculate split dates
        total_months = min_train_months + test_months + (fold * 2)  # Advancing by 2 months each fold
        train_start_idx = 0
        train_end_idx = int(len(unique_dates) * (min_train_months + fold * 2) / (total_months))
        test_start_idx = min(train_end_idx + gap_days, len(unique_dates) - test_months * 30)
        test_end_idx = min(test_start_idx + test_months * 30, len(unique_dates))

        if test_start_idx >= len(unique_dates) or test_end_idx <= test_start_idx:
            logger.warning(f"Insufficient data for fold {fold + 1}, skipping")
            continue

        train_dates = unique_dates[train_start_idx:train_end_idx]
        test_dates = unique_dates[test_start_idx:test_end_idx]

        logger.info(f"Fold {fold + 1}: Train {train_dates[0]} to {train_dates[-1]}, "
                   f"Test {test_dates[0]} to {test_dates[-1]}")

        # Split data based on dates
        train_mask = df['trade_date'].isin(train_dates)
        test_mask = df['trade_date'].isin(test_dates)

        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()

        if len(train_df) < 100 or len(test_df) < 20:
            logger.warning(f"Insufficient samples in fold {fold + 1}: train={len(train_df)}, test={len(test_df)}")
            continue

        # Feature selection INSIDE the fold to prevent leakage
        fold_results = train_and_evaluate_fold(train_df, test_df, config, fold + 1)
        if fold_results is not None:
            results.append(fold_results)

    if not results:
        raise ValueError("No valid folds generated - check data size and split parameters")

    # Combine all fold results
    backtest_results = pd.concat(results, ignore_index=True)

    # Save results
    output_path = os.path.join(config['paths']['model_dir'], 'strict_walk_forward_results.csv')
    backtest_results.to_csv(output_path, index=False)
    logger.info(f"Strict walk-forward results saved to {output_path}")

    return backtest_results


def train_and_evaluate_fold(train_df: pd.DataFrame, test_df: pd.DataFrame, config: Dict, fold_num: int) -> pd.DataFrame:
    """
    Train models and evaluate on a single fold with feature selection INSIDE the fold.
    This is critical to prevent leakage from feature selection.
    """
    logger.info(f"Training fold {fold_num} with in-fold feature selection...")

    # Separate features and targets
    exclude_cols = ['account_id', 'trade_date', 'target_pnl', 'target_large_loss', 'daily_pnl', 'large_loss_threshold']
    all_feature_cols = [col for col in train_df.columns if col not in exclude_cols]

    # Remove any rows with missing targets
    train_df = train_df.dropna(subset=['target_pnl', 'target_large_loss'])
    test_df = test_df.dropna(subset=['target_pnl', 'target_large_loss'])

    if len(train_df) < 50 or len(test_df) < 10:
        logger.warning(f"Insufficient data after cleanup: train={len(train_df)}, test={len(test_df)}")
        return None

    # Feature selection INSIDE fold to prevent leakage
    X_train_all = train_df[all_feature_cols].fillna(0)

    # Handle infinity values that can break sklearn models
    X_train_all = X_train_all.replace([np.inf, -np.inf], [1e6, -1e6])

    y_train_var = train_df['target_pnl']
    y_train_loss = train_df['target_large_loss']

    # Forward feature selection for stability
    selected_features = forward_feature_selection(X_train_all, y_train_var, y_train_loss,
                                                 config.get('model_quality', {}).get('max_features', 16))

    logger.info(f"Fold {fold_num}: Selected {len(selected_features)} features")

    # Train on selected features only
    X_train = X_train_all[selected_features]
    X_test = test_df[selected_features].fillna(0)
    X_test = X_test.replace([np.inf, -np.inf], [1e6, -1e6])
    y_test_var = test_df['target_pnl']
    y_test_loss = test_df['target_large_loss']

    # Train VaR model (LightGBM Quantile Regression)
    var_model = lgb.LGBMRegressor(
        objective='quantile',
        alpha=0.05,  # 5% quantile for VaR
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1
    )

    var_model.fit(X_train, y_train_var)
    var_predictions = var_model.predict(X_test)

    # Train Loss model (LightGBM Binary Classification)
    loss_model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1
    )

    loss_model.fit(X_train, y_train_loss)
    loss_predictions = loss_model.predict_proba(X_test)[:, 1]

    # Create results dataframe
    fold_results = pd.DataFrame({
        'fold': fold_num,
        'account_id': test_df['account_id'].values,
        'trade_date': test_df['trade_date'].values,
        'true_pnl': y_test_var.values,
        'pred_var': var_predictions,
        'true_large_loss': y_test_loss.values,
        'pred_loss_proba': loss_predictions,
        'train_samples': len(train_df),
        'test_samples': len(test_df),
        'n_features_selected': len(selected_features)
    })

    # Calculate fold performance metrics
    var_violations = (y_test_var < var_predictions).sum()
    var_violation_rate = var_violations / len(y_test_var)
    loss_auc = roc_auc_score(y_test_loss, loss_predictions) if len(np.unique(y_test_loss)) > 1 else 0.5

    logger.info(f"Fold {fold_num} performance: VaR violation rate {var_violation_rate:.3f}, Loss AUC {loss_auc:.3f}")

    return fold_results


def forward_feature_selection(X: pd.DataFrame, y_var: pd.Series, y_loss: pd.Series, max_features: int) -> List[str]:
    """
    Fast importance-based feature selection for time series data.
    Much faster than iterative forward selection.
    """
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

    # Quick importance-based selection using small random forests
    rf_var = RandomForestRegressor(n_estimators=30, max_depth=6, random_state=42)
    rf_loss = RandomForestClassifier(n_estimators=30, max_depth=6, random_state=42)

    rf_var.fit(X, y_var)
    rf_loss.fit(X, y_loss)

    # Combine importance scores with equal weighting
    var_importance = pd.Series(rf_var.feature_importances_, index=X.columns)
    loss_importance = pd.Series(rf_loss.feature_importances_, index=X.columns)

    # Combined score - equally weight both tasks
    combined_importance = (var_importance + loss_importance) / 2

    # Select top features based on combined importance
    selected_features = combined_importance.nlargest(max_features).index.tolist()

    return selected_features


def run_walk_forward_backtest(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Legacy walk-forward implementation - use run_strict_walk_forward_backtest for better temporal control.
    """
    logger.warning("Using legacy walk-forward - consider run_strict_walk_forward_backtest for better temporal control")

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

    # Basic metrics
    alpha = config['production_model']['var_model']['alpha']
    violations = (backtest_results['true_pnl'] < backtest_results['pred_var']).mean()
    logger.info(f"VaR violation rate: {violations:.4f} (expected: {alpha})")

    auc = roc_auc_score(backtest_results['true_large_loss'], backtest_results['pred_loss_proba'])
    logger.info(f"Large loss prediction AUC: {auc:.4f}")

    # Advanced statistical validation
    advanced_metrics = calculate_advanced_metrics(backtest_results, config)

    # Save backtest results and validation metrics
    results_path = os.path.join(config['paths']['model_dir'], 'backtest_results.csv')
    backtest_results.to_csv(results_path, index=False)
    logger.info(f"Backtest results saved to {results_path}")

    # Save validation metrics
    validation_path = os.path.join(config['paths']['model_dir'], 'validation_metrics.json')
    import json
    with open(validation_path, 'w') as f:
        json.dump(advanced_metrics, f, indent=2, default=str)
    logger.info(f"Validation metrics saved to {validation_path}")

    return backtest_results


def train_production_model(df: pd.DataFrame, config: Dict) -> Tuple[lgb.LGBMRegressor, lgb.LGBMClassifier]:
    """
    Train final production models with enhanced validation and feature selection.

    Args:
        df: Feature DataFrame
        config: Configuration dictionary

    Returns:
        Tuple of trained VaR and loss models
    """
    logger.info("Training production models with enhanced validation...")

    # Data quality validation
    validation_results = validate_data_quality(df, config)

    # Stop if critical data quality issues found
    critical_issues = [issue for issue in validation_results['issues']
                      if 'correlation issue' in issue or 'leakage' in issue.lower()]
    if critical_issues:
        logger.error(f"Critical data quality issues found: {critical_issues}")
        logger.error("Model training aborted. Please fix data quality issues first.")
        raise ValueError("Critical data quality issues detected")

    # Prepare initial features and targets
    feature_cols = [col for col in df.columns if col not in [
        'account_id', 'trade_date', 'target_pnl', 'target_large_loss',
        'daily_pnl', 'large_loss_threshold'
    ]]

    X_full = df[feature_cols]
    y_var = df['target_pnl']
    y_loss = df['target_large_loss']

    # Feature selection for overfitting prevention
    max_features = config.get('model_quality', {}).get('max_features', 15)

    if len(feature_cols) > max_features:
        logger.info(f"Performing feature selection: {len(feature_cols)} -> {max_features} features")

        # Feature selection for VaR model
        var_params = config['production_model']['var_model'].copy()
        selected_var_features = feature_selection_pipeline(
            X_full, y_var, var_params, 'var', max_features
        )

        # Feature selection for loss model
        loss_params = config['production_model']['loss_model'].copy()
        selected_loss_features = feature_selection_pipeline(
            X_full, y_loss, loss_params, 'loss', max_features
        )

        # Use union of selected features for both models
        selected_features = list(set(selected_var_features + selected_loss_features))
        logger.info(f"Final feature set: {len(selected_features)} features")

        X = X_full[selected_features]
    else:
        X = X_full
        selected_features = feature_cols

    # Split data for validation
    validation_split = config.get('model_quality', {}).get('validation_split', 0.2)
    split_point = int(len(df) * (1 - validation_split))
    X_train, X_val = X.iloc[:split_point], X.iloc[split_point:]
    y_var_train, y_var_val = y_var.iloc[:split_point], y_var.iloc[split_point:]
    y_loss_train, y_loss_val = y_loss.iloc[:split_point], y_loss.iloc[split_point:]

    # Train VaR model with validation
    logger.info("Training VaR model with validation...")
    var_params = config['production_model']['var_model'].copy()
    var_model = lgb.LGBMRegressor(**var_params)
    var_model.fit(
        X_train, y_var_train,
        eval_set=[(X_val, y_var_val)],
        callbacks=[lgb.early_stopping(config.get('model_quality', {}).get('early_stopping_rounds', 20)),
                   lgb.log_evaluation(0)]  # Silent evaluation
    )

    # Validate VaR model convergence
    var_convergence = validate_model_convergence(var_model, 'var')

    # Train loss model with validation
    logger.info("Training loss model with validation...")
    loss_params = config['production_model']['loss_model'].copy()
    loss_model = lgb.LGBMClassifier(**loss_params)
    loss_model.fit(
        X_train, y_loss_train,
        eval_set=[(X_val, y_loss_val)],
        callbacks=[lgb.early_stopping(config.get('model_quality', {}).get('early_stopping_rounds', 20)),
                   lgb.log_evaluation(0)]  # Silent evaluation
    )

    # Validate loss model convergence
    loss_convergence = validate_model_convergence(loss_model, 'loss')

    # Check for overfitting and retrain if necessary
    if var_convergence.get('likely_overfitting', False) or loss_convergence.get('likely_overfitting', False):
        logger.warning("Potential overfitting detected. Retraining with increased regularization...")

        # Increase regularization parameters
        var_params_reg = var_params.copy()
        var_params_reg['learning_rate'] *= 0.7
        var_params_reg['reg_alpha'] = var_params_reg.get('reg_alpha', 0) + 0.1
        var_params_reg['reg_lambda'] = var_params_reg.get('reg_lambda', 0) + 0.1

        loss_params_reg = loss_params.copy()
        loss_params_reg['learning_rate'] *= 0.7
        loss_params_reg['reg_alpha'] = loss_params_reg.get('reg_alpha', 0) + 0.1
        loss_params_reg['reg_lambda'] = loss_params_reg.get('reg_lambda', 0) + 0.1

        # Retrain with regularization
        var_model = lgb.LGBMRegressor(**var_params_reg)
        var_model.fit(X_train, y_var_train, eval_set=[(X_val, y_var_val)],
                     callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])

        loss_model = lgb.LGBMClassifier(**loss_params_reg)
        loss_model.fit(X_train, y_loss_train, eval_set=[(X_val, y_loss_val)],
                      callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])

    # Final training on full dataset
    logger.info("Final training on full dataset...")
    var_model.fit(X, y_var)
    loss_model.fit(X, y_loss)

    # Save models
    var_model_path = os.path.join(config['paths']['model_dir'], 'lgbm_var_model.joblib')
    loss_model_path = os.path.join(config['paths']['model_dir'], 'lgbm_loss_model.joblib')

    joblib.dump(var_model, var_model_path)
    joblib.dump(loss_model, loss_model_path)

    logger.info(f"Models saved to {config['paths']['model_dir']}")

    # Save enhanced feature importance and metadata
    feature_importance = pd.DataFrame({
        'feature': selected_features,
        'var_importance': var_model.feature_importances_,
        'loss_importance': loss_model.feature_importances_
    }).sort_values('var_importance', ascending=False)

    importance_path = os.path.join(config['paths']['model_dir'], 'feature_importance.csv')
    feature_importance.to_csv(importance_path, index=False)

    # Save model metadata
    model_metadata = {
        'training_date': datetime.now().isoformat(),
        'selected_features': selected_features,
        'n_features': len(selected_features),
        'n_samples': len(df),
        'var_convergence': var_convergence,
        'loss_convergence': loss_convergence,
        'data_quality_issues': validation_results['n_issues'],
        'validation_results': validation_results
    }

    metadata_path = os.path.join(config['paths']['model_dir'], 'model_metadata.json')
    import json
    with open(metadata_path, 'w') as f:
        json.dump(model_metadata, f, indent=2, default=str)

    logger.info("Enhanced model training completed successfully:")
    logger.info(f"  Features used: {len(selected_features)}")
    logger.info(f"  VaR model convergence: {'GOOD' if not var_convergence.get('likely_overfitting', False) else 'OVERFITTING'}")
    logger.info(f"  Loss model convergence: {'GOOD' if not loss_convergence.get('likely_overfitting', False) else 'OVERFITTING'}")
    logger.info(f"  Data quality issues: {validation_results['n_issues']}")

    return var_model, loss_model


def validate_model_convergence(model, model_type: str = 'var') -> Dict:
    """
    Validate model convergence and training quality.

    Args:
        model: Trained LightGBM model
        model_type: 'var' or 'loss'

    Returns:
        Dict with convergence metrics
    """
    logger.info(f"Validating {model_type} model convergence...")

    metrics = {}

    # Get training history if available
    if hasattr(model, 'evals_result_') and model.evals_result_:
        eval_results = model.evals_result_

        # Check for validation set results
        if 'valid_0' in eval_results:
            valid_scores = eval_results['valid_0']

            # Get the metric name (depends on model type)
            metric_name = list(valid_scores.keys())[0]
            scores = valid_scores[metric_name]

            # Calculate convergence metrics
            metrics['final_score'] = scores[-1]
            metrics['best_score'] = min(scores) if 'rmse' in metric_name or 'quantile' in metric_name else max(scores)
            metrics['best_iteration'] = np.argmin(scores) if 'rmse' in metric_name or 'quantile' in metric_name else np.argmax(scores)
            metrics['total_iterations'] = len(scores)
            metrics['early_stopped'] = metrics['best_iteration'] < metrics['total_iterations'] - 1

            # Check for overfitting (performance degradation after best iteration)
            if metrics['early_stopped']:
                after_best = scores[metrics['best_iteration']:]
                if len(after_best) > 5:  # Need at least 5 points
                    slope, _, r_value, p_value, _ = stats.linregress(range(len(after_best)), after_best)
                    metrics['overfitting_slope'] = slope
                    metrics['overfitting_r_squared'] = r_value ** 2
                    metrics['overfitting_p_value'] = p_value

                    # Flag potential overfitting
                    if model_type == 'var':
                        metrics['likely_overfitting'] = slope > 0 and p_value < 0.1  # Increasing loss
                    else:
                        metrics['likely_overfitting'] = slope < 0 and p_value < 0.1  # Decreasing AUC/accuracy

            logger.info(f"{model_type.upper()} Model Convergence:")
            logger.info(f"  Final Score: {metrics['final_score']:.4f}")
            logger.info(f"  Best Score: {metrics['best_score']:.4f}")
            logger.info(f"  Best Iteration: {metrics['best_iteration']}/{metrics['total_iterations']}")
            if 'likely_overfitting' in metrics:
                logger.info(f"  Likely Overfitting: {metrics['likely_overfitting']}")

    # Feature importance stability check
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        metrics['feature_importance_concentration'] = np.sum(importances[:5]) / np.sum(importances)  # Top 5 features
        metrics['zero_importance_features'] = np.sum(importances == 0)

        logger.info(f"  Top 5 features account for {metrics['feature_importance_concentration']:.1%} of importance")
        logger.info(f"  Features with zero importance: {metrics['zero_importance_features']}")

    return metrics


def calculate_advanced_metrics(backtest_results: pd.DataFrame, config: Dict) -> Dict:
    """
    Calculate advanced statistical metrics for model validation.

    Args:
        backtest_results: DataFrame with backtest results
        config: Configuration dictionary

    Returns:
        Dict with advanced metrics
    """
    logger.info("Calculating advanced validation metrics...")

    metrics = {}
    alpha = config['production_model']['var_model']['alpha']

    # VaR Model Metrics
    violations = backtest_results['true_pnl'] < backtest_results['pred_var']
    violation_rate = violations.mean()

    # Unconditional Coverage Test (Kupiec Test)
    n = len(backtest_results)
    violations_count = violations.sum()
    expected_violations = n * alpha

    if violations_count > 0 and violations_count < n:
        lr_uc = -2 * np.log((alpha**violations_count * (1-alpha)**(n-violations_count)) /
                           ((violations_count/n)**violations_count * (1-violations_count/n)**(n-violations_count)))
        p_value_uc = 1 - stats.chi2.cdf(lr_uc, df=1)
    else:
        lr_uc = np.inf
        p_value_uc = 0.0

    metrics['var_violation_rate'] = violation_rate
    metrics['var_kupiec_lr'] = lr_uc
    metrics['var_kupiec_p_value'] = p_value_uc
    metrics['var_test_passed'] = p_value_uc > 0.05  # Accept null hypothesis if p > 0.05

    # Conditional Coverage Test (independence of violations)
    violations_shifted = violations.shift(1).fillna(False)
    contingency_table = pd.crosstab(violations_shifted, violations)

    if contingency_table.shape == (2, 2):
        chi2_stat, p_value_ind, _, _ = stats.chi2_contingency(contingency_table)
        metrics['var_independence_chi2'] = chi2_stat
        metrics['var_independence_p_value'] = p_value_ind
        metrics['var_independence_passed'] = p_value_ind > 0.05
    else:
        metrics['var_independence_passed'] = True  # Not enough variation to test

    # VaR Economic Metrics
    avg_violation_size = backtest_results.loc[violations, 'true_pnl'].mean() - backtest_results.loc[violations, 'pred_var'].mean()
    metrics['var_avg_violation_size'] = avg_violation_size

    # Loss Model Metrics
    y_true = backtest_results['true_large_loss']
    y_pred_proba = backtest_results['pred_loss_proba']

    # AUC and Precision-Recall
    auc_score = roc_auc_score(y_true, y_pred_proba)
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)

    metrics['loss_auc'] = auc_score
    metrics['loss_pr_auc'] = pr_auc

    # Calibration test (Hosmer-Lemeshow style)
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    observed_freq = []
    expected_freq = []

    for i in range(n_bins):
        mask = (y_pred_proba >= bin_edges[i]) & (y_pred_proba < bin_edges[i+1])
        if i == n_bins - 1:  # Include upper bound for last bin
            mask = (y_pred_proba >= bin_edges[i]) & (y_pred_proba <= bin_edges[i+1])

        if mask.sum() > 0:
            obs_rate = y_true[mask].mean()
            exp_rate = y_pred_proba[mask].mean()
            observed_freq.append(obs_rate)
            expected_freq.append(exp_rate)
        else:
            observed_freq.append(0)
            expected_freq.append(bin_centers[i])

    # Calculate calibration slope and intercept
    if len(observed_freq) > 1:
        slope, intercept, r_value, p_value, std_err = stats.linregress(expected_freq, observed_freq)
        metrics['loss_calibration_slope'] = slope
        metrics['loss_calibration_intercept'] = intercept
        metrics['loss_calibration_r_squared'] = r_value ** 2
        metrics['loss_well_calibrated'] = abs(slope - 1.0) < 0.2 and abs(intercept) < 0.05

    # Stability metrics across folds
    fold_metrics = []
    for fold in backtest_results['fold'].unique():
        fold_data = backtest_results[backtest_results['fold'] == fold]
        fold_auc = roc_auc_score(fold_data['true_large_loss'], fold_data['pred_loss_proba'])
        fold_violation_rate = (fold_data['true_pnl'] < fold_data['pred_var']).mean()
        fold_metrics.append({'fold': fold, 'auc': fold_auc, 'violation_rate': fold_violation_rate})

    fold_df = pd.DataFrame(fold_metrics)
    metrics['auc_stability_std'] = fold_df['auc'].std()
    metrics['violation_rate_stability_std'] = fold_df['violation_rate'].std()
    metrics['auc_stability_cv'] = fold_df['auc'].std() / fold_df['auc'].mean()

    # Log key metrics
    logger.info("Advanced Validation Results:")
    logger.info(f"  VaR Kupiec Test: {'PASS' if metrics['var_test_passed'] else 'FAIL'} (p={metrics['var_kupiec_p_value']:.3f})")
    logger.info(f"  VaR Independence Test: {'PASS' if metrics['var_independence_passed'] else 'FAIL'}")
    logger.info(f"  Loss Model Calibration: {'PASS' if metrics.get('loss_well_calibrated', False) else 'FAIL'}")
    logger.info(f"  AUC Stability (CV): {metrics['auc_stability_cv']:.3f}")

    return metrics


def feature_selection_pipeline(X: pd.DataFrame, y: pd.Series, model_params: Dict,
                             model_type: str = 'var', max_features: int = 15) -> List[str]:
    """
    Select features using recursive feature elimination to prevent overfitting.

    Args:
        X: Feature matrix
        y: Target variable
        model_params: Model parameters
        model_type: 'var' or 'loss'
        max_features: Maximum number of features to select

    Returns:
        List of selected feature names
    """
    logger.info(f"Starting feature selection for {model_type} model...")
    logger.info(f"Input features: {X.shape[1]}, Target features: {max_features}")

    # Create base model
    if model_type == 'var':
        base_model = lgb.LGBMRegressor(**model_params)
    else:
        base_model = lgb.LGBMClassifier(**model_params)

    # Use RFE with cross-validation
    selector = RFE(
        estimator=base_model,
        n_features_to_select=max_features,
        step=1,
        verbose=1
    )

    # Fit selector
    selector.fit(X, y)

    # Get selected features
    selected_features = X.columns[selector.support_].tolist()
    feature_rankings = dict(zip(X.columns, selector.ranking_))

    logger.info(f"Selected {len(selected_features)} features:")
    for i, feature in enumerate(selected_features[:10]):  # Show top 10
        logger.info(f"  {i+1}. {feature}")

    # Log feature correlation analysis
    if len(selected_features) > 1:
        corr_matrix = X[selected_features].corr()
        high_corr_pairs = []

        for i in range(len(selected_features)):
            for j in range(i+1, len(selected_features)):
                corr = corr_matrix.iloc[i, j]
                if abs(corr) > 0.8:
                    high_corr_pairs.append((selected_features[i], selected_features[j], corr))

        if high_corr_pairs:
            logger.warning(f"Found {len(high_corr_pairs)} highly correlated feature pairs (|r| > 0.8):")
            for feat1, feat2, corr in high_corr_pairs[:5]:  # Show top 5
                logger.warning(f"  {feat1} - {feat2}: {corr:.3f}")

    return selected_features


def validate_data_quality(df: pd.DataFrame, config: Dict) -> Dict:
    """
    Comprehensive data quality validation to prevent leakage and ensure integrity.

    Args:
        df: Feature DataFrame
        config: Configuration dictionary

    Returns:
        Dict with validation results
    """
    logger.info("Performing data quality validation...")

    validation_results = {}
    issues = []

    # Check for future information leakage
    feature_cols = [col for col in df.columns if col not in [
        'account_id', 'trade_date', 'target_pnl', 'target_large_loss',
        'daily_pnl', 'large_loss_threshold'
    ]]

    # Temporal consistency check
    df_sorted = df.sort_values(['account_id', 'trade_date'])

    # Check if target variables are properly shifted
    for account in df['account_id'].unique():
        account_data = df_sorted[df_sorted['account_id'] == account].copy()

        if len(account_data) > 1:
            # Check if targets align with next day's actual PnL
            target_vs_actual_corr = account_data['target_pnl'].corr(account_data['daily_pnl'].shift(-1))
            if abs(target_vs_actual_corr - 1.0) > 0.01:  # Should be perfect correlation
                issues.append(f"Target-actual correlation issue for trader {account}: {target_vs_actual_corr:.3f}")

    # Check for data completeness
    missing_data = df[feature_cols].isnull().sum()
    high_missing_features = missing_data[missing_data > len(df) * 0.1].index.tolist()

    if high_missing_features:
        issues.append(f"Features with >10% missing data: {high_missing_features}")

    # Check for extreme outliers (beyond 5 standard deviations)
    for col in feature_cols:
        if df[col].dtype in ['int64', 'float64']:
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            extreme_outliers = (z_scores > 5).sum()
            if extreme_outliers > len(df) * 0.01:  # More than 1% extreme outliers
                issues.append(f"Feature {col} has {extreme_outliers} extreme outliers (>5 sigma)")

    # Check for feature-target leakage using correlation
    target_correlations = {}
    for col in feature_cols:
        if df[col].dtype in ['int64', 'float64']:
            corr_var = df[col].corr(df['target_pnl'])
            corr_loss = df[col].corr(df['target_large_loss'])

            # Flag suspiciously high correlations
            if abs(corr_var) > 0.8:
                issues.append(f"Suspiciously high correlation between {col} and target_pnl: {corr_var:.3f}")
            if abs(corr_loss) > 0.8:
                issues.append(f"Suspiciously high correlation between {col} and target_large_loss: {corr_loss:.3f}")

            target_correlations[col] = {'var_corr': corr_var, 'loss_corr': corr_loss}

    # Check panel balance
    panel_balance = df.groupby('account_id').size()
    min_obs = panel_balance.min()
    max_obs = panel_balance.max()
    balance_ratio = min_obs / max_obs if max_obs > 0 else 0

    if balance_ratio < 0.5:
        issues.append(f"Unbalanced panel: min {min_obs} vs max {max_obs} observations per trader")

    # Store results
    validation_results['issues'] = issues
    validation_results['n_issues'] = len(issues)
    validation_results['missing_data_summary'] = missing_data.to_dict()
    validation_results['target_correlations'] = target_correlations
    validation_results['panel_balance_ratio'] = balance_ratio
    validation_results['panel_observations'] = panel_balance.to_dict()

    # Log results
    if issues:
        logger.warning(f"Found {len(issues)} data quality issues:")
        for issue in issues[:10]:  # Show first 10 issues
            logger.warning(f"  {issue}")
    else:
        logger.info("Data quality validation passed - no issues found")

    return validation_results


class PurgedTimeSeriesSplit:
    """
    Purged Time Series Cross-Validation for financial data.

    This implementation adds a purging period after the training set to prevent
    leakage from overlapping data or features that look into the future.
    """

    def __init__(self, n_splits=5, test_size=None, purge_days=2, embargo_days=1):
        """
        Args:
            n_splits: Number of splits
            test_size: Size of test set (if None, uses equal splits)
            purge_days: Days to purge after training set (removes overlap)
            embargo_days: Additional embargo period to prevent leakage
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.purge_days = purge_days
        self.embargo_days = embargo_days

    def split(self, X, y=None, groups=None):
        """
        Generate indices for purged time series splits.

        Args:
            X: Feature matrix or date index
            y: Target (not used)
            groups: Not used

        Yields:
            train_idx, test_idx: Arrays of training and test indices
        """
        if isinstance(X, pd.DataFrame):
            dates = X.index if isinstance(X.index, pd.DatetimeIndex) else X['trade_date']
        else:
            dates = X  # Assume X is array of dates

        unique_dates = sorted(dates.unique())
        n_dates = len(unique_dates)

        if self.test_size is None:
            test_size = n_dates // (self.n_splits + 1)
        else:
            test_size = self.test_size

        purge_size = self.purge_days
        embargo_size = self.embargo_days

        for i in range(self.n_splits):
            # Calculate test period
            test_start_idx = (i + 1) * test_size + i * (purge_size + embargo_size)
            test_end_idx = test_start_idx + test_size

            if test_end_idx > n_dates:
                break

            # Training period ends before purge period
            train_end_idx = test_start_idx - purge_size - embargo_size

            if train_end_idx <= 0:
                continue

            # Get date ranges
            train_dates = unique_dates[:train_end_idx]
            test_dates = unique_dates[test_start_idx:test_end_idx]

            # Convert to indices
            if isinstance(X, pd.DataFrame):
                train_mask = X['trade_date'].isin(train_dates) if 'trade_date' in X.columns else X.index.isin(train_dates)
                test_mask = X['trade_date'].isin(test_dates) if 'trade_date' in X.columns else X.index.isin(test_dates)
                train_idx = np.where(train_mask)[0]
                test_idx = np.where(test_mask)[0]
            else:
                train_idx = np.where(np.isin(dates, train_dates))[0]
                test_idx = np.where(np.isin(dates, test_dates))[0]

            yield train_idx, test_idx


def run_purged_walk_forward_backtest(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Enhanced walk-forward backtesting with purged cross-validation.

    Args:
        df: Feature DataFrame
        config: Configuration dictionary

    Returns:
        pd.DataFrame: Backtest results with predictions
    """
    logger.info("Starting purged walk-forward backtesting...")

    # Data quality validation first
    validation_results = validate_data_quality(df, config)
    if validation_results['n_issues'] > 0:
        logger.warning(f"Data quality issues detected: {validation_results['n_issues']}")

    # Prepare features and targets
    feature_cols = [col for col in df.columns if col not in [
        'account_id', 'trade_date', 'target_pnl', 'target_large_loss',
        'daily_pnl', 'large_loss_threshold'
    ]]

    X = df[feature_cols]
    y_var = df['target_pnl']
    y_loss = df['target_large_loss']

    # Initialize Purged TimeSeriesSplit
    purge_days = config.get('model_quality', {}).get('purge_days', 2)
    embargo_days = config.get('model_quality', {}).get('embargo_days', 1)

    pctscv = PurgedTimeSeriesSplit(
        n_splits=config['backtesting']['n_splits'],
        test_size=config['backtesting']['test_days'],
        purge_days=purge_days,
        embargo_days=embargo_days
    )

    # Storage for out-of-sample predictions
    oos_predictions = []
    fold_convergence_metrics = []

    # Get unique dates for proper splitting
    unique_dates = df['trade_date'].unique()
    date_to_idx = {date: idx for idx, date in enumerate(unique_dates)}
    df['date_idx'] = df['trade_date'].map(date_to_idx)

    # Perform purged walk-forward validation
    fold = 0
    for train_idx, test_idx in pctscv.split(df):
        fold += 1
        logger.info(f"Processing purged fold {fold}/{config['backtesting']['n_splits']}")

        # Split data using purged indices
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_var_train, y_var_test = y_var.iloc[train_idx], y_var.iloc[test_idx]
        y_loss_train, y_loss_test = y_loss.iloc[train_idx], y_loss.iloc[test_idx]

        # Log fold statistics
        train_dates = df.iloc[train_idx]['trade_date']
        test_dates = df.iloc[test_idx]['trade_date']
        logger.info(f"  Train period: {train_dates.min()} to {train_dates.max()} ({len(train_idx)} samples)")
        logger.info(f"  Test period: {test_dates.min()} to {test_dates.max()} ({len(test_idx)} samples)")

        # Check for sufficient data
        if len(train_idx) < 100 or len(test_idx) < 10:
            logger.warning(f"Insufficient data in fold {fold}, skipping")
            continue

        # Train VaR model with validation split
        var_params = config['production_model']['var_model'].copy()
        var_model = lgb.LGBMRegressor(**var_params)

        # Create internal validation split
        val_split = int(len(X_train) * 0.8)
        X_train_fold, X_val_fold = X_train.iloc[:val_split], X_train.iloc[val_split:]
        y_var_train_fold, y_var_val_fold = y_var_train.iloc[:val_split], y_var_train.iloc[val_split:]

        var_model.fit(
            X_train_fold, y_var_train_fold,
            eval_set=[(X_val_fold, y_var_val_fold)],
            callbacks=[lgb.early_stopping(config.get('model_quality', {}).get('early_stopping_rounds', 20)),
                       lgb.log_evaluation(0)]
        )

        # Validate convergence
        var_convergence = validate_model_convergence(var_model, 'var')

        # Train loss model with validation split
        loss_params = config['production_model']['loss_model'].copy()
        loss_model = lgb.LGBMClassifier(**loss_params)

        y_loss_train_fold, y_loss_val_fold = y_loss_train.iloc[:val_split], y_loss_train.iloc[val_split:]

        loss_model.fit(
            X_train_fold, y_loss_train_fold,
            eval_set=[(X_val_fold, y_loss_val_fold)],
            callbacks=[lgb.early_stopping(config.get('model_quality', {}).get('early_stopping_rounds', 20)),
                       lgb.log_evaluation(0)]
        )

        # Validate convergence
        loss_convergence = validate_model_convergence(loss_model, 'loss')

        # Store convergence metrics
        fold_convergence_metrics.append({
            'fold': fold,
            'var_convergence': var_convergence,
            'loss_convergence': loss_convergence
        })

        # Make predictions
        var_pred = var_model.predict(X_test)
        loss_pred_proba = loss_model.predict_proba(X_test)[:, 1]

        # Store results with additional metadata
        fold_results = pd.DataFrame({
            'fold': fold,
            'account_id': df.iloc[test_idx]['account_id'].values,
            'trade_date': df.iloc[test_idx]['trade_date'].values,
            'true_pnl': y_var_test.values,
            'pred_var': var_pred,
            'true_large_loss': y_loss_test.values,
            'pred_loss_proba': loss_pred_proba,
            'train_samples': len(train_idx),
            'test_samples': len(test_idx)
        })

        oos_predictions.append(fold_results)

    # Combine all out-of-sample predictions
    if not oos_predictions:
        logger.error("No valid folds generated - check data quality and configuration")
        raise ValueError("Purged backtesting failed - no valid folds")

    backtest_results = pd.concat(oos_predictions, ignore_index=True)

    # Calculate enhanced performance metrics
    logger.info("Calculating enhanced backtest performance metrics...")

    # Basic metrics
    alpha = config['production_model']['var_model']['alpha']
    violations = (backtest_results['true_pnl'] < backtest_results['pred_var']).mean()
    logger.info(f"VaR violation rate: {violations:.4f} (expected: {alpha})")

    auc = roc_auc_score(backtest_results['true_large_loss'], backtest_results['pred_loss_proba'])
    logger.info(f"Large loss prediction AUC: {auc:.4f}")

    # Advanced statistical validation
    advanced_metrics = calculate_advanced_metrics(backtest_results, config)

    # Add convergence analysis
    convergence_summary = {
        'var_overfitting_folds': sum(1 for m in fold_convergence_metrics
                                   if m['var_convergence'].get('likely_overfitting', False)),
        'loss_overfitting_folds': sum(1 for m in fold_convergence_metrics
                                    if m['loss_convergence'].get('likely_overfitting', False)),
        'avg_var_early_stopping': np.mean([m['var_convergence'].get('best_iteration', 0)
                                         for m in fold_convergence_metrics]),
        'avg_loss_early_stopping': np.mean([m['loss_convergence'].get('best_iteration', 0)
                                          for m in fold_convergence_metrics])
    }

    advanced_metrics.update(convergence_summary)

    # Save enhanced backtest results
    results_path = os.path.join(config['paths']['model_dir'], 'purged_backtest_results.csv')
    backtest_results.to_csv(results_path, index=False)
    logger.info(f"Purged backtest results saved to {results_path}")

    # Save enhanced validation metrics
    validation_path = os.path.join(config['paths']['model_dir'], 'purged_validation_metrics.json')
    import json
    with open(validation_path, 'w') as f:
        json.dump(advanced_metrics, f, indent=2, default=str)
    logger.info(f"Enhanced validation metrics saved to {validation_path}")

    # Save convergence details
    convergence_path = os.path.join(config['paths']['model_dir'], 'fold_convergence_metrics.json')
    with open(convergence_path, 'w') as f:
        json.dump(fold_convergence_metrics, f, indent=2, default=str)

    # Generate advanced risk metrics if enabled
    if config.get('advanced_metrics', {}).get('enable_cvar', True):
        logger.info("Generating advanced risk metrics...")

        # We need models for advanced metrics - retrain on full dataset for this analysis
        feature_cols = [col for col in df.columns if col not in [
            'account_id', 'trade_date', 'target_pnl', 'target_large_loss',
            'daily_pnl', 'large_loss_threshold'
        ]]

        X_full = df[feature_cols]
        y_var_full = df['target_pnl']
        y_loss_full = df['target_large_loss']

        # Quick model training for advanced metrics
        var_params = config['production_model']['var_model'].copy()
        loss_params = config['production_model']['loss_model'].copy()

        temp_var_model = lgb.LGBMRegressor(**var_params)
        temp_loss_model = lgb.LGBMClassifier(**loss_params)

        temp_var_model.fit(X_full, y_var_full)
        temp_loss_model.fit(X_full, y_loss_full)

        # Generate advanced risk report
        try:
            advanced_report = generate_advanced_risk_report(
                backtest_results, df, config, temp_var_model, temp_loss_model
            )

            # Save advanced metrics
            advanced_path = os.path.join(config['paths']['model_dir'], 'advanced_risk_metrics.json')
            import json
            with open(advanced_path, 'w') as f:
                json.dump(advanced_report, f, indent=2, default=str)
            logger.info(f"Advanced risk metrics saved to {advanced_path}")

        except Exception as e:
            logger.warning(f"Failed to generate advanced risk metrics: {str(e)}")

    logger.info("Purged walk-forward backtesting completed successfully")
    logger.info(f"Model convergence: VaR overfitting in {convergence_summary['var_overfitting_folds']} folds, "
               f"Loss overfitting in {convergence_summary['loss_overfitting_folds']} folds")

    return backtest_results
