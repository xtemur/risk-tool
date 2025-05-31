#!/usr/bin/env python
"""
Model Training Verification Script
Run this after retraining to verify models are working correctly
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append('src')

from database import Database
from feature_engineer import FeatureEngineer
from model_trainer import ModelTrainer

def verify_model_training():
    """Verify that model training worked correctly"""

    print("="*60)
    print("MODEL TRAINING VERIFICATION")
    print("="*60)

    # Initialize components
    db = Database()
    feature_engineer = FeatureEngineer()
    model_trainer = ModelTrainer()

    # Get all models
    models = model_trainer.get_all_models()
    print(f"\n1. MODELS LOADED: {len(models)}")
    print("-" * 30)

    if len(models) == 0:
        print("❌ No models found! Run: python scripts/train_models.py")
        return False

    # Check model quality
    print("\n2. MODEL QUALITY CHECK")
    print("-" * 30)

    model_stats = []
    good_models = 0

    for account_id, model_data in models.items():
        val_metrics = model_data.get('validation_metrics', {})
        test_metrics = model_data.get('test_metrics', {})

        rmse = val_metrics.get('rmse', 0)
        mae = val_metrics.get('mae', 0)
        test_rmse = test_metrics.get('test_rmse', 0)
        threshold = model_data.get('threshold', 0)

        # Check if model is good
        is_good = rmse > 0.1 and rmse < 1000 and threshold != 0
        if is_good:
            good_models += 1
            status = "✅"
        else:
            status = "❌"

        model_stats.append({
            'account_id': account_id,
            'rmse': rmse,
            'mae': mae,
            'test_rmse': test_rmse,
            'threshold': threshold,
            'good': is_good
        })

        print(f"  {status} {account_id}: RMSE={rmse:.2f}, MAE={mae:.2f}, Threshold={threshold:.3f}")

    print(f"\nGood models: {good_models}/{len(models)} ({good_models/len(models)*100:.1f}%)")

    # Test predictions on sample trader
    print("\n3. PREDICTION TEST")
    print("-" * 30)

    # Get a trader with good model
    good_model_accounts = [m['account_id'] for m in model_stats if m['good']]

    if not good_model_accounts:
        print("❌ No good models available for testing")
        return False

    test_account = good_model_accounts[0]
    print(f"Testing predictions for: {test_account}")

    # Get recent data
    totals_df, fills_df = db.get_trader_data(test_account)
    if totals_df.empty:
        print("❌ No data available for testing")
        return False

    # Create features
    features_df = feature_engineer.create_features(totals_df, fills_df)
    print(f"Features available: {len(features_df)} days")

    # Load model and make predictions
    model_data = models[test_account]
    model = model_data['model']
    feature_columns = model_data['feature_columns']

    # Test on last 5 days
    if len(features_df) >= 5:
        test_features = features_df[feature_columns].tail(5).values
        predictions = model.predict(test_features, num_iteration=model.best_iteration)

        print(f"Sample predictions:")
        for i, pred in enumerate(predictions):
            actual = features_df['target'].iloc[-(5-i)] if 'target' in features_df.columns else 'N/A'
            print(f"  Day {i+1}: Predicted=${pred:.2f}, Actual=${actual}")

        # Check prediction quality
        pred_std = np.std(predictions)
        pred_mean = np.mean(predictions)

        print(f"\nPrediction statistics:")
        print(f"  Mean: ${pred_mean:.2f}")
        print(f"  Std: ${pred_std:.2f}")
        print(f"  Range: ${np.min(predictions):.2f} to ${np.max(predictions):.2f}")

        # Quality checks
        if pred_std < 0.01:
            print("  ⚠️ WARNING: Predictions have very low variance")
        elif abs(pred_mean) > 1000:
            print("  ⚠️ WARNING: Predictions seem unrealistically large")
        else:
            print("  ✅ Predictions look reasonable")

    # Test risk scoring
    print("\n4. RISK SCORING TEST")
    print("-" * 30)

    from predictor import RiskPredictor

    try:
        predictor = RiskPredictor()

        # Generate predictions for all traders
        predictions = predictor.predict_all_traders()
        print(f"Generated predictions for: {len(predictions)} traders")

        if predictions:
            # Check risk distribution
            risk_counts = {}
            prediction_values = []

            for pred in predictions:
                risk_level = pred['risk_level']
                risk_counts[risk_level] = risk_counts.get(risk_level, 0) + 1
                prediction_values.append(pred['predicted_pnl'])

            print(f"Risk distribution:")
            for risk, count in risk_counts.items():
                print(f"  {risk}: {count}")

            print(f"Prediction summary:")
            print(f"  Mean: ${np.mean(prediction_values):.2f}")
            print(f"  Std: ${np.std(prediction_values):.2f}")
            print(f"  Range: ${np.min(prediction_values):.2f} to ${np.max(prediction_values):.2f}")

            # Quality checks
            has_variety = len(set(pred['risk_level'] for pred in predictions)) > 1
            has_nonzero = any(abs(pred['predicted_pnl']) > 0.01 for pred in predictions)

            if not has_variety:
                print("  ⚠️ WARNING: All predictions have same risk level")
            elif not has_nonzero:
                print("  ⚠️ WARNING: All predictions are near zero")
            else:
                print("  ✅ Risk scoring working properly")

    except Exception as e:
        print(f"❌ Error in risk prediction: {str(e)}")
        return False

    # Overall assessment
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)

    success_criteria = [
        (good_models > 0, f"Good models available: {good_models}"),
        (len(models) > 0, f"Models loaded: {len(models)}"),
        (len(predictions) > 0, f"Predictions generated: {len(predictions)}"),
    ]

    all_good = all(criteria[0] for criteria in success_criteria)

    print("\nStatus checks:")
    for passed, description in success_criteria:
        status = "✅" if passed else "❌"
        print(f"  {status} {description}")

    if all_good:
        print(f"\n🎉 SUCCESS: Models are working correctly!")
        print(f"   Ready to run daily predictions")
        return True
    else:
        print(f"\n❌ ISSUES FOUND: Check the warnings above")
        print(f"   May need to retrain models or check data")
        return False

if __name__ == "__main__":
    success = verify_model_training()
    exit(0 if success else 1)
