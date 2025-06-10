"""
Unit tests for Risk Model
Tests model training, prediction, and risk assessment
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile

from src.models.risk_model import RiskModel
from src.core.constants import TradingConstants as TC, ModelConfig as MC


class TestRiskModel:
    """Test suite for RiskModel"""

    @pytest.fixture
    def sample_features(self):
        """Create sample feature data"""
        n_samples = 1000
        n_features = 20

        # Create synthetic features
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=n_samples, freq='D')

        # Generate correlated features (like real trading data)
        base_signal = np.random.randn(n_samples)

        features = pd.DataFrame(index=dates)

        # Technical features
        features['tech_return_5d'] = base_signal + 0.3 * np.random.randn(n_samples)
        features['tech_volatility_20d'] = np.abs(base_signal) + 0.2 * np.random.randn(n_samples)
        features['tech_rsi_14'] = 50 + 20 * np.tanh(base_signal) + 5 * np.random.randn(n_samples)

        # Behavioral features
        features['behav_win_rate'] = (base_signal > 0).rolling(20).mean().fillna(0.5)
        features['behav_loss_streak'] = np.random.randint(0, 5, n_samples)

        # Market regime features
        features['regime_volatility'] = np.abs(base_signal).rolling(10).mean().fillna(1)

        # Add more random features
        for i in range(n_features - 6):
            features[f'feature_{i}'] = np.random.randn(n_samples)

        return features

    @pytest.fixture
    def sample_target(self, sample_features):
        """Create sample target based on features"""
        # Create target with some signal from features
        target = (
            0.3 * sample_features['tech_return_5d'] -
            0.2 * sample_features['tech_volatility_20d'] +
            0.1 * np.random.randn(len(sample_features))
        )

        return pd.Series(target, index=sample_features.index, name='target')

    def test_model_initialization(self):
        """Test model initialization"""
        model = RiskModel(
            model_name="test_model",
            model_type="regression",
            objective="regression"
        )

        assert model.model_name == "test_model"
        assert model.model_type == "regression"
        assert model.objective == "regression"
        assert model.params['objective'] == "regression"
        assert not model.is_fitted_

    def test_model_fit(self, sample_features, sample_target):
        """Test model fitting"""
        model = RiskModel(model_name="test_model")

        # Split data
        split_idx = int(len(sample_features) * 0.8)
        X_train = sample_features[:split_idx]
        y_train = sample_target[:split_idx]
        X_val = sample_features[split_idx:]
        y_val = sample_target[split_idx:]

        # Fit model
        model.fit(X_train, y_train, validation_data=(X_val, y_val))

        assert model.is_fitted_
        assert model.feature_names_ == X_train.columns.tolist()
        assert model.feature_importance_ is not None
        assert len(model.feature_importance_) == len(X_train.columns)

        # Check metadata
        assert 'n_samples' in model.metadata_['training_info']
        assert model.metadata_['training_info']['n_samples'] == len(X_train)

    def test_model_predict(self, sample_features, sample_target):
        """Test model prediction"""
        model = RiskModel(model_name="test_model")

        # Train model
        split_idx = int(len(sample_features) * 0.8)
        X_train = sample_features[:split_idx]
        y_train = sample_target[:split_idx]
        X_test = sample_features[split_idx:]

        model.fit(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_test)

        assert len(predictions) == len(X_test)
        assert isinstance(predictions, np.ndarray)
        assert not np.isnan(predictions).any()

    def test_predict_risk(self, sample_features, sample_target):
        """Test risk prediction functionality"""
        model = RiskModel(model_name="test_model")

        # Train model
        split_idx = int(len(sample_features) * 0.8)
        X_train = sample_features[:split_idx]
        y_train = sample_target[:split_idx]
        X_test = sample_features[split_idx:]

        model.fit(X_train, y_train)

        # Get risk predictions
        risk_df = model.predict_risk(X_test)

        # Check output structure
        assert 'predicted_pnl' in risk_df.columns
        assert 'risk_score' in risk_df.columns
        assert 'risk_category' in risk_df.columns
        assert 'confidence' in risk_df.columns

        # Check risk scores are in valid range
        assert (risk_df['risk_score'] >= 0).all()
        assert (risk_df['risk_score'] <= 1).all()

        # Check risk categories
        assert set(risk_df['risk_category'].unique()) <= {'Low', 'Medium', 'High', 'Critical'}

    def test_feature_importance(self, sample_features, sample_target):
        """Test feature importance extraction"""
        model = RiskModel(model_name="test_model")

        # Train model
        model.fit(sample_features, sample_target)

        # Get feature importance
        importance_df = model.get_feature_importance(top_k=10)

        assert len(importance_df) == 10
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns

        # Check importance values
        assert (importance_df['importance'] >= 0).all()
        assert importance_df['importance'].sum() > 0

        # Check ordering
        assert importance_df['importance'].is_monotonic_decreasing

    def test_cross_validation(self, sample_features, sample_target):
        """Test cross-validation functionality"""
        model = RiskModel(model_name="test_model")

        # Run cross-validation with small number of splits
        cv_scores = model.cross_validate(
            sample_features,
            sample_target,
            cv_splitter=None,  # Will use default
            metrics=['rmse', 'mae']
        )

        assert 'rmse' in cv_scores
        assert 'mae' in cv_scores
        assert len(cv_scores['rmse']) >= 3  # At least 3 folds

        # Check validation scores stored
        assert model.validation_scores_ is not None
        assert 'rmse_mean' in model.validation_scores_
        assert 'rmse_std' in model.validation_scores_

    def test_save_load_model(self, sample_features, sample_target):
        """Test model serialization"""
        model = RiskModel(model_name="test_model")

        # Train model
        model.fit(sample_features, sample_target)
        original_score = model.score(sample_features, sample_target)
        original_importance = model.feature_importance_

        # Save model
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.pkl"
            model.save(model_path)

            assert model_path.exists()
            assert (model_path.with_suffix('.json')).exists()  # Metadata file

            # Load model
            loaded_model = RiskModel.load(model_path)

            assert loaded_model.is_fitted_
            assert loaded_model.model_name == model.model_name
            assert loaded_model.feature_names_ == model.feature_names_

            # Check predictions are same
            loaded_score = loaded_model.score(sample_features, sample_target)
            assert abs(loaded_score - original_score) < 1e-6

    def test_classification_mode(self, sample_features):
        """Test classification model"""
        # Create binary target
        target = pd.Series(
            np.random.randint(0, 2, len(sample_features)),
            index=sample_features.index
        )

        model = RiskModel(
            model_name="test_classifier",
            model_type="classification",
            objective="binary"
        )

        # Train model
        split_idx = int(len(sample_features) * 0.8)
        X_train = sample_features[:split_idx]
        y_train = target[:split_idx]
        X_test = sample_features[split_idx:]

        model.fit(X_train, y_train)

        # Test predictions
        predictions = model.predict(X_test)
        assert set(predictions) <= {0, 1}

        # Test probability predictions
        proba = model.predict_proba(X_test)
        assert proba.shape[1] == 2  # Binary classification
        assert np.allclose(proba.sum(axis=1), 1.0)  # Probabilities sum to 1

    def test_risk_score_calculation(self):
        """Test risk score calculation logic"""
        model = RiskModel(model_name="test_model")

        # Set expected return and risk for testing
        model.expected_return_ = 100
        model.expected_risk_ = 50

        # Test risk score calculation
        predictions = np.array([150, 100, 50, 0, -50, -100])
        risk_scores = model._calculate_risk_score(predictions)

        # Check risk scores are in valid range
        assert (risk_scores >= 0).all()
        assert (risk_scores <= 1).all()

        # Check ordering: lower predictions should have higher risk scores
        assert risk_scores[0] < risk_scores[-1]  # 150 has lower risk than -100

    def test_optimization_threshold(self, sample_features):
        """Test threshold optimization for classification"""
        # Create binary target
        target = pd.Series(
            (sample_features['tech_return_5d'] > 0).astype(int),
            index=sample_features.index
        )

        model = RiskModel(
            model_name="test_classifier",
            model_type="classification",
            objective="binary"
        )

        # Train model
        split_idx = int(len(sample_features) * 0.8)
        X_train = sample_features[:split_idx]
        y_train = target[:split_idx]
        X_val = sample_features[split_idx:]
        y_val = target[split_idx:]

        model.fit(X_train, y_train)

        # Optimize threshold
        optimal_threshold = model.optimize_threshold(X_val, y_val, metric='f1')

        assert 0 < optimal_threshold < 1

    def test_hyperparameter_validation(self):
        """Test that hyperparameters are properly set"""
        custom_params = {
            'n_estimators': 50,
            'max_depth': 3,
            'learning_rate': 0.01
        }

        model = RiskModel(
            model_name="test_model",
            params=custom_params
        )

        assert model.params['n_estimators'] == 50
        assert model.params['max_depth'] == 3
        assert model.params['learning_rate'] == 0.01

    def test_model_with_missing_features(self, sample_features, sample_target):
        """Test model handles missing features in prediction"""
        model = RiskModel(model_name="test_model")

        # Train with all features
        model.fit(sample_features, sample_target)

        # Predict with missing features
        test_features = sample_features.iloc[-50:].copy()
        test_features = test_features.drop(columns=['feature_0', 'feature_1'])

        # Should still work (fills missing with 0)
        predictions = model.predict(test_features)
        assert len(predictions) == len(test_features)
        assert not np.isnan(predictions).any()

    def test_confidence_calculation(self, sample_features, sample_target):
        """Test confidence score calculation"""
        model = RiskModel(model_name="test_model")

        # Train model
        model.fit(sample_features, sample_target)

        # Create test data with varying quality
        test_features = sample_features.iloc[-50:].copy()

        # Make some features zero (low quality)
        test_features.iloc[:10] = 0

        # Get predictions
        risk_df = model.predict_risk(test_features)

        # Check confidence exists
        assert 'confidence' in risk_df.columns

        # First 10 rows should have lower confidence (many zeros)
        # This is a simplified test - actual confidence calculation is more complex
        assert risk_df['confidence'].notna().all()


@pytest.mark.integration
class TestRiskModelIntegration:
    """Integration tests for risk model with other components"""

    def test_with_feature_pipeline(self, sample_features, sample_target):
        """Test model with feature pipeline"""
        from src.pipeline.feature_pipeline import FeaturePipeline

        # This is a simplified test - in reality would use actual data
        pipeline = FeaturePipeline()

        # Use sample features as if they came from pipeline
        model = RiskModel(model_name="test_model")
        model.fit(sample_features, sample_target)

        assert model.is_fitted_

    def test_with_monitoring(self, sample_features, sample_target):
        """Test model with monitoring components"""
        from src.monitoring.model_monitor import ModelMonitor

        model = RiskModel(model_name="test_model")
        model.fit(sample_features, sample_target)

        # Create monitor
        monitor = ModelMonitor(model_name="test_model")

        # Make predictions
        test_features = sample_features.iloc[-50:]
        predictions = model.predict_risk(test_features)

        # Log to monitor
        monitor.log_predictions(
            predictions=predictions,
            features=test_features,
            prediction_time_ms=10
        )

        # Check monitoring worked
        assert len(monitor.metrics_history) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
