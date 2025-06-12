"""
Feature Pipeline - Orchestrates feature generation
Simplified and fixed for actual usage
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path
import joblib

from src.features.base_features import BaseFeatures
from src.features.technical_features import TechnicalFeatures
from src.features.behavioral_features import BehavioralFeatures

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """
    Orchestrates feature generation from multiple feature classes
    """

    def __init__(self,
                 feature_generators: Optional[List[BaseFeatures]] = None,
                 cache_dir: Optional[str] = None):
        """
        Initialize feature pipeline

        Args:
            feature_generators: List of feature generators
            cache_dir: Directory for caching features (optional)
        """
        self.feature_generators = feature_generators or self._get_default_generators()
        self.cache_dir = Path(cache_dir) if cache_dir else None

        if self.cache_dir:
            self.cache_dir.mkdir(exist_ok=True, parents=True)

        # Feature metadata
        self.feature_names_ = []
        self.feature_importance_ = {}

    def _get_default_generators(self) -> List[BaseFeatures]:
        """Get default feature generators"""
        return [
            TechnicalFeatures(),
            BehavioralFeatures()
        ]

    def generate_features(self,
                         daily_summary: pd.DataFrame,
                         fills: Optional[pd.DataFrame] = None,
                         account_id: Optional[str] = None) -> pd.DataFrame:
        """
        Generate all features from input data

        Args:
            daily_summary: Daily summary data from database
            fills: Fills data (optional)
            account_id: Account ID for caching

        Returns:
            DataFrame with all features
        """
        if daily_summary.empty:
            logger.warning("Empty daily summary data")
            return pd.DataFrame()

        # Check cache if enabled
        if self.cache_dir and account_id:
            cached = self._load_from_cache(account_id)
            if cached is not None:
                return cached

        all_features = []

        # Generate features from each generator
        for generator in self.feature_generators:
            generator_name = generator.__class__.__name__
            logger.info(f"Generating features with {generator_name}")

            try:
                features = generator.create_features(daily_summary, fills)

                if not features.empty:
                    all_features.append(features)
                    logger.info(f"Generated {len(features.columns)-1} features from {generator_name}")
                else:
                    logger.warning(f"No features generated from {generator_name}")

            except Exception as e:
                logger.error(f"Error in {generator_name}: {str(e)}", exc_info=True)
                continue

        if not all_features:
            logger.warning("No features generated")
            return pd.DataFrame()

        # Merge all features on date
        result = all_features[0]
        for features in all_features[1:]:
            result = pd.merge(result, features, on='date', how='outer')

        # Sort by date
        result = result.sort_values('date')

        # Store feature names
        self.feature_names_ = [col for col in result.columns if col != 'date']

        # Basic validation
        result = self._validate_features(result)

        # Cache if enabled
        if self.cache_dir and account_id:
            self._save_to_cache(result, account_id)

        logger.info(f"Generated {len(self.feature_names_)} total features")

        return result

    def _validate_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Basic feature validation"""

        # Remove features with too many missing values
        missing_pct = features.isnull().sum() / len(features)
        high_missing = missing_pct[missing_pct > 0.5]

        if len(high_missing) > 0:
            logger.warning(f"Dropping {len(high_missing)} features with >50% missing")
            features = features.drop(columns=high_missing.index)

        # Remove constant features
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        constant_cols = [col for col in numeric_cols
                        if features[col].nunique() <= 1]

        if constant_cols:
            logger.warning(f"Dropping {len(constant_cols)} constant features")
            features = features.drop(columns=constant_cols)

        # Replace infinities
        features = features.replace([np.inf, -np.inf], np.nan)

        return features

    def select_features(self,
                       features: pd.DataFrame,
                       target: pd.Series,
                       method: str = 'importance',
                       top_k: int = 30) -> List[str]:
        """
        Select top features

        Args:
            features: Feature DataFrame
            target: Target variable
            method: Selection method ('importance' or 'correlation')
            top_k: Number of features to select

        Returns:
            List of selected feature names
        """
        feature_cols = [col for col in features.columns if col != 'date']

        if method == 'importance':
            # Use RandomForest for feature importance
            from sklearn.ensemble import RandomForestRegressor

            X = features[feature_cols].fillna(0)

            # Align target with features
            common_idx = features.index.intersection(target.index)
            X = X.loc[common_idx]
            y = target.loc[common_idx]

            rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
            rf.fit(X, y)

            importance = pd.Series(rf.feature_importances_, index=feature_cols)
            importance = importance.sort_values(ascending=False)

            self.feature_importance_ = importance.to_dict()

        elif method == 'correlation':
            # Simple correlation with target
            correlations = features[feature_cols].corrwith(target).abs()
            importance = correlations.sort_values(ascending=False)

        else:
            raise ValueError(f"Unknown method: {method}")

        selected = importance.head(top_k).index.tolist()
        logger.info(f"Selected {len(selected)} features using {method}")

        return selected

    def _save_to_cache(self, features: pd.DataFrame, account_id: str):
        """Save features to cache"""
        if not self.cache_dir:
            return

        cache_path = self.cache_dir / f"features_{account_id}.pkl"
        try:
            joblib.dump(features, cache_path)
            logger.debug(f"Cached features for {account_id}")
        except Exception as e:
            logger.warning(f"Failed to cache features: {e}")

    def _load_from_cache(self, account_id: str) -> Optional[pd.DataFrame]:
        """Load features from cache"""
        if not self.cache_dir:
            return None

        cache_path = self.cache_dir / f"features_{account_id}.pkl"
        if not cache_path.exists():
            return None

        try:
            # Check cache age (expire after 1 day)
            age_days = (pd.Timestamp.now() - pd.Timestamp(cache_path.stat().st_mtime, unit='s')).days
            if age_days > 1:
                return None

            features = joblib.load(cache_path)
            logger.debug(f"Loaded features from cache for {account_id}")
            return features

        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return None
