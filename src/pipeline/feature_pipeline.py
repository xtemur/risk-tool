"""
Feature Pipeline
Orchestrates feature generation from multiple feature classes
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import logging
import joblib
from pathlib import Path
from datetime import datetime
import json

from src.features.base_features import BaseFeatures
from src.features.technical_features import TechnicalFeatures
from src.features.behavioral_features import BehavioralFeatures
from src.features.market_regime_features import MarketRegimeFeatures
from src.pipeline.data_validator import DataValidator, ValidationResult
from src.core.constants import TradingConstants as TC

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """
    Orchestrates feature generation across multiple feature classes
    Handles caching, versioning, and feature selection
    """

    def __init__(self,
                 feature_generators: Optional[List[BaseFeatures]] = None,
                 cache_dir: str = "data/features",
                 validate_data: bool = True):
        """
        Initialize feature pipeline

        Args:
            feature_generators: List of feature generator instances
            cache_dir: Directory for caching features
            validate_data: Whether to validate input data
        """
        self.feature_generators = feature_generators or self._get_default_generators()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.validate_data = validate_data

        # Feature metadata
        self.feature_metadata_ = {}
        self.feature_importance_ = {}
        self.selected_features_ = None

        # Data validator
        self.validator = DataValidator(strict_mode=False)

    def _get_default_generators(self) -> List[BaseFeatures]:
        """Get default set of feature generators"""
        return [
            TechnicalFeatures(),
            BehavioralFeatures(),
            MarketRegimeFeatures()
        ]

    def generate_features(self,
                         totals_df: pd.DataFrame,
                         fills_df: Optional[pd.DataFrame] = None,
                         as_of_date: Optional[pd.Timestamp] = None,
                         use_cache: bool = True) -> pd.DataFrame:
        """
        Generate all features from input data

        Args:
            totals_df: Daily totals data
            fills_df: Optional fills data
            as_of_date: Point-in-time cutoff
            use_cache: Whether to use cached features

        Returns:
            DataFrame with all features
        """
        # Validate input data
        if self.validate_data:
            validation_result = self._validate_input_data(totals_df, fills_df)
            if not validation_result.is_valid:
                raise ValueError(f"Data validation failed: {validation_result.errors}")

        # Check cache
        cache_key = self._get_cache_key(totals_df, fills_df, as_of_date)
        if use_cache:
            cached_features = self._load_from_cache(cache_key)
            if cached_features is not None:
                logger.info("Loaded features from cache")
                return cached_features

        # Generate features from each generator
        all_features = []
        feature_times = {}

        for generator in self.feature_generators:
            generator_name = generator.__class__.__name__
            logger.info(f"Generating features with {generator_name}")

            start_time = datetime.now()

            try:
                features = generator.create_features(totals_df, fills_df, as_of_date)

                if not features.empty:
                    all_features.append(features)

                    # Track metadata
                    self.feature_metadata_[generator_name] = {
                        'n_features': len(generator.get_feature_names()),
                        'feature_names': generator.get_feature_names(),
                        'generation_time': (datetime.now() - start_time).total_seconds()
                    }

                    logger.info(f"Generated {len(generator.get_feature_names())} features from {generator_name}")

            except Exception as e:
                logger.error(f"Error in {generator_name}: {str(e)}")
                # Continue with other generators

        # Combine all features
        if not all_features:
            logger.warning("No features generated")
            return pd.DataFrame()

        # Merge on date and account_id
        combined_features = all_features[0]
        for features in all_features[1:]:
            # Get common columns (date/account_id)
            merge_cols = [col for col in ['date', 'account_id'] if col in features.columns]

            # Merge features
            combined_features = pd.merge(
                combined_features,
                features,
                on=merge_cols,
                how='outer',
                suffixes=('', '_dup')
            )

            # Remove duplicate columns
            dup_cols = [col for col in combined_features.columns if col.endswith('_dup')]
            combined_features = combined_features.drop(columns=dup_cols)

        # Sort by date and account
        if 'date' in combined_features.columns:
            sort_cols = ['date', 'account_id'] if 'account_id' in combined_features.columns else ['date']
            combined_features = combined_features.sort_values(sort_cols)

        # Post-process features
        combined_features = self._post_process_features(combined_features)

        # Cache features
        if use_cache:
            self._save_to_cache(combined_features, cache_key)

        # Log summary
        logger.info(f"Generated {combined_features.shape[1]} total features for {len(combined_features)} samples")

        return combined_features

    def _validate_input_data(self,
                           totals_df: pd.DataFrame,
                           fills_df: Optional[pd.DataFrame]) -> ValidationResult:
        """Validate input data before feature generation"""

        # Validate totals
        totals_result = self.validator.validate_totals(totals_df)

        # Validate fills if provided
        if fills_df is not None and not fills_df.empty:
            fills_result = self.validator.validate_fills(fills_df)

            # Combine results
            combined_result = self.validator.validate_combined(totals_df, fills_df)
            return combined_result

        return totals_result

    def _post_process_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Post-process generated features"""

        # Remove features with too many missing values
        missing_pct = features.isnull().sum() / len(features)
        high_missing_cols = missing_pct[missing_pct > TC.MAX_MISSING_PCT].index

        if len(high_missing_cols) > 0:
            logger.warning(f"Removing {len(high_missing_cols)} features with >{TC.MAX_MISSING_PCT*100}% missing")
            features = features.drop(columns=high_missing_cols)

        # Remove constant features
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        constant_cols = [col for col in numeric_cols if features[col].nunique() <= 1]

        if constant_cols:
            logger.warning(f"Removing {len(constant_cols)} constant features")
            features = features.drop(columns=constant_cols)

        # Remove highly correlated features
        features = self._remove_correlated_features(features)

        # Ensure no infinite values
        features = features.replace([np.inf, -np.inf], np.nan)

        # Final validation
        if features.empty:
            logger.error("No features remaining after post-processing")
        else:
            logger.info(f"Post-processing complete: {features.shape[1]} features retained")

        return features

    def _remove_correlated_features(self,
                                   features: pd.DataFrame,
                                   threshold: float = TC.HIGH_CORRELATION) -> pd.DataFrame:
        """Remove highly correlated features"""

        # Get numeric columns only
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['account_id']]

        if len(numeric_cols) < 2:
            return features

        # Calculate correlation matrix
        corr_matrix = features[numeric_cols].corr().abs()

        # Find highly correlated pairs
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Find features to drop
        to_drop = set()
        for column in upper_tri.columns:
            if column in to_drop:
                continue

            # Find correlated features
            correlated = upper_tri.index[upper_tri[column] > threshold].tolist()

            # Keep the feature with fewer missing values
            if correlated:
                candidates = [column] + correlated
                missing_counts = features[candidates].isnull().sum()

                # Keep the one with least missing
                keep = missing_counts.idxmin()
                drop = [c for c in candidates if c != keep]
                to_drop.update(drop)

        if to_drop:
            logger.info(f"Removing {len(to_drop)} highly correlated features")
            features = features.drop(columns=list(to_drop))

        return features

    def select_features(self,
                       features: pd.DataFrame,
                       target: pd.Series,
                       method: str = 'importance',
                       top_k: Optional[int] = None,
                       importance_threshold: float = TC.FEATURE_IMPORTANCE_THRESHOLD) -> List[str]:
        """
        Select most important features

        Args:
            features: Feature DataFrame
            target: Target variable
            method: Selection method ('importance', 'mutual_info', 'correlation')
            top_k: Number of top features to select
            importance_threshold: Minimum importance to retain feature

        Returns:
            List of selected feature names
        """
        # Get numeric features only
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        exclude_cols = ['account_id', 'date']
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]

        if method == 'importance':
            # Use tree-based feature importance
            from sklearn.ensemble import RandomForestRegressor

            # Prepare data
            X = features[feature_cols].fillna(0)
            y = target.loc[X.index]

            # Fit random forest
            rf = RandomForestRegressor(
                n_estimators=100,
                max_depth=6,
                random_state=42,
                n_jobs=-1
            )
            rf.fit(X, y)

            # Get importances
            importances = pd.Series(rf.feature_importances_, index=feature_cols)

        elif method == 'mutual_info':
            from sklearn.feature_selection import mutual_info_regression

            X = features[feature_cols].fillna(0)
            y = target.loc[X.index]

            # Calculate mutual information
            mi_scores = mutual_info_regression(X, y, random_state=42)
            importances = pd.Series(mi_scores, index=feature_cols)

        elif method == 'correlation':
            # Simple correlation with target
            correlations = features[feature_cols].corrwith(target).abs()
            importances = correlations

        else:
            raise ValueError(f"Unknown method: {method}")

        # Sort by importance
        importances = importances.sort_values(ascending=False)

        # Apply threshold
        important_features = importances[importances > importance_threshold]

        # Apply top_k if specified
        if top_k and len(important_features) > top_k:
            important_features = important_features.head(top_k)

        # Store importance scores
        self.feature_importance_ = importances.to_dict()
        self.selected_features_ = important_features.index.tolist()

        logger.info(f"Selected {len(self.selected_features_)} features using {method}")

        return self.selected_features_

    def transform_features(self,
                          features: pd.DataFrame,
                          selected_features: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Transform features to selected subset

        Args:
            features: Full feature DataFrame
            selected_features: List of features to keep

        Returns:
            Transformed DataFrame with selected features
        """
        if selected_features is None:
            selected_features = self.selected_features_

        if selected_features is None:
            return features

        # Always keep identifier columns
        keep_cols = ['date', 'account_id'] if 'date' in features.columns else ['account_id']
        keep_cols.extend([col for col in selected_features if col in features.columns])

        # Remove duplicates while preserving order
        keep_cols = list(dict.fromkeys(keep_cols))

        return features[keep_cols]

    def _get_cache_key(self,
                      totals_df: pd.DataFrame,
                      fills_df: Optional[pd.DataFrame],
                      as_of_date: Optional[pd.Timestamp]) -> str:
        """Generate cache key for feature set"""

        # Create hash of data characteristics
        key_parts = [
            f"totals_{len(totals_df)}_{totals_df.index.min()}_{totals_df.index.max()}",
            f"fills_{len(fills_df) if fills_df is not None else 0}",
            f"asof_{as_of_date if as_of_date else 'none'}",
            f"generators_{len(self.feature_generators)}"
        ]

        return "_".join(str(p).replace(" ", "_").replace(":", "") for p in key_parts)

    def _save_to_cache(self, features: pd.DataFrame, cache_key: str):
        """Save features to cache"""
        try:
            cache_path = self.cache_dir / f"{cache_key}.pkl"
            joblib.dump(features, cache_path)

            # Save metadata
            metadata_path = self.cache_dir / f"{cache_key}_metadata.json"
            metadata = {
                'created_at': datetime.now().isoformat(),
                'shape': features.shape,
                'columns': features.columns.tolist(),
                'feature_metadata': self.feature_metadata_
            }

            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.debug(f"Cached features to {cache_path}")

        except Exception as e:
            logger.warning(f"Failed to cache features: {e}")

    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Load features from cache"""
        try:
            cache_path = self.cache_dir / f"{cache_key}.pkl"

            if cache_path.exists():
                # Check age
                age_days = (datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)).days

                if age_days > 7:  # Cache expires after 7 days
                    logger.debug("Cache is stale, regenerating features")
                    return None

                features = joblib.load(cache_path)

                # Load metadata
                metadata_path = self.cache_dir / f"{cache_key}_metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        self.feature_metadata_ = json.load(f).get('feature_metadata', {})

                return features

        except Exception as e:
            logger.warning(f"Failed to load from cache: {e}")

        return None

    def get_feature_summary(self) -> pd.DataFrame:
        """Get summary of all features"""

        summary_data = []

        for generator_name, metadata in self.feature_metadata_.items():
            for feature_name in metadata['feature_names']:
                summary_data.append({
                    'generator': generator_name,
                    'feature': feature_name,
                    'importance': self.feature_importance_.get(feature_name, 0),
                    'selected': feature_name in (self.selected_features_ or [])
                })

        summary_df = pd.DataFrame(summary_data)

        if not summary_df.empty:
            summary_df = summary_df.sort_values('importance', ascending=False)

        return summary_df
