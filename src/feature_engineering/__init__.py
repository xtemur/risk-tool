"""
Feature Engineering Module

This module contains classes and functions for preprocessing trading data
and extracting features for PnL prediction models.
"""

from .feature_processor import FeatureProcessor
from .feature_extractor import FeatureExtractor
from .pipeline import FeatureEngineeringPipeline
from .config import FeatureConfig

__all__ = ['FeatureProcessor', 'FeatureExtractor', 'FeatureEngineeringPipeline', 'FeatureConfig']
