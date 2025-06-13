"""
Modeling Module

This module contains classes and functions for training and validating
trader PnL prediction models with proper time series validation and
state-of-the-art causal inference capabilities.
"""

from .time_series_validator import TimeSeriesValidator
from .model_trainer import ModelTrainer
from .performance_evaluator import PerformanceEvaluator
from .prediction_pipeline import PredictionPipeline
from .causal_impact_analyzer import CausalImpactAnalyzer
from .config import ModelConfig

# Advanced causal inference modules
from .advanced_causal_analyzer import AdvancedCausalAnalyzer
from .double_ml_estimator import DoubleMachineLearningEstimator, TradingDMLAnalyzer
from .synthetic_control_analyzer import SyntheticControlAnalyzer
from .causal_validation_suite import CausalValidationSuite
from .causal_interpretability import CausalInterpretabilityAnalyzer

__all__ = [
    # Core modeling components
    'TimeSeriesValidator',
    'ModelTrainer',
    'PerformanceEvaluator',
    'PredictionPipeline',
    'CausalImpactAnalyzer',
    'ModelConfig',

    # Advanced causal inference components
    'AdvancedCausalAnalyzer',
    'DoubleMachineLearningEstimator',
    'TradingDMLAnalyzer',
    'SyntheticControlAnalyzer',
    'CausalValidationSuite',
    'CausalInterpretabilityAnalyzer'
]
