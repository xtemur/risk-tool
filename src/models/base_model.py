# src/models/base_model.py
from abc import ABC, abstractmethod


class BaseRiskModel(ABC):
    """
    Abstract base class for all risk prediction models
    """

    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.feature_names = []
        self.is_trained = False

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Train the model"""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions"""
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate prediction probabilities"""
        pass

    def save_model(self, path: str):
        """Save model to disk"""
        pass

    def load_model(self, path: str):
        """Load model from disk"""
        pass
