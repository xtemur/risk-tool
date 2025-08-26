"""Dependency injection container for the risk tool."""

import logging
from typing import Optional, Dict, Any
from pathlib import Path

from .config.config_manager import ConfigManager
from .repositories.trader_repository import TraderRepository
from .repositories.model_repository import ModelRepository
from .repositories.fills_repository import FillsRepository
from .services.risk_service import RiskService
from .services.trader_service import TraderService
from .services.prediction_service import PredictionService
from .services.metrics_service import MetricsService
from .constants import DB_PATH, TRADER_MODELS_DIR

logger = logging.getLogger(__name__)


class ServiceContainer:
    """
    Dependency injection container for managing service instances.

    This container ensures that services are properly initialized
    with their dependencies and provides singleton instances.
    """

    def __init__(self, config_path: Optional[str] = None, env: str = 'production'):
        """
        Initialize the service container.

        Args:
            config_path: Optional path to configuration file
            env: Environment name (production, development, test)
        """
        self.env = env
        self._instances: Dict[str, Any] = {}

        # Initialize configuration first
        self._config = ConfigManager(env=env, config_file=config_path)
        self._config.validate()

        logger.info(f"Service container initialized for environment: {env}")

    @property
    def config(self) -> ConfigManager:
        """Get configuration manager."""
        return self._config

    @property
    def trader_repository(self) -> TraderRepository:
        """Get trader repository (singleton)."""
        if 'trader_repository' not in self._instances:
            db_path = self._config.database.path
            self._instances['trader_repository'] = TraderRepository(db_path)
            logger.debug("TraderRepository initialized")
        return self._instances['trader_repository']

    @property
    def model_repository(self) -> ModelRepository:
        """Get model repository (singleton)."""
        if 'model_repository' not in self._instances:
            models_dir = Path(TRADER_MODELS_DIR)
            self._instances['model_repository'] = ModelRepository(models_dir)
            logger.debug("ModelRepository initialized")
        return self._instances['model_repository']

    @property
    def fills_repository(self) -> FillsRepository:
        """Get fills repository (singleton)."""
        if 'fills_repository' not in self._instances:
            db_path = self._config.database.path
            self._instances['fills_repository'] = FillsRepository(db_path)
            logger.debug("FillsRepository initialized")
        return self._instances['fills_repository']

    @property
    def risk_service(self) -> RiskService:
        """Get risk service (singleton)."""
        if 'risk_service' not in self._instances:
            self._instances['risk_service'] = RiskService(
                trader_repo=self.trader_repository,
                model_repo=self.model_repository,
                config=self._config
            )
            logger.debug("RiskService initialized")
        return self._instances['risk_service']

    @property
    def trader_service(self) -> TraderService:
        """Get trader service (singleton)."""
        if 'trader_service' not in self._instances:
            self._instances['trader_service'] = TraderService(
                trader_repo=self.trader_repository,
                fills_repo=self.fills_repository,
                config=self._config
            )
            logger.debug("TraderService initialized")
        return self._instances['trader_service']

    @property
    def prediction_service(self) -> PredictionService:
        """Get prediction service (singleton)."""
        if 'prediction_service' not in self._instances:
            self._instances['prediction_service'] = PredictionService(
                model_repo=self.model_repository,
                trader_repo=self.trader_repository,
                config=self._config
            )
            logger.debug("PredictionService initialized")
        return self._instances['prediction_service']

    @property
    def metrics_service(self) -> MetricsService:
        """Get metrics service (singleton)."""
        if 'metrics_service' not in self._instances:
            self._instances['metrics_service'] = MetricsService(
                fills_repo=self.fills_repository,
                trader_repo=self.trader_repository,
                config=self._config
            )
            logger.debug("MetricsService initialized")
        return self._instances['metrics_service']

    def reset(self) -> None:
        """Reset all service instances."""
        self._instances.clear()
        logger.info("Service container reset - all instances cleared")

    def get_service(self, service_name: str) -> Any:
        """
        Get service by name.

        Args:
            service_name: Name of the service

        Returns:
            Service instance

        Raises:
            ValueError: If service not found
        """
        service_map = {
            'trader_repository': self.trader_repository,
            'model_repository': self.model_repository,
            'fills_repository': self.fills_repository,
            'risk_service': self.risk_service,
            'trader_service': self.trader_service,
            'prediction_service': self.prediction_service,
            'metrics_service': self.metrics_service,
            'config': self.config
        }

        if service_name not in service_map:
            raise ValueError(f"Service not found: {service_name}")

        return service_map[service_name]

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup if needed."""
        # Could add cleanup logic here if needed
        pass


# Global container instance (can be overridden for testing)
_global_container: Optional[ServiceContainer] = None


def get_container(config_path: Optional[str] = None,
                 env: str = 'production',
                 reset: bool = False) -> ServiceContainer:
    """
    Get the global service container.

    Args:
        config_path: Optional configuration file path
        env: Environment name
        reset: Whether to reset existing container

    Returns:
        ServiceContainer instance
    """
    global _global_container

    if reset or _global_container is None:
        _global_container = ServiceContainer(config_path=config_path, env=env)

    return _global_container


def reset_container():
    """Reset the global container."""
    global _global_container
    if _global_container:
        _global_container.reset()
    _global_container = None
    logger.info("Global container reset")
