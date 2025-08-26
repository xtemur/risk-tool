"""Base repository interface for data access layer."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class Repository(ABC):
    """Abstract base class for repository pattern."""

    @abstractmethod
    def find_by_id(self, id: Any) -> Optional[Any]:
        """Find entity by its identifier."""
        pass

    @abstractmethod
    def find_all(self) -> List[Any]:
        """Retrieve all entities."""
        pass

    @abstractmethod
    def save(self, entity: Any) -> Any:
        """Save or update entity."""
        pass

    @abstractmethod
    def delete(self, id: Any) -> bool:
        """Delete entity by identifier."""
        pass

    @abstractmethod
    def exists(self, id: Any) -> bool:
        """Check if entity exists."""
        pass
