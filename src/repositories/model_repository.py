"""Repository for model management."""

import pickle
import logging
from pathlib import Path
from typing import Dict, Optional, Any, List
from datetime import datetime

from .base import Repository
from ..constants import TRADER_MODELS_DIR, MODEL_FILE_SUFFIX
from ..exceptions import ModelNotFoundError, ModelLoadError

logger = logging.getLogger(__name__)


class ModelRepository(Repository):
    """Repository for managing ML models."""

    def __init__(self, models_dir: Path = TRADER_MODELS_DIR):
        """Initialize model repository."""
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._cache = {}  # Simple in-memory cache

    def find_by_id(self, trader_id: int) -> Optional[Dict[str, Any]]:
        """Load model for a specific trader."""
        return self.load_model(trader_id)

    def find_all(self) -> List[Dict[str, Any]]:
        """Load all available models."""
        models = []
        for model_file in self.models_dir.glob(f"*{MODEL_FILE_SUFFIX}"):
            try:
                trader_id = int(model_file.stem.split('_')[0])
                model_data = self.load_model(trader_id)
                if model_data:
                    models.append(model_data)
            except (ValueError, IndexError) as e:
                logger.warning(f"Skipping invalid model file {model_file}: {e}")
        return models

    def load_model(self, trader_id: int) -> Optional[Dict[str, Any]]:
        """
        Load model for a specific trader.

        Args:
            trader_id: The trader ID

        Returns:
            Dictionary containing model and metadata

        Raises:
            ModelNotFoundError: If model doesn't exist
            ModelLoadError: If model loading fails
        """
        # Check cache first
        if trader_id in self._cache:
            logger.debug(f"Loading model for trader {trader_id} from cache")
            return self._cache[trader_id]

        model_path = self._get_model_path(trader_id)

        if not model_path.exists():
            raise ModelNotFoundError(trader_id, str(model_path))

        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)

            # Add metadata
            model_data['trader_id'] = trader_id
            model_data['loaded_at'] = datetime.now()
            model_data['model_path'] = str(model_path)

            # Cache the model
            self._cache[trader_id] = model_data

            logger.info(f"Successfully loaded model for trader {trader_id}")
            return model_data

        except Exception as e:
            raise ModelLoadError(str(model_path), e)

    def save(self, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Save or update a model.

        Args:
            model_data: Dictionary containing model and metadata

        Returns:
            The saved model data
        """
        trader_id = model_data.get('trader_id')
        if not trader_id:
            raise ValueError("model_data must contain 'trader_id'")

        model_path = self._get_model_path(trader_id)

        # Add save metadata
        model_data['saved_at'] = datetime.now()

        try:
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)

            # Update cache
            self._cache[trader_id] = model_data

            logger.info(f"Successfully saved model for trader {trader_id}")
            return model_data

        except Exception as e:
            logger.error(f"Failed to save model for trader {trader_id}: {e}")
            raise

    def delete(self, trader_id: int) -> bool:
        """
        Delete a model (archives it instead of deleting).

        Args:
            trader_id: The trader ID

        Returns:
            True if successful
        """
        model_path = self._get_model_path(trader_id)

        if not model_path.exists():
            return False

        # Archive instead of delete
        archive_path = model_path.parent / "archive"
        archive_path.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archived_name = f"{trader_id}_archived_{timestamp}.pkl"
        archived_path = archive_path / archived_name

        try:
            model_path.rename(archived_path)

            # Remove from cache
            if trader_id in self._cache:
                del self._cache[trader_id]

            logger.info(f"Archived model for trader {trader_id} to {archived_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to archive model for trader {trader_id}: {e}")
            return False

    def exists(self, trader_id: int) -> bool:
        """Check if model exists for trader."""
        return self._get_model_path(trader_id).exists()

    def get_model_info(self, trader_id: int) -> Optional[Dict[str, Any]]:
        """
        Get model metadata without loading the full model.

        Args:
            trader_id: The trader ID

        Returns:
            Dictionary with model metadata
        """
        model_path = self._get_model_path(trader_id)

        if not model_path.exists():
            return None

        stat = model_path.stat()

        return {
            'trader_id': trader_id,
            'path': str(model_path),
            'size_bytes': stat.st_size,
            'modified_time': datetime.fromtimestamp(stat.st_mtime),
            'exists': True
        }

    def list_available_models(self) -> List[int]:
        """Get list of trader IDs with available models."""
        trader_ids = []

        for model_file in self.models_dir.glob(f"*{MODEL_FILE_SUFFIX}"):
            try:
                trader_id = int(model_file.stem.split('_')[0])
                trader_ids.append(trader_id)
            except (ValueError, IndexError):
                continue

        return sorted(trader_ids)

    def clear_cache(self):
        """Clear the model cache."""
        self._cache.clear()
        logger.info("Model cache cleared")

    def _get_model_path(self, trader_id: int) -> Path:
        """Get the path for a trader's model file."""
        return self.models_dir / f"{trader_id}{MODEL_FILE_SUFFIX}"
