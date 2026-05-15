"""Redis-backed distributed embedding store.

TODO[Phase-2]: Full implementation with Redis caching.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from reason_reduce.monitoring.logger import get_logger

logger = get_logger(__name__)


class EmbeddingStore:
    """In-memory embedding cache (Redis-backed in production).

    Keys are SHA-256 hashes of text. Values are numpy arrays.

    TODO[Phase-2]: Replace in-memory dict with Redis backend.
    """

    def __init__(self) -> None:
        self._cache: dict[str, np.ndarray[Any, Any]] = {}

    def get(self, text: str) -> np.ndarray[Any, Any] | None:
        """Retrieve a cached embedding.

        Args:
            text: Input text whose embedding to look up.

        Returns:
            Cached embedding array or None if not found.
        """
        key = hashlib.sha256(text.encode()).hexdigest()
        return self._cache.get(key)

    def put(self, text: str, embedding: np.ndarray[Any, Any]) -> None:
        """Store an embedding in the cache.

        Args:
            text: Input text (used to compute cache key).
            embedding: The embedding vector to store.
        """
        key = hashlib.sha256(text.encode()).hexdigest()
        self._cache[key] = embedding

    def __len__(self) -> int:
        return len(self._cache)
