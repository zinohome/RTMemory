"""Profile cache — in-memory cache with graph-change invalidation.

Cache key = (entity_id, space_id). Invalidated when any entity/relation/
memory for that entity changes. No TTL — pure invalidation-driven.
"""
from __future__ import annotations

from datetime import datetime

from app.core.profile_models import ConfidenceMap, ProfileData


class ProfileCache:
    """In-memory profile cache.

    Stores computed profiles keyed by (entity_id, space_id).
    Invalidated explicitly when the underlying graph data changes.
    """

    def __init__(self) -> None:
        # key: (entity_id, space_id)
        # value: (profile_data, confidence_map, computed_at, timing_ms)
        self._store: dict[tuple[str, str], tuple[ProfileData, ConfidenceMap, datetime, float]] = {}

    def get(
        self, entity_id: str, space_id: str
    ) -> tuple[ProfileData, ConfidenceMap, datetime, float] | None:
        """Return cached profile or None on miss."""
        return self._store.get((entity_id, space_id))

    def put(
        self,
        entity_id: str,
        space_id: str,
        profile: ProfileData,
        confidence: ConfidenceMap,
        computed_at: datetime,
        timing_ms: float,
    ) -> None:
        """Store a computed profile in the cache."""
        self._store[(entity_id, space_id)] = (profile, confidence, computed_at, timing_ms)

    def invalidate(self, entity_id: str, space_id: str) -> None:
        """Invalidate cached profile for a specific entity+space."""
        self._store.pop((entity_id, space_id), None)

    def invalidate_space(self, space_id: str) -> None:
        """Invalidate all cached profiles for a space."""
        keys_to_remove = [k for k in self._store if k[1] == space_id]
        for k in keys_to_remove:
            del self._store[k]

    def size(self) -> int:
        """Return number of cached profiles."""
        return len(self._store)

    def clear(self) -> None:
        """Clear the entire cache."""
        self._store.clear()