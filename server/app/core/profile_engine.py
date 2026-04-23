"""ProfileEngine — computes user profiles from the knowledge graph.

Orchestrates: graph reads -> projection -> confidence decay -> cache.
Profile is NOT stored — it's computed from the graph on demand and cached.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Protocol

from app.core.confidence_decay import (
    compute_memory_confidence,
    is_forgotten,
)
from app.core.profile_cache import ProfileCache
from app.core.profile_models import (
    ConfidenceMap,
    ProfileData,
    ProfileResponse,
    MemoryType,
    FORGETTING_THRESHOLD,
)
from app.core.profile_projection import project_relations


class GraphEngineProtocol(Protocol):
    async def get_entity(self, entity_id: str) -> Any | None: ...
    async def get_current_relations(self, entity_id: str, space_id: str) -> list[Any]: ...
    async def get_recent_memories(
        self, entity_id: str, space_id: str, limit: int = 10
    ) -> list[Any]: ...


class SearchEngineProtocol(Protocol):
    async def search(
        self, q: str, space_id: str, entity_id: str | None = None, limit: int = 5
    ) -> list[dict[str, Any]]: ...


class ProfileEngine:
    """Computes structured user profiles from the knowledge graph.

    Flow:
      1. Check cache (unless fresh=True)
      2. Read entity + current relations + recent memories from GraphEngine
      3. Project relations -> four-layer profile
      4. Compute confidence decay for dynamic memories
      5. Optionally search via SearchEngine if q is provided
      6. Cache result and return

    Cache invalidation: call invalidate_cache() when graph data changes.
    """

    def __init__(
        self,
        graph_engine: GraphEngineProtocol,
        search_engine: SearchEngineProtocol | None = None,
        cache: ProfileCache | None = None,
    ) -> None:
        self._graph_engine = graph_engine
        self._search_engine = search_engine
        self._cache = cache or ProfileCache()

    async def compute(
        self,
        entity_id: str,
        space_id: str,
        q: str | None = None,
        fresh: bool = False,
    ) -> ProfileResponse:
        """Compute and return the profile for an entity.

        Args:
            entity_id: The entity to compute profile for.
            space_id: Space isolation.
            q: Optional search query — triggers search and attaches results.
            fresh: If True, bypass cache and recompute.
        """
        start = time.monotonic()

        # 1. Check cache
        if not fresh:
            cached = self._cache.get(entity_id, space_id)
            if cached is not None:
                profile, confidence, computed_at, timing_ms = cached
                # Even cached results may need search results attached
                search_results = await self._maybe_search(q, entity_id, space_id)
                return ProfileResponse(
                    profile=profile,
                    confidence=confidence,
                    search_results=search_results,
                    computed_at=computed_at,
                    timing_ms=timing_ms,
                )

        # 2. Read from graph
        entity = await self._graph_engine.get_entity(entity_id)
        relations = await self._graph_engine.get_current_relations(entity_id, space_id)
        recent_memories = await self._graph_engine.get_recent_memories(
            entity_id, space_id, limit=10
        )

        # 3. Project relations -> profile
        entity_name = entity.name if entity else None
        profile, confidence = project_relations(
            relations, entity_id, entity_name=entity_name
        )

        # 4. Compute dynamic memories with confidence decay
        profile.dynamic_memories = await self._compute_dynamic_memories(
            recent_memories
        )

        # 5. Optional search
        search_results = await self._maybe_search(q, entity_id, space_id)

        # 6. Build response
        elapsed = (time.monotonic() - start) * 1000.0
        now = datetime.now(timezone.utc)

        response = ProfileResponse(
            profile=profile,
            confidence=confidence,
            search_results=search_results,
            computed_at=now,
            timing_ms=round(elapsed, 2),
        )

        # 7. Cache result
        self._cache.put(
            entity_id, space_id, profile, confidence, now, round(elapsed, 2)
        )

        return response

    async def _compute_dynamic_memories(
        self, memories: list[Any]
    ) -> list[str]:
        """Filter, decay, and sort dynamic memories by confidence.

        - Excludes is_forgotten=True memories
        - Computes decayed confidence at read time
        - Excludes memories below forgetting threshold
        - Sorts by decayed confidence descending
        - Returns content strings
        """
        now = datetime.now(timezone.utc)
        scored: list[tuple[float, str]] = []

        for mem in memories:
            # Skip explicitly forgotten
            if getattr(mem, "is_forgotten", False):
                continue

            # Compute decayed confidence
            mem_type_str = getattr(mem, "memory_type", "fact")
            try:
                mem_type = MemoryType(mem_type_str)
            except ValueError:
                mem_type = MemoryType.inference  # default for unknown types

            initial_conf = getattr(mem, "confidence", 0.5)
            created_at = getattr(mem, "created_at", now)
            ref_count = getattr(mem, "source_count", 0)

            decayed = compute_memory_confidence(
                initial_confidence=initial_conf,
                memory_type=mem_type,
                created_at=created_at,
                now=now,
                ref_count=ref_count,
            )

            # Skip if below forgetting threshold
            if is_forgotten(decayed):
                continue

            scored.append((decayed, mem.content))

        # Sort by decayed confidence descending
        scored.sort(key=lambda x: -x[0])
        return [content for _, content in scored]

    async def _maybe_search(
        self, q: str | None, entity_id: str, space_id: str
    ) -> list[dict[str, Any]]:
        """Run search if query is provided and search engine is available."""
        if not q or not self._search_engine:
            return []
        try:
            return await self._search_engine.search(
                q=q, space_id=space_id, entity_id=entity_id, limit=5
            )
        except Exception:
            # Search failure should not break profile computation
            return []

    def invalidate_cache(self, entity_id: str, space_id: str) -> None:
        """Invalidate cached profile — call when graph data changes."""
        self._cache.invalidate(entity_id, space_id)

    def invalidate_cache_space(self, space_id: str) -> None:
        """Invalidate all cached profiles for a space."""
        self._cache.invalidate_space(space_id)

    async def on_graph_change(
        self,
        entity_ids: list[str],
        space_id: str,
    ) -> None:
        """Hook called when graph data changes.

        Invalidates cached profiles for all affected entities.
        Intended to be called by GraphEngine after entity/relation/memory
        mutations.

        Args:
            entity_ids: List of entity IDs whose data changed.
            space_id: The space the changes belong to.
        """
        for eid in entity_ids:
            self.invalidate_cache(eid, space_id)