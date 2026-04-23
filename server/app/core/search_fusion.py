"""RRF (Reciprocal Rank Fusion) and Profile Boost for search result merging."""

from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class FusedResult:
    """A single search result after RRF fusion and optional profile boosting."""

    id: uuid.UUID
    rrf_score: float = 0.0
    boosted_score: float = 0.0
    content: str = ""
    type: str = ""
    entity_id: Optional[uuid.UUID] = None
    entity_name: Optional[str] = None
    entity_type: Optional[str] = None
    document_id: Optional[uuid.UUID] = None
    document: Optional[dict[str, Any]] = None
    source_channels: list[str] = field(default_factory=list)
    metadata: Optional[dict[str, Any]] = None
    created_at: Optional[Any] = None
    depth: Optional[int] = None
    relation_type: Optional[str] = None


def reciprocal_rank_fusion(
    channel_results: dict[str, list[dict[str, Any]]],
    k: int = 60,
) -> list[FusedResult]:
    """Merge results from multiple search channels using Reciprocal Rank Fusion.

    Formula: score += 1 / (k + rank + 1)   where k=60, rank is 0-indexed.

    Args:
        channel_results: Map of channel name to list of result dicts.
            Each result must have at least "id" key.
        k: RRF constant (default 60).

    Returns:
        List of FusedResult sorted by rrf_score descending.
    """
    scores: dict[uuid.UUID, float] = defaultdict(float)
    item_data: dict[uuid.UUID, dict[str, Any]] = {}
    item_channels: dict[uuid.UUID, list[str]] = defaultdict(list)

    for channel_name, results in channel_results.items():
        for rank, item in enumerate(results):
            item_id = item["id"]
            scores[item_id] += 1.0 / (k + rank + 1)
            item_channels[item_id].append(channel_name)
            if item_id not in item_data:
                item_data[item_id] = item

    fused: list[FusedResult] = []
    for item_id, rrf_score in scores.items():
        data = item_data.get(item_id, {})
        fused.append(FusedResult(
            id=item_id,
            rrf_score=rrf_score,
            boosted_score=rrf_score,
            content=data.get("content", ""),
            type=data.get("type", ""),
            entity_id=data.get("entity_id"),
            entity_name=data.get("entity_name"),
            entity_type=data.get("entity_type"),
            document_id=data.get("document_id"),
            document=data.get("document"),
            source_channels=item_channels[item_id],
            metadata=data.get("metadata"),
            created_at=data.get("created_at"),
            depth=data.get("depth"),
            relation_type=data.get("relation_type"),
        ))

    fused.sort(key=lambda r: r.rrf_score, reverse=True)
    return fused


# ── Profile Boost ────────────────────────────────────────────────

ENTITY_MATCH_BOOST = 1.5
PREFERENCE_MATCH_BOOST = 1.2


def apply_profile_boost(
    results: list[FusedResult],
    user_entity_id: Optional[uuid.UUID] = None,
    user_preference_entity_ids: list[uuid.UUID] | None = None,
) -> list[FusedResult]:
    """Apply profile-based score boosting to fused results.

    Rules (from spec):
    - Entity match:   if result.entity_id == user_entity_id -> x1.5
    - Preference match: if result.entity_id in user_preference_entity_ids -> x1.2
    - Both match: multiplicative -> x1.5 * x1.2 = x1.8
    """
    if user_preference_entity_ids is None:
        user_preference_entity_ids = []

    for result in results:
        boost = 1.0
        if user_entity_id is not None and result.entity_id == user_entity_id:
            boost *= ENTITY_MATCH_BOOST
        if result.entity_id in user_preference_entity_ids:
            boost *= PREFERENCE_MATCH_BOOST
        result.boosted_score = result.rrf_score * boost

    results.sort(key=lambda r: r.boosted_score, reverse=True)
    return results