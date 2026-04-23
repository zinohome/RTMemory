"""Profile projection — map relation types to profile fields.

The projection config maps relation_type strings to profile field paths
like "identity.location". The `prefers` and `knows` relation types use
the relation's `value` field as a sub-field discriminator (e.g.,
prefers+value="style" -> preferences.style, knows+value="collaborators"
-> relationships.collaborators).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from app.core.profile_models import (
    ConfidenceMap,
    CurrentStatusLayer,
    IdentityLayer,
    PreferencesLayer,
    ProfileData,
    RelationshipsLayer,
)


# ── Configurable projection mapping ──

DEFAULT_RELATION_MAP: dict[str, str] = {
    "lives_in": "identity.location",
    "works_at": "identity.company",
    "has_role": "identity.role",
    "prefers": "preferences.stack",       # default sub-field; value field overrides
    "current_focus": "current_status.focus",
    "current_project": "current_status.project",
    "current_mood": "current_status.mood",
    "knows": "relationships.team",         # default sub-field; value field overrides
}

# Sub-field overrides: when relation.value matches, remap to this path
PREFERS_SUBFIELDS: dict[str, str] = {
    "style": "preferences.style",
    "languages": "preferences.languages",
    "stack": "preferences.stack",
}

KNOWS_SUBFIELDS: dict[str, str] = {
    "collaborators": "relationships.collaborators",
    "team": "relationships.team",
}


@dataclass
class ProfileProjectionConfig:
    """Configurable mapping from relation_type -> profile field path."""
    relation_map: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_RELATION_MAP))
    prefers_subfields: dict[str, str] = field(default_factory=lambda: dict(PREFERS_SUBFIELDS))
    knows_subfields: dict[str, str] = field(default_factory=lambda: dict(KNOWS_SUBFIELDS))


def default_projection_config() -> ProfileProjectionConfig:
    """Return the default projection configuration."""
    return ProfileProjectionConfig()


class RelationStub(Protocol):
    """Protocol for relation objects used in projection."""
    relation_type: str
    target_name: str
    value: str | None
    confidence: float
    is_current: bool
    updated_at: datetime
    source_count: int


def _resolve_field_path(
    relation_type: str,
    value: str | None,
    config: ProfileProjectionConfig,
) -> str | None:
    """Resolve the full profile field path for a relation.

    Handles sub-field overrides for `prefers` and `knows` via the
    relation's value field.
    """
    if relation_type == "prefers" and value:
        subfield = config.prefers_subfields.get(value)
        if subfield:
            return subfield

    if relation_type == "knows" and value:
        subfield = config.knows_subfields.get(value)
        if subfield:
            return subfield

    return config.relation_map.get(relation_type)


def _is_identity_field(field_path: str) -> bool:
    return field_path.startswith("identity.")


def _is_status_field(field_path: str) -> bool:
    return field_path.startswith("current_status.")


def project_relations(
    relations: list[Any],
    entity_id: str,
    entity_name: str | None = None,
    config: ProfileProjectionConfig | None = None,
) -> tuple[ProfileData, ConfidenceMap]:
    """Project a list of relations into a four-layer profile.

    For identity and current_status layers (scalar fields), only
    is_current=True relations are used, and highest-confidence wins.

    For preferences and relationships (list fields), all relations
    are accumulated regardless of is_current.

    Returns (ProfileData, ConfidenceMap).
    """
    if config is None:
        config = default_projection_config()

    identity = IdentityLayer(name=entity_name)
    preferences = PreferencesLayer()
    current_status = CurrentStatusLayer()
    relationships = RelationshipsLayer()
    confidence = ConfidenceMap()

    # Track best scalar values for identity/status
    # key=field_path, value=(target_name, confidence)
    best_scalar: dict[str, tuple[str, float]] = {}

    # Track list accumulators for preferences/relationships
    # key=field_path, value=list of (target_name, confidence)
    list_accum: dict[str, list[tuple[str, float]]] = {}

    for rel in relations:
        field_path = _resolve_field_path(rel.relation_type, rel.value, config)
        if field_path is None:
            continue  # unknown relation type, skip

        is_scalar = _is_identity_field(field_path) or _is_status_field(field_path)

        if is_scalar:
            # For scalar fields, only use is_current=True
            if not rel.is_current:
                continue
            existing = best_scalar.get(field_path)
            if existing is None or rel.confidence > existing[1]:
                best_scalar[field_path] = (rel.target_name, rel.confidence)
        else:
            # For list fields, accumulate all (even non-current)
            if field_path not in list_accum:
                list_accum[field_path] = []
            list_accum[field_path].append((rel.target_name, rel.confidence))

    # ── Populate identity from best_scalar ──
    for field_path, (value, conf) in best_scalar.items():
        _set_scalar_field(identity, current_status, field_path, value)
        _set_confidence_field(confidence, field_path, conf)

    # ── Populate preferences and relationships from list_accum ──
    for field_path, items in list_accum.items():
        # Sort by confidence descending for deterministic ordering
        items_sorted = sorted(items, key=lambda x: -x[1])
        values = [name for name, _ in items_sorted]
        best_conf = items_sorted[0][1] if items_sorted else 0.0
        _set_list_field(preferences, relationships, field_path, values)
        _set_confidence_field(confidence, field_path, best_conf)

    profile = ProfileData(
        identity=identity,
        preferences=preferences,
        current_status=current_status,
        relationships=relationships,
    )
    return profile, confidence


def _set_scalar_field(
    identity: IdentityLayer,
    current_status: CurrentStatusLayer,
    field_path: str,
    value: str,
) -> None:
    """Set a scalar field on identity or current_status."""
    mapping = {
        "identity.location": lambda v: setattr(identity, "location", v),
        "identity.company": lambda v: setattr(identity, "company", v),
        "identity.role": lambda v: setattr(identity, "role", v),
        "current_status.focus": lambda v: setattr(current_status, "focus", v),
        "current_status.project": lambda v: setattr(current_status, "project", v),
        "current_status.mood": lambda v: setattr(current_status, "mood", v),
    }
    setter = mapping.get(field_path)
    if setter:
        setter(value)


def _set_list_field(
    preferences: PreferencesLayer,
    relationships: RelationshipsLayer,
    field_path: str,
    values: list[str],
) -> None:
    """Set a list field on preferences or relationships."""
    mapping = {
        "preferences.stack": lambda v: setattr(preferences, "stack", v),
        "preferences.languages": lambda v: setattr(preferences, "languages", v),
        "preferences.style": lambda v: setattr(preferences, "style", v[0] if v else None),
        "relationships.team": lambda v: setattr(relationships, "team", v),
        "relationships.collaborators": lambda v: setattr(relationships, "collaborators", v),
    }
    setter = mapping.get(field_path)
    if setter:
        setter(values)


def _set_confidence_field(
    confidence: ConfidenceMap,
    field_path: str,
    value: float,
) -> None:
    """Set a confidence value for a profile field."""
    # Extract the leaf field name from path like "identity.location" -> "location"
    field_name = field_path.split(".")[-1]
    if hasattr(confidence, field_name):
        setattr(confidence, field_name, value)