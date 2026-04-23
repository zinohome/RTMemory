"""Profile API route — POST /v1/profile.

Computes user profiles from the knowledge graph on demand.
Profile is NOT stored — computed and cached.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.core.profile_engine import ProfileEngine
from app.core.profile_models import ProfileRequest, ProfileResponse


# ── Dependency injection placeholder ──
# In production, these are wired in main.py with real engine instances.
# For tests, override via app.dependency_overrides.

_profile_engine: ProfileEngine | None = None


def set_profile_engine(engine: ProfileEngine) -> None:
    """Set the global ProfileEngine instance (called during app startup)."""
    global _profile_engine
    _profile_engine = engine


def get_profile_engine() -> ProfileEngine:
    """FastAPI dependency — returns the configured ProfileEngine."""
    if _profile_engine is None:
        raise HTTPException(status_code=503, detail="ProfileEngine not initialized")
    return _profile_engine


def create_profile_router() -> APIRouter:
    """Create and return the profile API router."""
    router = APIRouter(tags=["profile"])

    @router.post("/profile", response_model=ProfileResponse)
    async def get_profile(
        request: ProfileRequest,
        engine: ProfileEngine = Depends(get_profile_engine),
    ) -> ProfileResponse:
        """Compute and return the user profile for an entity.

        The profile is computed from the knowledge graph on demand and
        cached in memory. Use `fresh=True` to bypass the cache.

        If `q` is provided, a search is triggered and results are
        attached to the response.
        """
        result = await engine.compute(
            entity_id=request.entity_id,
            space_id=request.space_id,
            q=request.q,
            fresh=request.fresh,
        )
        return result

    return router