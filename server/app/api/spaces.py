"""Spaces CRUD API router."""

from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import db_session
from app.db.models import Space


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class SpaceCreate(BaseModel):
    """Request body for creating a space."""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    org_id: uuid.UUID
    owner_id: Optional[uuid.UUID] = None
    container_tag: Optional[str] = None
    is_default: bool = False


class SpaceUpdate(BaseModel):
    """Request body for updating a space."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    container_tag: Optional[str] = None
    is_default: Optional[bool] = None


class SpaceRead(BaseModel):
    """Response body for reading a space."""
    id: uuid.UUID
    name: str
    description: Optional[str] = None
    org_id: uuid.UUID
    owner_id: Optional[uuid.UUID] = None
    container_tag: Optional[str] = None
    is_default: bool
    created_at: str
    updated_at: str

    model_config = {"from_attributes": True}

    @classmethod
    def from_orm_space(cls, space: Space) -> "SpaceRead":
        return cls(
            id=space.id,
            name=space.name,
            description=space.description,
            org_id=space.org_id,
            owner_id=space.owner_id,
            container_tag=space.container_tag,
            is_default=space.is_default,
            created_at=space.created_at.isoformat() if space.created_at else "",
            updated_at=space.updated_at.isoformat() if space.updated_at else "",
        )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter()


@router.post("/", response_model=SpaceRead, status_code=status.HTTP_201_CREATED)
async def create_space(
    body: SpaceCreate,
    session: AsyncSession = Depends(db_session),
):
    """Create a new space."""
    space = Space(
        name=body.name,
        description=body.description,
        org_id=body.org_id,
        owner_id=body.owner_id,
        container_tag=body.container_tag,
        is_default=body.is_default,
    )
    session.add(space)
    await session.flush()
    await session.refresh(space)
    return SpaceRead.from_orm_space(space)


@router.get("/", response_model=list[SpaceRead])
async def list_spaces(
    session: AsyncSession = Depends(db_session),
):
    """List all spaces."""
    result = await session.execute(select(Space).order_by(Space.created_at.desc()))
    spaces = result.scalars().all()
    return [SpaceRead.from_orm_space(s) for s in spaces]


@router.get("/{space_id}", response_model=SpaceRead)
async def get_space(
    space_id: uuid.UUID,
    session: AsyncSession = Depends(db_session),
):
    """Get a space by ID."""
    space = await session.get(Space, space_id)
    if space is None:
        raise HTTPException(status_code=404, detail="Space not found")
    return SpaceRead.from_orm_space(space)


@router.delete("/{space_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_space(
    space_id: uuid.UUID,
    session: AsyncSession = Depends(db_session),
):
    """Delete a space and all its data (cascade)."""
    space = await session.get(Space, space_id)
    if space is None:
        raise HTTPException(status_code=404, detail="Space not found")
    await session.delete(space)
    await session.flush()


@router.patch("/{space_id}", response_model=SpaceRead)
async def update_space(
    space_id: uuid.UUID,
    body: SpaceUpdate,
    session: AsyncSession = Depends(db_session),
):
    """Update a space."""
    space = await session.get(Space, space_id)
    if space is None:
        raise HTTPException(status_code=404, detail="Space not found")
    update_data = body.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(space, key, value)
    await session.flush()
    await session.refresh(space)
    return SpaceRead.from_orm_space(space)