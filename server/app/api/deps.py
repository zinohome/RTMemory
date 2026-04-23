"""FastAPI dependency injection helpers."""

from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.db.session import get_session


async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async DB session. Use as FastAPI Depends()."""
    async for session in get_session():
        yield session


def settings() -> Settings:
    """Return the application Settings. Use as FastAPI Depends()."""
    return get_settings()