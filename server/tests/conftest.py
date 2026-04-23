"""Shared test fixtures for RTMemory tests."""

import uuid

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.db.models import Space
from app.main import app
from app.api.deps import db_session as _db_dep


# ---------------------------------------------------------------------------
# SQLite-based test DB (no pgvector needed for basic CRUD tests)
# ---------------------------------------------------------------------------

TEST_DB_URL = "sqlite+aiosqlite:///./test.db"


@pytest.fixture
async def test_engine():
    """Create a fresh async engine for each test module."""
    engine = create_async_engine(TEST_DB_URL, echo=False)
    async with engine.begin() as conn:
        # Only create the spaces table — other tables use pgvector/JSONB
        await conn.run_sync(Space.__table__.create, checkfirst=True)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Space.__table__.drop, checkfirst=True)
    await engine.dispose()
    import os
    try:
        os.remove("./test.db")
    except FileNotFoundError:
        pass


@pytest.fixture
async def test_session(test_engine):
    """Yield an async DB session for testing."""
    factory = async_sessionmaker(
        test_engine, class_=AsyncSession, expire_on_commit=False
    )
    async with factory() as session:
        yield session


@pytest.fixture
async def test_client(test_session):
    """Async HTTP client with DB dependency overridden."""
    async def _override_db():
        yield test_session

    app.dependency_overrides[_db_dep] = _override_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_org_id():
    """Return a fixed org UUID for tests."""
    return uuid.UUID("00000000-0000-0000-0000-000000000001")


@pytest.fixture
def sample_owner_id():
    """Return a fixed owner UUID for tests."""
    return uuid.UUID("00000000-0000-0000-0000-000000000002")