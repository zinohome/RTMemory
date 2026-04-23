"""Shared test fixtures for RTMemory tests."""

import uuid
import unittest.mock

import httpx
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


# ---------------------------------------------------------------------------
# LLM / Embedding adapter test helpers
# ---------------------------------------------------------------------------


def make_httpx_response(
    status_code: int = 200,
    json_data: dict | None = None,
) -> httpx.Response:
    """Create an httpx.Response for testing without making real requests.

    Args:
        status_code: HTTP status code.
        json_data: JSON response body.

    Returns:
        httpx.Response with the given status and JSON body.
    """
    request = httpx.Request("POST", "https://api.example.com")
    return httpx.Response(
        status_code=status_code,
        json=json_data or {},
        request=request,
    )


@pytest.fixture
def mock_client():
    """Provide a mock httpx.AsyncClient for testing adapters.

    Returns an AsyncMock with a configured .post() method.
    Callers should set mock_client.post.return_value before use.
    """
    client = unittest.mock.AsyncMock(spec=httpx.AsyncClient)
    return client