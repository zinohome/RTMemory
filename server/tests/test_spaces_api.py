"""Tests for Spaces CRUD API endpoints."""

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import String, event
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from app.db.models import Space
from app.main import app


# Use SQLite for API-level tests (no pgvector needed for CRUD)
TEST_DB_URL = "sqlite+aiosqlite:///./test_spaces.db"


@pytest.fixture
async def db_engine():
    engine = create_async_engine(TEST_DB_URL, echo=False)

    # Register SQLite type compilers so UUID columns work on SQLite
    from sqlalchemy.dialects.postgresql import UUID as PG_UUID
    from sqlalchemy.ext.compiler import compiles

    @compiles(PG_UUID, "sqlite")
    def compile_uuid_sqlite(type_, compiler, **kw):
        return "CHAR(36)"

    # Only create the spaces table — other tables use pgvector/JSONB which SQLite doesn't support
    async with engine.begin() as conn:
        await conn.run_sync(Space.__table__.create, checkfirst=True)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Space.__table__.drop, checkfirst=True)
    await engine.dispose()
    # Cleanup test db file
    import os
    try:
        os.remove("./test_spaces.db")
    except FileNotFoundError:
        pass


@pytest.fixture
async def db_session(db_engine):
    factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        yield session


@pytest.fixture
async def client(db_session):
    """Override the app's DB dependency with the test session."""

    async def _override_db():
        yield db_session

    from app.api.deps import db_session as _db_dep
    app.dependency_overrides[_db_dep] = _override_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


class TestCreateSpace:
    async def test_create_space(self, client, db_session):
        response = await client.post(
            "/v1/spaces/",
            json={
                "name": "Test Space",
                "description": "A test space",
                "org_id": "00000000-0000-0000-0000-000000000001",
                "owner_id": "00000000-0000-0000-0000-000000000002",
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Test Space"
        assert data["description"] == "A test space"
        assert data["is_default"] is False
        assert "id" in data

    async def test_create_space_minimal(self, client, db_session):
        response = await client.post(
            "/v1/spaces/",
            json={
                "name": "Minimal",
                "org_id": "00000000-0000-0000-0000-000000000001",
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Minimal"


class TestListSpaces:
    async def test_list_spaces_empty(self, client, db_session):
        response = await client.get("/v1/spaces/")
        assert response.status_code == 200
        assert response.json() == []

    async def test_list_spaces_with_data(self, client, db_session):
        # Create two spaces
        await client.post(
            "/v1/spaces/",
            json={"name": "Space A", "org_id": "00000000-0000-0000-0000-000000000001"},
        )
        await client.post(
            "/v1/spaces/",
            json={"name": "Space B", "org_id": "00000000-0000-0000-0000-000000000001"},
        )
        response = await client.get("/v1/spaces/")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        names = {s["name"] for s in data}
        assert names == {"Space A", "Space B"}


class TestGetSpace:
    async def test_get_space_by_id(self, client, db_session):
        create_resp = await client.post(
            "/v1/spaces/",
            json={"name": "Detail Space", "org_id": "00000000-0000-0000-0000-000000000001"},
        )
        space_id = create_resp.json()["id"]
        response = await client.get(f"/v1/spaces/{space_id}")
        assert response.status_code == 200
        assert response.json()["name"] == "Detail Space"

    async def test_get_space_not_found(self, client, db_session):
        response = await client.get("/v1/spaces/00000000-0000-0000-0000-000000000099")
        assert response.status_code == 404


class TestDeleteSpace:
    async def test_delete_space(self, client, db_session):
        create_resp = await client.post(
            "/v1/spaces/",
            json={"name": "Delete Me", "org_id": "00000000-0000-0000-0000-000000000001"},
        )
        space_id = create_resp.json()["id"]
        delete_resp = await client.delete(f"/v1/spaces/{space_id}")
        assert delete_resp.status_code == 204
        # Verify it's gone
        get_resp = await client.get(f"/v1/spaces/{space_id}")
        assert get_resp.status_code == 404

    async def test_delete_space_not_found(self, client, db_session):
        response = await client.delete("/v1/spaces/00000000-0000-0000-0000-000000000099")
        assert response.status_code == 404