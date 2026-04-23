# RTMemory Foundation — 项目骨架与数据库

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish the project skeleton, database models, configuration system, and Docker Compose deployment.

**Architecture:** Single FastAPI service with PostgreSQL+pgvector. SQLAlchemy async ORM with Alembic migrations. Config via pydantic-settings (YAML + env overrides).

**Tech Stack:** Python 3.12, FastAPI, SQLAlchemy 2.0 (async), asyncpg, pgvector-python, Alembic, pydantic-settings, Docker Compose

## File Map

```
server/
  app/
    __init__.py
    main.py                  # FastAPI entry point + health check
    config.py                # pydantic-settings config (YAML + env)
    db/
      __init__.py
      base.py               # SQLAlchemy Base + custom VECTOR type
      session.py             # async engine + sessionmaker
      models.py              # ALL 7 table models
    api/
      __init__.py
      deps.py                # Dependency injection (DB session, config)
      spaces.py              # Spaces CRUD router
  alembic/
    env.py                   # Alembic env with async support
    versions/
      001_initial.py         # Initial migration (all 7 tables)
    alembic.ini               # (symlink or copied to server/)
  tests/
    __init__.py
    conftest.py              # Shared fixtures (async DB, test client)
    test_health.py           # Health check test
    test_spaces_api.py        # Spaces CRUD API test
    test_config.py            # Config loading test
    test_models.py            # Model creation test
  pyproject.toml

config.yaml                  # Default config
docker-compose.yml
Dockerfile
```

---

## Phase 1: Project Skeleton

### Step 1: Create directory structure

```bash
cd /home/ubuntu/ReToneProjects/RTMemory
mkdir -p server/app/db
mkdir -p server/app/api
mkdir -p server/tests
mkdir -p server/alembic/versions
```

**Run:**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory && mkdir -p server/app/db server/app/api server/tests server/alembic/versions
```

**Expected:** All directories created, no errors.

---

### Step 2: Create pyproject.toml

**File:** `/home/ubuntu/ReToneProjects/RTMemory/server/pyproject.toml`

```toml
[project]
name = "rtmemory-server"
version = "0.1.0"
description = "RTMemory — Temporal Knowledge Graph-Driven AI Memory System"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.30",
    "sqlalchemy[asyncio]>=2.0",
    "asyncpg>=0.30",
    "pgvector-python>=0.3",
    "alembic>=1.13",
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "httpx>=0.27",
    "pymupdf>=1.24",
    "trafilatura>=1.8",
    "sentence-transformers>=3.0",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
    "pytest-cov>=5.0",
    "respx>=0.21",
]

[build-system]
requires = ["setuptools>=75.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.setuptools.packages.find]
include = ["app*"]
```

**Run:**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory/server && pip install -e ".[dev]" 2>&1 | tail -5
```

**Expected:** Package installed in editable mode. Last lines show `Successfully installed rtmemory-server-0.1.0`.

---

### Step 3: Create all __init__.py files

**File:** `/home/ubuntu/ReToneProjects/RTMemory/server/app/__init__.py`

```python
# RTMemory Server
```

**File:** `/home/ubuntu/ReToneProjects/RTMemory/server/app/db/__init__.py`

```python
from .base import Base
from .session import get_engine, get_session_factory, get_session
from .models import (
    Space,
    Entity,
    Relation,
    Memory,
    Document,
    Chunk,
    MemorySource,
)

__all__ = [
    "Base",
    "get_engine",
    "get_session_factory",
    "get_session",
    "Space",
    "Entity",
    "Relation",
    "Memory",
    "Document",
    "Chunk",
    "MemorySource",
]
```

**File:** `/home/ubuntu/ReToneProjects/RTMemory/server/app/api/__init__.py`

```python
# RTMemory API Routers
```

**File:** `/home/ubuntu/ReToneProjects/RTMemory/server/tests/__init__.py`

```python
# RTMemory Tests
```

**Run:**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory/server && touch app/__init__.py app/db/__init__.py app/api/__init__.py tests/__init__.py
```

Then overwrite each with the content above using the Write tool.

**Expected:** All `__init__.py` files exist.

**Commit:**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory && git add server/pyproject.toml server/app/__init__.py server/app/db/__init__.py server/app/api/__init__.py server/tests/__init__.py && git commit -m "chore: add project skeleton with pyproject.toml and directory structure"
```

---

## Phase 2: Configuration System

### Step 4: Write failing test for config loading

**File:** `/home/ubuntu/ReToneProjects/RTMemory/server/tests/test_config.py`

```python
"""Tests for configuration loading via pydantic-settings."""

import os
import tempfile

import pytest
import yaml

from app.config import Settings, DatabaseConfig, LLMConfig, EmbeddingConfig, ServerConfig


class TestDatabaseConfig:
    def test_default_database_config(self):
        cfg = DatabaseConfig()
        assert cfg.host == "localhost"
        assert cfg.port == 5432
        assert cfg.user == "rtmemory"
        assert cfg.password == "secret"
        assert cfg.database == "rtmemory"
        assert "postgresql+asyncpg" in cfg.url

    def test_database_url_from_components(self):
        cfg = DatabaseConfig(host="db", port=5433, user="admin", password="pw", database="testdb")
        assert cfg.url == "postgresql+asyncpg://admin:pw@db:5433/testdb"

    def test_database_url_env_override(self, monkeypatch):
        monkeypatch.setenv("RTMEM_DATABASE_URL", "postgresql+asyncpg://u:p@h:5432/db")
        cfg = DatabaseConfig()
        assert cfg.url == "postgresql+asyncpg://u:p@h:5432/db"


class TestLLMConfig:
    def test_default_llm_config(self):
        cfg = LLMConfig()
        assert cfg.provider == "ollama"
        assert cfg.model == "qwen2.5:7b"
        assert cfg.base_url == "http://localhost:11434"

    def test_llm_env_override(self, monkeypatch):
        monkeypatch.setenv("RTMEM_LLM_PROVIDER", "openai")
        monkeypatch.setenv("RTMEM_LLM_API_KEY", "sk-test")
        monkeypatch.setenv("RTMEM_LLM_MODEL", "gpt-4o")
        cfg = LLMConfig()
        assert cfg.provider == "openai"
        assert cfg.api_key == "sk-test"
        assert cfg.model == "gpt-4o"


class TestEmbeddingConfig:
    def test_default_embedding_config(self):
        cfg = EmbeddingConfig()
        assert cfg.provider == "local"
        assert cfg.model == "BAAI/bge-base-zh-v1.5"
        assert cfg.vector_dimension == 768

    def test_embedding_dimension_env_override(self, monkeypatch):
        monkeypatch.setenv("RTMEM_EMBEDDING_VECTOR_DIMENSION", "1536")
        cfg = EmbeddingConfig()
        assert cfg.vector_dimension == 1536


class TestSettings:
    def test_settings_from_yaml(self, tmp_path):
        config_data = {
            "server": {"host": "0.0.0.0", "port": 9000},
            "database": {"host": "db-server", "port": 5433},
            "embedding": {"vector_dimension": 1536, "provider": "openai"},
        }
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(yaml.dump(config_data))
        settings = Settings.from_yaml(str(yaml_path))
        assert settings.server.host == "0.0.0.0"
        assert settings.server.port == 9000
        assert settings.database.host == "db-server"
        assert settings.embedding.vector_dimension == 1536

    def test_settings_defaults(self):
        settings = Settings()
        assert settings.server.host == "0.0.0.0"
        assert settings.server.port == 8000
        assert settings.embedding.vector_dimension == 768

    def test_settings_env_overrides_yaml(self, tmp_path, monkeypatch):
        config_data = {
            "server": {"port": 9000},
            "embedding": {"vector_dimension": 768},
        }
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(yaml.dump(config_data))
        monkeypatch.setenv("RTMEM_SERVER_PORT", "3000")
        monkeypatch.setenv("RTMEM_EMBEDDING_VECTOR_DIMENSION", "1536")
        settings = Settings.from_yaml(str(yaml_path))
        assert settings.server.port == 3000
        assert settings.embedding.vector_dimension == 1536
```

**Run:**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/test_config.py -v 2>&1 | head -30
```

**Expected:** All tests FAIL (ImportError: cannot import `Settings` from `app.config`).

---

### Step 5: Implement configuration system

**File:** `/home/ubuntu/ReToneProjects/RTMemory/server/app/config.py`

```python
"""Configuration system using pydantic-settings with YAML + env var overrides.

YAML config.yaml provides defaults. Environment variables prefixed with RTMEM_ override.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings


class DatabaseConfig(BaseSettings):
    """PostgreSQL connection settings."""

    host: str = Field(default="localhost", description="PostgreSQL host")
    port: int = Field(default=5432, description="PostgreSQL port")
    user: str = Field(default="rtmemory", description="PostgreSQL user")
    password: str = Field(default="secret", description="PostgreSQL password")
    database: str = Field(default="rtmemory", description="PostgreSQL database name")
    url: str = Field(
        default="",
        description="Full database URL (overrides individual fields if set)",
    )

    model_config = {"env_prefix": "RTMEM_DATABASE_"}

    @property
    def effective_url(self) -> str:
        """Return the URL to use: explicit url if set, otherwise built from components."""
        if self.url:
            return self.url
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


class ServerConfig(BaseSettings):
    """FastAPI server settings."""

    host: str = Field(default="0.0.0.0", description="Server bind host")
    port: int = Field(default=8000, description="Server bind port")
    debug: bool = Field(default=False, description="Enable debug mode")
    cors_origins: list[str] = Field(
        default=["*"], description="Allowed CORS origins"
    )

    model_config = {"env_prefix": "RTMEM_SERVER_"}


class LLMConfig(BaseSettings):
    """LLM provider settings."""

    provider: str = Field(default="ollama", description="LLM provider: openai, anthropic, ollama")
    model: str = Field(default="qwen2.5:7b", description="Model name")
    api_key: Optional[str] = Field(default=None, description="API key for cloud providers")
    base_url: str = Field(default="http://localhost:11434", description="LLM API base URL")
    temperature: float = Field(default=0.1, description="Sampling temperature")
    max_tokens: int = Field(default=2048, description="Max output tokens")

    model_config = {"env_prefix": "RTMEM_LLM_"}


class EmbeddingConfig(BaseSettings):
    """Embedding model settings."""

    provider: str = Field(default="local", description="Embedding provider: local, openai")
    model: str = Field(default="BAAI/bge-base-zh-v1.5", description="Embedding model name")
    vector_dimension: int = Field(default=768, description="Vector embedding dimension")
    base_url: Optional[str] = Field(default=None, description="API base URL for remote providers")
    api_key: Optional[str] = Field(default=None, description="API key for remote providers")

    model_config = {"env_prefix": "RTMEM_EMBEDDING_"}


class Settings(BaseSettings):
    """Root settings object aggregating all sub-configs."""

    server: ServerConfig = Field(default_factory=ServerConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)

    model_config = {"env_prefix": "RTMEM_"}

    @classmethod
    def from_yaml(cls, path: str) -> "Settings":
        """Load settings from a YAML file, then apply env var overrides.

        Args:
            path: Path to config.yaml file.

        Returns:
            Settings instance with YAML defaults and env var overrides applied.
        """
        yaml_path = Path(path)
        if not yaml_path.exists():
            return cls()

        with open(yaml_path) as f:
            data = yaml.safe_load(f) or {}

        return cls(**data)


# Module-level singleton — lazy loaded on first access
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Return the global Settings singleton. Loads from config.yaml in CWD if present."""
    global _settings
    if _settings is None:
        config_path = Path.cwd() / "config.yaml"
        if config_path.exists():
            _settings = Settings.from_yaml(str(config_path))
        else:
            _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Reset the cached settings singleton (useful for testing)."""
    global _settings
    _settings = None
```

**Run:**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/test_config.py -v
```

**Expected:** All tests PASS.

---

### Step 6: Create default config.yaml

**File:** `/home/ubuntu/ReToneProjects/RTMemory/config.yaml`

```yaml
# RTMemory default configuration
# Environment variables with RTMEM_ prefix override these values.

server:
  host: "0.0.0.0"
  port: 8000
  debug: false
  cors_origins:
    - "*"

database:
  host: "localhost"
  port: 5432
  user: "rtmemory"
  password: "secret"
  database: "rtmemory"
  # url: ""  # Set to override host/port/user/password/database

llm:
  provider: "ollama"
  model: "qwen2.5:7b"
  base_url: "http://localhost:11434"
  temperature: 0.1
  max_tokens: 2048

embedding:
  provider: "local"
  model: "BAAI/bge-base-zh-v1.5"
  vector_dimension: 768
  # For OpenAI embeddings:
  # provider: "openai"
  # model: "text-embedding-3-small"
  # vector_dimension: 1536
```

**Run:**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory && python -c "import yaml; d = yaml.safe_load(open('config.yaml')); print(d['embedding']['vector_dimension'])"
```

**Expected:** `768`

**Commit:**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory && git add server/app/config.py server/tests/test_config.py config.yaml && git commit -m "feat: add pydantic-settings config system with YAML + env overrides"
```

---

## Phase 3: Database Models

### Step 7: Write failing test for database models

**File:** `/home/ubuntu/ReToneProjects/RTMemory/server/tests/test_models.py`

```python
"""Tests for SQLAlchemy database models."""

import uuid
from datetime import datetime, timezone

import pytest

from app.db.base import Base
from app.db.models import (
    Space,
    Entity,
    Relation,
    Memory,
    Document,
    Chunk,
    MemorySource,
    EntityType,
    MemoryType,
    DocType,
    DocStatus,
)


class TestSpaceModel:
    def test_space_creation(self):
        space = Space(
            name="Test Space",
            description="A test space",
            org_id=uuid.uuid4(),
            owner_id=uuid.uuid4(),
        )
        assert space.name == "Test Space"
        assert space.is_default is False
        assert space.container_tag is None
        assert isinstance(space.id, uuid.UUID)

    def test_space_defaults(self):
        space = Space(name="Default")
        assert space.is_default is False


class TestEntityModel:
    def test_entity_creation(self):
        entity = Entity(
            name="张军",
            entity_type=EntityType.person,
            description="A software engineer",
            org_id=uuid.uuid4(),
            space_id=uuid.uuid4(),
        )
        assert entity.name == "张军"
        assert entity.entity_type == EntityType.person
        assert entity.confidence == 1.0

    def test_entity_types(self):
        assert EntityType.person == "person"
        assert EntityType.org == "org"
        assert EntityType.location == "location"
        assert EntityType.concept == "concept"
        assert EntityType.project == "project"
        assert EntityType.technology == "technology"


class TestRelationModel:
    def test_relation_creation(self):
        src = uuid.uuid4()
        tgt = uuid.uuid4()
        relation = Relation(
            source_entity_id=src,
            target_entity_id=tgt,
            relation_type="lives_in",
            value="北京",
            org_id=uuid.uuid4(),
            space_id=uuid.uuid4(),
        )
        assert relation.relation_type == "lives_in"
        assert relation.is_current is True
        assert relation.valid_to is None
        assert relation.source_count == 1

    def test_relation_temporal_defaults(self):
        relation = Relation(
            source_entity_id=uuid.uuid4(),
            target_entity_id=uuid.uuid4(),
            relation_type="works_at",
            org_id=uuid.uuid4(),
            space_id=uuid.uuid4(),
        )
        assert relation.is_current is True
        assert relation.confidence == 1.0
        assert relation.source_count == 1
        assert relation.valid_to is None


class TestMemoryModel:
    def test_memory_creation(self):
        memory = Memory(
            content="张军最近在研究知识图谱",
            memory_type=MemoryType.fact,
            org_id=uuid.uuid4(),
            space_id=uuid.uuid4(),
        )
        assert memory.content == "张军最近在研究知识图谱"
        assert memory.memory_type == MemoryType.fact
        assert memory.is_forgotten is False
        assert memory.version == 1
        assert memory.confidence == 1.0

    def test_memory_types(self):
        assert MemoryType.fact == "fact"
        assert MemoryType.preference == "preference"
        assert MemoryType.status == "status"
        assert MemoryType.inference == "inference"

    def test_memory_decay_defaults(self):
        memory = Memory(
            content="test",
            memory_type=MemoryType.fact,
            org_id=uuid.uuid4(),
            space_id=uuid.uuid4(),
        )
        assert memory.decay_rate == 0.02
        assert memory.is_forgotten is False
        assert memory.forget_at is None


class TestDocumentModel:
    def test_document_creation(self):
        doc = Document(
            title="Next.js Guide",
            content="Full guide content here...",
            doc_type=DocType.text,
            org_id=uuid.uuid4(),
            space_id=uuid.uuid4(),
        )
        assert doc.title == "Next.js Guide"
        assert doc.doc_type == DocType.text
        assert doc.status == DocStatus.queued

    def test_document_types(self):
        assert DocType.text == "text"
        assert DocType.pdf == "pdf"
        assert DocType.webpage == "webpage"

    def test_document_status_flow(self):
        assert DocStatus.queued == "queued"
        assert DocStatus.extracting == "extracting"
        assert DocStatus.chunking == "chunking"
        assert DocStatus.embedding == "embedding"
        assert DocStatus.done == "done"
        assert DocStatus.failed == "failed"


class TestChunkModel:
    def test_chunk_creation(self):
        chunk = Chunk(
            document_id=uuid.uuid4(),
            content="A chunk of text",
            position=0,
        )
        assert chunk.content == "A chunk of text"
        assert chunk.position == 0


class TestMemorySourceModel:
    def test_memory_source_creation(self):
        ms = MemorySource(
            memory_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            relevance_score=0.85,
        )
        assert ms.relevance_score == 0.85
```

**Run:**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/test_models.py -v 2>&1 | head -20
```

**Expected:** All tests FAIL (ImportError: cannot import models).

---

### Step 8: Implement SQLAlchemy Base with VECTOR type

**File:** `/home/ubuntu/ReToneProjects/RTMemory/server/app/db/base.py`

```python
"""SQLAlchemy Base class and custom VECTOR type for pgvector support."""

from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.types import TypeDecorator
from pgvector.sqlalchemy import Vector


# Naming convention for constraints — required for Alembic autogenerate
convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

metadata = MetaData(naming_convention=convention)


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    metadata = metadata


__all__ = ["Base", "metadata"]
```

**Run:**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory/server && python -c "from app.db.base import Base; print(Base.metadata)"
```

**Expected:** Prints `<MetaData object at 0x...>`.

---

### Step 9: Implement ALL 7 database models

**File:** `/home/ubuntu/ReToneProjects/RTMemory/server/app/db/models.py`

```python
"""SQLAlchemy ORM models for all 7 RTMemory tables.

Tables:
  1. spaces       — Isolation boundary for all data
  2. entities     — Nodes in the temporal knowledge graph
  3. relations    — Temporal edges between entities
  4. memories     — Discrete memory entries with decay
  5. documents    — Uploaded documents for knowledge base
  6. chunks       — Segments of documents
  7. memory_sources — Traceability: memory -> document/chunk provenance
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    Index,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.config import get_settings
from app.db.base import Base


# ---------------------------------------------------------------------------
# Enum-like string types
# ---------------------------------------------------------------------------

class EntityType(str):
    """Entity type enum."""
    person = "person"
    org = "org"
    location = "location"
    concept = "concept"
    project = "project"
    technology = "technology"


class MemoryType(str):
    """Memory type enum."""
    fact = "fact"
    preference = "preference"
    status = "status"
    inference = "inference"


class DocType(str):
    """Document type enum."""
    text = "text"
    pdf = "pdf"
    webpage = "webpage"


class DocStatus(str):
    """Document processing status enum."""
    queued = "queued"
    extracting = "extracting"
    chunking = "chunking"
    embedding = "embedding"
    done = "done"
    failed = "failed"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _vector_dimension() -> int:
    """Read vector dimension from config. Falls back to 768 if config not ready."""
    try:
        return get_settings().embedding.vector_dimension
    except Exception:
        return 768


# ---------------------------------------------------------------------------
# 1. Spaces
# ---------------------------------------------------------------------------

class Space(Base):
    __tablename__ = "spaces"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True, default=None)
    org_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    owner_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), nullable=True, default=None
    )
    container_tag: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True, default=None
    )
    is_default: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow
    )

    # Relationships
    entities: Mapped[list["Entity"]] = relationship(
        "Entity", back_populates="space", passive_deletes=True
    )
    relations: Mapped[list["Relation"]] = relationship(
        "Relation", back_populates="space", passive_deletes=True
    )
    memories: Mapped[list["Memory"]] = relationship(
        "Memory", back_populates="space", passive_deletes=True
    )
    documents: Mapped[list["Document"]] = relationship(
        "Document", back_populates="space", passive_deletes=True
    )

    def __repr__(self) -> str:
        return f"<Space id={self.id} name={self.name!r}>"


# ---------------------------------------------------------------------------
# 2. Entities
# ---------------------------------------------------------------------------

class Entity(Base):
    __tablename__ = "entities"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(512), nullable=False)
    entity_type: Mapped[str] = mapped_column(
        String(50), nullable=False, default=EntityType.person
    )
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True, default=None)
    embedding = mapped_column(Vector(_vector_dimension()), nullable=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    org_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("spaces.org_id", use_alter=True, name="fk_entities_org_id"),
        nullable=False,
    )
    space_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("spaces.id", ondelete="CASCADE"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow
    )

    # Relationships
    space: Mapped["Space"] = relationship("Space", back_populates="entities")
    source_relations: Mapped[list["Relation"]] = relationship(
        "Relation",
        foreign_keys="Relation.source_entity_id",
        back_populates="source_entity",
        passive_deletes=True,
    )
    target_relations: Mapped[list["Relation"]] = relationship(
        "Relation",
        foreign_keys="Relation.target_entity_id",
        back_populates="target_entity",
        passive_deletes=True,
    )
    memories: Mapped[list["Memory"]] = relationship(
        "Memory", back_populates="entity", passive_deletes=True
    )

    __table_args__ = (
        Index("ix_entities_space_id", "space_id"),
        Index("ix_entities_org_id", "org_id"),
        Index("ix_entities_entity_type", "entity_type"),
    )

    def __repr__(self) -> str:
        return f"<Entity id={self.id} name={self.name!r} type={self.entity_type}>"


# ---------------------------------------------------------------------------
# 3. Relations (temporal edges)
# ---------------------------------------------------------------------------

class Relation(Base):
    __tablename__ = "relations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    source_entity_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("entities.id", ondelete="CASCADE"), nullable=False
    )
    target_entity_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("entities.id", ondelete="CASCADE"), nullable=False
    )
    relation_type: Mapped[str] = mapped_column(String(255), nullable=False)
    value: Mapped[Optional[str]] = mapped_column(Text, nullable=True, default=None)
    valid_from: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    valid_to: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None
    )
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    is_current: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    source_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    embedding = mapped_column(Vector(_vector_dimension()), nullable=True)
    org_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    space_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("spaces.id", ondelete="CASCADE"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow
    )

    # Relationships
    space: Mapped["Space"] = relationship("Space", back_populates="relations")
    source_entity: Mapped["Entity"] = relationship(
        "Entity", foreign_keys=[source_entity_id], back_populates="source_relations"
    )
    target_entity: Mapped["Entity"] = relationship(
        "Entity", foreign_keys=[target_entity_id], back_populates="target_relations"
    )
    memories: Mapped[list["Memory"]] = relationship(
        "Memory", back_populates="relation", passive_deletes=True
    )

    __table_args__ = (
        Index("ix_relations_source_entity_id", "source_entity_id"),
        Index("ix_relations_target_entity_id", "target_entity_id"),
        Index("ix_relations_space_id", "space_id"),
        Index("ix_relations_relation_type", "relation_type"),
        Index("ix_relations_is_current", "is_current"),
    )

    def __repr__(self) -> str:
        return f"<Relation id={self.id} type={self.relation_type} current={self.is_current}>"


# ---------------------------------------------------------------------------
# 4. Memories
# ---------------------------------------------------------------------------

class Memory(Base):
    __tablename__ = "memories"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    custom_id: Mapped[Optional[str]] = mapped_column(
        String(512), nullable=True, default=None
    )
    memory_type: Mapped[str] = mapped_column(
        String(50), nullable=False, default=MemoryType.fact
    )
    entity_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("entities.id", ondelete="SET NULL"), nullable=True
    )
    relation_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("relations.id", ondelete="SET NULL"), nullable=True
    )
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    decay_rate: Mapped[float] = mapped_column(Float, nullable=False, default=0.02)
    is_forgotten: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    forget_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None
    )
    forget_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True, default=None)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    parent_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("memories.id", ondelete="SET NULL"), nullable=True
    )
    root_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("memories.id", ondelete="SET NULL"), nullable=True
    )
    metadata_: Mapped[Optional[dict]] = mapped_column(
        "metadata", JSONB, nullable=True, default=None
    )
    embedding = mapped_column(Vector(_vector_dimension()), nullable=True)
    org_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    space_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("spaces.id", ondelete="CASCADE"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow
    )

    # Relationships
    space: Mapped["Space"] = relationship("Space", back_populates="memories")
    entity: Mapped[Optional["Entity"]] = relationship("Entity", back_populates="memories")
    relation: Mapped[Optional["Relation"]] = relationship("Relation", back_populates="memories")

    __table_args__ = (
        Index("ix_memories_space_id", "space_id"),
        Index("ix_memories_entity_id", "entity_id"),
        Index("ix_memories_memory_type", "memory_type"),
        Index("ix_memories_is_forgotten", "is_forgotten"),
        Index("ix_memories_custom_id", "custom_id"),
    )

    def __repr__(self) -> str:
        return f"<Memory id={self.id} type={self.memory_type} forgotten={self.is_forgotten}>"


# ---------------------------------------------------------------------------
# 5. Documents
# ---------------------------------------------------------------------------

class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    content: Mapped[Optional[str]] = mapped_column(Text, nullable=True, default=None)
    doc_type: Mapped[str] = mapped_column(
        String(50), nullable=False, default=DocType.text
    )
    url: Mapped[Optional[str]] = mapped_column(Text, nullable=True, default=None)
    status: Mapped[str] = mapped_column(
        String(50), nullable=False, default=DocStatus.queued
    )
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True, default=None)
    summary_embedding = mapped_column(Vector(_vector_dimension()), nullable=True)
    metadata_: Mapped[Optional[dict]] = mapped_column(
        "metadata", JSONB, nullable=True, default=None
    )
    org_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    space_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("spaces.id", ondelete="CASCADE"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow
    )

    # Relationships
    space: Mapped["Space"] = relationship("Space", back_populates="documents")
    chunks: Mapped[list["Chunk"]] = relationship(
        "Chunk", back_populates="document", passive_deletes=True
    )

    __table_args__ = (
        Index("ix_documents_space_id", "space_id"),
        Index("ix_documents_status", "status"),
        Index("ix_documents_doc_type", "doc_type"),
    )

    def __repr__(self) -> str:
        return f"<Document id={self.id} title={self.title!r} status={self.status}>"


# ---------------------------------------------------------------------------
# 6. Chunks
# ---------------------------------------------------------------------------

class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding = mapped_column(Vector(_vector_dimension()), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    # Relationships
    document: Mapped["Document"] = relationship("Document", back_populates="chunks")

    __table_args__ = (
        Index("ix_chunks_document_id", "document_id"),
    )

    def __repr__(self) -> str:
        return f"<Chunk id={self.id} doc={self.document_id} pos={self.position}>"


# ---------------------------------------------------------------------------
# 7. Memory Sources (traceability)
# ---------------------------------------------------------------------------

class MemorySource(Base):
    __tablename__ = "memory_sources"

    memory_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("memories.id", ondelete="CASCADE"),
        nullable=False,
        primary_key=True,
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        primary_key=True,
    )
    chunk_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chunks.id", ondelete="SET NULL"),
        nullable=True,
        default=None,
    )
    relevance_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    __table_args__ = (
        Index("ix_memory_sources_memory_id", "memory_id"),
        Index("ix_memory_sources_document_id", "document_id"),
    )

    def __repr__(self) -> str:
        return f"<MemorySource memory={self.memory_id} doc={self.document_id} score={self.relevance_score}>"
```

**Run:**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/test_models.py -v
```

**Expected:** All 14 tests PASS.

**Commit:**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory && git add server/app/db/ server/tests/test_models.py && git commit -m "feat: add SQLAlchemy models for all 7 tables with pgvector support"
```

---

### Step 10: Implement async database session

**File:** `/home/ubuntu/ReToneProjects/RTMemory/server/app/db/session.py`

```python
"""Async SQLAlchemy engine and session factory."""

from __future__ import annotations

from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import get_settings


_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """Return the global async engine, creating it if necessary."""
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(
            settings.database.effective_url,
            echo=settings.server.debug,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
        )
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return the global session factory, creating it if necessary."""
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _session_factory


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields an async database session."""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def close_engine() -> None:
    """Dispose the engine. Call on application shutdown."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
```

**Run:**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory/server && python -c "from app.db.session import get_engine, get_session, close_engine; print('session module OK')"
```

**Expected:** `session module OK`

---

## Phase 4: Alembic Migration Setup

### Step 11: Create alembic.ini

**File:** `/home/ubuntu/ReToneProjects/RTMemory/server/alembic.ini`

```ini
[alembic]
script_location = alembic
prepend_sys_path = .
sqlalchemy.url = postgresql+asyncpg://rtmemory:secret@localhost:5432/rtmemory

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
```

---

### Step 12: Create alembic env.py for async support

**File:** `/home/ubuntu/ReToneProjects/RTMemory/server/alembic/env.py`

```python
"""Alembic env.py with async SQLAlchemy support and pgvector extension."""

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config

# Import all models so Alembic can detect them
from app.db.base import Base
from app.db.models import (  # noqa: F401 — ensure models registered on Base
    Space,
    Entity,
    Relation,
    Memory,
    Document,
    Chunk,
    MemorySource,
)

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (emit SQL without connecting)."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=False,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:
    """Run migrations synchronously using the provided connection."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        render_as_batch=False,
    )
    with context.begin_transaction():
        # Ensure pgvector extension is available
        connection.execute(context._proxy.sql("CREATE EXTENSION IF NOT EXISTS vector"))
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations in 'online' mode with async engine."""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    """Entry point for online migrations — dispatches to async."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

---

### Step 13: Create alembic script.py.mako

**File:** `/home/ubuntu/ReToneProjects/RTMemory/server/alembic/script.py.mako`

```mako
"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Create Date: ${create_date}
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
${imports if imports else ""}

# revision identifiers, used by Alembic.
revision: str = ${repr(up_revision)}
down_revision: Union[str, None] = ${repr(down_revision)}
branch_labels: Union[str, Sequence[str], None] = ${repr(branch_labels)}
depends_on: Union[str, Sequence[str], None] = ${repr(depends_on)}


def upgrade() -> None:
    ${upgrades if upgrades else "pass"}


def downgrade() -> None:
    ${downgrades if downgrades else "pass"}
```

---

### Step 14: Generate initial migration

**File:** `/home/ubuntu/ReToneProjects/RTMemory/server/alembic/versions/001_initial.py`

```python
"""Initial schema — all 7 RTMemory tables.

Revision ID: 001_initial
Revises: None
Create Date: 2026-04-23
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = "001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Ensure pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # 1. spaces
    op.create_table(
        "spaces",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("org_id", UUID(as_uuid=True), nullable=False),
        sa.Column("owner_id", UUID(as_uuid=True), nullable=True),
        sa.Column("container_tag", sa.String(255), nullable=True),
        sa.Column("is_default", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )

    # 2. entities
    op.create_table(
        "entities",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(512), nullable=False),
        sa.Column("entity_type", sa.String(50), nullable=False, server_default="person"),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("embedding", Vector(768), nullable=True),
        sa.Column("confidence", sa.Float, nullable=False, server_default="1.0"),
        sa.Column("org_id", UUID(as_uuid=True), nullable=False),
        sa.Column("space_id", UUID(as_uuid=True), sa.ForeignKey("spaces.id", ondelete="CASCADE"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_entities_space_id", "entities", ["space_id"])
    op.create_index("ix_entities_org_id", "entities", ["org_id"])
    op.create_index("ix_entities_entity_type", "entities", ["entity_type"])

    # 3. relations
    op.create_table(
        "relations",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("source_entity_id", UUID(as_uuid=True), sa.ForeignKey("entities.id", ondelete="CASCADE"), nullable=False),
        sa.Column("target_entity_id", UUID(as_uuid=True), sa.ForeignKey("entities.id", ondelete="CASCADE"), nullable=False),
        sa.Column("relation_type", sa.String(255), nullable=False),
        sa.Column("value", sa.Text, nullable=True),
        sa.Column("valid_from", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("valid_to", sa.DateTime(timezone=True), nullable=True),
        sa.Column("confidence", sa.Float, nullable=False, server_default="1.0"),
        sa.Column("is_current", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("source_count", sa.Integer, nullable=False, server_default="1"),
        sa.Column("embedding", Vector(768), nullable=True),
        sa.Column("org_id", UUID(as_uuid=True), nullable=False),
        sa.Column("space_id", UUID(as_uuid=True), sa.ForeignKey("spaces.id", ondelete="CASCADE"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_relations_source_entity_id", "relations", ["source_entity_id"])
    op.create_index("ix_relations_target_entity_id", "relations", ["target_entity_id"])
    op.create_index("ix_relations_space_id", "relations", ["space_id"])
    op.create_index("ix_relations_relation_type", "relations", ["relation_type"])
    op.create_index("ix_relations_is_current", "relations", ["is_current"])

    # 4. memories
    op.create_table(
        "memories",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("custom_id", sa.String(512), nullable=True),
        sa.Column("memory_type", sa.String(50), nullable=False, server_default="fact"),
        sa.Column("entity_id", UUID(as_uuid=True), sa.ForeignKey("entities.id", ondelete="SET NULL"), nullable=True),
        sa.Column("relation_id", UUID(as_uuid=True), sa.ForeignKey("relations.id", ondelete="SET NULL"), nullable=True),
        sa.Column("confidence", sa.Float, nullable=False, server_default="1.0"),
        sa.Column("decay_rate", sa.Float, nullable=False, server_default="0.02"),
        sa.Column("is_forgotten", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("forget_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("forget_reason", sa.Text, nullable=True),
        sa.Column("version", sa.Integer, nullable=False, server_default="1"),
        sa.Column("parent_id", UUID(as_uuid=True), sa.ForeignKey("memories.id", ondelete="SET NULL"), nullable=True),
        sa.Column("root_id", UUID(as_uuid=True), sa.ForeignKey("memories.id", ondelete="SET NULL"), nullable=True),
        sa.Column("metadata", JSONB, nullable=True),
        sa.Column("embedding", Vector(768), nullable=True),
        sa.Column("org_id", UUID(as_uuid=True), nullable=False),
        sa.Column("space_id", UUID(as_uuid=True), sa.ForeignKey("spaces.id", ondelete="CASCADE"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_memories_space_id", "memories", ["space_id"])
    op.create_index("ix_memories_entity_id", "memories", ["entity_id"])
    op.create_index("ix_memories_memory_type", "memories", ["memory_type"])
    op.create_index("ix_memories_is_forgotten", "memories", ["is_forgotten"])
    op.create_index("ix_memories_custom_id", "memories", ["custom_id"])

    # 5. documents
    op.create_table(
        "documents",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("title", sa.String(512), nullable=False),
        sa.Column("content", sa.Text, nullable=True),
        sa.Column("doc_type", sa.String(50), nullable=False, server_default="text"),
        sa.Column("url", sa.Text, nullable=True),
        sa.Column("status", sa.String(50), nullable=False, server_default="queued"),
        sa.Column("summary", sa.Text, nullable=True),
        sa.Column("summary_embedding", Vector(768), nullable=True),
        sa.Column("metadata", JSONB, nullable=True),
        sa.Column("org_id", UUID(as_uuid=True), nullable=False),
        sa.Column("space_id", UUID(as_uuid=True), sa.ForeignKey("spaces.id", ondelete="CASCADE"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_documents_space_id", "documents", ["space_id"])
    op.create_index("ix_documents_status", "documents", ["status"])
    op.create_index("ix_documents_doc_type", "documents", ["doc_type"])

    # 6. chunks
    op.create_table(
        "chunks",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("document_id", UUID(as_uuid=True), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("position", sa.Integer, nullable=False),
        sa.Column("embedding", Vector(768), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_chunks_document_id", "chunks", ["document_id"])

    # 7. memory_sources
    op.create_table(
        "memory_sources",
        sa.Column("memory_id", UUID(as_uuid=True), sa.ForeignKey("memories.id", ondelete="CASCADE"), nullable=False, primary_key=True),
        sa.Column("document_id", UUID(as_uuid=True), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, primary_key=True),
        sa.Column("chunk_id", UUID(as_uuid=True), sa.ForeignKey("chunks.id", ondelete="SET NULL"), nullable=True),
        sa.Column("relevance_score", sa.Float, nullable=False, server_default="0.0"),
    )
    op.create_index("ix_memory_sources_memory_id", "memory_sources", ["memory_id"])
    op.create_index("ix_memory_sources_document_id", "memory_sources", ["document_id"])


def downgrade() -> None:
    op.drop_table("memory_sources")
    op.drop_table("chunks")
    op.drop_table("documents")
    op.drop_table("memories")
    op.drop_table("relations")
    op.drop_table("entities")
    op.drop_table("spaces")
    op.execute("DROP EXTENSION IF EXISTS vector")
```

**Commit:**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory && git add server/alembic/ server/alembic.ini && git commit -m "feat: add Alembic migration setup with async support and initial 7-table schema"
```

---

## Phase 5: FastAPI App Entry Point

### Step 15: Write failing test for health check

**File:** `/home/ubuntu/ReToneProjects/RTMemory/server/tests/test_health.py`

```python
"""Tests for the FastAPI health check endpoint."""

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestHealthCheck:
    async def test_health_returns_ok(self, client):
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data

    async def test_root_returns_info(self, client):
        response = await client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "RTMemory"
        assert "version" in data
```

**Run:**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/test_health.py -v 2>&1 | head -20
```

**Expected:** Tests FAIL (ImportError: cannot import `app`).

---

### Step 16: Implement FastAPI app with health check

**File:** `/home/ubuntu/ReToneProjects/RTMemory/server/app/api/deps.py`

```python
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
```

**File:** `/home/ubuntu/ReToneProjects/RTMemory/server/app/main.py`

```python
"""RTMemory FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.db.session import close_engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown hooks."""
    # Startup — no special actions needed; engine is lazy-created
    yield
    # Shutdown — close database engine
    await close_engine()


app = FastAPI(
    title="RTMemory",
    description="Temporal Knowledge Graph-Driven AI Memory System",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
_settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.server.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and register routers
from app.api.spaces import router as spaces_router  # noqa: E402

app.include_router(spaces_router, prefix="/v1/spaces", tags=["spaces"])


@app.get("/")
async def root():
    """Root endpoint — service info."""
    return {
        "name": "RTMemory",
        "version": "0.1.0",
        "description": "Temporal Knowledge Graph-Driven AI Memory System",
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "version": "0.1.0",
    }
```

**Run:**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/test_health.py -v
```

**Expected:** Both health tests PASS.

**Commit:**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory && git add server/app/main.py server/app/api/deps.py server/tests/test_health.py && git commit -m "feat: add FastAPI app entry point with health check and CORS"
```

---

## Phase 6: Spaces CRUD API

### Step 17: Write failing tests for Spaces CRUD

**File:** `/home/ubuntu/ReToneProjects/RTMemory/server/tests/test_spaces_api.py`

```python
"""Tests for Spaces CRUD API endpoints."""

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from app.db.base import Base
from app.db.models import Space
from app.main import app


# Use SQLite for API-level tests (no pgvector needed for CRUD)
TEST_DB_URL = "sqlite+aiosqlite:///./test_spaces.db"


@pytest.fixture
async def db_engine():
    engine = create_async_engine(TEST_DB_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
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
```

**Run:**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory/server && pip install aiosqlite 2>&1 | tail -2 && python -m pytest tests/test_spaces_api.py -v 2>&1 | head -30
```

**Expected:** Tests FAIL (ImportError or 404s because spaces router not yet implemented).

---

### Step 18: Implement Spaces CRUD router

**File:** `/home/ubuntu/ReToneProjects/RTMemory/server/app/api/spaces.py`

```python
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
```

**Run:**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory/server && pip install aiosqlite 2>&1 | tail -2 && python -m pytest tests/test_spaces_api.py -v
```

**Expected:** All 7 Spaces CRUD tests PASS.

**Commit:**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory && git add server/app/api/spaces.py server/tests/test_spaces_api.py && git commit -m "feat: add Spaces CRUD API with create, list, get, delete endpoints"
```

---

## Phase 7: Docker Compose + Dockerfile

### Step 19: Create Dockerfile

**File:** `/home/ubuntu/ReToneProjects/RTMemory/Dockerfile`

```dockerfile
# RTMemory Server Dockerfile
FROM python:3.12-slim

WORKDIR /app

# System deps for pgvector and PDF processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY server/pyproject.toml ./
RUN pip install --no-cache-dir -e ".[dev]"

# Copy source code
COPY server/app/ ./app/
COPY server/alembic/ ./alembic/
COPY server/alembic.ini ./alembic.ini
COPY config.yaml ./config.yaml

# Expose port
EXPOSE 8000

# Run migrations on startup, then serve
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
```

---

### Step 20: Create docker-compose.yml

**File:** `/home/ubuntu/ReToneProjects/RTMemory/docker-compose.yml`

```yaml
version: "3.9"

services:
  rtmemory-api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - RTMEM_DATABASE_HOST=postgres
      - RTMEM_DATABASE_PORT=5432
      - RTMEM_DATABASE_USER=rtmemory
      - RTMEM_DATABASE_PASSWORD=secret
      - RTMEM_DATABASE_DATABASE=rtmemory
      - RTMEM_LLM_PROVIDER=ollama
      - RTMEM_LLM_MODEL=qwen2.5:7b
      - RTMEM_LLM_BASE_URL=http://ollama:11434
      - RTMEM_EMBEDDING_PROVIDER=local
      - RTMEM_EMBEDDING_MODEL=BAAI/bge-base-zh-v1.5
      - RTMEM_EMBEDDING_VECTOR_DIMENSION=768
    depends_on:
      postgres:
        condition: service_healthy
    restart: unless-stopped

  rtmemory-worker:
    build:
      context: .
      dockerfile: Dockerfile
    command: python -m app.worker
    environment:
      - RTMEM_DATABASE_HOST=postgres
      - RTMEM_DATABASE_PORT=5432
      - RTMEM_DATABASE_USER=rtmemory
      - RTMEM_DATABASE_PASSWORD=secret
      - RTMEM_DATABASE_DATABASE=rtmemory
      - RTMEM_LLM_PROVIDER=ollama
      - RTMEM_LLM_BASE_URL=http://ollama:11434
    depends_on:
      postgres:
        condition: service_healthy
    restart: unless-stopped

  postgres:
    image: pgvector/pgvector:pg17
    volumes:
      - pgdata:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    environment:
      - POSTGRES_DB=rtmemory
      - POSTGRES_USER=rtmemory
      - POSTGRES_PASSWORD=secret
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U rtmemory"]
      interval: 5s
      timeout: 5s
      retries: 5

  ollama:
    image: ollama/ollama
    volumes:
      - ollama_data:/root/.ollama
    ports:
      - "11434:11434"
    restart: unless-stopped

volumes:
  pgdata:
  ollama_data:
```

**Commit:**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory && git add Dockerfile docker-compose.yml && git commit -m "feat: add Dockerfile and docker-compose.yml with PG+pgvector and Ollama"
```

---

## Phase 8: Test Infrastructure

### Step 21: Create conftest with shared fixtures

**File:** `/home/ubuntu/ReToneProjects/RTMemory/server/tests/conftest.py`

```python
"""Shared test fixtures for RTMemory tests."""

import uuid

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.db.base import Base
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
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
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
```

**Run:**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory/server && pip install aiosqlite 2>&1 | tail -2
```

**Expected:** `aiosqlite` installed.

---

### Step 22: Add pytest-asyncio and aiosqlite to dev deps

**File:** `/home/ubuntu/ReToneProjects/RTMemory/server/pyproject.toml` (update dev deps)

Edit the `[project.optional-dependencies]` section to add `aiosqlite`:

```toml
[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
    "pytest-cov>=5.0",
    "respx>=0.21",
    "aiosqlite>=0.20",
]
```

**Run:**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory/server && pip install -e ".[dev]" 2>&1 | tail -3
```

**Expected:** `Successfully installed rtmemory-server-0.1.0`.

---

### Step 23: Run all tests

**Run:**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/ -v --tb=short
```

**Expected:** All tests pass:

```
tests/test_config.py   — 10 passed
tests/test_health.py   — 2 passed
tests/test_models.py   — 14 passed
tests/test_spaces_api.py — 7 passed
TOTAL: 33 passed
```

**Commit:**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory && git add server/tests/conftest.py server/pyproject.toml && git commit -m "feat: add test infrastructure with conftest fixtures and aiosqlite"
```

---

## Phase 9: Final Verification

### Step 24: Verify full project structure

**Run:**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory && find server -type f | sort && echo "---" && ls -1 config.yaml Dockerfile docker-compose.yml
```

**Expected:**

```
server/alembic.ini
server/alembic/env.py
server/alembic/script.py.mako
server/alembic/versions/001_initial.py
server/app/__init__.py
server/app/api/__init__.py
server/app/api/deps.py
server/app/api/spaces.py
server/app/config.py
server/app/db/__init__.py
server/app/db/base.py
server/app/db/models.py
server/app/db/session.py
server/app/main.py
server/pyproject.toml
server/tests/__init__.py
server/tests/conftest.py
server/tests/test_config.py
server/tests/test_health.py
server/tests/test_models.py
server/tests/test_spaces_api.py
---
config.yaml
Dockerfile
docker-compose.yml
```

---

### Step 25: Verify Docker Compose config

**Run:**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory && docker compose config 2>&1 | head -20
```

**Expected:** Valid compose config output (services, volumes, etc. rendered).

---

### Step 26: Final commit — tag as foundation-complete

**Run:**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory && git add -A && git commit -m "feat: complete RTMemory foundation — skeleton, models, config, migrations, Spaces CRUD, Docker"
```

---

## Checklist

- [ ] Step 1: Create directory structure
- [ ] Step 2: Create pyproject.toml with all dependencies
- [ ] Step 3: Create all __init__.py files
- [ ] Step 4: Write failing test for config loading
- [ ] Step 5: Implement pydantic-settings config system
- [ ] Step 6: Create default config.yaml
- [ ] Step 7: Write failing test for database models
- [ ] Step 8: Implement SQLAlchemy Base with pgvector VECTOR type
- [ ] Step 9: Implement ALL 7 database models (spaces, entities, relations, memories, documents, chunks, memory_sources)
- [ ] Step 10: Implement async database session (engine + sessionmaker)
- [ ] Step 11: Create alembic.ini
- [ ] Step 12: Create alembic/env.py with async + pgvector extension support
- [ ] Step 13: Create alembic/script.py.mako
- [ ] Step 14: Create initial migration (all 7 tables)
- [ ] Step 15: Write failing test for health check
- [ ] Step 16: Implement FastAPI app with health check + CORS
- [ ] Step 17: Write failing tests for Spaces CRUD API
- [ ] Step 18: Implement Spaces CRUD router (create, list, get, delete, update)
- [ ] Step 19: Create Dockerfile
- [ ] Step 20: Create docker-compose.yml (api + worker + postgres+pgvector + ollama)
- [ ] Step 21: Create conftest with shared fixtures
- [ ] Step 22: Add aiosqlite to dev dependencies
- [ ] Step 23: Run all tests (33 expected)
- [ ] Step 24: Verify full project structure
- [ ] Step 25: Verify Docker Compose config
- [ ] Step 26: Final commit

## Key Design Decisions

1. **Vector dimension is configurable**: `EmbeddingConfig.vector_dimension` defaults to 768 (bge-base-zh-v1.5), overridable via `RTMEM_EMBEDDING_VECTOR_DIMENSION` env var or config.yaml. Set to 1536 for OpenAI embeddings.

2. **Migration uses hardcoded 768**: The initial Alembic migration uses `Vector(768)` explicitly. After running the first migration, change the dimension by creating a new ALTER migration. The ORM models read dimension from config dynamically.

3. **SQLite for unit tests**: Basic CRUD tests use aiosqlite (no pgvector). pgvector-dependent tests (vector similarity search) require a real PostgreSQL+pgvector and belong in integration tests (Phase 5+).

4. **Tenant isolation via org_id**: Every table has `org_id` for multi-tenant data isolation. `space_id` provides a secondary isolation boundary within a tenant.

5. **Temporal relations**: The `relations` table uses `valid_from`/`valid_to`/`is_current` to model temporal edges. When a contradiction is detected, the old relation gets `valid_to=now(), is_current=false` and a new relation is inserted with `valid_from=now(), is_current=true`.

6. **Confidence decay**: Memories carry `decay_rate` and `confidence`. The formula `C(t) = C0 * e^(-lambda * delta_days) * (1 + alpha * log(n+1))` is implemented in the profile engine (Plan 06), not at the model layer.