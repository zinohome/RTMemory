# RTMemory LLM Adapter — 多模型适配层

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a unified LLM and embedding adapter layer supporting OpenAI, Anthropic, Ollama, and local models.

**Architecture:** Protocol-based adapter pattern. Each provider implements a common interface. Factory creates adapters from config. Embedding service separate from chat service.

**Tech Stack:** Python 3.12, httpx, sentence-transformers, openai (optional), anthropic (optional), pydantic

## File Map

```
server/
  app/
    __init__.py
    config.py
    core/
      __init__.py
      llm/
        __init__.py
        adapter.py
        openai_adapter.py
        anthropic_adapter.py
        ollama_adapter.py
      embedding/
        __init__.py
        service.py
        local_embedding.py
        openai_embedding.py
  tests/
    __init__.py
    conftest.py
    core/
      __init__.py
      llm/
        __init__.py
        test_adapter.py
        test_openai_adapter.py
        test_anthropic_adapter.py
        test_ollama_adapter.py
        test_llm_factory.py
      embedding/
        __init__.py
        test_service.py
        test_local_embedding.py
        test_openai_embedding.py
        test_embedding_factory.py
  pyproject.toml

config.yaml
```

---

## Phase 1: Project Skeleton

### Step 1: Create directory structure

```bash
cd /home/ubuntu/ReToneProjects/RTMemory
mkdir -p server/app/core/llm
mkdir -p server/app/core/embedding
mkdir -p server/tests/core/llm
mkdir -p server/tests/core/embedding
```

**Run:**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory && mkdir -p server/app/core/llm server/app/core/embedding server/tests/core/llm server/tests/core/embedding
```

**Expected:** All directories created.

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
    "uvicorn>=0.30",
    "httpx>=0.27",
    "pydantic>=2.0",
    "pyyaml>=6.0",
    "sqlalchemy[asyncio]>=2.0",
    "asyncpg>=0.30",
    "sentence-transformers>=3.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
    "pytest-cov>=5.0",
    "numpy>=1.26",
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

**Expected:** Package installed in editable mode with dev dependencies.

---

### Step 3: Create all __init__.py files

**File:** `/home/ubuntu/ReToneProjects/RTMemory/server/app/__init__.py`

```python
# RTMemory Server
```

**File:** `/home/ubuntu/ReToneProjects/RTMemory/server/app/core/__init__.py`

```python
# RTMemory Core
```

**File:** `/home/ubuntu/ReToneProjects/RTMemory/server/app/core/llm/__init__.py`

```python
from .adapter import LLMAdapter
from .openai_adapter import OpenAIAdapter
from .anthropic_adapter import AnthropicAdapter
from .ollama_adapter import OllamaAdapter

__all__ = [
    "LLMAdapter",
    "OpenAIAdapter",
    "AnthropicAdapter",
    "OllamaAdapter",
]


def create_llm_adapter(config, client=None):
    """Factory: create an LLM adapter from config.

    Args:
        config: LLMConfig instance with provider, model, etc.
        client: Optional httpx.AsyncClient for testing.

    Returns:
        LLMAdapter instance.
    """
    provider = config.provider.lower()
    if provider == "openai":
        from .openai_adapter import OpenAIAdapter

        return OpenAIAdapter(
            api_key=config.api_key,
            model=config.model,
            base_url=config.base_url or "https://api.openai.com/v1",
            client=client,
        )
    elif provider == "anthropic":
        from .anthropic_adapter import AnthropicAdapter

        return AnthropicAdapter(
            api_key=config.api_key,
            model=config.model,
            base_url=config.base_url or "https://api.anthropic.com",
            client=client,
        )
    elif provider == "ollama":
        from .ollama_adapter import OllamaAdapter

        return OllamaAdapter(
            model=config.model,
            base_url=config.base_url or "http://localhost:11434",
            client=client,
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
```

**File:** `/home/ubuntu/ReToneProjects/RTMemory/server/app/core/embedding/__init__.py`

```python
from .service import EmbeddingService
from .local_embedding import LocalEmbeddingService
from .openai_embedding import OpenAIEmbeddingService

__all__ = [
    "EmbeddingService",
    "LocalEmbeddingService",
    "OpenAIEmbeddingService",
]


def create_embedding_service(config, client=None):
    """Factory: create an embedding service from config.

    Args:
        config: EmbeddingConfig instance with provider, model, etc.
        client: Optional httpx.AsyncClient for testing.

    Returns:
        EmbeddingService instance.
    """
    provider = config.provider.lower()
    if provider == "local":
        from .local_embedding import LocalEmbeddingService

        return LocalEmbeddingService(model_name=config.model)
    elif provider == "openai":
        from .openai_embedding import OpenAIEmbeddingService

        return OpenAIEmbeddingService(
            api_key=config.api_key,
            model=config.model,
            base_url=config.base_url or "https://api.openai.com/v1",
            client=client,
        )
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")
```

**File:** `/home/ubuntu/ReToneProjects/RTMemory/server/tests/__init__.py`

```python
# RTMemory Tests
```

**File:** `/home/ubuntu/ReToneProjects/RTMemory/server/tests/core/__init__.py`

```python
# Core Tests
```

**File:** `/home/ubuntu/ReToneProjects/RTMemory/server/tests/core/llm/__init__.py`

```python
# LLM Adapter Tests
```

**File:** `/home/ubuntu/ReToneProjects/RTMemory/server/tests/core/embedding/__init__.py`

```python
# Embedding Service Tests
```

**Run:**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory/server && find app tests -name __init__.py | sort
```

**Expected:**

```
app/__init__.py
app/core/__init__.py
app/core/embedding/__init__.py
app/core/llm/__init__.py
tests/__init__.py
tests/core/__init__.py
tests/core/embedding/__init__.py
tests/core/llm/__init__.py
```

---

### Step 4: Create conftest.py with test helpers

**File:** `/home/ubuntu/ReToneProjects/RTMemory/server/tests/conftest.py`

```python
"""Shared test fixtures and helpers for RTMemory tests."""

import httpx
import pytest


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
    import unittest.mock

    client = unittest.mock.AsyncMock(spec=httpx.AsyncClient)
    return client
```

---

### Step 5: Commit project skeleton

```bash
cd /home/ubuntu/ReToneProjects/RTMemory
git add server/pyproject.toml server/app/__init__.py server/app/core/__init__.py \
  server/app/core/llm/__init__.py server/app/core/embedding/__init__.py \
  server/tests/__init__.py server/tests/conftest.py \
  server/tests/core/__init__.py server/tests/core/llm/__init__.py \
  server/tests/core/embedding/__init__.py
git commit -m "feat(llm-adapter): add project skeleton with directory structure and dependencies"
```

---

## Phase 2: LLM Adapter Protocol

### Step 6: Write LLMAdapter ABC test

**File:** `/home/ubuntu/ReToneProjects/RTMemory/server/tests/core/llm/test_adapter.py`

```python
"""Tests for LLMAdapter abstract base class."""

import pytest

from app.core.llm.adapter import LLMAdapter


def test_llm_adapter_cannot_be_instantiated_directly():
    """LLMAdapter is abstract and cannot be instantiated."""
    with pytest.raises(TypeError, match="abstract method"):
        LLMAdapter()


def test_incomplete_subclass_cannot_be_instantiated():
    """Subclass without all abstract methods cannot be instantiated."""

    class IncompleteAdapter(LLMAdapter):
        async def complete(self, messages, temperature=0.7, max_tokens=1024, response_format=None):
            return "hello"

    with pytest.raises(TypeError, match="abstract method"):
        IncompleteAdapter()


def test_complete_subclass_can_be_instantiated():
    """Subclass implementing all abstract methods can be instantiated."""

    class CompleteAdapter(LLMAdapter):
        async def complete(self, messages, temperature=0.7, max_tokens=1024, response_format=None):
            return "test response"

        async def complete_structured(self, messages, schema, temperature=0.1):
            return {"result": True}

    adapter = CompleteAdapter()
    assert isinstance(adapter, LLMAdapter)


def test_llm_adapter_has_required_methods():
    """LLMAdapter defines complete and complete_structured abstract methods."""
    abstract_methods = LLMAdapter.__abstractmethods__
    assert "complete" in abstract_methods
    assert "complete_structured" in abstract_methods
```

**Run:**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/core/llm/test_adapter.py -v 2>&1 | tail -15
```

**Expected:** Tests FAIL because `adapter.py` does not exist yet.

---

### Step 7: Implement LLMAdapter ABC

**File:** `/home/ubuntu/ReToneProjects/RTMemory/server/app/core/llm/adapter.py`

```python
"""LLM Adapter abstract base class.

Defines the unified interface for all LLM providers.
Each provider (OpenAI, Anthropic, Ollama) implements this protocol.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class LLMAdapter(ABC):
    """Abstract base class for LLM chat completion adapters.

    All providers must implement:
    - complete: standard chat completion returning raw text
    - complete_structured: chat completion returning parsed JSON
    """

    @abstractmethod
    async def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """Send messages to the LLM and return the text response.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
                Example: [{"role": "user", "content": "Hello"}]
            temperature: Sampling temperature (0.0 - 2.0).
            max_tokens: Maximum tokens in the response.
            response_format: Optional format specification for the response.
                For OpenAI: {"type": "json_object"} or {"type": "json_schema", ...}
                For Ollama: {"type": "json_object"}
                Anthropic: Not natively supported; ignored.

        Returns:
            The assistant's response text.
        """
        ...

    @abstractmethod
    async def complete_structured(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        temperature: float = 0.1,
    ) -> dict:
        """Send messages to the LLM and return a parsed JSON response.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            schema: JSON Schema dict describing the expected output structure.
            temperature: Sampling temperature, lower for more deterministic output.

        Returns:
            Parsed JSON dict matching the provided schema.

        Raises:
            ValueError: If the LLM response cannot be parsed as valid JSON.
        """
        ...
```

**Run:**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/core/llm/test_adapter.py -v 2>&1 | tail -15
```

**Expected:**

```
tests/core/llm/test_adapter.py::test_llm_adapter_cannot_be_instantiated_directly PASSED
tests/core/llm/test_adapter.py::test_incomplete_subclass_cannot_be_instantiated PASSED
tests/core/llm/test_adapter.py::test_complete_subclass_can_be_instantiated PASSED
tests/core/llm/test_adapter.py::test_llm_adapter_has_required_methods PASSED

4 passed
```

---

### Step 8: Commit LLMAdapter ABC

```bash
cd /home/ubuntu/ReToneProjects/RTMemory
git add server/app/core/llm/adapter.py server/tests/core/llm/test_adapter.py
git commit -m "feat(llm-adapter): add LLMAdapter abstract base class with tests"
```

---

## Phase 3: Embedding Service Protocol

### Step 9: Write EmbeddingService ABC test

**File:** `/home/ubuntu/ReToneProjects/RTMemory/server/tests/core/embedding/test_service.py`

```python
"""Tests for EmbeddingService abstract base class."""

import pytest

from app.core.embedding.service import EmbeddingService


def test_embedding_service_cannot_be_instantiated_directly():
    """EmbeddingService is abstract and cannot be instantiated."""
    with pytest.raises(TypeError, match="abstract method"):
        EmbeddingService()


def test_incomplete_subclass_cannot_be_instantiated():
    """Subclass without all abstract methods cannot be instantiated."""

    class IncompleteService(EmbeddingService):
        async def embed(self, texts):
            return [[0.1]]

    with pytest.raises(TypeError, match="abstract method"):
        IncompleteService()


def test_complete_subclass_can_be_instantiated():
    """Subclass implementing all abstract methods can be instantiated."""

    class CompleteService(EmbeddingService):
        async def embed(self, texts):
            return [[0.1, 0.2, 0.3]]

        def get_dimension(self):
            return 3

    service = CompleteService()
    assert isinstance(service, EmbeddingService)


def test_embedding_service_has_required_methods():
    """EmbeddingService defines embed and get_dimension abstract methods."""
    abstract_methods = EmbeddingService.__abstractmethods__
    assert "embed" in abstract_methods
    assert "get_dimension" in abstract_methods
```

**Run:**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/core/embedding/test_service.py -v 2>&1 | tail -15
```

**Expected:** Tests FAIL because `service.py` does not exist yet.

---

### Step 10: Implement EmbeddingService ABC

**File:** `/home/ubuntu/ReToneProjects/RTMemory/server/app/core/embedding/service.py`

```python
"""Embedding Service abstract base class.

Defines the unified interface for all embedding providers.
Each provider (local sentence-transformers, OpenAI) implements this protocol.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingService(ABC):
    """Abstract base class for text embedding services.

    All providers must implement:
    - embed: batch embedding of text strings
    - get_dimension: return the embedding vector dimension
    """

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of text strings into vector representations.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors, one per input string.
            Each vector is a list of floats with dimension get_dimension().
        """
        ...

    @abstractmethod
    def get_dimension(self) -> int:
        """Return the dimension of embedding vectors produced by this service.

        Returns:
            Integer dimension of the embedding vectors.
        """
        ...
```

**Run:**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/core/embedding/test_service.py -v 2>&1 | tail -15
```

**Expected:**

```
tests/core/embedding/test_service.py::test_embedding_service_cannot_be_instantiated_directly PASSED
tests/core/embedding/test_service.py::test_incomplete_subclass_cannot_be_instantiated PASSED
tests/core/embedding/test_service.py::test_complete_subclass_can_be_instantiated PASSED
tests/core/embedding/test_service.py::test_embedding_service_has_required_methods PASSED

4 passed
```

---

### Step 11: Commit EmbeddingService ABC

```bash
cd /home/ubuntu/ReToneProjects/RTMemory
git add server/app/core/embedding/service.py server/tests/core/embedding/test_service.py
git commit -m "feat(embedding): add EmbeddingService abstract base class with tests"
```

---

## Phase 4: Config System

### Step 12: Write config loading test

**File:** `/home/ubuntu/ReToneProjects/RTMemory/server/tests/test_config.py`

```python
"""Tests for configuration loading."""

import pytest
import yaml
from pathlib import Path

from app.config import LLMConfig, EmbeddingConfig, AppConfig, load_config


class TestLLMConfig:
    def test_llm_config_defaults(self):
        config = LLMConfig(provider="openai", model="gpt-4o", api_key="sk-test")
        assert config.provider == "openai"
        assert config.model == "gpt-4o"
        assert config.api_key == "sk-test"
        assert config.base_url is None
        assert config.temperature == 0.7
        assert config.max_tokens == 1024

    def test_llm_config_with_base_url(self):
        config = LLMConfig(
            provider="ollama",
            model="qwen2.5:7b",
            base_url="http://localhost:11434",
            api_key=None,
            temperature=0.1,
            max_tokens=2048,
        )
        assert config.provider == "ollama"
        assert config.base_url == "http://localhost:11434"
        assert config.temperature == 0.1
        assert config.max_tokens == 2048


class TestEmbeddingConfig:
    def test_embedding_config_local(self):
        config = EmbeddingConfig(provider="local", model="BAAI/bge-base-zh-v1.5")
        assert config.provider == "local"
        assert config.model == "BAAI/bge-base-zh-v1.5"
        assert config.api_key is None

    def test_embedding_config_openai(self):
        config = EmbeddingConfig(
            provider="openai",
            model="text-embedding-3-small",
            api_key="sk-test",
        )
        assert config.provider == "openai"
        assert config.api_key == "sk-test"


class TestAppConfig:
    def test_app_config_combines_llm_and_embedding(self):
        llm = LLMConfig(provider="openai", model="gpt-4o", api_key="sk-test")
        embedding = EmbeddingConfig(provider="local", model="BAAI/bge-base-zh-v1.5")
        config = AppConfig(llm=llm, embedding=embedding)
        assert config.llm.provider == "openai"
        assert config.embedding.provider == "local"


class TestLoadConfig:
    def test_load_config_from_yaml(self, tmp_path):
        config_data = {
            "llm": {
                "provider": "ollama",
                "model": "qwen2.5:7b",
                "base_url": "http://localhost:11434",
                "temperature": 0.1,
                "max_tokens": 2048,
            },
            "embedding": {
                "provider": "local",
                "model": "BAAI/bge-base-zh-v1.5",
            },
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        config = load_config(config_file)
        assert config.llm.provider == "ollama"
        assert config.llm.model == "qwen2.5:7b"
        assert config.llm.base_url == "http://localhost:11434"
        assert config.llm.temperature == 0.1
        assert config.llm.max_tokens == 2048
        assert config.embedding.provider == "local"
        assert config.embedding.model == "BAAI/bge-base-zh-v1.5"

    def test_load_config_with_openai_defaults(self, tmp_path):
        config_data = {
            "llm": {
                "provider": "openai",
                "model": "gpt-4o",
                "api_key": "sk-test-key",
            },
            "embedding": {
                "provider": "openai",
                "model": "text-embedding-3-small",
                "api_key": "sk-test-key",
            },
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        config = load_config(config_file)
        assert config.llm.api_key == "sk-test-key"
        assert config.llm.base_url is None
        assert config.embedding.api_key == "sk-test-key"
```

**Run:**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/test_config.py -v 2>&1 | tail -15
```

**Expected:** Tests FAIL because `app/config.py` does not exist yet.

---

### Step 13: Implement config loading

**File:** `/home/ubuntu/ReToneProjects/RTMemory/server/app/config.py`

```python
"""Configuration loading for RTMemory server.

Reads config.yaml and provides typed configuration objects using pydantic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


class LLMConfig(BaseModel):
    """Configuration for the LLM adapter.

    Attributes:
        provider: LLM provider — "openai", "anthropic", or "ollama".
        model: Model identifier (e.g., "gpt-4o", "claude-sonnet-4-20250514", "qwen2.5:7b").
        base_url: Optional API base URL override. Defaults to provider's standard URL.
        api_key: Optional API key. Required for OpenAI and Anthropic. Not needed for Ollama.
        temperature: Default sampling temperature.
        max_tokens: Default maximum response tokens.
    """

    provider: str
    model: str
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = 0.7
    max_tokens: int = 1024


class EmbeddingConfig(BaseModel):
    """Configuration for the embedding service.

    Attributes:
        provider: Embedding provider — "local" (sentence-transformers) or "openai".
        model: Model name (e.g., "BAAI/bge-base-zh-v1.5" for local, "text-embedding-3-small" for OpenAI).
        base_url: Optional API base URL override for OpenAI.
        api_key: Optional API key. Required for OpenAI. Not needed for local.
    """

    provider: str
    model: str
    base_url: str | None = None
    api_key: str | None = None


class AppConfig(BaseModel):
    """Top-level application configuration.

    Attributes:
        llm: LLM adapter configuration.
        embedding: Embedding service configuration.
    """

    llm: LLMConfig
    embedding: EmbeddingConfig


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    """Load application configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        AppConfig instance with validated configuration.

    Raises:
        FileNotFoundError: If the config file does not exist.
        pydantic.ValidationError: If the config data is invalid.
    """
    path = Path(path)
    with open(path) as f:
        data = yaml.safe_load(f)
    return AppConfig(**data)
```

**Run:**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/test_config.py -v 2>&1 | tail -15
```

**Expected:**

```
tests/test_config.py::TestLLMConfig::test_llm_config_defaults PASSED
tests/test_config.py::TestLLMConfig::test_llm_config_with_base_url PASSED
tests/test_config.py::TestEmbeddingConfig::test_embedding_config_local PASSED
tests/test_config.py::TestEmbeddingConfig::test_embedding_config_openai PASSED
tests/test_config.py::TestAppConfig::test_app_config_combines_llm_and_embedding PASSED
tests/test_config.py::TestLoadConfig::test_load_config_from_yaml PASSED
tests/test_config.py::TestLoadConfig::test_load_config_with_openai_defaults PASSED

7 passed
```

---

### Step 14: Commit config system

```bash
cd /home/ubuntu/ReToneProjects/RTMemory
git add server/app/config.py server/tests/test_config.py
git commit -m "feat(config): add pydantic config models and YAML loading with tests"
```

---

## Phase 5: OpenAI Adapter

### Step 15: Write test for OpenAIAdapter.complete

**File:** `/home/ubuntu/ReToneProjects/RTMemory/server/tests/core/llm/test_openai_adapter.py`

```python
"""Tests for OpenAI LLM adapter."""

import json
import unittest.mock

import httpx
import pytest

from app.core.llm.openai_adapter import OpenAIAdapter
from tests.conftest import make_httpx_response


class TestOpenAIAdapterComplete:
    """Tests for OpenAIAdapter.complete method."""

    async def test_complete_basic(self, mock_client):
        """complete() sends messages to OpenAI and returns assistant text."""
        mock_client.post.return_value = make_httpx_response(
            200,
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello!"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            },
        )

        adapter = OpenAIAdapter(
            api_key="sk-test-key",
            model="gpt-4o",
            client=mock_client,
        )
        result = await adapter.complete(
            messages=[{"role": "user", "content": "Hi"}],
        )

        assert result == "Hello!"
        mock_client.post.assert_awaited_once()
        call_args = mock_client.post.call_args
        assert call_args.kwargs["json"]["model"] == "gpt-4o"
        assert call_args.kwargs["json"]["messages"] == [{"role": "user", "content": "Hi"}]
        assert call_args.kwargs["json"]["temperature"] == 0.7
        assert call_args.kwargs["json"]["max_tokens"] == 1024

    async def test_complete_with_system_message(self, mock_client):
        """complete() sends system + user messages to OpenAI."""
        mock_client.post.return_value = make_httpx_response(
            200,
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "I can help."},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 15, "completion_tokens": 3, "total_tokens": 18},
            },
        )

        adapter = OpenAIAdapter(
            api_key="sk-test-key",
            model="gpt-4o",
            client=mock_client,
        )
        result = await adapter.complete(
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Help me"},
            ],
        )

        assert result == "I can help."
        call_args = mock_client.post.call_args
        assert call_args.kwargs["json"]["messages"] == [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Help me"},
        ]

    async def test_complete_custom_parameters(self, mock_client):
        """complete() uses custom temperature and max_tokens."""
        mock_client.post.return_value = make_httpx_response(
            200,
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Precise answer"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            },
        )

        adapter = OpenAIAdapter(
            api_key="sk-test-key",
            model="gpt-4o",
            client=mock_client,
        )
        result = await adapter.complete(
            messages=[{"role": "user", "content": "Be precise"}],
            temperature=0.1,
            max_tokens=2048,
        )

        assert result == "Precise answer"
        call_args = mock_client.post.call_args
        assert call_args.kwargs["json"]["temperature"] == 0.1
        assert call_args.kwargs["json"]["max_tokens"] == 2048

    async def test_complete_with_response_format(self, mock_client):
        """complete() passes response_format to OpenAI API."""
        mock_client.post.return_value = make_httpx_response(
            200,
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": '{"key": "value"}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )

        adapter = OpenAIAdapter(
            api_key="sk-test-key",
            model="gpt-4o",
            client=mock_client,
        )
        result = await adapter.complete(
            messages=[{"role": "user", "content": "Return JSON"}],
            response_format={"type": "json_object"},
        )

        call_args = mock_client.post.call_args
        assert call_args.kwargs["json"]["response_format"] == {"type": "json_object"}

    async def test_complete_api_error(self, mock_client):
        """complete() raises httpx.HTTPStatusError on API error."""
        mock_client.post.side_effect = httpx.HTTPStatusError(
            message="401 Unauthorized",
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
            response=httpx.Response(401, json={"error": {"message": "Invalid API key"}}),
        )

        adapter = OpenAIAdapter(
            api_key="sk-invalid",
            model="gpt-4o",
            client=mock_client,
        )
        with pytest.raises(httpx.HTTPStatusError):
            await adapter.complete(messages=[{"role": "user", "content": "Hi"}])

    async def test_complete_uses_authorization_header(self, mock_client):
        """complete() includes Bearer token in Authorization header."""
        mock_client.post.return_value = make_httpx_response(
            200,
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "OK"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
            },
        )

        adapter = OpenAIAdapter(
            api_key="sk-my-secret-key",
            model="gpt-4o",
            client=mock_client,
        )
        await adapter.complete(messages=[{"role": "user", "content": "Hi"}])

        call_args = mock_client.post.call_args
        assert call_args.kwargs["headers"]["Authorization"] == "Bearer sk-my-secret-key"
        assert call_args.kwargs["headers"]["Content-Type"] == "application/json"

    async def test_complete_custom_base_url(self, mock_client):
        """complete() uses custom base_url when provided."""
        mock_client.post.return_value = make_httpx_response(
            200,
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Proxy response"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            },
        )

        adapter = OpenAIAdapter(
            api_key="sk-test",
            model="gpt-4o",
            base_url="https://proxy.example.com/v1",
            client=mock_client,
        )
        await adapter.complete(messages=[{"role": "user", "content": "Hi"}])

        call_args = mock_client.post.call_args
        assert call_args.args[0] == "https://proxy.example.com/v1/chat/completions"
```

**Run:**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/core/llm/test_openai_adapter.py -v 2>&1 | tail -15
```

**Expected:** Tests FAIL because `openai_adapter.py` does not exist yet.

---

### Step 16: Implement OpenAIAdapter.complete

**File:** `/home/ubuntu/ReToneProjects/RTMemory/server/app/core/llm/openai_adapter.py`

```python
"""OpenAI LLM adapter.

Implements LLMAdapter for the OpenAI chat completions API.
Supports GPT-4o, GPT-4o-mini, and other OpenAI models.
Uses native json_schema response_format for structured output.
"""

from __future__ import annotations

import json
from typing import Any

import httpx


class OpenAIAdapter:
    """LLM adapter for OpenAI chat completions API.

    Args:
        api_key: OpenAI API key (starts with "sk-").
        model: Model identifier, e.g. "gpt-4o", "gpt-4o-mini".
        base_url: API base URL. Defaults to "https://api.openai.com/v1".
            Can be set to a proxy or Azure OpenAI endpoint.
        client: Optional httpx.AsyncClient for dependency injection in tests.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        base_url: str = "https://api.openai.com/v1",
        client: httpx.AsyncClient | None = None,
    ):
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.AsyncClient()

    async def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """Send messages to OpenAI and return the assistant text response.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            temperature: Sampling temperature (0.0 - 2.0).
            max_tokens: Maximum tokens in the response.
            response_format: Optional format spec, e.g. {"type": "json_object"}.

        Returns:
            The assistant's response text.

        Raises:
            httpx.HTTPStatusError: On API error responses.
        """
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        response = await self._client.post(
            f"{self._base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    async def complete_structured(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        temperature: float = 0.1,
    ) -> dict:
        """Send messages to OpenAI and return a parsed JSON response.

        Uses OpenAI's native json_schema response_format for structured output.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            schema: JSON Schema dict describing the expected output structure.
            temperature: Sampling temperature, lower for deterministic output.

        Returns:
            Parsed JSON dict matching the provided schema.

        Raises:
            ValueError: If the response cannot be parsed as valid JSON.
        """
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "schema": schema,
            },
        }
        text = await self.complete(
            messages=messages,
            temperature=temperature,
            response_format=response_format,
        )
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"OpenAI did not return valid JSON: {text}") from e
```

**Run:**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/core/llm/test_openai_adapter.py::TestOpenAIAdapterComplete -v 2>&1 | tail -15
```

**Expected:**

```
tests/core/llm/test_openai_adapter.py::TestOpenAIAdapterComplete::test_complete_basic PASSED
tests/core/llm/test_openai_adapter.py::TestOpenAIAdapterComplete::test_complete_with_system_message PASSED
tests/core/llm/test_openai_adapter.py::TestCompleteCustomParameters::test_complete_custom_parameters PASSED
tests/core/llm/test_openai_adapter.py::TestOpenAIAdapterComplete::test_complete_with_response_format PASSED
tests/core/llm/test_openai_adapter.py::TestOpenAIAdapterComplete::test_complete_api_error PASSED
tests/core/llm/test_openai_adapter.py::TestOpenAIAdapterComplete::test_complete_uses_authorization_header PASSED
tests/core/llm/test_openai_adapter.py::TestOpenAIAdapterComplete::test_complete_custom_base_url PASSED

7 passed
```

---

### Step 17: Write test for OpenAIAdapter.complete_structured

Add to the existing test file:

**File:** `/home/ubuntu/ReToneProjects/RTMemory/server/tests/core/llm/test_openai_adapter.py`

Append the following class after `TestOpenAIAdapterComplete`:

```python
class TestOpenAIAdapterCompleteStructured:
    """Tests for OpenAIAdapter.complete_structured method."""

    async def test_complete_structured_basic(self, mock_client):
        """complete_structured() returns parsed JSON matching the schema."""
        mock_client.post.return_value = make_httpx_response(
            200,
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": '{"name": "Alice", "age": 30}',
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
            },
        )

        adapter = OpenAIAdapter(
            api_key="sk-test-key",
            model="gpt-4o",
            client=mock_client,
        )
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        }
        result = await adapter.complete_structured(
            messages=[{"role": "user", "content": "Extract person info"}],
            schema=schema,
        )

        assert result == {"name": "Alice", "age": 30}

        # Verify json_schema response_format is passed
        call_args = mock_client.post.call_args
        payload = call_args.kwargs["json"]
        assert payload["response_format"]["type"] == "json_schema"
        assert payload["response_format"]["json_schema"]["name"] == "response"
        assert payload["response_format"]["json_schema"]["schema"] == schema

    async def test_complete_structured_uses_low_temperature(self, mock_client):
        """complete_structured() defaults to temperature=0.1."""
        mock_client.post.return_value = make_httpx_response(
            200,
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": '{"result": true}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )

        adapter = OpenAIAdapter(
            api_key="sk-test-key",
            model="gpt-4o",
            client=mock_client,
        )
        await adapter.complete_structured(
            messages=[{"role": "user", "content": "Test"}],
            schema={"type": "object", "properties": {"result": {"type": "boolean"}}},
        )

        call_args = mock_client.post.call_args
        assert call_args.kwargs["json"]["temperature"] == 0.1

    async def test_complete_structured_invalid_json_raises(self, mock_client):
        """complete_structured() raises ValueError when LLM returns invalid JSON."""
        mock_client.post.return_value = make_httpx_response(
            200,
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Not JSON at all"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )

        adapter = OpenAIAdapter(
            api_key="sk-test-key",
            model="gpt-4o",
            client=mock_client,
        )
        with pytest.raises(ValueError, match="OpenAI did not return valid JSON"):
            await adapter.complete_structured(
                messages=[{"role": "user", "content": "Test"}],
                schema={"type": "object"},
            )
```

**Run:**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/core/llm/test_openai_adapter.py -v 2>&1 | tail -15
```

**Expected:**

```
tests/core/llm/test_openai_adapter.py::TestOpenAIAdapterComplete::test_complete_basic PASSED
tests/core/llm/test_openai_adapter.py::TestOpenAIAdapterComplete::test_complete_with_system_message PASSED
...
tests/core/llm/test_openai_adapter.py::TestOpenAIAdapterCompleteStructured::test_complete_structured_basic PASSED
tests/core/llm/test_openai_adapter.py::TestOpenAIAdapterCompleteStructured::test_complete_structured_uses_low_temperature PASSED
tests/core/llm/test_openai_adapter.py::TestOpenAIAdapterCompleteStructured::test_complete_structured_invalid_json_raises PASSED

10 passed
```

---

### Step 18: Commit OpenAI adapter

```bash
cd /home/ubuntu/ReToneProjects/RTMemory
git add server/app/core/llm/openai_adapter.py server/tests/core/llm/test_openai_adapter.py
git commit -m "feat(llm-adapter): implement OpenAI adapter with complete and complete_structured"
```

---

## Phase 6: Anthropic Adapter

### Step 19: Write test for AnthropicAdapter.complete

**File:** `/home/ubuntu/ReToneProjects/RTMemory/server/tests/core/llm/test_anthropic_adapter.py`

```python
"""Tests for Anthropic LLM adapter."""

import json
import unittest.mock

import httpx
import pytest

from app.core.llm.anthropic_adapter import AnthropicAdapter
from tests.conftest import make_httpx_response


class TestAnthropicAdapterComplete:
    """Tests for AnthropicAdapter.complete method."""

    async def test_complete_basic(self, mock_client):
        """complete() sends messages to Anthropic and returns assistant text."""
        mock_client.post.return_value = make_httpx_response(
            200,
            {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "Hello from Claude!"}],
                "model": "claude-sonnet-4-20250514",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        )

        adapter = AnthropicAdapter(
            api_key="sk-ant-test-key",
            model="claude-sonnet-4-20250514",
            client=mock_client,
        )
        result = await adapter.complete(
            messages=[{"role": "user", "content": "Hi"}],
        )

        assert result == "Hello from Claude!"

    async def test_complete_extracts_system_message(self, mock_client):
        """complete() sends system message as a top-level parameter."""
        mock_client.post.return_value = make_httpx_response(
            200,
            {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "I am helpful."}],
                "model": "claude-sonnet-4-20250514",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 15, "output_tokens": 4},
            },
        )

        adapter = AnthropicAdapter(
            api_key="sk-ant-test-key",
            model="claude-sonnet-4-20250514",
            client=mock_client,
        )
        await adapter.complete(
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi"},
            ],
        )

        call_args = mock_client.post.call_args
        payload = call_args.kwargs["json"]
        # System message should be in top-level "system" field, not in messages
        assert payload["system"] == "You are helpful."
        assert all(msg["role"] != "system" for msg in payload["messages"])

    async def test_complete_sends_anthropic_headers(self, mock_client):
        """complete() includes x-api-key and anthropic-version headers."""
        mock_client.post.return_value = make_httpx_response(
            200,
            {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "OK"}],
                "model": "claude-sonnet-4-20250514",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 5, "output_tokens": 1},
            },
        )

        adapter = AnthropicAdapter(
            api_key="sk-ant-test-key",
            model="claude-sonnet-4-20250514",
            client=mock_client,
        )
        await adapter.complete(messages=[{"role": "user", "content": "Hi"}])

        call_args = mock_client.post.call_args
        assert call_args.kwargs["headers"]["x-api-key"] == "sk-ant-test-key"
        assert call_args.kwargs["headers"]["anthropic-version"] == "2023-06-01"

    async def test_complete_custom_parameters(self, mock_client):
        """complete() uses custom temperature and max_tokens."""
        mock_client.post.return_value = make_httpx_response(
            200,
            {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "Precise"}],
                "model": "claude-sonnet-4-20250514",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 1},
            },
        )

        adapter = AnthropicAdapter(
            api_key="sk-ant-test-key",
            model="claude-sonnet-4-20250514",
            client=mock_client,
        )
        await adapter.complete(
            messages=[{"role": "user", "content": "Be precise"}],
            temperature=0.1,
            max_tokens=2048,
        )

        call_args = mock_client.post.call_args
        assert call_args.kwargs["json"]["temperature"] == 0.1
        assert call_args.kwargs["json"]["max_tokens"] == 2048

    async def test_complete_api_error(self, mock_client):
        """complete() raises httpx.HTTPStatusError on API error."""
        mock_client.post.side_effect = httpx.HTTPStatusError(
            message="401 Unauthorized",
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
            response=httpx.Response(401, json={"error": {"message": "Invalid API key"}}),
        )

        adapter = AnthropicAdapter(
            api_key="sk-ant-invalid",
            model="claude-sonnet-4-20250514",
            client=mock_client,
        )
        with pytest.raises(httpx.HTTPStatusError):
            await adapter.complete(messages=[{"role": "user", "content": "Hi"}])

    async def test_complete_custom_base_url(self, mock_client):
        """complete() uses custom base_url when provided."""
        mock_client.post.return_value = make_httpx_response(
            200,
            {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "Proxy"}],
                "model": "claude-sonnet-4-20250514",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 5, "output_tokens": 1},
            },
        )

        adapter = AnthropicAdapter(
            api_key="sk-ant-test",
            model="claude-sonnet-4-20250514",
            base_url="https://proxy.anthropic.example.com",
            client=mock_client,
        )
        await adapter.complete(messages=[{"role": "user", "content": "Hi"}])

        call_args = mock_client.post.call_args
        assert call_args.args[0] == "https://proxy.anthropic.example.com/v1/messages"
```

**Run:**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/core/llm/test_anthropic_adapter.py -v 2>&1 | tail -15
```

**Expected:** Tests FAIL because `anthropic_adapter.py` does not exist yet.

---

### Step 20: Implement AnthropicAdapter.complete

**File:** `/home/ubuntu/ReToneProjects/RTMemory/server/app/core/llm/anthropic_adapter.py`

```python
"""Anthropic LLM adapter.

Implements LLMAdapter for the Anthropic messages API.
Uses prompt-based JSON instructions for structured output
since Anthropic does not natively support response_format.
"""

from __future__ import annotations

import json
from typing import Any

import httpx


class AnthropicAdapter:
    """LLM adapter for the Anthropic messages API.

    Handles the Anthropic-specific API format:
    - System messages are sent as a top-level "system" parameter
    - Messages list contains only user/assistant messages
    - Structured output uses prompt engineering with JSON schema instructions

    Args:
        api_key: Anthropic API key.
        model: Model identifier, e.g. "claude-sonnet-4-20250514".
        base_url: API base URL. Defaults to "https://api.anthropic.com".
        client: Optional httpx.AsyncClient for dependency injection in tests.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        base_url: str = "https://api.anthropic.com",
        client: httpx.AsyncClient | None = None,
    ):
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.AsyncClient()

    def _prepare_messages(
        self, messages: list[dict[str, str]]
    ) -> tuple[str | None, list[dict[str, str]]]:
        """Extract system message from messages list.

        Anthropic requires the system message as a separate top-level parameter.
        This method extracts it and returns (system, filtered_messages).

        Args:
            messages: List of message dicts that may include a system message.

        Returns:
            Tuple of (system_content_or_None, filtered_messages_without_system).
        """
        system = None
        filtered = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                filtered.append(msg)
        return system, filtered

    async def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """Send messages to Anthropic and return the assistant text response.

        Args:
            messages: List of message dicts with 'role' and 'content'.
                System messages are extracted and sent as a top-level parameter.
            temperature: Sampling temperature (0.0 - 1.0).
            max_tokens: Maximum tokens in the response.
            response_format: Ignored for Anthropic. Anthropic does not natively
                support response_format; use complete_structured for JSON output.

        Returns:
            The assistant's response text.

        Raises:
            httpx.HTTPStatusError: On API error responses.
        """
        system, filtered = self._prepare_messages(messages)

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": filtered,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system is not None:
            payload["system"] = system

        response = await self._client.post(
            f"{self._base_url}/v1/messages",
            headers={
                "x-api-key": self._api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        # Anthropic returns content as a list of content blocks
        return data["content"][0]["text"]

    async def complete_structured(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        temperature: float = 0.1,
    ) -> dict:
        """Send messages to Anthropic and return a parsed JSON response.

        Uses prompt engineering: adds the JSON schema instruction to the system
        message (or creates one) since Anthropic does not natively support
        structured output via response_format.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            schema: JSON Schema dict describing the expected output structure.
            temperature: Sampling temperature, lower for deterministic output.

        Returns:
            Parsed JSON dict matching the provided schema.

        Raises:
            ValueError: If the response cannot be parsed as valid JSON.
        """
        schema_instruction = (
            "You must respond with valid JSON matching this schema:\n"
            f"{json.dumps(schema, indent=2)}\n"
            "Do not include any text outside the JSON object."
        )

        modified = []
        has_system = False
        for msg in messages:
            if msg["role"] == "system":
                modified.append({
                    "role": "system",
                    "content": msg["content"] + "\n\n" + schema_instruction,
                })
                has_system = True
            else:
                modified.append(msg)

        if not has_system:
            modified.insert(0, {"role": "system", "content": schema_instruction})

        text = await self.complete(
            messages=modified,
            temperature=temperature,
        )
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Anthropic did not return valid JSON: {text}") from e
```

**Run:**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/core/llm/test_anthropic_adapter.py::TestAnthropicAdapterComplete -v 2>&1 | tail -15
```

**Expected:**

```
tests/core/llm/test_anthropic_adapter.py::TestAnthropicAdapterComplete::test_complete_basic PASSED
tests/core/llm/test_anthropic_adapter.py::TestAnthropicAdapterComplete::test_complete_extracts_system_message PASSED
tests/core/llm/test_anthropic_adapter.py::TestAnthropicAdapterComplete::test_complete_sends_anthropic_headers PASSED
tests/core/llm/test_anthropic_adapter.py::TestAnthropicAdapterComplete::test_complete_custom_parameters PASSED
tests/core/llm/test_anthropic_adapter.py::TestAnthropicAdapterComplete::test_complete_api_error PASSED
tests/core/llm/test_anthropic_adapter.py::TestAnthropicAdapterComplete::test_complete_custom_base_url PASSED

6 passed
```

---

### Step 21: Write test for AnthropicAdapter.complete_structured

Append to the existing test file:

**File:** `/home/ubuntu/ReToneProjects/RTMemory/server/tests/core/llm/test_anthropic_adapter.py`

Add the following class:

```python
class TestAnthropicAdapterCompleteStructured:
    """Tests for AnthropicAdapter.complete_structured method."""

    async def test_complete_structured_basic(self, mock_client):
        """complete_structured() returns parsed JSON matching the schema."""
        mock_client.post.return_value = make_httpx_response(
            200,
            {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": '{"name": "Bob", "age": 25}'}],
                "model": "claude-sonnet-4-20250514",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 20, "output_tokens": 10},
            },
        )

        adapter = AnthropicAdapter(
            api_key="sk-ant-test-key",
            model="claude-sonnet-4-20250514",
            client=mock_client,
        )
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        }
        result = await adapter.complete_structured(
            messages=[{"role": "user", "content": "Extract person info"}],
            schema=schema,
        )

        assert result == {"name": "Bob", "age": 25}

        # Verify schema instruction was added to system message
        call_args = mock_client.post.call_args
        payload = call_args.kwargs["json"]
        assert "system" in payload
        assert "valid JSON matching this schema" in payload["system"]

    async def test_complete_structured_adds_system_if_missing(self, mock_client):
        """complete_structured() adds a system message if none exists."""
        mock_client.post.return_value = make_httpx_response(
            200,
            {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": '{"result": true}'}],
                "model": "claude-sonnet-4-20250514",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 15, "output_tokens": 5},
            },
        )

        adapter = AnthropicAdapter(
            api_key="sk-ant-test-key",
            model="claude-sonnet-4-20250514",
            client=mock_client,
        )
        await adapter.complete_structured(
            messages=[{"role": "user", "content": "Test"}],
            schema={"type": "object", "properties": {"result": {"type": "boolean"}}},
        )

        call_args = mock_client.post.call_args
        payload = call_args.kwargs["json"]
        assert "system" in payload
        assert "valid JSON" in payload["system"]

    async def test_complete_structured_appends_to_existing_system(self, mock_client):
        """complete_structured() appends schema instruction to existing system message."""
        mock_client.post.return_value = make_httpx_response(
            200,
            {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": '{"items": []}'}],
                "model": "claude-sonnet-4-20250514",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 20, "output_tokens": 5},
            },
        )

        adapter = AnthropicAdapter(
            api_key="sk-ant-test-key",
            model="claude-sonnet-4-20250514",
            client=mock_client,
        )
        await adapter.complete_structured(
            messages=[
                {"role": "system", "content": "You are a data extractor."},
                {"role": "user", "content": "Extract items"},
            ],
            schema={"type": "object", "properties": {"items": {"type": "array"}}},
        )

        call_args = mock_client.post.call_args
        payload = call_args.kwargs["json"]
        assert payload["system"].startswith("You are a data extractor.")
        assert "valid JSON" in payload["system"]

    async def test_complete_structured_invalid_json_raises(self, mock_client):
        """complete_structured() raises ValueError when Anthropic returns invalid JSON."""
        mock_client.post.return_value = make_httpx_response(
            200,
            {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "I cannot produce JSON."}],
                "model": "claude-sonnet-4-20250514",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        )

        adapter = AnthropicAdapter(
            api_key="sk-ant-test-key",
            model="claude-sonnet-4-20250514",
            client=mock_client,
        )
        with pytest.raises(ValueError, match="Anthropic did not return valid JSON"):
            await adapter.complete_structured(
                messages=[{"role": "user", "content": "Test"}],
                schema={"type": "object"},
            )
```

**Run:**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/core/llm/test_anthropic_adapter.py -v 2>&1 | tail -15
```

**Expected:**

```
tests/core/llm/test_anthropic_adapter.py::TestAnthropicAdapterComplete::test_complete_basic PASSED
...
tests/core/llm/test_anthropic_adapter.py::TestAnthropicAdapterCompleteStructured::test_complete_structured_basic PASSED
tests/core/llm/test_anthropic_adapter.py::TestAnthropicAdapterCompleteStructured::test_complete_structured_adds_system_if_missing PASSED
tests/core/llm/test_anthropic_adapter.py::TestAnthropicAdapterCompleteStructured::test_complete_structured_appends_to_existing_system PASSED
tests/core/llm/test_anthropic_adapter.py::TestAnthropicAdapterCompleteStructured::test_complete_structured_invalid_json_raises PASSED

10 passed
```

---

### Step 22: Commit Anthropic adapter

```bash
cd /home/ubuntu/ReToneProjects/RTMemory
git add server/app/core/llm/anthropic_adapter.py server/tests/core/llm/test_anthropic_adapter.py
git commit -m "feat(llm-adapter): implement Anthropic adapter with complete and complete_structured"
```

---

## Phase 7: Ollama Adapter

### Step 23: Write test for OllamaAdapter.complete

**File:** `/home/ubuntu/ReToneProjects/RTMemory/server/tests/core/llm/test_ollama_adapter.py`

```python
"""Tests for Ollama LLM adapter."""

import json
import unittest.mock

import httpx
import pytest

from app.core.llm.ollama_adapter import OllamaAdapter
from tests.conftest import make_httpx_response


class TestOllamaAdapterComplete:
    """Tests for OllamaAdapter.complete method."""

    async def test_complete_basic(self, mock_client):
        """complete() sends messages to Ollama and returns assistant text."""
        mock_client.post.return_value = make_httpx_response(
            200,
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": "qwen2.5:7b",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello from Ollama!"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )

        adapter = OllamaAdapter(
            model="qwen2.5:7b",
            base_url="http://localhost:11434",
            client=mock_client,
        )
        result = await adapter.complete(
            messages=[{"role": "user", "content": "Hi"}],
        )

        assert result == "Hello from Ollama!"

    async def test_complete_uses_openai_compatible_endpoint(self, mock_client):
        """complete() uses /v1/chat/completions endpoint (OpenAI-compatible)."""
        mock_client.post.return_value = make_httpx_response(
            200,
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": "qwen2.5:7b",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "OK"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
            },
        )

        adapter = OllamaAdapter(
            model="qwen2.5:7b",
            client=mock_client,
        )
        await adapter.complete(messages=[{"role": "user", "content": "Hi"}])

        call_args = mock_client.post.call_args
        assert call_args.args[0] == "http://localhost:11434/v1/chat/completions"

    async def test_complete_sends_model_and_stream_false(self, mock_client):
        """complete() sends model, stream=false, and other parameters."""
        mock_client.post.return_value = make_httpx_response(
            200,
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": "qwen2.5:7b",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Reply"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
            },
        )

        adapter = OllamaAdapter(
            model="qwen2.5:7b",
            client=mock_client,
        )
        await adapter.complete(
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.5,
            max_tokens=512,
        )

        call_args = mock_client.post.call_args
        payload = call_args.kwargs["json"]
        assert payload["model"] == "qwen2.5:7b"
        assert payload["stream"] is False
        assert payload["temperature"] == 0.5
        assert payload["max_tokens"] == 512

    async def test_complete_with_response_format(self, mock_client):
        """complete() passes response_format to the Ollama API."""
        mock_client.post.return_value = make_httpx_response(
            200,
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": "qwen2.5:7b",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": '{"key": "value"}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )

        adapter = OllamaAdapter(
            model="qwen2.5:7b",
            client=mock_client,
        )
        await adapter.complete(
            messages=[{"role": "user", "content": "Return JSON"}],
            response_format={"type": "json_object"},
        )

        call_args = mock_client.post.call_args
        assert call_args.kwargs["json"]["response_format"] == {"type": "json_object"}

    async def test_complete_no_api_key(self, mock_client):
        """complete() does not send Authorization header (Ollama is local)."""
        mock_client.post.return_value = make_httpx_response(
            200,
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": "qwen2.5:7b",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hi"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
            },
        )

        adapter = OllamaAdapter(
            model="qwen2.5:7b",
            client=mock_client,
        )
        await adapter.complete(messages=[{"role": "user", "content": "Hi"}])

        call_args = mock_client.post.call_args
        headers = call_args.kwargs["headers"]
        assert "Authorization" not in headers
        assert headers["Content-Type"] == "application/json"

    async def test_complete_api_error(self, mock_client):
        """complete() raises httpx.HTTPStatusError on API error."""
        mock_client.post.side_effect = httpx.HTTPStatusError(
            message="500 Internal Server Error",
            request=httpx.Request("POST", "http://localhost:11434/v1/chat/completions"),
            response=httpx.Response(500, json={"error": "model not found"}),
        )

        adapter = OllamaAdapter(
            model="qwen2.5:7b",
            client=mock_client,
        )
        with pytest.raises(httpx.HTTPStatusError):
            await adapter.complete(messages=[{"role": "user", "content": "Hi"}])

    async def test_complete_custom_base_url(self, mock_client):
        """complete() uses custom base_url when provided."""
        mock_client.post.return_value = make_httpx_response(
            200,
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": "qwen2.5:7b",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Remote"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
            },
        )

        adapter = OllamaAdapter(
            model="qwen2.5:7b",
            base_url="http://remote-ollama.example.com",
            client=mock_client,
        )
        await adapter.complete(messages=[{"role": "user", "content": "Hi"}])

        call_args = mock_client.post.call_args
        assert call_args.args[0] == "http://remote-ollama.example.com/v1/chat/completions"
```

**Run:**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/core/llm/test_ollama_adapter.py::TestOllamaAdapterComplete -v 2>&1 | tail -15
```

**Expected:** Tests FAIL because `ollama_adapter.py` does not exist yet.

---

### Step 24: Implement OllamaAdapter.complete

**File:** `/home/ubuntu/ReToneProjects/RTMemory/server/app/core/llm/ollama_adapter.py`

```python
"""Ollama LLM adapter.

Implements LLMAdapter for Ollama using the OpenAI-compatible API endpoint.
Ollama provides an OpenAI-compatible API at /v1/chat/completions, which
allows us to use the same request format as the OpenAI adapter.

For structured output, uses response_format={"type": "json_object"} and
adds "Output valid JSON" instruction to the prompt.
"""

from __future__ import annotations

import json
from typing import Any

import httpx


class OllamaAdapter:
    """LLM adapter for Ollama using the OpenAI-compatible API.

    Uses Ollama's /v1/chat/completions endpoint which is compatible with
    the OpenAI chat completions format. No API key is required since
    Ollama runs locally.

    Args:
        model: Ollama model identifier, e.g. "qwen2.5:7b", "llama3.1:8b".
        base_url: Ollama server URL. Defaults to "http://localhost:11434".
        client: Optional httpx.AsyncClient for dependency injection in tests.
    """

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        base_url: str = "http://localhost:11434",
        client: httpx.AsyncClient | None = None,
    ):
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.AsyncClient()

    async def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """Send messages to Ollama and return the assistant text response.

        Uses the OpenAI-compatible /v1/chat/completions endpoint.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in the response.
            response_format: Optional format spec, e.g. {"type": "json_object"}.

        Returns:
            The assistant's response text.

        Raises:
            httpx.HTTPStatusError: On API error responses.
        """
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        response = await self._client.post(
            f"{self._base_url}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    async def complete_structured(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        temperature: float = 0.1,
    ) -> dict:
        """Send messages to Ollama and return a parsed JSON response.

        Uses Ollama's response_format={"type": "json_object"} and adds
        "Output valid JSON" instruction to the prompt along with the schema.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            schema: JSON Schema dict describing the expected output structure.
            temperature: Sampling temperature, lower for deterministic output.

        Returns:
            Parsed JSON dict matching the provided schema.

        Raises:
            ValueError: If the response cannot be parsed as valid JSON.
        """
        schema_instruction = (
            "\n\nOutput valid JSON matching this schema:\n"
            f"{json.dumps(schema, indent=2)}"
        )

        # Add the instruction to the last user message
        modified = [msg.copy() for msg in messages]
        if modified and modified[-1]["role"] == "user":
            modified[-1]["content"] = modified[-1]["content"] + schema_instruction
        else:
            modified.append({"role": "user", "content": "Output valid JSON." + schema_instruction})

        text = await self.complete(
            messages=modified,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Ollama did not return valid JSON: {text}") from e
```

**Run:**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/core/llm/test_ollama_adapter.py::TestOllamaAdapterComplete -v 2>&1 | tail -15
```

**Expected:**

```
tests/core/llm/test_ollama_adapter.py::TestOllamaAdapterComplete::test_complete_basic PASSED
tests/core/llm/test_ollama_adapter.py::TestOllamaAdapterComplete::test_complete_uses_openai_compatible_endpoint PASSED
tests/core/llm/test_ollama_adapter.py::TestOllamaAdapterComplete::test_complete_sends_model_and_stream_false PASSED
tests/core/llm/test_ollama_adapter.py::TestOllamaAdapterComplete::test_complete_with_response_format PASSED
tests/core/llm/test_ollama_adapter.py::TestOllamaAdapterComplete::test_complete_no_api_key PASSED
tests/core/llm/test_ollama_adapter.py::TestOllamaAdapterComplete::test_complete_api_error PASSED
tests/core/llm/test_ollama_adapter.py::TestOllamaAdapterComplete::test_complete_custom_base_url PASSED

7 passed
```

---

### Step 25: Write test for OllamaAdapter.complete_structured

Append to the existing test file:

**File:** `/home/ubuntu/ReToneProjects/RTMemory/server/tests/core/llm/test_ollama_adapter.py`

Add the following class:

```python
class TestOllamaAdapterCompleteStructured:
    """Tests for OllamaAdapter.complete_structured method."""

    async def test_complete_structured_basic(self, mock_client):
        """complete_structured() returns parsed JSON matching the schema."""
        mock_client.post.return_value = make_httpx_response(
            200,
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": "qwen2.5:7b",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": '{"name": "Carol", "age": 35}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
            },
        )

        adapter = OllamaAdapter(
            model="qwen2.5:7b",
            client=mock_client,
        )
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        }
        result = await adapter.complete_structured(
            messages=[{"role": "user", "content": "Extract person info"}],
            schema=schema,
        )

        assert result == {"name": "Carol", "age": 35}

        # Verify response_format={"type": "json_object"} was sent
        call_args = mock_client.post.call_args
        payload = call_args.kwargs["json"]
        assert payload["response_format"] == {"type": "json_object"}

        # Verify schema instruction was added to last user message
        last_msg = payload["messages"][-1]
        assert "Output valid JSON" in last_msg["content"]
        assert "schema" in last_msg["content"].lower() or "properties" in last_msg["content"]

    async def test_complete_structured_appends_to_user_message(self, mock_client):
        """complete_structured() appends instruction to the last user message."""
        mock_client.post.return_value = make_httpx_response(
            200,
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": "qwen2.5:7b",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": '{"items": []}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
            },
        )

        adapter = OllamaAdapter(
            model="qwen2.5:7b",
            client=mock_client,
        )
        await adapter.complete_structured(
            messages=[
                {"role": "system", "content": "You are a data extractor."},
                {"role": "user", "content": "Extract items from: apples, oranges"},
            ],
            schema={"type": "object", "properties": {"items": {"type": "array"}}},
        )

        call_args = mock_client.post.call_args
        payload = call_args.kwargs["json"]
        # System message should be unchanged
        assert payload["messages"][0] == {"role": "system", "content": "You are a data extractor."}
        # User message should have schema instruction appended
        user_msg = payload["messages"][1]
        assert user_msg["content"].startswith("Extract items from: apples, oranges")
        assert "Output valid JSON" in user_msg["content"]

    async def test_complete_structured_adds_user_message_if_none(self, mock_client):
        """complete_structured() adds a user message if none exists."""
        mock_client.post.return_value = make_httpx_response(
            200,
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": "qwen2.5:7b",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": '{"result": true}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 15, "completion_tokens": 5, "total_tokens": 20},
            },
        )

        adapter = OllamaAdapter(
            model="qwen2.5:7b",
            client=mock_client,
        )
        await adapter.complete_structured(
            messages=[{"role": "system", "content": "You are helpful."}],
            schema={"type": "object", "properties": {"result": {"type": "boolean"}}},
        )

        call_args = mock_client.post.call_args
        payload = call_args.kwargs["json"]
        # Should have original system message plus a new user message
        assert len(payload["messages"]) == 2
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][1]["role"] == "user"
        assert "Output valid JSON" in payload["messages"][1]["content"]

    async def test_complete_structured_invalid_json_raises(self, mock_client):
        """complete_structured() raises ValueError when Ollama returns invalid JSON."""
        mock_client.post.return_value = make_httpx_response(
            200,
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": "qwen2.5:7b",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "I cannot produce JSON."},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )

        adapter = OllamaAdapter(
            model="qwen2.5:7b",
            client=mock_client,
        )
        with pytest.raises(ValueError, match="Ollama did not return valid JSON"):
            await adapter.complete_structured(
                messages=[{"role": "user", "content": "Test"}],
                schema={"type": "object"},
            )
```

**Run:**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/core/llm/test_ollama_adapter.py -v 2>&1 | tail -15
```

**Expected:**

```
tests/core/llm/test_ollama_adapter.py::TestOllamaAdapterComplete::test_complete_basic PASSED
...
tests/core/llm/test_ollama_adapter.py::TestOllamaAdapterCompleteStructured::test_complete_structured_basic PASSED
tests/core/llm/test_ollama_adapter.py::TestOllamaAdapterCompleteStructured::test_complete_structured_appends_to_user_message PASSED
tests/core/llm/test_ollama_adapter.py::TestOllamaAdapterCompleteStructured::test_complete_structured_adds_user_message_if_none PASSED
tests/core/llm/test_ollama_adapter.py::TestOllamaAdapterCompleteStructured::test_complete_structured_invalid_json_raises PASSED

11 passed
```

---

### Step 26: Commit Ollama adapter

```bash
cd /home/ubuntu/ReToneProjects/RTMemory
git add server/app/core/llm/ollama_adapter.py server/tests/core/llm/test_ollama_adapter.py
git commit -m "feat(llm-adapter): implement Ollama adapter with complete and complete_structured"
```

---

## Phase 8: LLM Adapter Factory

### Step 27: Write test for create_llm_adapter factory

**File:** `/home/ubuntu/ReToneProjects/RTMemory/server/tests/core/llm/test_llm_factory.py`

```python
"""Tests for create_llm_adapter factory function."""

import pytest

from app.config import LLMConfig
from app.core.llm import create_llm_adapter
from app.core.llm.adapter import LLMAdapter
from app.core.llm.openai_adapter import OpenAIAdapter
from app.core.llm.anthropic_adapter import AnthropicAdapter
from app.core.llm.ollama_adapter import OllamaAdapter


class TestCreateLLMAdapter:
    """Tests for the create_llm_adapter factory function."""

    def test_creates_openai_adapter(self):
        """create_llm_adapter creates OpenAIAdapter for provider='openai'."""
        config = LLMConfig(
            provider="openai",
            model="gpt-4o",
            api_key="sk-test-key",
        )
        adapter = create_llm_adapter(config)
        assert isinstance(adapter, OpenAIAdapter)
        assert isinstance(adapter, LLMAdapter)
        assert adapter._api_key == "sk-test-key"
        assert adapter._model == "gpt-4o"
        assert adapter._base_url == "https://api.openai.com/v1"

    def test_creates_openai_adapter_with_custom_base_url(self):
        """create_llm_adapter respects custom base_url for OpenAI."""
        config = LLMConfig(
            provider="openai",
            model="gpt-4o",
            api_key="sk-test-key",
            base_url="https://proxy.example.com/v1",
        )
        adapter = create_llm_adapter(config)
        assert isinstance(adapter, OpenAIAdapter)
        assert adapter._base_url == "https://proxy.example.com/v1"

    def test_creates_anthropic_adapter(self):
        """create_llm_adapter creates AnthropicAdapter for provider='anthropic'."""
        config = LLMConfig(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            api_key="sk-ant-test-key",
        )
        adapter = create_llm_adapter(config)
        assert isinstance(adapter, AnthropicAdapter)
        assert isinstance(adapter, LLMAdapter)
        assert adapter._api_key == "sk-ant-test-key"
        assert adapter._model == "claude-sonnet-4-20250514"
        assert adapter._base_url == "https://api.anthropic.com"

    def test_creates_anthropic_adapter_with_custom_base_url(self):
        """create_llm_adapter respects custom base_url for Anthropic."""
        config = LLMConfig(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            api_key="sk-ant-test-key",
            base_url="https://proxy.anthropic.example.com",
        )
        adapter = create_llm_adapter(config)
        assert isinstance(adapter, AnthropicAdapter)
        assert adapter._base_url == "https://proxy.anthropic.example.com"

    def test_creates_ollama_adapter(self):
        """create_llm_adapter creates OllamaAdapter for provider='ollama'."""
        config = LLMConfig(
            provider="ollama",
            model="qwen2.5:7b",
            base_url="http://localhost:11434",
        )
        adapter = create_llm_adapter(config)
        assert isinstance(adapter, OllamaAdapter)
        assert isinstance(adapter, LLMAdapter)
        assert adapter._model == "qwen2.5:7b"
        assert adapter._base_url == "http://localhost:11434"

    def test_creates_ollama_adapter_with_default_base_url(self):
        """create_llm_adapter uses default Ollama base_url when not specified."""
        config = LLMConfig(
            provider="ollama",
            model="qwen2.5:7b",
        )
        adapter = create_llm_adapter(config)
        assert isinstance(adapter, OllamaAdapter)
        assert adapter._base_url == "http://localhost:11434"

    def test_raises_for_unknown_provider(self):
        """create_llm_adapter raises ValueError for unknown provider."""
        config = LLMConfig(
            provider="gemini",
            model="gemini-pro",
            api_key="test-key",
        )
        with pytest.raises(ValueError, match="Unknown LLM provider: gemini"):
            create_llm_adapter(config)

    def test_case_insensitive_provider(self):
        """create_llm_adapter is case-insensitive for provider name."""
        config = LLMConfig(
            provider="OpenAI",
            model="gpt-4o",
            api_key="sk-test-key",
        )
        adapter = create_llm_adapter(config)
        assert isinstance(adapter, OpenAIAdapter)

    async def test_created_openai_adapter_can_complete(self):
        """Factory-created OpenAI adapter has working complete method."""
        import unittest.mock
        import httpx
        from tests.conftest import make_httpx_response

        mock_client = unittest.mock.AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.return_value = make_httpx_response(
            200,
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Factory works!"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            },
        )

        config = LLMConfig(provider="openai", model="gpt-4o", api_key="sk-test")
        adapter = create_llm_adapter(config, client=mock_client)
        result = await adapter.complete(messages=[{"role": "user", "content": "Hi"}])
        assert result == "Factory works!"
```

**Run:**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/core/llm/test_llm_factory.py -v 2>&1 | tail -15
```

**Expected:** Tests should PASS since `__init__.py` already contains the factory function from Step 3.

```
tests/core/llm/test_llm_factory.py::TestCreateLLMAdapter::test_creates_openai_adapter PASSED
tests/core/llm/test_llm_factory.py::TestCreateLLMAdapter::test_creates_openai_adapter_with_custom_base_url PASSED
tests/core/llm/test_llm_factory.py::TestCreateLLMAdapter::test_creates_anthropic_adapter PASSED
tests/core/llm/test_llm_factory.py::TestCreateLLMAdapter::test_creates_anthropic_adapter_with_custom_base_url PASSED
tests/core/llm/test_llm_factory.py::TestCreateLLMAdapter::test_creates_ollama_adapter PASSED
tests/core/llm/test_llm_factory.py::TestCreateLLMAdapter::test_creates_ollama_adapter_with_default_base_url PASSED
tests/core/llm/test_llm_factory.py::TestCreateLLMAdapter::test_raises_for_unknown_provider PASSED
tests/core/llm/test_llm_factory.py::test_case_insensitive_provider PASSED
tests/core/llm/test_llm_factory.py::test_created_openai_adapter_can_complete PASSED

9 passed
```

---

### Step 28: Commit LLM adapter factory

```bash
cd /home/ubuntu/ReToneProjects/RTMemory
git add server/tests/core/llm/test_llm_factory.py
git commit -m "feat(llm-adapter): add create_llm_adapter factory tests"
```

---

## Phase 9: Local Embedding Service

### Step 29: Write test for LocalEmbeddingService

**File:** `/home/ubuntu/ReToneProjects/RTMemory/server/tests/core/embedding/test_local_embedding.py`

```python
"""Tests for LocalEmbeddingService using sentence-transformers."""

import unittest.mock

import numpy as np
import pytest

from app.core.embedding.local_embedding import LocalEmbeddingService


class TestLocalEmbeddingService:
    """Tests for LocalEmbeddingService."""

    def test_get_dimension_before_embed(self):
        """get_dimension() lazy-loads the model and returns the dimension."""
        mock_model = unittest.mock.MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768

        with unittest.mock.patch(
            "app.core.embedding.local_embedding.SentenceTransformer",
            return_value=mock_model,
        ):
            service = LocalEmbeddingService(model_name="BAAI/bge-base-zh-v1.5")
            dim = service.get_dimension()
            assert dim == 768

    async def test_embed_single_text(self):
        """embed() returns embeddings for a single text."""
        mock_model = unittest.mock.MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_model.get_sentence_embedding_dimension.return_value = 3

        with unittest.mock.patch(
            "app.core.embedding.local_embedding.SentenceTransformer",
            return_value=mock_model,
        ):
            service = LocalEmbeddingService(model_name="test-model")
            result = await service.embed(["hello"])
            assert result == [[0.1, 0.2, 0.3]]
            mock_model.encode.assert_called_once_with(["hello"], normalize_embeddings=True)

    async def test_embed_multiple_texts(self):
        """embed() returns embeddings for multiple texts."""
        mock_model = unittest.mock.MagicMock()
        mock_model.encode.return_value = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ])
        mock_model.get_sentence_embedding_dimension.return_value = 3

        with unittest.mock.patch(
            "app.core.embedding.local_embedding.SentenceTransformer",
            return_value=mock_model,
        ):
            service = LocalEmbeddingService(model_name="test-model")
            result = await service.embed(["hello", "world"])
            assert len(result) == 2
            assert result[0] == [0.1, 0.2, 0.3]
            assert result[1] == [0.4, 0.5, 0.6]

    async def test_embed_lazy_loads_model(self):
        """embed() lazy-loads the model on first call."""
        mock_model = unittest.mock.MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2]])
        mock_model.get_sentence_embedding_dimension.return_value = 2

        with unittest.mock.patch(
            "app.core.embedding.local_embedding.SentenceTransformer",
            return_value=mock_model,
        ) as mock_cls:
            service = LocalEmbeddingService(model_name="test-model")
            # Model should not be loaded yet
            mock_cls.assert_not_called()

            # Model should be loaded on first embed call
            await service.embed(["hello"])
            mock_cls.assert_called_once_with("test-model")

            # Second call should not reload
            await service.embed(["world"])
            assert mock_cls.call_count == 1

    async def test_embed_empty_list(self):
        """embed() returns empty list for empty input."""
        mock_model = unittest.mock.MagicMock()
        mock_model.encode.return_value = np.array([])
        mock_model.get_sentence_embedding_dimension.return_value = 768

        with unittest.mock.patch(
            "app.core.embedding.local_embedding.SentenceTransformer",
            return_value=mock_model,
        ):
            service = LocalEmbeddingService(model_name="test-model")
            result = await service.embed([])
            assert result == []

    def test_default_model_name(self):
        """LocalEmbeddingService defaults to bge-base-zh-v1.5."""
        service = LocalEmbeddingService()
        assert service._model_name == "BAAI/bge-base-zh-v1.5"

    async def test_get_dimension_matches_embed_dimension(self):
        """get_dimension() returns the same dimension as the embeddings."""
        mock_model = unittest.mock.MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])
        mock_model.get_sentence_embedding_dimension.return_value = 6

        with unittest.mock.patch(
            "app.core.embedding.local_embedding.SentenceTransformer",
            return_value=mock_model,
        ):
            service = LocalEmbeddingService(model_name="test-model")
            dim = service.get_dimension()
            result = await service.embed(["hello"])
            assert dim == len(result[0])
```

**Run:**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/core/embedding/test_local_embedding.py -v 2>&1 | tail -15
```

**Expected:** Tests FAIL because `local_embedding.py` does not exist yet.

---

### Step 30: Implement LocalEmbeddingService

**File:** `/home/ubuntu/ReToneProjects/RTMemory/server/app/core/embedding/local_embedding.py`

```python
"""Local embedding service using sentence-transformers.

Provides local text embedding by lazy-loading a sentence-transformers model.
No API calls are made — everything runs on the local machine.
"""

from __future__ import annotations

import numpy as np

from app.core.embedding.service import EmbeddingService


class LocalEmbeddingService(EmbeddingService):
    """Embedding service using a locally-loaded sentence-transformers model.

    The model is lazy-loaded on the first call to embed() or get_dimension(),
    avoiding the overhead of loading the model at import time.

    Args:
        model_name: HuggingFace model name or path.
            Defaults to "BAAI/bge-base-zh-v1.5" (Chinese-optimized embedding model).
    """

    def __init__(self, model_name: str = "BAAI/bge-base-zh-v1.5"):
        self._model_name = model_name
        self._model = None
        self._dimension: int | None = None

    def _load_model(self):
        """Lazy-load the sentence-transformers model on first use."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
            self._dimension = self._model.get_sentence_embedding_dimension()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of text strings using the local model.

        The model is lazy-loaded on the first call.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors, one per input string.
        """
        if not texts:
            return []

        self._load_model()
        embeddings: np.ndarray = self._model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    def get_dimension(self) -> int:
        """Return the dimension of embedding vectors.

        The model is lazy-loaded if not already loaded.

        Returns:
            Integer dimension of the embedding vectors.
        """
        self._load_model()
        return self._dimension
```

**Run:**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/core/embedding/test_local_embedding.py -v 2>&1 | tail -15
```

**Expected:**

```
tests/core/embedding/test_local_embedding.py::TestLocalEmbeddingService::test_get_dimension_before_embed PASSED
tests/core/embedding/test_local_embedding.py::TestLocalEmbeddingService::test_embed_single_text PASSED
tests/core/embedding/test_local_embedding.py::TestLocalEmbeddingService::test_embed_multiple_texts PASSED
tests/core/embedding/test_local_embedding.py::TestLocalEmbeddingService::test_embed_lazy_loads_model PASSED
tests/core/embedding/test_local_embedding.py::TestLocalEmbeddingService::test_embed_empty_list PASSED
tests/core/embedding/test_local_embedding.py::TestLocalEmbeddingService::test_default_model_name PASSED
tests/core/embedding/test_local_embedding.py::TestLocalEmbeddingService::test_get_dimension_matches_embed_dimension PASSED

7 passed
```

---

### Step 31: Commit Local embedding service

```bash
cd /home/ubuntu/ReToneProjects/RTMemory
git add server/app/core/embedding/local_embedding.py server/tests/core/embedding/test_local_embedding.py
git commit -m "feat(embedding): implement LocalEmbeddingService with lazy-loaded sentence-transformers"
```

---

## Phase 10: OpenAI Embedding Service

### Step 32: Write test for OpenAIEmbeddingService

**File:** `/home/ubuntu/ReToneProjects/RTMemory/server/tests/core/embedding/test_openai_embedding.py`

```python
"""Tests for OpenAIEmbeddingService."""

import unittest.mock

import httpx
import pytest

from app.core.embedding.openai_embedding import OpenAIEmbeddingService
from tests.conftest import make_httpx_response


class TestOpenAIEmbeddingService:
    """Tests for OpenAIEmbeddingService."""

    async def test_embed_single_text(self, mock_client):
        """embed() returns embeddings for a single text."""
        mock_client.post.return_value = make_httpx_response(
            200,
            {
                "object": "list",
                "data": [
                    {"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]},
                ],
                "model": "text-embedding-3-small",
                "usage": {"prompt_tokens": 5, "total_tokens": 5},
            },
        )

        service = OpenAIEmbeddingService(
            api_key="sk-test",
            model="text-embedding-3-small",
            client=mock_client,
        )
        result = await service.embed(["hello"])

        assert result == [[0.1, 0.2, 0.3]]

        call_args = mock_client.post.call_args
        assert call_args.kwargs["json"]["model"] == "text-embedding-3-small"
        assert call_args.kwargs["json"]["input"] == ["hello"]

    async def test_embed_multiple_texts(self, mock_client):
        """embed() returns embeddings for multiple texts."""
        mock_client.post.return_value = make_httpx_response(
            200,
            {
                "object": "list",
                "data": [
                    {"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]},
                    {"object": "embedding", "index": 1, "embedding": [0.4, 0.5, 0.6]},
                ],
                "model": "text-embedding-3-small",
                "usage": {"prompt_tokens": 10, "total_tokens": 10},
            },
        )

        service = OpenAIEmbeddingService(
            api_key="sk-test",
            model="text-embedding-3-small",
            client=mock_client,
        )
        result = await service.embed(["hello", "world"])

        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]
        assert result[1] == [0.4, 0.5, 0.6]

    async def test_embed_sends_authorization_header(self, mock_client):
        """embed() includes Bearer token in Authorization header."""
        mock_client.post.return_value = make_httpx_response(
            200,
            {
                "object": "list",
                "data": [
                    {"object": "embedding", "index": 0, "embedding": [0.1]},
                ],
                "model": "text-embedding-3-small",
                "usage": {"prompt_tokens": 5, "total_tokens": 5},
            },
        )

        service = OpenAIEmbeddingService(
            api_key="sk-my-secret",
            model="text-embedding-3-small",
            client=mock_client,
        )
        await service.embed(["test"])

        call_args = mock_client.post.call_args
        assert call_args.kwargs["headers"]["Authorization"] == "Bearer sk-my-secret"

    async def test_embed_api_error(self, mock_client):
        """embed() raises httpx.HTTPStatusError on API error."""
        mock_client.post.side_effect = httpx.HTTPStatusError(
            message="401 Unauthorized",
            request=httpx.Request("POST", "https://api.openai.com/v1/embeddings"),
            response=httpx.Response(401, json={"error": {"message": "Invalid API key"}}),
        )

        service = OpenAIEmbeddingService(
            api_key="sk-invalid",
            model="text-embedding-3-small",
            client=mock_client,
        )
        with pytest.raises(httpx.HTTPStatusError):
            await service.embed(["test"])

    def test_get_dimension_text_embedding_3_small(self):
        """get_dimension() returns 1536 for text-embedding-3-small."""
        service = OpenAIEmbeddingService(
            api_key="sk-test",
            model="text-embedding-3-small",
        )
        assert service.get_dimension() == 1536

    def test_get_dimension_text_embedding_3_large(self):
        """get_dimension() returns 3072 for text-embedding-3-large."""
        service = OpenAIEmbeddingService(
            api_key="sk-test",
            model="text-embedding-3-large",
        )
        assert service.get_dimension() == 3072

    def test_get_dimension_text_embedding_ada_002(self):
        """get_dimension() returns 1536 for text-embedding-ada-002."""
        service = OpenAIEmbeddingService(
            api_key="sk-test",
            model="text-embedding-ada-002",
        )
        assert service.get_dimension() == 1536

    def test_get_dimension_unknown_model_defaults_to_1536(self):
        """get_dimension() returns 1536 for unknown models."""
        service = OpenAIEmbeddingService(
            api_key="sk-test",
            model="my-custom-model",
        )
        assert service.get_dimension() == 1536

    async def test_embed_uses_custom_base_url(self, mock_client):
        """embed() uses custom base_url when provided."""
        mock_client.post.return_value = make_httpx_response(
            200,
            {
                "object": "list",
                "data": [
                    {"object": "embedding", "index": 0, "embedding": [0.1]},
                ],
                "model": "text-embedding-3-small",
                "usage": {"prompt_tokens": 5, "total_tokens": 5},
            },
        )

        service = OpenAIEmbeddingService(
            api_key="sk-test",
            model="text-embedding-3-small",
            base_url="https://proxy.example.com/v1",
            client=mock_client,
        )
        await service.embed(["test"])

        call_args = mock_client.post.call_args
        assert call_args.args[0] == "https://proxy.example.com/v1/embeddings"
```

**Run:**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/core/embedding/test_openai_embedding.py -v 2>&1 | tail -15
```

**Expected:** Tests FAIL because `openai_embedding.py` does not exist yet.

---

### Step 33: Implement OpenAIEmbeddingService

**File:** `/home/ubuntu/ReToneProjects/RTMemory/server/app/core/embedding/openai_embedding.py`

```python
"""OpenAI embedding service.

Provides text embeddings via the OpenAI embeddings API.
Supports text-embedding-3-small, text-embedding-3-large, and text-embedding-ada-002.
"""

from __future__ import annotations

import httpx

from app.core.embedding.service import EmbeddingService


class OpenAIEmbeddingService(EmbeddingService):
    """Embedding service using the OpenAI embeddings API.

    Args:
        api_key: OpenAI API key.
        model: Embedding model name. Defaults to "text-embedding-3-small".
        base_url: API base URL. Defaults to "https://api.openai.com/v1".
        client: Optional httpx.AsyncClient for dependency injection in tests.
    """

    # Known dimensions for OpenAI embedding models
    _MODEL_DIMENSIONS: dict[str, int] = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        base_url: str = "https://api.openai.com/v1",
        client: httpx.AsyncClient | None = None,
    ):
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.AsyncClient()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of text strings using the OpenAI embeddings API.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors, one per input string.

        Raises:
            httpx.HTTPStatusError: On API error responses.
        """
        response = await self._client.post(
            f"{self._base_url}/embeddings",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self._model,
                "input": texts,
            },
        )
        response.raise_for_status()
        data = response.json()
        # Sort by index to ensure order matches input
        sorted_data = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in sorted_data]

    def get_dimension(self) -> int:
        """Return the dimension of embedding vectors.

        Uses a lookup table for known OpenAI models.
        Returns 1536 (the most common dimension) for unknown models.

        Returns:
            Integer dimension of the embedding vectors.
        """
        return self._MODEL_DIMENSIONS.get(self._model, 1536)
```

**Run:**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/core/embedding/test_openai_embedding.py -v 2>&1 | tail -15
```

**Expected:**

```
tests/core/embedding/test_openai_embedding.py::TestOpenAIEmbeddingService::test_embed_single_text PASSED
tests/core/embedding/test_openai_embedding.py::TestOpenAIEmbeddingService::test_embed_multiple_texts PASSED
tests/core/embedding/test_openai_embedding.py::TestOpenAIEmbeddingService::test_embed_sends_authorization_header PASSED
tests/core/embedding/test_openai_embedding.py::TestOpenAIEmbeddingService::test_embed_api_error PASSED
tests/core/embedding/test_openai_embedding.py::TestOpenAIEmbeddingService::test_get_dimension_text_embedding_3_small PASSED
tests/core/embedding/test_openai_embedding.py::TestOpenAIEmbeddingService::test_get_dimension_text_embedding_3_large PASSED
tests/core/embedding/test_openAI_embedding.py::TestOpenAIEmbeddingService::test_get_dimension_text_embedding_ada_002 PASSED
tests/core/embedding/test_openai_embedding.py::TestOpenAIEmbeddingService::test_get_dimension_unknown_model_defaults_to_1536 PASSED
tests/core/embedding/test_openai_embedding.py::TestOpenAIEmbeddingService::test_embed_uses_custom_base_url PASSED

9 passed
```

---

### Step 34: Commit OpenAI embedding service

```bash
cd /home/ubuntu/ReToneProjects/RTMemory
git add server/app/core/embedding/openai_embedding.py server/tests/core/embedding/test_openai_embedding.py
git commit -m "feat(embedding): implement OpenAIEmbeddingService with API-backed embeddings"
```

---

## Phase 11: Embedding Factory

### Step 35: Write test for create_embedding_service factory

**File:** `/home/ubuntu/ReToneProjects/RTMemory/server/tests/core/embedding/test_embedding_factory.py`

```python
"""Tests for create_embedding_service factory function."""

import unittest.mock

import pytest

from app.config import EmbeddingConfig
from app.core.embedding import create_embedding_service
from app.core.embedding.service import EmbeddingService
from app.core.embedding.local_embedding import LocalEmbeddingService
from app.core.embedding.openai_embedding import OpenAIEmbeddingService


class TestCreateEmbeddingService:
    """Tests for the create_embedding_service factory function."""

    def test_creates_local_embedding_service(self):
        """create_embedding_service creates LocalEmbeddingService for provider='local'."""
        config = EmbeddingConfig(
            provider="local",
            model="BAAI/bge-base-zh-v1.5",
        )
        service = create_embedding_service(config)
        assert isinstance(service, LocalEmbeddingService)
        assert isinstance(service, EmbeddingService)
        assert service._model_name == "BAAI/bge-base-zh-v1.5"

    def test_creates_openai_embedding_service(self):
        """create_embedding_service creates OpenAIEmbeddingService for provider='openai'."""
        config = EmbeddingConfig(
            provider="openai",
            model="text-embedding-3-small",
            api_key="sk-test-key",
        )
        service = create_embedding_service(config)
        assert isinstance(service, OpenAIEmbeddingService)
        assert isinstance(service, EmbeddingService)
        assert service._api_key == "sk-test-key"
        assert service._model == "text-embedding-3-small"

    def test_creates_openai_with_custom_base_url(self):
        """create_embedding_service uses custom base_url for OpenAI."""
        config = EmbeddingConfig(
            provider="openai",
            model="text-embedding-3-small",
            api_key="sk-test-key",
            base_url="https://proxy.example.com/v1",
        )
        service = create_embedding_service(config)
        assert isinstance(service, OpenAIEmbeddingService)
        assert service._base_url == "https://proxy.example.com/v1"

    def test_openai_default_base_url(self):
        """create_embedding_service defaults to OpenAI's base URL."""
        config = EmbeddingConfig(
            provider="openai",
            model="text-embedding-3-small",
            api_key="sk-test-key",
        )
        service = create_embedding_service(config)
        assert service._base_url == "https://api.openai.com/v1"

    def test_raises_for_unknown_provider(self):
        """create_embedding_service raises ValueError for unknown provider."""
        config = EmbeddingConfig(
            provider="cohere",
            model="embed-english-v3",
            api_key="test-key",
        )
        with pytest.raises(ValueError, match="Unknown embedding provider: cohere"):
            create_embedding_service(config)

    def test_case_insensitive_provider(self):
        """create_embedding_service is case-insensitive for provider name."""
        config = EmbeddingConfig(
            provider="Local",
            model="BAAI/bge-base-zh-v1.5",
        )
        service = create_embedding_service(config)
        assert isinstance(service, LocalEmbeddingService)

    async def test_created_local_service_can_embed(self):
        """Factory-created local embedding service has working embed method."""
        mock_model = unittest.mock.MagicMock()
        mock_model.encode.return_value = __import__("numpy").array([[0.1, 0.2, 0.3]])
        mock_model.get_sentence_embedding_dimension.return_value = 3

        with unittest.mock.patch(
            "app.core.embedding.local_embedding.SentenceTransformer",
            return_value=mock_model,
        ):
            config = EmbeddingConfig(provider="local", model="test-model")
            service = create_embedding_service(config)
            result = await service.embed(["hello"])
            assert result == [[0.1, 0.2, 0.3]]

    async def test_created_openai_service_can_embed(self):
        """Factory-created OpenAI embedding service has working embed method."""
        import httpx
        from tests.conftest import make_httpx_response

        mock_client = unittest.mock.AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.return_value = make_httpx_response(
            200,
            {
                "object": "list",
                "data": [
                    {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]},
                ],
                "model": "text-embedding-3-small",
                "usage": {"prompt_tokens": 5, "total_tokens": 5},
            },
        )

        config = EmbeddingConfig(
            provider="openai",
            model="text-embedding-3-small",
            api_key="sk-test",
        )
        service = create_embedding_service(config, client=mock_client)
        result = await service.embed(["hello"])
        assert result == [[0.1, 0.2]]
```

**Run:**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/core/embedding/test_embedding_factory.py -v 2>&1 | tail -15
```

**Expected:** Tests should PASS since `__init__.py` already contains the factory function.

```
tests/core/embedding/test_embedding_factory.py::TestCreateEmbeddingService::test_creates_local_embedding_service PASSED
tests/core/embedding/test_embedding_factory.py::TestCreateEmbeddingService::test_creates_openai_embedding_service PASSED
tests/core/embedding/test_embedding_factory.py::TestCreateEmbeddingService::test_creates_openai_with_custom_base_url PASSED
tests/core/embedding/test_embedding_factory.py::TestCreateEmbeddingService::test_openai_default_base_url PASSED
tests/core/embedding/test_embedding_factory.py::TestCreateEmbeddingService::test_raises_for_unknown_provider PASSED
tests/core/embedding/test_embedding_factory.py::TestCreateEmbeddingService::test_case_insensitive_provider PASSED
tests/core/embedding/test_embedding_factory.py::TestCreateEmbeddingService::test_created_local_service_can_embed PASSED
tests/core/embedding/test_embedding_factory.py::TestCreateEmbeddingService::test_created_openai_service_can_embed PASSED

8 passed
```

---

### Step 36: Commit embedding factory

```bash
cd /home/ubuntu/ReToneProjects/RTMemory
git add server/tests/core/embedding/test_embedding_factory.py
git commit -m "feat(embedding): add create_embedding_service factory tests"
```

---

## Phase 12: Config File and Integration

### Step 37: Write config.yaml

**File:** `/home/ubuntu/ReToneProjects/RTMemory/config.yaml`

```yaml
# RTMemory Configuration
# Switch providers by changing the provider field.

llm:
  # Options: openai, anthropic, ollama
  provider: ollama
  model: qwen2.5:7b
  base_url: http://localhost:11434
  # api_key: sk-...  # Uncomment for OpenAI/Anthropic
  temperature: 0.1
  max_tokens: 2048

embedding:
  # Options: local, openai
  provider: local
  model: BAAI/bge-base-zh-v1.5
  # For OpenAI embedding:
  # provider: openai
  # model: text-embedding-3-small
  # api_key: sk-...
  # base_url: https://api.openai.com/v1
```

---

### Step 38: Write integration test

**File:** `/home/ubuntu/ReToneProjects/RTMemory/server/tests/test_integration.py`

```python
"""Integration tests for LLM adapter and embedding service configuration.

These tests verify that the config.yaml can be loaded and used to create
adapters and services through the factory functions. All HTTP calls are mocked.
"""

import unittest.mock
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest
import yaml

from app.config import AppConfig, load_config
from app.core.llm import create_llm_adapter
from app.core.llm.ollama_adapter import OllamaAdapter
from app.core.llm.openai_adapter import OpenAIAdapter
from app.core.llm.anthropic_adapter import AnthropicAdapter
from app.core.embedding import create_embedding_service
from app.core.embedding.local_embedding import LocalEmbeddingService
from app.core.embedding.openai_embedding import OpenAIEmbeddingService
from tests.conftest import make_httpx_response


class TestConfigIntegration:
    """Tests for loading config.yaml and creating adapters."""

    def test_load_default_config(self):
        """Load the project's config.yaml and verify structure."""
        config_path = Path(__file__).parent.parent.parent / "config.yaml"
        if not config_path.exists():
            pytest.skip("config.yaml not found at project root")

        config = load_config(config_path)
        assert config.llm.provider in ("openai", "anthropic", "ollama")
        assert config.embedding.provider in ("local", "openai")
        assert config.llm.model
        assert config.embedding.model

    def test_create_ollama_adapter_from_config(self):
        """Create OllamaAdapter from an Ollama config."""
        config_yaml = {
            "llm": {
                "provider": "ollama",
                "model": "qwen2.5:7b",
                "base_url": "http://localhost:11434",
                "temperature": 0.1,
                "max_tokens": 2048,
            },
            "embedding": {
                "provider": "local",
                "model": "BAAI/bge-base-zh-v1.5",
            },
        }
        config = AppConfig(**config_yaml)
        adapter = create_llm_adapter(config.llm)
        assert isinstance(adapter, OllamaAdapter)
        assert adapter._model == "qwen2.5:7b"

    def test_create_openai_adapter_from_config(self):
        """Create OpenAIAdapter from an OpenAI config."""
        config_yaml = {
            "llm": {
                "provider": "openai",
                "model": "gpt-4o",
                "api_key": "sk-test-key",
                "temperature": 0.7,
                "max_tokens": 1024,
            },
            "embedding": {
                "provider": "openai",
                "model": "text-embedding-3-small",
                "api_key": "sk-test-key",
            },
        }
        config = AppConfig(**config_yaml)
        adapter = create_llm_adapter(config.llm)
        assert isinstance(adapter, OpenAIAdapter)

    def test_create_anthropic_adapter_from_config(self):
        """Create AnthropicAdapter from an Anthropic config."""
        config_yaml = {
            "llm": {
                "provider": "anthropic",
                "model": "claude-sonnet-4-20250514",
                "api_key": "sk-ant-test-key",
                "temperature": 0.5,
                "max_tokens": 2048,
            },
            "embedding": {
                "provider": "local",
                "model": "BAAI/bge-base-zh-v1.5",
            },
        }
        config = AppConfig(**config_yaml)
        adapter = create_llm_adapter(config.llm)
        assert isinstance(adapter, AnthropicAdapter)

    def test_create_local_embedding_from_config(self):
        """Create LocalEmbeddingService from a local config."""
        config_yaml = {
            "llm": {
                "provider": "ollama",
                "model": "qwen2.5:7b",
                "base_url": "http://localhost:11434",
                "temperature": 0.1,
                "max_tokens": 2048,
            },
            "embedding": {
                "provider": "local",
                "model": "BAAI/bge-base-zh-v1.5",
            },
        }
        config = AppConfig(**config_yaml)
        service = create_embedding_service(config.embedding)
        assert isinstance(service, LocalEmbeddingService)

    def test_create_openai_embedding_from_config(self):
        """Create OpenAIEmbeddingService from an OpenAI config."""
        config_yaml = {
            "llm": {
                "provider": "openai",
                "model": "gpt-4o",
                "api_key": "sk-test-key",
                "temperature": 0.7,
                "max_tokens": 1024,
            },
            "embedding": {
                "provider": "openai",
                "model": "text-embedding-3-small",
                "api_key": "sk-test-key",
            },
        }
        config = AppConfig(**config_yaml)
        service = create_embedding_service(config.embedding)
        assert isinstance(service, OpenAIEmbeddingService)


class TestEndToEndWithMocking:
    """End-to-end tests with mocked HTTP calls."""

    async def test_ollama_adapter_full_flow(self):
        """Full flow: config -> factory -> adapter -> complete + complete_structured."""
        config_yaml = {
            "llm": {
                "provider": "ollama",
                "model": "qwen2.5:7b",
                "base_url": "http://localhost:11434",
                "temperature": 0.1,
                "max_tokens": 2048,
            },
            "embedding": {
                "provider": "local",
                "model": "BAAI/bge-base-zh-v1.5",
            },
        }
        config = AppConfig(**config_yaml)

        mock_client = AsyncMock(spec=httpx.AsyncClient)

        # Mock complete response
        mock_client.post.return_value = make_httpx_response(
            200,
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": "qwen2.5:7b",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello from Ollama!"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

        adapter = create_llm_adapter(config.llm, client=mock_client)
        result = await adapter.complete(
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert result == "Hello from Ollama!"

    async def test_openai_adapter_then_embedding_flow(self):
        """Full flow: LLM complete -> embedding with mocked calls."""
        config_yaml = {
            "llm": {
                "provider": "openai",
                "model": "gpt-4o",
                "api_key": "sk-test-key",
                "temperature": 0.7,
                "max_tokens": 1024,
            },
            "embedding": {
                "provider": "openai",
                "model": "text-embedding-3-small",
                "api_key": "sk-test-key",
            },
        }
        config = AppConfig(**config_yaml)

        mock_client = AsyncMock(spec=httpx.AsyncClient)

        # Mock LLM response
        mock_client.post.return_value = make_httpx_response(
            200,
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Extracted: Alice likes Python"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

        adapter = create_llm_adapter(config.llm, client=mock_client)
        result = await adapter.complete(
            messages=[{"role": "user", "content": "What does Alice like?"}],
        )
        assert "Alice" in result

        # Mock embedding response
        mock_client.post.return_value = make_httpx_response(
            200,
            {
                "object": "list",
                "data": [
                    {"object": "embedding", "index": 0, "embedding": [0.1] * 1536},
                ],
                "model": "text-embedding-3-small",
            },
        )

        embedding_service = create_embedding_service(config.embedding, client=mock_client)
        embeddings = await embedding_service.embed(["Alice likes Python"])
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 1536
        assert embedding_service.get_dimension() == 1536

    async def test_anthropic_adapter_structured_output_flow(self):
        """Full flow: Anthropic complete_structured with mocked response."""
        config_yaml = {
            "llm": {
                "provider": "anthropic",
                "model": "claude-sonnet-4-20250514",
                "api_key": "sk-ant-test-key",
                "temperature": 0.1,
                "max_tokens": 1024,
            },
            "embedding": {
                "provider": "local",
                "model": "BAAI/bge-base-zh-v1.5",
            },
        }
        config = AppConfig(**config_yaml)

        mock_client = AsyncMock(spec=httpx.AsyncClient)

        # Mock structured response
        mock_client.post.return_value = make_httpx_response(
            200,
            {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": '{"entities": [{"name": "Bob", "type": "person"}], "relations": [], "memories": []}'}],
                "model": "claude-sonnet-4-20250514",
                "stop_reason": "end_turn",
            },
        )

        adapter = create_llm_adapter(config.llm, client=mock_client)
        schema = {
            "type": "object",
            "properties": {
                "entities": {"type": "array"},
                "relations": {"type": "array"},
                "memories": {"type": "array"},
            },
        }
        result = await adapter.complete_structured(
            messages=[{"role": "user", "content": "Extract info from: Bob lives in NYC"}],
            schema=schema,
        )
        assert result["entities"][0]["name"] == "Bob"
        assert result["entities"][0]["type"] == "person"
```

**Run:**

```bash
cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/test_integration.py -v 2>&1 | tail -15
```

**Expected:**

```
tests/test_integration.py::TestConfigIntegration::test_load_default_config PASSED
tests/test_integration.py::TestConfigIntegration::test_create_ollama_adapter_from_config PASSED
tests/test_integration.py::TestConfigIntegration::test_create_openai_adapter_from_config PASSED
tests/test_integration.py::TestConfigIntegration::test_create_anthropic_adapter_from_config PASSED
tests/test_integration.py::TestConfigIntegration::test_create_local_embedding_from_config PASSED
tests/test_integration.py::TestConfigIntegration::test_create_openai_embedding_from_config PASSED
tests/test_integration.py::TestEndToEndWithMocking::test_ollama_adapter_full_flow PASSED
tests/test_integration.py::TestEndToEndWithMocking::test_openai_adapter_then_embedding_flow PASSED
tests/test_integration.py::TestEndToEndWithMocking::test_anthropic_adapter_structured_output_flow PASSED

9 passed
```

---

### Step 39: Run full test suite

Verify that all tests pass together:

```bash
cd /home/ubuntu/ReToneProjects/RTMemory/server && python -m pytest tests/ -v 2>&1 | tail -40
```

**Expected:**

```
tests/core/embedding/test_embedding_factory.py::TestCreateEmbeddingService::test_case_insensitive_provider PASSED
tests/core/embedding/test_embedding_factory.py::TestCreateEmbeddingService::test_created_local_service_can_embed PASSED
tests/core/embedding/test_embedding_factory.py::TestCreateEmbeddingService::test_created_openai_service_can_embed PASSED
tests/core/llm/test_adapter.py::test_llm_adapter_cannot_be_instantiated_directly PASSED
tests/core/llm/test_adapter.py::test_incomplete_subclass_cannot_be_instantiated PASSED
tests/core/llm/test_adapter.py::test_complete_subclass_can_be_instantiated PASSED
tests/core/llm/test_adapter.py::test_llm_adapter_has_required_methods PASSED
tests/core/llm/test_anthropic_adapter.py::TestAnthropicAdapterComplete::test_complete_basic PASSED
...
tests/test_integration.py::TestEndToEndWithMocking::test_anthropic_adapter_structured_output_flow PASSED

=== X passed ===
```

All tests should pass with no failures.

---

### Step 40: Commit config file and integration tests

```bash
cd /home/ubuntu/ReToneProjects/RTMemory
git add config.yaml server/tests/test_integration.py
git commit -m "feat(llm-adapter): add config.yaml and integration tests for full adapter flow"
```

---

### Step 41: Final commit — verify all files are tracked

```bash
cd /home/ubuntu/ReToneProjects/RTMemory
git add -A
git status
```

Verify that all files in `server/app/core/llm/`, `server/app/core/embedding/`, `server/app/config.py`, `server/tests/`, and `config.yaml` are tracked. If there are untracked files, add and commit them.

```bash
cd /home/ubuntu/ReToneProjects/RTMemory && git add -A && git commit -m "chore(llm-adapter): ensure all implementation files are tracked"
```

---

## Summary

| Component | File | Lines of Code (approx) |
|-----------|------|------------------------|
| LLMAdapter ABC | `server/app/core/llm/adapter.py` | 65 |
| OpenAIAdapter | `server/app/core/llm/openai_adapter.py` | 85 |
| AnthropicAdapter | `server/app/core/llm/anthropic_adapter.py` | 110 |
| OllamaAdapter | `server/app/core/llm/ollama_adapter.py` | 95 |
| EmbeddingService ABC | `server/app/core/embedding/service.py` | 35 |
| LocalEmbeddingService | `server/app/core/embedding/local_embedding.py` | 50 |
| OpenAIEmbeddingService | `server/app/core/embedding/openai_embedding.py` | 65 |
| Config system | `server/app/config.py` | 70 |
| Factory: LLM | `server/app/core/llm/__init__.py` | 45 |
| Factory: Embedding | `server/app/core/embedding/__init__.py` | 30 |
| Config YAML | `config.yaml` | 15 |
| **Test files** | `server/tests/` | ~500 |

**Key design decisions:**
- All adapters accept an optional `httpx.AsyncClient` for dependency injection (testability)
- Ollama uses the OpenAI-compatible `/v1/chat/completions` endpoint
- Anthropic structured output uses prompt engineering (no native `response_format`)
- Ollama structured output uses `response_format={"type": "json_object"}` + prompt instruction
- OpenAI structured output uses native `json_schema` response_format
- Local embedding lazy-loads the sentence-transformers model on first call
- Factory functions use lazy imports to avoid circular dependencies
- All HTTP calls are mocked in tests (no real API keys required)