"""Tests for configuration loading via pydantic-settings."""

import os
import tempfile

import pytest
import yaml

from app.config import Settings, DatabaseConfig, LLMConfig, EmbeddingConfig, ServerConfig, AppConfig, load_config


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
        assert cfg.base_url is None

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
        assert config.llm.base_url is None  # default from LLMConfig
        assert config.embedding.api_key == "sk-test-key"