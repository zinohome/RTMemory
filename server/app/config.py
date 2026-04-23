"""Configuration system using pydantic-settings with YAML + env var overrides.

YAML config.yaml provides defaults. Environment variables prefixed with RTMEM_ override.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, model_validator
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

    @model_validator(mode="after")
    def _compute_url(self) -> "DatabaseConfig":
        """If url is not set explicitly, build it from components."""
        if not self.url:
            object.__setattr__(
                self,
                "url",
                f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}",
            )
        return self

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
    base_url: Optional[str] = Field(default=None, description="LLM API base URL")
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


# Map of sub-config names to their config classes and env prefixes
_SUB_CONFIGS = {
    "server": (ServerConfig, "RTMEM_SERVER_"),
    "database": (DatabaseConfig, "RTMEM_DATABASE_"),
    "llm": (LLMConfig, "RTMEM_LLM_"),
    "embedding": (EmbeddingConfig, "RTMEM_EMBEDDING_"),
}


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

        Env vars take precedence over YAML values for sub-config fields.

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

        # Apply env var overrides to each sub-config dict from YAML
        for sub_name, (sub_cls, env_prefix) in _SUB_CONFIGS.items():
            if sub_name not in data or not isinstance(data[sub_name], dict):
                continue
            sub_data = data[sub_name]
            for field_name in list(sub_data.keys()):
                env_key = f"{env_prefix}{field_name.upper()}"
                if env_key in os.environ:
                    # Env var overrides the YAML value
                    sub_data[field_name] = os.environ[env_key]

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


class AppConfig(BaseModel):
    """Top-level application configuration for LLM and embedding adapters.

    This is a simpler config model than Settings — it only covers LLM and
    embedding config, and is suitable for the adapter factory functions.

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