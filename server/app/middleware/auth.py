"""API Key authentication middleware for RTMemory.

Provides a lightweight API key check that can be enabled/disabled via config.
When RTMEM_AUTH_ENABLED=true, all /v1/* endpoints require a valid API key
passed via the Authorization header (Bearer <key>) or X-API-Key header.

The root (/) and health (/health) endpoints are always public.
API keys are configured via RTMEM_API_KEYS (comma-separated) environment variable.
"""
from __future__ import annotations

import os
from typing import Optional

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse


def _load_api_keys() -> set[str]:
    """Load valid API keys from environment variable."""
    keys_str = os.environ.get("RTMEM_API_KEYS", "")
    if not keys_str:
        return set()
    return {k.strip() for k in keys_str.split(",") if k.strip()}


def _is_auth_enabled() -> bool:
    """Check if authentication is enabled."""
    return os.environ.get("RTMEM_AUTH_ENABLED", "false").lower() in ("true", "1", "yes")


# Module-level config loaded at import time
_AUTH_ENABLED: bool = _is_auth_enabled()
_API_KEYS: set[str] = _load_api_keys()

# Public paths that never require auth
_PUBLIC_PATHS = {"/", "/health", "/docs", "/openapi.json", "/redoc"}


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Middleware that validates API keys on protected endpoints.

    Configuration via environment variables:
    - RTMEM_AUTH_ENABLED: Set to "true" to enable auth (default: false)
    - RTMEM_API_KEYS: Comma-separated list of valid API keys

    When disabled (default), all requests pass through without auth.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint):
        # Skip auth if not enabled
        if not _AUTH_ENABLED:
            return await call_next(request)

        # Skip auth for public paths
        path = request.url.path.rstrip("/")
        if path in _PUBLIC_PATHS:
            return await call_next(request)

        # Skip auth for non-API paths
        if not path.startswith("/v1/"):
            return await call_next(request)

        # Extract API key from headers
        api_key = self._extract_api_key(request)
        if api_key is None:
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing API key. Use Authorization: Bearer <key> or X-API-Key header."},
            )

        if api_key not in _API_KEYS:
            return JSONResponse(
                status_code=403,
                content={"detail": "Invalid API key."},
            )

        return await call_next(request)

    @staticmethod
    def _extract_api_key(request: Request) -> Optional[str]:
        """Extract API key from request headers.

        Supports:
        - Authorization: Bearer <key>
        - X-API-Key: <key>
        """
        # Check Authorization: Bearer <key>
        auth_header = request.headers.get("authorization", "")
        if auth_header.lower().startswith("bearer "):
            return auth_header[7:].strip()

        # Check X-API-Key header
        api_key = request.headers.get("x-api-key")
        if api_key:
            return api_key.strip()

        return None