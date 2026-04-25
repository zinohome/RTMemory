"""RTMemory FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.db.session import close_engine
from app.middleware.auth import APIKeyMiddleware
from app.worker import Worker


# ── Worker instance (shared across the app) ────────────────────────────────

worker = Worker(max_concurrent=4)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown hooks."""
    # Startup — initialize worker
    worker.start()
    # Yield control to the application
    yield
    # Shutdown — stop worker and close database engine
    await worker.stop()
    await close_engine()


app = FastAPI(
    title="RTMemory",
    description="Temporal Knowledge Graph-Driven AI Memory System",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
_settings = get_settings()
_cors_origins = _settings.server.cors_origins
# Per CORS spec, allow_credentials=True is incompatible with allow_origins=["*"].
# When using wildcard origins, credentials must be False.
# When specific origins are listed, credentials can be True.
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_cors_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key authentication middleware (disabled by default, enable via RTMEM_AUTH_ENABLED=true)
app.add_middleware(APIKeyMiddleware)

# Import and register routers
from app.api.spaces import router as spaces_router  # noqa: E402
from app.api.entities import router as entities_router  # noqa: E402
from app.api.relations import router as relations_router  # noqa: E402
from app.api.memories import router as memories_router  # noqa: E402
from app.api.search import router as search_router  # noqa: E402
from app.api.conversations import router as conversations_router  # noqa: E402
from app.api.documents import router as documents_router  # noqa: E402
from app.api.profile import create_profile_router  # noqa: E402
from app.api.graph import router as graph_router  # noqa: E402
from app.api.tasks import router as tasks_router  # noqa: E402
from app.api.tasks import set_worker  # noqa: E402

# Wire worker to tasks API
set_worker(worker)

app.include_router(spaces_router, prefix="/v1/spaces", tags=["spaces"])
app.include_router(entities_router)
app.include_router(relations_router)
app.include_router(memories_router)
app.include_router(search_router)
app.include_router(conversations_router)
app.include_router(documents_router)
app.include_router(create_profile_router())
app.include_router(graph_router)
app.include_router(tasks_router)


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