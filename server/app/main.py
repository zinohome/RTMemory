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
from app.api.entities import router as entities_router  # noqa: E402
from app.api.relations import router as relations_router  # noqa: E402
from app.api.memories import router as memories_router  # noqa: E402

app.include_router(spaces_router, prefix="/v1/spaces", tags=["spaces"])
app.include_router(entities_router)
app.include_router(relations_router)
app.include_router(memories_router)


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