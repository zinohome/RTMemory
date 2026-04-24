# ── Stage 1: Builder ────────────────────────────────────────────────────
# Install Python dependencies in a separate stage so the final image
# doesn't carry build tools (gcc, etc.).
FROM python:3.12-slim AS builder

WORKDIR /build

# System build deps (needed for asyncpg C extension)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (much smaller than CUDA build — ~200MB vs 2GB+).
# This must come before sentence-transformers so pip doesn't pull the CUDA wheel.
RUN pip install --no-cache-dir --prefix=/install \
    torch --index-url https://download.pytorch.org/whl/cpu

# Copy server project definition and install remaining deps.
# Use --no-deps for the server itself, then install only the deps we need
# to avoid pulling in nvidia-* CUDA packages through transitive dependencies.
COPY server/pyproject.toml ./
RUN pip install --no-cache-dir --prefix=/install --no-deps . && \
    pip install --no-cache-dir --prefix=/install \
    fastapi>=0.115 uvicorn[standard]>=0.30 sqlalchemy[asyncio]>=2.0 asyncpg>=0.30 \
    pgvector>=0.3 alembic>=1.13 pydantic>=2.0 pydantic-settings>=2.0 httpx>=0.27 \
    pymupdf>=1.24 trafilatura>=1.8 sentence-transformers>=3.0 pyyaml>=6.0 \
    openai>=1.0 anthropic>=0.30

# Strip unnecessary CUDA packages — keep only what torch needs at runtime:
# nvidia-cublas (libcublasLt) and nvidia-cuda-runtime (libcudart).
# Everything else (cudnn, cufft, curand, cusolver, cusparse, nccl, nvshmem,
# cusparselt, triton, cuda-toolkit, cuda-bindings) saves ~1.5GB when removed.
RUN cd /install/lib/python3.12/site-packages && \
    rm -rf \
    nvidia_cudnn* \
    nvidia_cufft* \
    nvidia_cufile* \
    nvidia_curand* \
    nvidia_cusolver* \
    nvidia_cusparse* \
    nvidia_cusparselt* \
    nvidia_nccl* \
    nvidia_nvshmem* \
    nvidia_nvjitlink* \
    nvidia_nvtx* \
    cuda_toolkit* \
    cuda_bindings* \
    cuda_pathfinder* \
    triton* \
    nvidia_cuda_cupti* \
    nvidia_cuda_nvrtc* \
    2>/dev/null; true

# ── Stage 2: Runtime ──────────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

# Runtime system deps (libpq for asyncpg, no gcc needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder (with CUDA trimmed)
COPY --from=builder /install /usr/local

# Copy application source
COPY server/app/ ./app/
COPY server/alembic/ ./alembic/
COPY server/alembic.ini ./alembic.ini
COPY config.yaml ./config.yaml

# Non-root user for security
RUN useradd --create-home rtmemory && chown -R rtmemory:rtmemory /app
USER rtmemory

EXPOSE 8000

# Run migrations on startup, then serve
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]