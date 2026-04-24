# ── Stage 1: Builder ────────────────────────────────────────────────────
# Install Python dependencies in a separate stage so the final image
# doesn't carry build tools (gcc, etc.).
FROM python:3.12-slim AS builder

WORKDIR /build

# System build deps (needed for asyncpg, sentence-transformers C extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy server project definition and install
COPY server/pyproject.toml ./
RUN pip install --no-cache-dir --prefix=/install .

# ── Stage 2: Runtime ──────────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

# Runtime system deps (libpq for asyncpg, no gcc needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
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