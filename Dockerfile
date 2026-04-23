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