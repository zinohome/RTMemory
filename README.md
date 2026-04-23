# RTMemory

**Temporal Knowledge Graph-Driven AI Memory System**

RTMemory is an open-source memory system for AI applications that provides intelligent memory extraction, temporal knowledge graph management, hybrid search (vector + graph + keyword), and auto-maintained user profiles — all self-hosted via Docker Compose.

## Architecture

```
┌─────────────┐    ┌──────────────┐    ┌───────────────┐
│  SDK (Python │    │  MCP Server  │    │  LangChain    │
│  / JS/TS)    │    │  (Claude/Cursor)│   │  Integration  │
└──────┬───────┘    └──────┬───────┘    └───────┬───────┘
       │                   │                    │
       └───────────────────┼────────────────────┘
                           │ HTTP
                    ┌──────▼───────┐
                    │  FastAPI      │
                    │  Server       │
                    └──────┬───────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
    ┌────▼─────┐    ┌──────▼──────┐   ┌─────▼──────┐
    │ Extraction│    │ Graph Engine │   │ Search      │
    │ Pipeline  │    │ (TKG Core)   │   │ Engine      │
    └────┬──────┘    └──────┬──────┘   └─────┬──────┘
         │                  │                 │
         └──────────────────┼─────────────────┘
                            │
                    ┌───────▼───────┐
                    │  PostgreSQL +  │
                    │  pgvector      │
                    └───────────────┘
```

## Key Features

- **Temporal Knowledge Graph**: Entities and relations with `valid_from`/`valid_to` time intervals, automatic contradiction handling, and version chains for memories
- **Three-Layer Memory Extraction**: FactDetector (regex) → Extractor (LLM structured) → DeepScanner (batch) with confidence scoring
- **Hybrid Search**: pgvector + recursive CTE graph traversal + tsvector keyword search, fused via Reciprocal Rank Fusion (RRF) with profile boost
- **Auto-Maintained Profiles**: Identity, preferences, status, and relationships computed from the knowledge graph on demand with in-memory caching
- **Confidence Decay**: `C(t) = C0 * e^(-lambda * delta_t) * (1 + alpha * log(n+1))` — memories naturally fade unless reaffirmed
- **Document Pipeline**: Ingest text, PDFs, or web pages with automatic chunking and embedding
- **Multi-LLM Support**: OpenAI, Anthropic, and local Ollama models via a unified adapter

## Quick Start

### Docker Compose

```bash
git clone https://github.com/your-org/rtmemory.git
cd rtmemory
docker compose up -d
```

### Manual Setup

```bash
# Start PostgreSQL with pgvector
docker run -d --name rtmemory-db \
  -e POSTGRES_DB=rtmemory \
  -e POSTGRES_USER=rtmemory \
  -e POSTGRES_PASSWORD=secret \
  -p 5432:5432 \
  pgvector/pgvector:pg16

# Install and run the server
cd server
pip install -e ".[dev]"
alembic upgrade head
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Configuration

Create `config.yaml` in the server directory:

```yaml
server:
  host: "0.0.0.0"
  port: 8000
  cors_origins: ["*"]

database:
  host: localhost
  port: 5432
  user: rtmemory
  password: secret
  database: rtmemory

llm:
  provider: ollama  # or openai, anthropic
  model: qwen2.5:7b
  base_url: http://localhost:11434

embedding:
  provider: local  # or openai
  model: BAAI/bge-base-zh-v1.5
  vector_dimension: 768
```

Environment variables with `RTMEM_` prefix override YAML values:

```bash
RTMEM_LLM_PROVIDER=openai RTMEM_LLM_API_KEY=sk-... uvicorn app.main:app
```

## SDK Usage

### Python

```python
import asyncio
from rtmemory import RTMemoryClient

async def main():
    async with RTMemoryClient(base_url="http://localhost:8000", api_key="sk-...") as client:
        # Add a memory
        result = await client.memories.add(
            content="用户喜欢深色主题",
            space_id="sp_001",
            entity_context="user_preferences",
        )
        print(f"Memory ID: {result.id}")

        # Search memories
        results = await client.search(
            q="用户主题偏好",
            space_id="sp_001",
            mode="hybrid",
            include_profile=True,
        )
        for r in results.results:
            print(f"[{r.source}] {r.content} (score: {r.score:.3f})")

        # Get user profile
        profile = await client.profile.get(
            entity_id="ent_001",
            space_id="sp_001",
        )
        print(profile.profile)

asyncio.run(main())
```

### JavaScript/TypeScript

```typescript
import { RTMemoryClient } from "@rtmemory/sdk";

const client = new RTMemoryClient({
  baseUrl: "http://localhost:8000",
  apiKey: "sk-...",
});

// Add a memory
const result = await client.addMemory({
  content: "用户喜欢深色主题",
  spaceId: "sp_001",
  entityContext: "user_preferences",
});

// Search
const results = await client.search({
  q: "用户主题偏好",
  spaceId: "sp_001",
  mode: "hybrid",
  includeProfile: true,
});
```

### LangChain Integration

```python
from rtmemory.langchain import RTMemoryTool, RTMemoryVectorStore

# As a tool
tool = RTMemoryTool(base_url="http://localhost:8000", space_id="sp_001")
result = tool.run("user preferences about themes")

# As a vector store
vs = RTMemoryVectorStore(base_url="http://localhost:8000", space_id="sp_001")
ids = vs.add_texts(["User prefers dark mode", "User speaks Chinese"])
docs = vs.similarity_search("主题偏好", k=5)
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/memories/` | Add a memory (triggers extraction) |
| GET | `/v1/memories/` | List memories |
| GET | `/v1/memories/:id` | Get memory by ID |
| PATCH | `/v1/memories/:id` | Update a memory |
| POST | `/v1/memories/forget` | Forget (soft-delete) a memory |
| POST | `/v1/search/` | Hybrid search |
| POST | `/v1/profile` | Compute user profile |
| POST | `/v1/documents/` | Add document (async processing) |
| GET | `/v1/documents/` | List documents |
| GET | `/v1/documents/:id` | Get document details |
| DELETE | `/v1/documents/:id` | Delete a document |
| POST | `/v1/conversations/` | Add conversation fragment |
| POST | `/v1/conversations/end` | End conversation (triggers deep scan) |
| POST | `/v1/spaces/` | Create a space |
| GET | `/v1/spaces/` | List spaces |
| GET | `/v1/graph/neighborhood` | Get graph neighborhood for visualization |
| GET | `/v1/tasks/` | List background tasks |
| GET | `/v1/tasks/:id` | Get task status |

## Project Structure

```
RTMemory/
├── server/                    # Python FastAPI server
│   ├── app/
│   │   ├── api/              # API routes
│   │   ├── core/             # Core engines (graph, search, profile, LLM)
│   │   ├── db/               # SQLAlchemy models & session
│   │   ├── extraction/       # Memory extraction pipeline
│   │   ├── integrations/     # Claude Code adapter
│   │   ├── mcp/              # MCP server
│   │   ├── schemas/          # Pydantic request/response schemas
│   │   ├── worker.py         # Async background task worker
│   │   └── main.py           # FastAPI entry point
│   ├── alembic/              # Database migrations
│   └── tests/                # Test suite
├── sdk-python/               # Python SDK
│   └── rtmemory/
│       ├── client.py         # RTMemoryClient
│       ├── langchain.py      # LangChain integration
│       ├── tools.py          # Generic LLM tool definitions
│       └── ...               # Domain namespaces
├── sdk-js/                   # JavaScript/TypeScript SDK
│   └── src/
│       ├── client.ts          # RTMemoryClient
│       ├── types.ts           # Zod-validated types
│       └── ...                # Domain modules
└── docs/superpowers/         # Design specs & implementation plans
```

## Testing

```bash
cd server
pytest tests/ -v
```

## License

MIT