# RTMemory

Temporal Knowledge Graph-Driven AI Memory System -- persistent memory, structured user profiles, and hybrid search for AI agents and chatbots.

## Key Features

- **Temporal Knowledge Graph** -- Entities, relations, and temporal edges make time a first-class citizen. Contradictions resolve naturally through time intervals instead of overwrites.
- **Hybrid Search** -- Three-channel retrieval (vector similarity via pgvector, graph traversal via recursive CTE, keyword via PostgreSQL tsvector) fused with Reciprocal Rank Fusion, with optional profile boosting.
- **Profile Engine** -- User profiles are computed on-demand from the knowledge graph, not stored. Always consistent with the underlying data.
- **Extraction Pipeline** -- Three-layer pipeline: fact detection filters messages, structured extraction produces entities/relations/memories, deep scanning captures implicit preferences across conversations.
- **Confidence Decay** -- Memories fade gradually via configurable decay curves instead of binary forgotten/not-forgotten flags.
- **Multi-LLM** -- Pluggable adapter layer for OpenAI, Anthropic, and Ollama. Switch providers via configuration.
- **Multi-tenant** -- Org and space isolation at the data layer.

## Quick Start

### Docker Compose (recommended)

```bash
git clone https://github.com/your-org/RTMemory.git
cd RTMemory
docker compose up -d
```

This starts:
- RTMemory API on `http://localhost:8000`
- PostgreSQL 17 + pgvector on port 5432
- Ollama on port 11434

### Manual setup

```bash
# Install server
pip install -e ./server

# Set environment variables
export RTMEM_DATABASE_HOST=localhost
export RTMEM_DATABASE_PORT=5432
export RTMEM_DATABASE_USER=rtmemory
export RTMEM_DATABASE_PASSWORD=secret
export RTMEM_DATABASE_DATABASE=rtmemory
export RTMEM_LLM_PROVIDER=ollama
export RTMEM_LLM_MODEL=qwen2.5:7b
export RTMEM_LLM_BASE_URL=http://localhost:11434
export RTMEM_EMBEDDING_PROVIDER=local
export RTMEM_EMBEDDING_MODEL=BAAI/bge-base-zh-v1.5

# Run database migrations and start server
cd server
alembic upgrade head
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/v1/memories/` | Add a new memory |
| GET | `/v1/memories/` | List memories (paginated, filterable) |
| GET | `/v1/memories/{id}` | Get a memory by ID |
| PATCH | `/v1/memories/{id}` | Update a memory (creates new version) |
| DELETE | `/v1/memories/{id}` | Soft-delete (forget) a memory |
| POST | `/v1/memories/traverse` | Traverse the knowledge graph |
| GET | `/v1/graph/neighborhood` | Graph neighborhood for visualization |
| POST | `/v1/search/` | Hybrid search (vector + graph + keyword) |
| POST | `/v1/profile` | Compute user profile from knowledge graph |
| POST | `/v1/conversations/` | Submit conversation messages |
| POST | `/v1/conversations/end` | End conversation (triggers deep scan) |
| POST | `/v1/documents/` | Add a document |
| GET | `/v1/documents/` | List documents |
| GET | `/v1/documents/{id}` | Get document details |
| DELETE | `/v1/documents/{id}` | Delete a document |
| POST | `/v1/spaces/` | Create a space |
| GET | `/v1/spaces/` | List spaces |
| GET | `/v1/spaces/{id}` | Get space details |
| DELETE | `/v1/spaces/{id}` | Delete a space |
| POST | `/v1/entities/` | Create an entity |
| GET | `/v1/entities/` | List entities |
| POST | `/v1/relations/` | Create a relation |
| GET | `/v1/relations/` | List relations |

## SDK Usage

### Python

```bash
pip install rtmemory
```

```python
import asyncio
from rtmemory import RTMemoryClient

async def main():
    client = RTMemoryClient(base_url="http://localhost:8000")

    # Add a memory
    result = await client.memories.add(
        content="User prefers dark mode for the IDE",
        space_id="sp_abc123",
    )
    print(f"Memory added: {result.id}")

    # Search
    results = await client.search(
        q="theme preference",
        space_id="sp_abc123",
    )
    for r in results.results:
        print(f"  [{r.score:.2f}] {r.content}")

    # Get profile
    profile = await client.profile.get(
        entity_id="ent_001",
        space_id="sp_abc123",
    )
    print(f"Identity: {profile.profile.identity}")

    # Forget
    await client.memories.forget(content_match="dark mode")

asyncio.run(main())
```

### JavaScript / TypeScript

```bash
npm install rtmemory-sdk
```

```typescript
import { RTMemoryClient } from "rtmemory-sdk";

const client = new RTMemoryClient({
  baseUrl: "http://localhost:8000",
  apiKey: "sk-...",  // optional
});

// Add a memory
const mem = await client.addMemory({
  content: "User prefers dark mode for the IDE",
  spaceId: "sp_abc123",
});
console.log("Memory added:", mem.id);

// Search
const results = await client.search({
  q: "theme preference",
  spaceId: "sp_abc123",
});
for (const r of results.results) {
  console.log(`  [${r.score.toFixed(2)}] ${r.content}`);
}

// Get profile
const profile = await client.getProfile({
  entityId: "ent_001",
  spaceId: "sp_abc123",
});

// Forget
await client.forgetMemory(mem.id, { reason: "outdated" });
```

### Domain-specific sub-clients (JS)

```typescript
import {
  MemoryAddClient,
  MemoryListClient,
  SearchClient,
  ProfileClient,
  DocumentClient,
  SpaceClient,
  ConversationClient,
  GraphClient,
} from "rtmemory-sdk";

const headers = { "Content-Type": "application/json" };
const base = "http://localhost:8000";

const adder = new MemoryAddClient(base, headers);
const lister = new MemoryListClient(base, headers);
const search = new SearchClient(base, headers);
const profile = new ProfileClient(base, headers);
const docs = new DocumentClient(base, headers);
const spaces = new SpaceClient(base, headers);
const convos = new ConversationClient(base, headers);
const graph = new GraphClient(base, headers);
```

## Architecture

```
+----------------------------------------------------------+
|                     RTMemory Server (FastAPI)             |
|                                                           |
|  +-------------+  +-------------+  +-----------------+   |
|  |  API Layer  |  | Auth &       |  |  SDK Layer     |   |
|  |  (REST)     |  | Tenant Mgr   |  |  (Python+JS)   |   |
|  +------+------|  +------+------+  +-------+--------+   |
|         |               |                |               |
|  +------+---------------+----------------+----------+    |
|  |             Core Service Layer                  |    |
|  |  +----------+ +-----------+ +----------------+ |    |
|  |  | Memory    | | Profile   | | Search Engine  | |    |
|  |  | Engine    | | Manager   | | (Vec+Graph+KW) | |    |
|  |  +----+------+ +-----+----+ +--------+-------+ |    |
|  |       |              |               |          |    |
|  |  +----+--------------+---------------+------+   |    |
|  |  |        Temporal Knowledge Graph           |   |    |
|  |  |   (Entities - Relations - Temporal Edges) |   |    |
|  |  +------------------+-----------------------+   |    |
|  +---------------------+---------------------------+   |
|                        |                                |
|  +---------------------+---------------------------+   |
|  |          Processing Pipeline                    |   |
|  |  +----------+ +----------+ +----------+        |   |
|  |  |Extraction| | Embedding| | Document |        |   |
|  |  |  Worker  | |  Worker  | |  Worker  |        |   |
|  |  +----------+ +----------+ +----------+        |   |
|  +---------------------+---------------------------+   |
|                        |                                |
|  +---------------------+---------------------------+   |
|  |          LLM Adapter Layer                      |   |
|  |   OpenAI - Anthropic - Ollama (configurable)   |   |
|  +------------------------------------------------+   |
+----------------------------------------------------------+
                          |
                   +------+-------+
                   |  PostgreSQL  |
                   |  + pgvector  |
                   +--------------+
```

## Configuration

RTMemory is configured via environment variables or a `config.yaml` file. Key settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `RTMEM_DATABASE_HOST` | `localhost` | PostgreSQL host |
| `RTMEM_DATABASE_PORT` | `5432` | PostgreSQL port |
| `RTMEM_DATABASE_USER` | `rtmemory` | Database user |
| `RTMEM_DATABASE_PASSWORD` | `secret` | Database password |
| `RTMEM_DATABASE_DATABASE` | `rtmemory` | Database name |
| `RTMEM_LLM_PROVIDER` | `ollama` | LLM provider: `openai`, `anthropic`, or `ollama` |
| `RTMEM_LLM_MODEL` | `qwen2.5:7b` | Model name |
| `RTMEM_LLM_BASE_URL` | `http://localhost:11434` | LLM API base URL |
| `RTMEM_EMBEDDING_PROVIDER` | `local` | Embedding provider: `local` or `openai` |
| `RTMEM_EMBEDDING_MODEL` | `BAAI/bge-base-zh-v1.5` | Embedding model name |
| `RTMEM_EMBEDDING_VECTOR_DIMENSION` | `768` | Vector dimension |

## License

MIT
