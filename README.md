# Personal Knowledge Base

Local-first RAG personal knowledge base using FastAPI, Next.js, PostgreSQL, Qdrant,
OpenSearch, Docling, and Docker Compose.

## Architecture

- Raw PDF files live in `storage/docs`.
- PostgreSQL stores business data: documents, ingestion jobs, and chunk metadata.
- Qdrant stores embeddings plus retrievable chunk payloads.
- OpenSearch stores chunk text for BM25 keyword search.
- The FastAPI backend owns ingestion orchestration, retrieval, and chat endpoints.
- The Next.js frontend provides document upload, document status, and chat UI.

## Local Setup

```bash
cp .env.example .env
uv sync --extra dev
pnpm install
docker compose -f infra/docker-compose.yml up -d
```

## Run

```bash
uv run uvicorn pkb_api.main:app --app-dir apps/api/src --reload
pnpm dev
```

Open the web app at `http://localhost:3000`. The API is available at
`http://localhost:8000`.

## Verify

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q
pnpm --filter @pkb/web typecheck
```

The current MVP includes tested ingestion primitives, storage contracts, hybrid retrieval
merge logic, API endpoint contracts, and a first-pass web interface. Real Qdrant,
OpenSearch, and PostgreSQL persistence are scaffolded for local Docker and can be wired
behind the existing contracts.
