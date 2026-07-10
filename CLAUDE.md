# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Install (Python via `uv`, JS via `pnpm`):

```bash
uv sync --extra dev
pnpm install
docker compose -f infra/docker-compose.yml up -d   # PostgreSQL, Qdrant, OpenSearch
```

Run the two apps (in separate terminals):

```bash
uv run uvicorn pkb_api.main:app --app-dir apps/api/src --reload   # API on :8000
pnpm dev                                                          # web on :3000
```

Verify (these are the canonical checks; run them before considering work done):

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q          # Python tests
pnpm --filter @pkb/web typecheck                           # TS typecheck
uv run ruff check .                                        # lint
```

Run a single test file / single test:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/ingestion/test_chunker.py -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/ingestion/test_chunker.py::test_chunker_uses_overlap_between_long_chunks -q
```

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` is required — keep it on every pytest invocation (see Conventions).

Alembic migrations. `alembic.ini` uses a relative `script_location = migrations`, so it **must be run from `apps/api`**:

```bash
cd apps/api
uv run alembic upgrade head
uv run alembic revision -m "describe change"     # autogenerate
```

## Architecture

Local-first RAG knowledge base. Three independent stores are intentionally **not** merged:

- **PostgreSQL** — business data only: `documents` (file metadata), `document_chunks` (maps business data to search-index IDs), `ingestion_jobs` (pipeline status). `document_chunks` deliberately stores **no chunk text**.
- **Qdrant** — embeddings + retrievable chunk payload (vector search).
- **OpenSearch** — chunk text for BM25 keyword search.
- **Filesystem** (`storage/docs`, `storage/derived`) — raw source files.

Chunk text is duplicated into both Qdrant and OpenSearch so retrieval returns ready-to-use context without a Postgres lookup per hit. This split is the central design decision — see `docs/architecture.md`.

### The stable chunk ID is the cross-store join key

`stable_chunk_id(document_id, chunk_index)` = `"{document_id}:{chunk_index:06d}"` (`packages/ingestion/src/pkb_ingestion/ids.py`). The **same** value is used as the Qdrant point ID, the OpenSearch `_id`, and `document_chunks.chunk_id` / `qdrant_point_id` / `opensearch_document_id` in Postgres. Dedup in hybrid retrieval keys on it. Preserve this identity when adding/changing stores.

### Code layout

- `apps/api/src/pkb_api/` — FastAPI app. `main.py` (routes + CORS), `schemas.py` (Pydantic), `models.py` (SQLAlchemy), `retrieval.py` (hybrid merge + `SimpleAnswerGenerator`/`LLMAnswerGenerator` + factories), `db.py` (engine/session), `settings.py` (pydantic-settings from `.env`), `storage.py` (uploads to `storage/docs`), `backends.py` (real Qdrant/OpenSearch indexers + search backends with graceful degradation), `services.py` (`Services` container + `get_services` lazy singleton + `make_embedding_provider`), `ingestion_service.py` (convert -> pipeline -> persist to all three stores + `list_documents`).
- `apps/api/migrations/` — Alembic. `env.py` overrides `sqlalchemy.url` from `Settings` at runtime, so the value in `alembic.ini` is a placeholder.
- `packages/ingestion/src/pkb_ingestion/` — the ingestion library, imported by the API. `pipeline.py` (orchestrates chunk → embed → build index batch), `chunker.py` (markdown chunker), `embeddings.py` (`EmbeddingProvider` Protocol + `HashEmbeddingProvider` + `OpenAICompatibleEmbeddingProvider`), `index_contracts.py` (Qdrant/OpenSearch payload builders), `docling_converter.py` (PDF→markdown boundary).
- `apps/web/` — Next.js 15 App Router. Server component `app/page.tsx` fetches documents; `lib/api.ts` is the typed API client; `@/` alias points at the web app root.

### Ingestion pipeline

`IngestionPipeline.build_index_batch(artifact)` produces an `IndexBatch` of three parallel lists — `chunk_records` (Postgres), `qdrant_points`, `opensearch_documents` — all keyed by the same chunk ID. The pipeline is pure data transformation; it does **not** write to any store. Persisting to Postgres/Qdrant/OpenSearch is done by `IngestionService` (`ingestion_service.py`), which the `POST /documents` endpoint calls.

`MarkdownChunker` splits on `## `+ headings (H1 is treated as the doc title and dropped), honors `<!-- page: N -->` markers for page metadata, and does char-based splitting with overlap. `HashEmbeddingProvider` is a deterministic local embedding (sha256-indexed bag of tokens, L2-normalized) used so tests don't need a network model; real embedding via an OpenAI-compatible API is implemented in `OpenAICompatibleEmbeddingProvider` and selected by `make_embedding_provider(settings)` when `EMBEDDING_PROVIDER=openai` and the `EMBEDDING_API_*` keys are set; otherwise the hash provider is used.

### Hybrid retrieval

`HybridRetriever` (`apps/api/src/pkb_api/retrieval.py`) embeds the query for Qdrant and sends the raw query to OpenSearch, then `merge_hybrid_hits` min-max-normalizes each backend's scores, applies weights (vector 0.6 / keyword 0.4), sums per chunk ID, dedups, and ranks. Backends are `Protocol`s (`VectorSearchBackend`, `KeywordSearchBackend`) — the merge logic is tested against fakes; real Qdrant/OpenSearch backends live in `backends.py` (`QdrantVectorBackend`, `OpenSearchKeywordBackend`) and degrade to empty results when a store is unreachable.

When no LLM is configured (`LLM_API_KEY` unset), `SimpleAnswerGenerator` returns the matched chunks themselves as the answer with citations. This keeps local smoke tests deterministic.

## Current state

The API endpoints are now wired end-to-end. `POST /documents` runs `IngestionService` (convert → chunk → embed → persist to Postgres + Qdrant + OpenSearch, synchronously within the request) and returns the completed job. `GET /documents` reads from Postgres. `POST /search` runs the real `HybridRetriever` over the live Qdrant/OpenSearch backends; when no LLM is configured (`LLM_API_KEY` unset) `SimpleAnswerGenerator` returns the matched chunks themselves as the answer with citations, keeping local smoke tests deterministic. An opt-in `LLMAnswerGenerator` and `OpenAICompatibleEmbeddingProvider` are used when the corresponding `*_API_KEY` settings are set. Ingestion is synchronous (no background jobs / polling yet); remaining work is in `docs/implementation-plan.md`.

## Conventions

- **Always prefix pytest with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.** `pytest-asyncio` is a dev dependency but all tests are synchronous; disabling autoload keeps the run clean and deterministic.
- There is no editable install (`[tool.uv] package = false`). `import pkb_api` / `import pkb_ingestion` resolve only because `pyproject.toml`'s `[tool.pytest.ini_options] pythonpath` lists `apps/api/src` and `packages/ingestion/src`. Keep those paths current if source moves.
- Ruff: line-length 100, target py311, rules `E, F, I, UP, B` (`pyproject.toml`). VS Code is set to 4-space indent and format-on-save.
- Config is environment-driven via `.env` (copy `.env.example`); never hardcode the DB/LLM/embedding endpoints — read from `Settings`.
