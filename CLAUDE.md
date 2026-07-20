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
uv run python scripts/dev_api.py   # API on :8000 (cross-platform, no PYTHONPATH needed)
pnpm dev                            # web on :3000
```

`main.py` imports `pkb_ingestion` from `packages/ingestion/src`, which is not on the
path that uvicorn's `--app-dir apps/api/src` sets. `scripts/dev_api.py` inserts both
source roots onto `sys.path` and exports `PYTHONPATH` for the reload child process, so
the same command works on macOS, Linux, and Windows. Without it you would need the
shell-specific `PYTHONPATH=packages/ingestion/src uv run uvicorn pkb_api.main:app --app-dir apps/api/src --reload`.

Local models (bge-small-zh-v1.5 embedding, Qwen3-Reranker-0.6B cross-encoder) download
on first use to `~/.cache/huggingface`. In China, download from the HF origin **through
the proxy** (the hf-mirror.com mirror fails on LFS redirects; `HTTPS_PROXY` works):

```bash
HTTPS_PROXY=http://127.0.0.1:7897 uv run python -c "from huggingface_hub import snapshot_download; \
snapshot_download('BAAI/bge-small-zh-v1.5'); snapshot_download('Qwen/Qwen3-Reranker-0.6B')"
```

At runtime set `HF_HUB_OFFLINE=1` so the app uses the cache without phoning home:

```bash
HF_HUB_OFFLINE=1 uv run python scripts/dev_api.py
```

Verify (these are the canonical checks; run them before considering work done):

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q          # Python tests (integration tests skip by default)
PKB_RUN_INTEGRATION=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/integration -q   # load real models
pnpm --filter @pkb/web typecheck                           # TS typecheck
uv run ruff check .                                        # lint
```

Run a single test file / single test:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/ingestion/test_chunker.py -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/ingestion/test_chunker.py::test_chunker_uses_overlap_between_long_chunks -q
```

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` is required - keep it on every pytest invocation (see Conventions).

Alembic migrations. `alembic.ini` uses a relative `script_location = migrations`, so it **must be run from `apps/api`**. `migrations/env.py` inserts both source roots onto `sys.path`, so no `PYTHONPATH` prefix is needed (cross-platform):

```bash
cd apps/api
uv run alembic upgrade head
uv run alembic revision -m "describe change"     # autogenerate
```

## Architecture

Local-first RAG knowledge base. Three independent stores are intentionally **not** merged:

- **PostgreSQL** - business data only: `documents` (file metadata), `document_chunks` (maps business data to search-index IDs), `ingestion_jobs` (pipeline status). `document_chunks` deliberately stores **no chunk text**.
- **Qdrant** - embeddings + retrievable chunk payload (vector search).
- **OpenSearch** - chunk text for BM25 keyword search.
- **Filesystem** (`storage/docs`) - raw source files.

Chunk text is duplicated into both Qdrant and OpenSearch so retrieval returns ready-to-use context without a Postgres lookup per hit. This split is the central design decision - see `docs/architecture.md`.

### The stable chunk ID is the cross-store join key

`stable_chunk_id(document_id, chunk_index)` = `"{document_id}:{chunk_index:06d}"` (`packages/ingestion/src/pkb_ingestion/ids.py`). This string is the cross-store join key: it is the OpenSearch `_id`, the Qdrant payload `chunkId`, and `document_chunks.chunk_id` / `opensearch_document_id` in Postgres. Dedup in hybrid retrieval keys on it. **The Qdrant point ID is NOT this string** - Qdrant requires a UUID or unsigned int, so `qdrant_point_id(chunk_id)` derives a deterministic uuid5; that UUID is the Qdrant point ID and `document_chunks.qdrant_point_id` in Postgres. Preserve these identities when adding/changing stores.

### Code layout

- `apps/api/src/pkb_api/` - FastAPI app. `main.py` (routes + CORS), `schemas.py` (Pydantic), `models.py` (SQLAlchemy), `retrieval.py` (`rrf_fuse` RRF fusion + `HybridRetriever` + `Reranker` Protocol + `SimpleAnswerGenerator`/`LLMAnswerGenerator` + factories), `reranker.py` (`QwenReranker` cross-encoder rerank via Qwen3-Reranker-0.6B), `db.py` (engine/session), `settings.py` (pydantic-settings from `.env`), `storage.py` (uploads to `storage/docs`), `backends.py` (real Qdrant/OpenSearch indexers + search backends with graceful degradation), `services.py` (`Services` container + `get_services` lazy singleton + `make_embedding_provider` + `make_reranker`), `ingestion_service.py` (convert -> pipeline -> persist to all three stores + `list_documents`).
- `apps/api/migrations/` - Alembic. `env.py` overrides `sqlalchemy.url` from `Settings` at runtime (so the value in `alembic.ini` is a placeholder) and inserts the source roots onto `sys.path` so no `PYTHONPATH` is needed.
- `packages/ingestion/src/pkb_ingestion/` - the ingestion library, imported by the API. `pipeline.py` (orchestrates chunk -> embed -> build index batch), `chunker.py` (markdown chunker), `embeddings.py` (`EmbeddingProvider` Protocol + `LocalEmbeddingProvider` (sentence-transformers/bge-small-zh-v1.5) + `OpenAICompatibleEmbeddingProvider`), `index_contracts.py` (Qdrant/OpenSearch payload builders), `ids.py` (`stable_chunk_id` + `qdrant_point_id`), `docling_converter.py` (PDF->markdown boundary).
- `apps/web/` - Next.js 15 App Router. Server component `app/page.tsx` fetches documents; `lib/api.ts` is the typed API client; `@/` alias points at the web app root.
- `scripts/dev_api.py` - cross-platform API launcher (sets `sys.path`/`PYTHONPATH`, then `uvicorn.run` with reload).

### Ingestion pipeline

`IngestionPipeline.build_index_batch(artifact)` produces an `IndexBatch` of three parallel lists - `chunk_records` (Postgres), `qdrant_points`, `opensearch_documents` - all keyed by the same stable chunk ID. The pipeline is pure data transformation; it does **not** write to any store. Persisting to Postgres/Qdrant/OpenSearch is done by `IngestionService` (`ingestion_service.py`), which the `POST /documents` endpoint calls. `IngestionService` accepts `.md`/`.markdown`/`.txt` directly (no converter) and `.pdf` via Docling; it is idempotent on re-ingest because all three stores key on the stable chunk ID (Qdrant by the derived UUID point ID).

`MarkdownChunker` splits on `## `+ headings (H1 is treated as the doc title and dropped), honors `<!-- page: N -->` markers for page metadata, and does char-based splitting with overlap. `LocalEmbeddingProvider` is the default - a real semantic embedding via sentence-transformers (bge-small-zh-v1.5, 512-dim, L2-normalized for Qdrant COSINE), loaded lazily on first use. An OpenAI-compatible API embedding is available via `OpenAICompatibleEmbeddingProvider`, selected by `make_embedding_provider(settings)` when `EMBEDDING_PROVIDER=openai` and the `EMBEDDING_API_*` keys are set; otherwise the local model is used. `EMBEDDING_DIMENSIONS` must match the model (512 for bge-small-zh).

### Hybrid retrieval + rerank

`HybridRetriever` (`apps/api/src/pkb_api/retrieval.py`) embeds the query (local bge model) for Qdrant and sends the raw query to OpenSearch BM25, then **RRF** (`rrf_fuse`, `score = Σ 1/(k + rank)`, default k=60) fuses the two ranked lists. Rank-only fusion is robust to BM25-vs-cosine score-scale differences (the previous weighted min-max merge was removed). When a `Reranker` is configured, each backend fetches `rerank_top_k` hits, RRF fuses them, then the cross-encoder re-scores and re-orders, returning the top `limit`. Backends are `Protocol`s (`VectorSearchBackend`, `KeywordSearchBackend`, `Reranker`); real Qdrant/OpenSearch backends live in `backends.py` and degrade to empty results when a store is unreachable.

`QwenReranker` (`reranker.py`) scores each (query, doc) pair via Qwen3-Reranker-0.6B's yes/no token probabilities (P(yes)), loads lazily, and uses MPS on Apple Silicon / CPU elsewhere. Disable via `RERANKER_ENABLED=false` to skip the model load for fast smoke tests.

When no LLM is configured (`LLM_API_KEY` unset), `SimpleAnswerGenerator` returns the matched chunks themselves as the answer with citations.

## Current state

The API endpoints are wired end-to-end. `POST /documents` runs `IngestionService` (convert -> chunk -> embed (bge) -> persist to Postgres + Qdrant + OpenSearch) and returns the job immediately (HTTP 202; the pipeline runs in a background task, status flips to ready/failed). `GET /documents` reads from Postgres. `POST /search` runs the real `HybridRetriever`: Qdrant vector + OpenSearch BM25 -> RRF fusion -> `QwenReranker` rerank -> answer. When no LLM is configured `SimpleAnswerGenerator` returns the matched chunks with citations; `LLMAnswerGenerator` is used when `LLM_API_*` are set. Embedding defaults to the local bge model; `OpenAICompatibleEmbeddingProvider` is opt-in. Ingestion is asynchronous: `POST /documents` returns 202 and runs the pipeline in a background task; the web client polls `GET /documents` for `ready`/`failed`. `DELETE /documents/{id}` cascades to all three stores; `POST /search/stream` streams cited answers as SSE. Remaining work is in `docs/implementation-plan.md`.

## Conventions

- **Always prefix pytest with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.** `pytest-asyncio` is a dev dependency but all tests are synchronous; disabling autoload keeps the run clean and deterministic. Integration tests (real models) are skipped unless `PKB_RUN_INTEGRATION=1`.
- There is no editable install (`[tool.uv] package = false`). `import pkb_api` / `import pkb_ingestion` resolve in **tests** via `pyproject.toml`'s `[tool.pytest.ini_options] pythonpath`. At runtime, `scripts/dev_api.py` sets the paths for the API server and `migrations/env.py` sets them for Alembic - both cross-platform, no `PYTHONPATH` shell prefix needed. Keep those paths current if source moves.
- `qdrant-client` is pinned to `>=1.11,<1.12` to match the Qdrant server in `infra/docker-compose.yml` (v1.11.3); newer clients removed `client.search()` which `QdrantVectorBackend` uses. Keep the client and server minor versions in sync if either is upgraded.
- ML deps: `sentence-transformers`, `transformers`, `torch` (torch is heavy ~2GB but cross-platform; MPS on Apple Silicon). Models cache in `~/.cache/huggingface` - use `HF_ENDPOINT=https://hf-mirror.com` (direct, unset proxy) in China.
- Ruff: line-length 100, target py311, rules `E, F, I, UP, B` (`pyproject.toml`). VS Code is set to 4-space indent and format-on-save.
- Config is environment-driven via `.env` (copy `.env.example`); never hardcode the DB/LLM/embedding endpoints - read from `Settings`.
- **Cross-platform (macOS + Windows).** Use `pathlib` (not string concatenation with `/`), read all endpoints/config from `.env` via `Settings`, and prefer `scripts/dev_api.py` over `PYTHONPATH=...` shell prefixes in documented commands. Avoid platform-specific paths or shell syntax in code.
