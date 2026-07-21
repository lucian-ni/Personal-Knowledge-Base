# Implementation Plan

## Goal

Build a local full-web personal knowledge base that can ingest private documents, index
chunks into vector and keyword stores, and answer questions with citations.

## Completed Foundation

- Monorepo scaffold with `apps/api`, `apps/web`, `packages/ingestion`, `infra`, `docs`,
  and local `storage` directories.
- Docker Compose for PostgreSQL, Qdrant, and OpenSearch.
- Python project managed by `uv`.
- Frontend workspace managed by `pnpm`.
- FastAPI health, document, upload, and search endpoint contracts.
- PostgreSQL SQLAlchemy models and Alembic initial schema.
- PyMuPDF PDF->markdown conversion boundary.
- Markdown chunking with section and page metadata.
- Local bge-small-zh-v1.5 sentence embedding provider (real semantic embeddings, default).
- Qdrant and OpenSearch index payload contracts.
- Ingestion pipeline that builds chunk records, vector points, and keyword documents.
- Hybrid retrieval merge and citation fallback answer generation.
- Next.js Web MVP for upload, document list, and chat.

## Completed Wiring (end-to-end)

The placeholder API endpoints are now wired to real Postgres/Qdrant/OpenSearch and the
ingestion pipeline. A `POST /documents` upload converts the file to markdown (direct
read for `.md`/`.txt`, PyMuPDF for PDF), runs `IngestionPipeline.build_index_batch`,
and persists chunk rows to Postgres plus vectors/keyword docs to Qdrant/OpenSearch.
`GET /documents` reads from Postgres. `POST /search` runs the real `HybridRetriever`
with the live Qdrant/OpenSearch backends. CORS is configured for the web origin.

- Document endpoints backed by PostgreSQL persistence (`IngestionService`).
- Uploaded files stored under `storage/docs/{document_id}/{filename}`.
- Qdrant collection creation + vector upsert (`QdrantIndexer`).
- OpenSearch index creation + bulk indexing (`OpenSearchIndexer`).
- Upload wired to the PyMuPDF/markdown ingestion pipeline.
- `document_chunks` rows persisted from `IndexBatch.chunk_records`.
- `/search` backed by real `HybridRetriever` backends with graceful degradation
  (returns no hits when a store is unreachable, so the endpoint stays up).
- OpenAI-compatible embedding (`OpenAICompatibleEmbeddingProvider`) and LLM
  (`LLMAnswerGenerator`) clients, opt-in via `EMBEDDING_*`/`LLM_*` settings;
  the local `LocalEmbeddingProvider` (bge-small-zh) and `SimpleAnswerGenerator`
  remain the defaults.

## Recently Completed

- Hybrid retrieval uses RRF fusion (`score = sum 1/(k + rank)`); the previous
  weighted min-max merge was removed.
- OpenSearch BM25 fixed to query the top-level `text` field (was `document.text`,
  which matched nothing, leaving retrieval effectively vector-only).
- OpenSearch `text`/`title` use the built-in `cjk` analyzer for Chinese BM25
  (no plugin required).
- Cross-encoder rerank (Qwen3-Reranker-0.6B) wired into the hybrid pipeline.
- Streaming answers via `POST /search/stream` (SSE: citations, then deltas).
- Async ingestion: `POST /documents` returns 202 and runs the pipeline in a
  background task; the web client polls `GET /documents` for `ready`/`failed`.
  Failed ingestion cleans up any Qdrant/OpenSearch chunks already written.
- `DELETE /documents/{id}` cascades to Postgres + Qdrant + OpenSearch + filesystem.
- `/health` probes Postgres, Qdrant, and OpenSearch (returns `ok`/`degraded`).
- `.env.example` aligned to the real local-model defaults (was stale: `hash`
  provider, 1024 dims, empty model, missing reranker keys).

## Remaining Implementation Steps

1. Run Alembic migrations against the local PostgreSQL container
   (`cd apps/api && uv run alembic upgrade head`) once Docker is up.
2. Add end-to-end smoke tests using the local Docker stack (the BM25 fix is
   unit-tested via request-body capture, but not against a live OpenSearch).

## Verification Commands

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q
pnpm --filter @pkb/web typecheck
```
