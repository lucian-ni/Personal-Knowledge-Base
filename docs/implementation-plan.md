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
- Docling conversion boundary.
- Markdown chunking with section and page metadata.
- Deterministic hash embedding provider for local tests.
- Qdrant and OpenSearch index payload contracts.
- Ingestion pipeline that builds chunk records, vector points, and keyword documents.
- Hybrid retrieval merge and citation fallback answer generation.
- Next.js Web MVP for upload, document list, and chat.

## Next Implementation Steps

1. Replace placeholder document endpoints with PostgreSQL-backed document persistence.
2. Store uploaded files under `storage/docs` with stable document IDs.
3. Run Alembic migrations against the local PostgreSQL container.
4. Add Qdrant collection creation and upsert client.
5. Add OpenSearch index creation and bulk indexing client.
6. Wire upload jobs to the Docling ingestion pipeline.
7. Persist `document_chunks` rows from `IndexBatch.chunk_records`.
8. Replace placeholder `/search` with real `HybridRetriever` backends.
9. Add an OpenAI-compatible LLM client for answer synthesis.
10. Add end-to-end smoke tests using the local Docker stack.
11. Add optional reranking and streaming once the synchronous MVP is stable.

## Verification Commands

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q
pnpm --filter @pkb/web typecheck
```
