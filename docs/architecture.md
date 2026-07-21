# Architecture

## Data Ownership

This project follows a production-style RAG storage split:

- PostgreSQL stores business data.
- Qdrant stores vector-searchable chunk payloads.
- OpenSearch stores keyword-searchable chunk text.
- Local filesystem storage keeps the raw source files.

The same chunk text is intentionally stored in Qdrant and OpenSearch. Retrieval can return
ready-to-use context without an extra PostgreSQL lookup for every chunk hit.

## Document Lifecycle

```mermaid
flowchart TD
    Upload[Upload PDF] --> FileStore[storage/docs]
    Upload --> DocumentRow[PostgreSQL documents]
    FileStore --> PdfConv[PyMuPDF to Markdown]
    PdfConv --> Chunker[Markdown Chunker]
    Chunker --> Embedding[Embedding Provider]
    Chunker --> ChunkRows[PostgreSQL document_chunks]
    Embedding --> Qdrant[Qdrant Vector Payload]
    Chunker --> OpenSearch[OpenSearch BM25 Document]
    Query[User Query] --> Hybrid[Hybrid Retriever]
    Hybrid --> Qdrant
    Hybrid --> OpenSearch
    Hybrid --> Answer[Answer With Citations]
```

## PostgreSQL

Uploads return immediately (HTTP 202); the convert/chunk/embed/persist work runs in a FastAPI
background task, and the document flips to `ready` or `failed`. On failure any chunks already
written to Qdrant/OpenSearch are cleaned up so a failed document leaves no searchable orphans.
`DELETE /documents/{id}` removes a document from Postgres, Qdrant, OpenSearch, and the
filesystem (search-store removal is best-effort).

`documents` contains metadata about original files:

- title
- original filename
- MIME type
- local file path
- status
- owner/category placeholders
- version and timestamps

`document_chunks` maps business metadata to retrievable chunk IDs:

- document ID
- chunk ID
- chunk index
- title, section, page
- checksum and token count
- Qdrant point ID
- OpenSearch document ID

It does not store the full chunk text. Qdrant and OpenSearch own the retrievable copy.

`ingestion_jobs` tracks pipeline progress and failures.

## Qdrant Payload

Each Qdrant point uses a UUID (uuid5 of the stable chunk ID) as its point ID and stores:

- vector
- document ID
- chunk ID
- chunk text
- title
- section
- page
- checksum

## OpenSearch Document

Each OpenSearch document uses the same stable chunk ID and stores:

- chunk text
- title
- section
- page
- document ID
- checksum

The `text` and `title` fields use the built-in `cjk` analyzer so Chinese BM25 tokenizes by
bigram (the default `standard` analyzer handles CJK poorly). No plugin is required. The
chunk text is queried as the top-level `text` field (not `document.text`).

## Retrieval

The backend embeds the query for vector search (Qdrant) and sends the raw query to keyword
search (OpenSearch BM25). The hybrid retriever fuses the two ranked lists with Reciprocal
Rank Fusion (RRF, `score = sum 1/(k + rank)`, rank-only so it is robust to BM25-vs-cosine
score-scale differences), deduplicates by chunk ID, and optionally re-orders with a
cross-encoder reranker (Qwen3-Reranker-0.6B). Answers stream as Server-Sent Events
(citations first, then token deltas) via `POST /search/stream`.

When an LLM endpoint is not configured, the API returns the most relevant local chunks as a
fallback answer. This keeps local smoke tests deterministic.
