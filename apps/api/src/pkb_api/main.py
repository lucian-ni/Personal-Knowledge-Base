from __future__ import annotations

import json
from collections.abc import Iterator

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy import text
from sqlalchemy.orm import Session

from pkb_api.db import get_session
from pkb_api.ingestion_service import (
    IngestionService,
    UnsupportedDocumentError,
)
from pkb_api.ingestion_service import (
    list_documents as list_documents_service,
)
from pkb_api.retrieval import HybridRetriever, build_citations
from pkb_api.schemas import (
    DocumentRead,
    HealthResponse,
    IngestionJobRead,
    SearchRequest,
    SearchResult,
)
from pkb_api.services import Services, get_services
from pkb_api.settings import settings

app = FastAPI(title="Personal Knowledge Base API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health(
    session: Session = Depends(get_session),
    services: Services = Depends(get_services),
) -> HealthResponse:
    stores = {
        "postgres": _check_postgres(session),
        "qdrant": services.qdrant_indexer.health(),
        "opensearch": services.opensearch_indexer.health(),
    }
    return HealthResponse(status="ok" if all(stores.values()) else "degraded", stores=stores)


def _check_postgres(session: Session) -> bool:
    try:
        session.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


@app.get("/documents", response_model=list[DocumentRead])
def list_documents(session: Session = Depends(get_session)) -> list[DocumentRead]:
    return list_documents_service(session)


@app.post("/documents", response_model=IngestionJobRead, status_code=202)
def upload_document(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session),
    services: Services = Depends(get_services),
) -> IngestionJobRead:
    """Accept an upload and return the ingestion job immediately (HTTP 202).

    Only the fast setup runs synchronously (validate, create rows, save file); the
    heavy convert/embed/persist runs as a background task. Clients poll
    ``GET /documents`` to see the document flip to ``ready``/``failed``.
    """
    data = file.file.read()
    service = IngestionService(services)
    try:
        prep = service.prepare_ingest(
            session, file.filename or "uploaded-document", file.content_type, data
        )
    except UnsupportedDocumentError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    background_tasks.add_task(service.run_ingestion, prep)
    return prep.job


@app.delete("/documents/{document_id}", status_code=204)
def delete_document(
    document_id: str,
    session: Session = Depends(get_session),
    services: Services = Depends(get_services),
) -> None:
    """Delete a document from Postgres + Qdrant + OpenSearch + the filesystem.

    Search-store and file removal are best-effort; only a missing document yields 404.
    """
    service = IngestionService(services)
    if not service.delete_document(session, document_id):
        raise HTTPException(status_code=404, detail="Document not found")


def _build_retriever(services: Services) -> HybridRetriever:
    return HybridRetriever(
        embedding_provider=services.pipeline.embedding_provider,
        vector_search=services.vector_backend,
        keyword_search=services.keyword_backend,
        reranker=services.reranker,
        rrf_k=settings.rrf_k,
        rerank_top_k=settings.rerank_top_k,
    )


@app.post("/search", response_model=SearchResult)
def search(
    request: SearchRequest,
    services: Services = Depends(get_services),
) -> SearchResult:
    hits = _build_retriever(services).search(request.query, limit=request.limit)
    return services.answer_generator.answer(request.query, hits)


@app.post("/search/stream")
def search_stream(
    request: SearchRequest,
    services: Services = Depends(get_services),
) -> StreamingResponse:
    """Stream a cited answer as Server-Sent Events.

    Emits ``citations`` (all at once), then ``delta`` events carrying answer text
    chunks, then ``done``. When no LLM is configured the SimpleAnswerGenerator
    yields the matched chunks as a single delta.
    """
    hits = _build_retriever(services).search(request.query, limit=request.limit)
    citations = build_citations(hits)
    generator = services.answer_generator
    query = request.query

    def event_stream() -> Iterator[str]:
        yield f"event: citations\ndata: {json.dumps([c.model_dump() for c in citations])}\n\n"
        for delta in generator.stream_answer(query, hits):
            yield f"event: delta\ndata: {json.dumps({'content': delta})}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
