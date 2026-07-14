from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from pkb_api.db import get_session
from pkb_api.ingestion_service import (
    IngestionService,
    UnsupportedDocumentError,
)
from pkb_api.ingestion_service import (
    list_documents as list_documents_service,
)
from pkb_api.retrieval import HybridRetriever
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
def health() -> HealthResponse:
    return HealthResponse()


@app.get("/documents", response_model=list[DocumentRead])
def list_documents(session: Session = Depends(get_session)) -> list[DocumentRead]:
    return list_documents_service(session)


@app.post("/documents", response_model=IngestionJobRead)
def upload_document(
    file: UploadFile,
    session: Session = Depends(get_session),
    services: Services = Depends(get_services),
) -> IngestionJobRead:
    data = file.file.read()
    service = IngestionService(services)
    try:
        return service.ingest(
            session,
            file.filename or "uploaded-document",
            file.content_type,
            data,
        )
    except UnsupportedDocumentError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc


@app.post("/search", response_model=SearchResult)
def search(
    request: SearchRequest,
    services: Services = Depends(get_services),
) -> SearchResult:
    retriever = HybridRetriever(
        embedding_provider=services.pipeline.embedding_provider,
        vector_search=services.vector_backend,
        keyword_search=services.keyword_backend,
        reranker=services.reranker,
        rrf_k=settings.rrf_k,
        rerank_top_k=settings.rerank_top_k,
    )
    hits = retriever.search(request.query, limit=request.limit)
    return services.answer_generator.answer(request.query, hits)
