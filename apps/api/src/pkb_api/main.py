from __future__ import annotations

from fastapi import FastAPI, UploadFile

from pkb_api.retrieval import SimpleAnswerGenerator
from pkb_api.schemas import (
    DocumentRead,
    HealthResponse,
    IngestionJobRead,
    SearchRequest,
    SearchResult,
)

app = FastAPI(title="Personal Knowledge Base API", version="0.1.0")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@app.get("/documents", response_model=list[DocumentRead])
def list_documents() -> list[DocumentRead]:
    return []


@app.post("/documents", response_model=IngestionJobRead, status_code=202)
async def upload_document(file: UploadFile) -> IngestionJobRead:
    return IngestionJobRead(
        id="local-placeholder-job",
        document_id=file.filename or "uploaded-document",
        status="queued",
    )


@app.post("/search", response_model=SearchResult)
def search(request: SearchRequest) -> SearchResult:
    return SimpleAnswerGenerator().answer(request.query, [])
