from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"


class DocumentRead(BaseModel):
    id: str
    title: str
    original_filename: str
    mime_type: str
    status: str
    version: int
    created_at: datetime | None = None
    updated_at: datetime | None = None


class IngestionJobRead(BaseModel):
    id: str
    document_id: str
    status: str
    failure_reason: str | None = None
    retry_count: int = 0


class ChunkCitation(BaseModel):
    document_id: str
    chunk_id: str
    title: str
    section: str | None = None
    page: int | None = None
    text: str
    score: float = Field(ge=0.0)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=8, ge=1, le=30)


class SearchResult(BaseModel):
    answer: str
    citations: list[ChunkCitation]
