from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DocumentArtifact:
    document_id: str
    title: str
    source_path: Path
    markdown: str


@dataclass(frozen=True)
class Chunk:
    document_id: str
    chunk_id: str
    chunk_index: int
    text: str
    title: str
    section: str | None
    page: int | None
    token_count: int
    checksum: str


@dataclass(frozen=True)
class ChunkRecord:
    document_id: str
    chunk_id: str
    chunk_index: int
    title: str
    section: str | None
    page: int | None
    token_count: int
    checksum: str
    qdrant_point_id: str
    opensearch_document_id: str


@dataclass(frozen=True)
class QdrantPoint:
    id: str
    vector: list[float]
    payload: dict[str, Any]


@dataclass(frozen=True)
class IndexBatch:
    document_id: str
    chunk_records: list[ChunkRecord]
    qdrant_points: list[QdrantPoint]
    opensearch_documents: list[dict[str, Any]]
