from __future__ import annotations

from typing import Any

from pkb_ingestion.models import Chunk


def build_qdrant_payload(chunk: Chunk) -> dict[str, Any]:
    return {
        "docId": chunk.document_id,
        "chunkId": chunk.chunk_id,
        "chunkIndex": chunk.chunk_index,
        "text": chunk.text,
        "title": chunk.title,
        "section": chunk.section,
        "page": chunk.page,
        "checksum": chunk.checksum,
    }


def build_opensearch_document(chunk: Chunk) -> dict[str, Any]:
    return {
        "_id": chunk.chunk_id,
        "document": {
            "docId": chunk.document_id,
            "chunkId": chunk.chunk_id,
            "chunkIndex": chunk.chunk_index,
            "text": chunk.text,
            "title": chunk.title,
            "section": chunk.section,
            "page": chunk.page,
            "checksum": chunk.checksum,
        },
    }
