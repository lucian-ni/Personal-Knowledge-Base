from __future__ import annotations

import hashlib
import uuid


def stable_chunk_id(document_id: str, chunk_index: int) -> str:
    return f"{document_id}:{chunk_index:06d}"


def text_checksum(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def qdrant_point_id(chunk_id: str) -> str:
    """Deterministic UUID for a Qdrant point derived from the stable chunk ID.

    Qdrant point IDs must be an unsigned integer or a UUID; the stable chunk ID
    (``{document_id}:{chunk_index}``) is neither, so we derive a stable UUID via
    uuid5. The stable chunk ID remains the cross-store join key (Qdrant payload
    ``chunkId``, OpenSearch ``_id``, Postgres ``chunk_id``); this UUID is only the
    Qdrant point address, also stored in Postgres ``qdrant_point_id``.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))
