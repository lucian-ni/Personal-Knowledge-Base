from __future__ import annotations

import hashlib


def stable_chunk_id(document_id: str, chunk_index: int) -> str:
    return f"{document_id}:{chunk_index:06d}"


def text_checksum(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
