from __future__ import annotations

import shutil
from pathlib import Path


class DocumentStorage:
    """Persists uploaded source files under the configured docs directory.

    Files are stored as ``storage/docs/{document_id}/{filename}`` so a document's
    raw source can be re-derived (e.g. re-chunking after a chunker change) without
    a re-upload. The path is the canonical ``Document.file_path`` value.
    """

    def __init__(self, docs_path: Path) -> None:
        self.docs_path = docs_path

    def save(self, document_id: str, filename: str, data: bytes) -> Path:
        directory = self.docs_path / document_id
        directory.mkdir(parents=True, exist_ok=True)
        # Guard against path traversal: keep only the file's base name.
        safe_name = Path(filename).name
        path = directory / safe_name
        path.write_bytes(data)
        return path

    def delete(self, document_id: str) -> None:
        """Remove the document's source directory. No-op if it never existed."""
        directory = self.docs_path / document_id
        if directory.exists():
            shutil.rmtree(directory, ignore_errors=True)
