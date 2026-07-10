from __future__ import annotations

import uuid
from pathlib import Path

from pkb_ingestion.docling_converter import DoclingMarkdownConverter
from pkb_ingestion.models import ChunkRecord, DocumentArtifact
from sqlalchemy import select
from sqlalchemy.orm import Session

from pkb_api.models import Document, DocumentChunk, DocumentStatus, IngestionJob, JobStatus
from pkb_api.schemas import DocumentRead, IngestionJobRead
from pkb_api.services import Services

MARKDOWN_SUFFIXES = {".md", ".markdown"}
TEXT_SUFFIXES = {".txt"}
PDF_SUFFIXES = {".pdf"}
MARKDOWN_MIME = {"text/markdown", "text/x-markdown"}
TEXT_MIME = {"text/plain"}
PDF_MIME = {"application/pdf"}

# Anything we can read directly as markdown text without a converter.
DIRECT_SUFFIXES = MARKDOWN_SUFFIXES | TEXT_SUFFIXES
DIRECT_MIME = MARKDOWN_MIME | TEXT_MIME


class UnsupportedDocumentError(Exception):
    """Raised when an uploaded file type cannot be converted to markdown."""


class IngestionService:
    """Orchestrates a single document ingestion: convert -> chunk -> embed -> persist.

    The pipeline itself is a pure transform; this service owns the side effects of
    writing to Postgres (document + chunks + job), Qdrant (vectors), and OpenSearch
    (keyword index). Cross-store writes are best-effort: Postgres chunk rows are
    committed only on full success, while Qdrant/OpenSearch upserts may leave
    orphaned points if a later step fails (dedup by stable_chunk_id makes a
    re-ingest idempotent).
    """

    def __init__(
        self,
        services: Services,
        converter: DoclingMarkdownConverter | None = None,
    ) -> None:
        self.services = services
        self.converter = converter or DoclingMarkdownConverter()

    def ingest(
        self,
        session: Session,
        filename: str,
        content_type: str | None,
        data: bytes,
    ) -> IngestionJobRead:
        mime_type = content_type or "application/octet-stream"
        # Validate up front so unsupported types 415 without leaving a failed row.
        kind = self._classify(filename, mime_type)

        document_id = uuid.uuid4().hex
        title = Path(filename).stem or document_id

        document = Document(
            id=document_id,
            title=title,
            original_filename=filename,
            mime_type=mime_type,
            file_path="",
            status=DocumentStatus.processing,
        )
        job = IngestionJob(id=uuid.uuid4().hex, document_id=document_id, status=JobStatus.running)
        session.add_all([document, job])
        session.commit()
        session.refresh(document)
        session.refresh(job)

        try:
            source_path = self.services.storage.save(document_id, filename, data)
            document.file_path = str(source_path)
            session.commit()

            artifact = self._to_artifact(document_id, title, source_path, kind, data)
            batch = self.services.pipeline.build_index_batch(artifact)

            # Ensure stores exist before writing anything, so a missing store fails early.
            self.services.qdrant_indexer.ensure_collection()
            self.services.opensearch_indexer.ensure_index()

            session.add_all([_chunk_to_orm(document_id, rec) for rec in batch.chunk_records])
            self.services.qdrant_indexer.upsert(batch.qdrant_points)
            self.services.opensearch_indexer.bulk_index(batch.opensearch_documents)

            document.status = DocumentStatus.ready
            job.status = JobStatus.completed
            session.commit()
        except Exception as exc:
            session.rollback()
            document.status = DocumentStatus.failed
            job.status = JobStatus.failed
            job.failure_reason = str(exc)[:1000]
            session.commit()
            raise

        return IngestionJobRead(
            id=job.id,
            document_id=document_id,
            status=str(job.status),
        )

    def _classify(self, filename: str, mime_type: str) -> str:
        suffix = Path(filename).suffix.lower()
        if suffix in DIRECT_SUFFIXES or mime_type in DIRECT_MIME:
            return "direct"
        if suffix in PDF_SUFFIXES or mime_type in PDF_MIME:
            return "pdf"
        raise UnsupportedDocumentError(
            f"Unsupported file type: filename={filename} mime={mime_type}"
        )

    def _to_artifact(
        self,
        document_id: str,
        title: str,
        source_path: Path,
        kind: str,
        data: bytes,
    ) -> DocumentArtifact:
        if kind == "pdf":
            return self.converter.convert(document_id, source_path, title=title)
        markdown = data.decode("utf-8", errors="replace")
        return DocumentArtifact(
            document_id=document_id,
            title=title,
            source_path=source_path,
            markdown=markdown,
        )


def _chunk_to_orm(document_id: str, record: ChunkRecord) -> DocumentChunk:
    return DocumentChunk(
        document_id=document_id,
        chunk_id=record.chunk_id,
        chunk_index=record.chunk_index,
        title=record.title,
        section=record.section,
        page=record.page,
        token_count=record.token_count,
        checksum=record.checksum,
        qdrant_point_id=record.qdrant_point_id,
        opensearch_document_id=record.opensearch_document_id,
    )


def list_documents(session: Session) -> list[DocumentRead]:
    rows = session.scalars(select(Document).order_by(Document.created_at.desc())).all()
    return [
        DocumentRead(
            id=row.id,
            title=row.title,
            original_filename=row.original_filename,
            mime_type=row.mime_type,
            status=str(row.status),
            version=row.version,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )
        for row in rows
    ]
