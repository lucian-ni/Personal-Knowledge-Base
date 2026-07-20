from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from pathlib import Path

from pkb_ingestion.docling_converter import DoclingMarkdownConverter
from pkb_ingestion.models import ChunkRecord, DocumentArtifact
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from pkb_api.models import Document, DocumentChunk, DocumentStatus, IngestionJob, JobStatus
from pkb_api.schemas import DocumentRead, IngestionJobRead
from pkb_api.services import Services

logger = logging.getLogger(__name__)

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


@dataclass
class PreparedIngestion:
    """State handed from the upload request to the background ingestion task.

    The request creates the document + job rows (status ``running``), saves the raw
    file, and returns immediately (HTTP 202). The heavy work (convert -> chunk ->
    embed -> persist) runs in the background via ``IngestionService.run_ingestion``.
    """

    job: IngestionJobRead
    document_id: str
    job_id: str
    kind: str
    data: bytes
    source_path: Path
    title: str


class IngestionService:
    """Orchestrates document ingestion and deletion across all three stores.

    Ingestion is split: ``prepare_ingest`` does the fast, request-synchronous work
    (validate, create rows, save file) and returns a ``PreparedIngestion``;
    ``run_ingestion`` does the heavy work in its own session as a FastAPI background
    task. Cross-store writes are best-effort: on failure any chunks already written
    to Qdrant/OpenSearch are cleaned up so a ``failed`` document never leaks
    searchable hits. Re-ingest is idempotent because all stores key on the stable
    chunk ID (Qdrant by the derived UUID point ID).
    """

    def __init__(
        self,
        services: Services,
        converter: DoclingMarkdownConverter | None = None,
    ) -> None:
        self.services = services
        self.converter = converter or DoclingMarkdownConverter()

    # -- ingestion -----------------------------------------------------------

    def prepare_ingest(
        self,
        session: Session,
        filename: str,
        content_type: str | None,
        data: bytes,
    ) -> PreparedIngestion:
        """Fast, request-synchronous setup: validate, create rows, save the file.

        Raises ``UnsupportedDocumentError`` (-> 415) for unknown types. Does NOT do
        the heavy convert/embed/persist work; that runs in ``run_ingestion``.
        """
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
        except Exception as exc:
            session.rollback()
            document.status = DocumentStatus.failed
            job.status = JobStatus.failed
            job.failure_reason = str(exc)[:1000]
            session.commit()
            raise

        return PreparedIngestion(
            job=IngestionJobRead(id=job.id, document_id=document_id, status=str(job.status)),
            document_id=document_id,
            job_id=job.id,
            kind=kind,
            data=data,
            source_path=source_path,
            title=title,
        )

    def run_ingestion(self, prep: PreparedIngestion) -> None:
        """Heavy ingestion (convert -> chunk -> embed -> persist) in its own session.

        Runs as a FastAPI background task after the upload response is sent. Updates
        the document/job to ``completed`` or, on failure, ``failed`` and cleans up
        any chunks already written to Qdrant/OpenSearch.
        """
        with self.services.session_factory() as session:
            document = session.get(Document, prep.document_id)
            job = session.get(IngestionJob, prep.job_id)
            if document is None or job is None:
                logger.warning("Ingestion target missing for document %s", prep.document_id)
                return
            try:
                artifact = self._to_artifact(
                    prep.document_id, prep.title, prep.source_path, prep.kind, prep.data
                )
                batch = self.services.pipeline.build_index_batch(artifact)

                # Ensure stores exist before writing anything, so a missing store fails early.
                self.services.qdrant_indexer.ensure_collection()
                self.services.opensearch_indexer.ensure_index()

                session.add_all(
                    [_chunk_to_orm(prep.document_id, rec) for rec in batch.chunk_records]
                )
                self.services.qdrant_indexer.upsert(batch.qdrant_points)
                self.services.opensearch_indexer.bulk_index(batch.opensearch_documents)

                document.status = DocumentStatus.ready
                job.status = JobStatus.completed
                session.commit()
            except Exception as exc:
                session.rollback()
                # Re-fetch after rollback so status updates land on fresh instances.
                self._cleanup_search_stores(prep.document_id)
                document = session.get(Document, prep.document_id)
                job = session.get(IngestionJob, prep.job_id)
                if document is not None:
                    document.status = DocumentStatus.failed
                if job is not None:
                    job.status = JobStatus.failed
                    job.failure_reason = str(exc)[:1000]
                session.commit()
                logger.exception("Ingestion failed for document %s", prep.document_id)

    def _cleanup_search_stores(self, document_id: str) -> None:
        """Best-effort removal of any Qdrant points / OpenSearch docs for a document.

        Both backends swallow transport errors, so this never raises. Called on
        ingestion failure (so a failed doc has no searchable orphans) and on delete.
        """
        self.services.qdrant_indexer.delete_by_document(document_id)
        self.services.opensearch_indexer.delete_by_document(document_id)

    # -- deletion ------------------------------------------------------------

    def delete_document(self, session: Session, document_id: str) -> bool:
        """Remove a document from Postgres + both search stores + the filesystem.

        Returns False if the document does not exist. Search-store and file removal
        are best-effort (never raise) so a delete succeeds even if a store is down.
        """
        document = session.get(Document, document_id)
        if document is None:
            return False
        # Chunks cascade via the Document.chunks relationship (delete-orphan). Jobs
        # have no relationship, so delete them explicitly. Both work on SQLite (FK
        # CASCADE is off in tests) and Postgres.
        session.execute(delete(IngestionJob).where(IngestionJob.document_id == document_id))
        session.delete(document)
        session.commit()

        self._cleanup_search_stores(document_id)
        self.services.storage.delete(document_id)
        return True

    # -- helpers -------------------------------------------------------------

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
