from __future__ import annotations

from pkb_ingestion.chunker import MarkdownChunker
from pkb_ingestion.embeddings import EmbeddingProvider
from pkb_ingestion.ids import qdrant_point_id
from pkb_ingestion.index_contracts import build_opensearch_document, build_qdrant_payload
from pkb_ingestion.models import Chunk, ChunkRecord, DocumentArtifact, IndexBatch, QdrantPoint


class IngestionPipeline:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        chunker: MarkdownChunker | None = None,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.chunker = chunker or MarkdownChunker()

    def build_index_batch(self, artifact: DocumentArtifact) -> IndexBatch:
        chunks = self.chunker.chunk(
            document_id=artifact.document_id,
            markdown=artifact.markdown,
            title=artifact.title,
        )
        vectors = self.embedding_provider.embed([chunk.text for chunk in chunks])

        return IndexBatch(
            document_id=artifact.document_id,
            chunk_records=[self._chunk_record(chunk) for chunk in chunks],
            qdrant_points=[
                QdrantPoint(
                    id=qdrant_point_id(chunk.chunk_id),
                    vector=vector,
                    payload=build_qdrant_payload(chunk),
                )
                for chunk, vector in zip(chunks, vectors, strict=True)
            ],
            opensearch_documents=[build_opensearch_document(chunk) for chunk in chunks],
        )

    def _chunk_record(self, chunk: Chunk) -> ChunkRecord:
        return ChunkRecord(
            document_id=chunk.document_id,
            chunk_id=chunk.chunk_id,
            chunk_index=chunk.chunk_index,
            title=chunk.title,
            section=chunk.section,
            page=chunk.page,
            # Estimated tokens from the chunker's TokenCounter (heuristic by
            # default: CJK ~1 token/char, Latin ~1/4) - the same budget that sized
            # the chunk, so the persisted value matches what was embedded.
            token_count=chunk.token_count,
            checksum=chunk.checksum,
            qdrant_point_id=qdrant_point_id(chunk.chunk_id),
            opensearch_document_id=chunk.chunk_id,
        )
