from pathlib import Path

from pkb_ingestion.ids import qdrant_point_id
from pkb_ingestion.models import DocumentArtifact
from pkb_ingestion.pipeline import IngestionPipeline


class _FakeEmbeddingProvider:
    dimensions = 8

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * self.dimensions for _ in texts]


def test_pipeline_builds_index_batch_with_vectors_and_chunk_text() -> None:
    artifact = DocumentArtifact(
        document_id="java-concurrency",
        title="Java Concurrency",
        source_path=Path("storage/docs/java.pdf"),
        markdown="""# Java Concurrency

<!-- page: 12 -->

## Lock

ReentrantLock is a reentrant lock.
""",
    )
    pipeline = IngestionPipeline(embedding_provider=_FakeEmbeddingProvider())

    batch = pipeline.build_index_batch(artifact)

    assert batch.document_id == "java-concurrency"
    assert len(batch.qdrant_points) == 1
    # Qdrant point IDs must be a UUID/uint, so they are a uuid5 of the stable
    # chunk ID; the stable chunk ID itself remains the cross-store join key.
    point_id = qdrant_point_id("java-concurrency:000001")
    assert batch.qdrant_points[0].id == point_id
    assert len(batch.qdrant_points[0].vector) == 8
    assert batch.qdrant_points[0].payload["chunkId"] == "java-concurrency:000001"
    assert batch.qdrant_points[0].payload["text"] == "ReentrantLock is a reentrant lock."
    assert batch.opensearch_documents[0]["_id"] == "java-concurrency:000001"
    assert batch.chunk_records[0].qdrant_point_id == point_id
    assert batch.chunk_records[0].opensearch_document_id == "java-concurrency:000001"
    # token_count is a character count (meaningful for CJK; split() would be ~1 for Chinese).
    assert batch.chunk_records[0].token_count == len("ReentrantLock is a reentrant lock.")
