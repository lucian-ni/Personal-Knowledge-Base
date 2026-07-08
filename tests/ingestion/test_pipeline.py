from pathlib import Path

from pkb_ingestion.embeddings import HashEmbeddingProvider
from pkb_ingestion.models import DocumentArtifact
from pkb_ingestion.pipeline import IngestionPipeline


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
    pipeline = IngestionPipeline(embedding_provider=HashEmbeddingProvider(dimensions=8))

    batch = pipeline.build_index_batch(artifact)

    assert batch.document_id == "java-concurrency"
    assert len(batch.qdrant_points) == 1
    assert batch.qdrant_points[0].id == "java-concurrency:000001"
    assert len(batch.qdrant_points[0].vector) == 8
    assert batch.qdrant_points[0].payload["text"] == "ReentrantLock is a reentrant lock."
    assert batch.opensearch_documents[0]["_id"] == "java-concurrency:000001"
    assert batch.chunk_records[0].qdrant_point_id == "java-concurrency:000001"
