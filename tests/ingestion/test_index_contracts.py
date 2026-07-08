from pkb_ingestion.index_contracts import build_opensearch_document, build_qdrant_payload
from pkb_ingestion.models import Chunk


def test_qdrant_payload_contains_retrievable_chunk_text() -> None:
    chunk = Chunk(
        document_id="java-concurrency",
        chunk_id="java-concurrency:000001",
        chunk_index=1,
        text="ReentrantLock is a reentrant lock.",
        title="Java Concurrency",
        section="Lock",
        page=12,
        checksum="abc123",
    )

    payload = build_qdrant_payload(chunk)

    assert payload["docId"] == "java-concurrency"
    assert payload["chunkId"] == "java-concurrency:000001"
    assert payload["text"] == "ReentrantLock is a reentrant lock."
    assert payload["title"] == "Java Concurrency"
    assert payload["section"] == "Lock"
    assert payload["page"] == 12


def test_opensearch_document_uses_same_stable_chunk_identity() -> None:
    chunk = Chunk(
        document_id="redis",
        chunk_id="redis:000003",
        chunk_index=3,
        text="Redis uses an event loop.",
        title="Redis",
        section="Networking",
        page=None,
        checksum="def456",
    )

    document = build_opensearch_document(chunk)

    assert document["_id"] == "redis:000003"
    assert document["document"]["chunkId"] == "redis:000003"
    assert document["document"]["text"] == "Redis uses an event loop."
