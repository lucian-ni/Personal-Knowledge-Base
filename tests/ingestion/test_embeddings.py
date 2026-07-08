from pkb_ingestion.embeddings import HashEmbeddingProvider


def test_hash_embedding_provider_is_deterministic_and_normalized() -> None:
    provider = HashEmbeddingProvider(dimensions=16)

    first = provider.embed(["ReentrantLock is reentrant"])[0]
    second = provider.embed(["ReentrantLock is reentrant"])[0]

    assert first == second
    assert len(first) == 16
    assert abs(sum(value * value for value in first) - 1.0) < 0.000001


def test_hash_embedding_provider_returns_zero_vector_for_empty_text() -> None:
    provider = HashEmbeddingProvider(dimensions=4)

    assert provider.embed([""])[0] == [0.0, 0.0, 0.0, 0.0]
