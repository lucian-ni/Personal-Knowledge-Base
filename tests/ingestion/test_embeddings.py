import pytest
from pkb_ingestion.embeddings import (
    LocalEmbeddingProvider,
    OpenAICompatibleEmbeddingProvider,
)


def test_local_embedding_provider_embeds_lazily_and_refreshes_dimensions(monkeypatch) -> None:
    """The model loads on first embed() and the real dimension overrides the default."""
    import sentence_transformers

    calls: dict = {}

    class FakeModel:
        def __init__(self, name, device=None) -> None:
            calls["name"] = name
            calls["device"] = device

        def get_embedding_dimension(self) -> int:
            return 3

        def encode(self, texts, normalize_embeddings=True):
            calls["normalize"] = normalize_embeddings
            return [[1.0, 0.0, 0.0] for _ in texts]

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FakeModel)

    provider = LocalEmbeddingProvider(model_name="BAAI/bge-small-zh-v1.5", dimensions=512)
    assert provider._model is None  # lazy

    vectors = provider.embed(["hello", "world"])

    assert calls["name"] == "BAAI/bge-small-zh-v1.5"
    assert provider.dimensions == 3  # refreshed from the model
    assert calls["normalize"] is True  # vectors are normalized for COSINE
    assert len(vectors) == 2
    assert len(vectors[0]) == 3


def test_local_embedding_provider_returns_empty_for_no_texts() -> None:
    provider = LocalEmbeddingProvider()

    assert provider.embed([]) == []


def test_openai_compatible_provider_requires_full_config() -> None:
    with pytest.raises(ValueError):
        OpenAICompatibleEmbeddingProvider(base_url="", api_key="k", model="m", dimensions=8)
    with pytest.raises(ValueError):
        OpenAICompatibleEmbeddingProvider(base_url="https://x", api_key="k", model="", dimensions=8)


def test_openai_compatible_provider_returns_empty_for_no_texts() -> None:
    provider = OpenAICompatibleEmbeddingProvider(
        base_url="https://x", api_key="k", model="m", dimensions=8
    )

    assert provider.embed([]) == []


def test_openai_compatible_provider_calls_endpoint_and_normalizes(monkeypatch) -> None:
    """Mocks the /embeddings HTTP call: correct URL, auth header, body, and L2 normalization."""
    import httpx

    captured: dict = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {"data": [{"embedding": [3.0, 4.0]}, {"embedding": [0.0, 5.0]}]}

    def fake_post(url, *, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["body"] = json
        return FakeResponse()

    monkeypatch.setattr(httpx, "post", fake_post)

    provider = OpenAICompatibleEmbeddingProvider(
        base_url="https://embed.example.com/",
        api_key="sk-test",
        model="text-embedding-3-small",
        dimensions=2,
    )
    vectors = provider.embed(["hello", "world"])

    assert captured["url"] == "https://embed.example.com/embeddings"
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
    assert captured["body"] == {"input": ["hello", "world"], "model": "text-embedding-3-small"}
    # L2-normalized: [3,4]/5 = [0.6,0.8]; [0,5]/5 = [0,1].
    assert vectors[0] == pytest.approx([0.6, 0.8])
    assert vectors[1] == pytest.approx([0.0, 1.0])
