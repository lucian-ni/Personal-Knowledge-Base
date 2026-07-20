from __future__ import annotations

import math
from typing import Any, Protocol


class EmbeddingProvider(Protocol):
    dimensions: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""


class LocalEmbeddingProvider:
    """Local sentence embeddings via sentence-transformers (default bge-small-zh-v1.5).

    The model loads lazily on the first ``embed`` call and is cached by
    HuggingFace (``~/.cache/huggingface``). In China set
    ``HF_ENDPOINT=https://hf-mirror.com`` (and/or an HTTPS proxy) for the
    download. Vectors are L2-normalized to match Qdrant's COSINE distance.

    ``dimensions`` should match the model (512 for bge-small-zh-v1.5); it is
    also refreshed from the loaded model so the Qdrant collection is created
    with the correct vector size.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        dimensions: int = 512,
        *,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.dimensions = dimensions
        self.device = device
        self._model: Any = None

    def _ensure_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
            self.dimensions = self._model.get_embedding_dimension()
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._ensure_model()
        vectors = model.encode(texts, normalize_embeddings=True)
        return [list(v) for v in vectors]


class OpenAICompatibleEmbeddingProvider:
    """Embedding provider backed by an OpenAI-compatible ``/embeddings`` endpoint.

    The provider returns whatever vector size the model produces and L2-normalizes
    each vector. The caller is responsible for configuring ``embedding_dimensions``
    to match the model so the Qdrant collection is created with the right size.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        dimensions: int,
        *,
        timeout: float = 30.0,
    ) -> None:
        if not base_url or not api_key or not model:
            raise ValueError("base_url, api_key, and model are required for OpenAI embeddings")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions
        self.timeout = timeout
        self._client: Any = None

    def _ensure_client(self) -> Any:
        # A shared client reuses the HTTP connection across embed() calls (query
        # embedding runs on every search). Created lazily so import needs no network.
        if self._client is None:
            import httpx

            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        client = self._ensure_client()
        response = client.post(
            f"{self.base_url}/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"input": texts, "model": self.model},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        vectors = [self._normalize(item["embedding"]) for item in payload["data"]]
        return vectors

    @staticmethod
    def _normalize(vector: list[float]) -> list[float]:
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]
