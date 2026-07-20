from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Protocol

from pkb_ingestion.embeddings import EmbeddingProvider

from pkb_api.schemas import ChunkCitation, SearchResult
from pkb_api.settings import Settings


@dataclass(frozen=True)
class RetrievalHit:
    document_id: str
    chunk_id: str
    title: str
    section: str | None
    page: int | None
    text: str
    score: float
    source: str


class VectorSearchBackend(Protocol):
    def search(self, vector: list[float], limit: int) -> list[RetrievalHit]:
        """Return vector search hits for an embedded query."""


class KeywordSearchBackend(Protocol):
    def search(self, query: str, limit: int) -> list[RetrievalHit]:
        """Return keyword search hits for a raw query."""


class Reranker(Protocol):
    def rerank(
        self, query: str, hits: list[RetrievalHit], limit: int
    ) -> list[RetrievalHit]:
        """Re-score and re-order hits by relevance to the query."""


class HybridRetriever:
    """Hybrid retrieval: vector (Qdrant) + keyword (OpenSearch BM25) -> RRF fusion,
    optionally followed by a cross-encoder rerank step.

    When a ``reranker`` is configured, each backend is asked for ``rerank_top_k``
    hits, RRF fuses them, the reranker re-scores, and the top ``limit`` are
    returned. Without a reranker, RRF output is truncated to ``limit``.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_search: VectorSearchBackend,
        keyword_search: KeywordSearchBackend,
        *,
        reranker: Reranker | None = None,
        rrf_k: int = 60,
        rerank_top_k: int = 20,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.vector_search = vector_search
        self.keyword_search = keyword_search
        self.reranker = reranker
        self.rrf_k = rrf_k
        self.rerank_top_k = rerank_top_k

    def search(self, query: str, limit: int = 8) -> list[RetrievalHit]:
        query_vector = self.embedding_provider.embed([query])[0]
        fetch = max(limit, self.rerank_top_k) if self.reranker else limit
        vector_hits = self.vector_search.search(query_vector, fetch)
        keyword_hits = self.keyword_search.search(query, fetch)
        fused = rrf_fuse(
            vector_hits, keyword_hits, k=self.rrf_k, limit=fetch
        )
        if self.reranker:
            return self.reranker.rerank(query, fused, limit=limit)
        return fused[:limit]


def rrf_fuse(
    *ranked_lists: list[RetrievalHit],
    k: int = 60,
    limit: int = 8,
) -> list[RetrievalHit]:
    """Reciprocal Rank Fusion: score = sum(1 / (k + rank)) across retrievers.

    ``rank`` is 1-based within each input list. Uses only rank (not score
    magnitude), so it is robust to BM25 vs cosine score-scale differences.
    Ties keep insertion order (first retriever wins).
    """
    scores: dict[str, float] = {}
    by_chunk: dict[str, RetrievalHit] = {}
    for hits in ranked_lists:
        for rank, hit in enumerate(hits, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank)
            by_chunk.setdefault(hit.chunk_id, hit)

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [
        RetrievalHit(
            document_id=by_chunk[chunk_id].document_id,
            chunk_id=chunk_id,
            title=by_chunk[chunk_id].title,
            section=by_chunk[chunk_id].section,
            page=by_chunk[chunk_id].page,
            text=by_chunk[chunk_id].text,
            score=score,
            source="rrf",
        )
        for chunk_id, score in ranked[:limit]
    ]


def build_citations(hits: list[RetrievalHit]) -> list[ChunkCitation]:
    return [
        ChunkCitation(
            document_id=hit.document_id,
            chunk_id=hit.chunk_id,
            title=hit.title,
            section=hit.section,
            page=hit.page,
            text=hit.text,
            score=hit.score,
        )
        for hit in hits
    ]


class SimpleAnswerGenerator:
    def answer(self, query: str, hits: list[RetrievalHit]) -> SearchResult:
        if not hits:
            return SearchResult(answer=f"No local context found for: {query}", citations=[])

        snippets = "\n\n".join(f"[{index}] {hit.text}" for index, hit in enumerate(hits, start=1))
        return SearchResult(
            answer=(
                "LLM is not configured, so here are the most relevant local chunks:\n\n"
                f"{snippets}"
            ),
            citations=build_citations(hits),
        )

    def stream_answer(self, query: str, hits: list[RetrievalHit]) -> Iterator[str]:
        """Yield the answer in one or two chunks (no LLM to stream token-by-token)."""
        if not hits:
            yield f"No local context found for: {query}"
            return
        yield "LLM is not configured, so here are the most relevant local chunks:\n\n"
        yield "\n\n".join(f"[{index}] {hit.text}" for index, hit in enumerate(hits, start=1))


class LLMAnswerGenerator:
    """Synthesizes a cited answer via an OpenAI-compatible ``/chat/completions`` endpoint.

    Falls back to returning the raw context on transport errors so the search endpoint
    never hard-fails purely because the LLM is unreachable. ``answer`` does a single
    request; ``stream_answer`` streams token deltas (``stream: true``) for the SSE
    endpoint. Both reuse a shared ``httpx.Client`` (one connection across searches).
    """

    def __init__(self, base_url: str, api_key: str, model: str, *, timeout: float = 60.0) -> None:
        if not base_url or not api_key or not model:
            raise ValueError("base_url, api_key, and model are required for LLM answers")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._client: Any = None

    def _ensure_client(self) -> Any:
        if self._client is None:
            import httpx

            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def _messages(self, query: str, hits: list[RetrievalHit]) -> tuple[list[dict[str, str]], str]:
        context = "\n\n".join(
            f"[{index}] (chunk_id={hit.chunk_id}) {hit.text}"
            for index, hit in enumerate(hits, start=1)
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "Answer the user's question using only the provided context. "
                    "Cite sources as [1], [2], ... matching the bracketed indices. "
                    "If the context does not contain the answer, say so."
                ),
            },
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ]
        return messages, context

    def answer(self, query: str, hits: list[RetrievalHit]) -> SearchResult:
        citations = build_citations(hits)
        if not hits:
            return SearchResult(answer=f"No local context found for: {query}", citations=[])

        messages, context = self._messages(query, hits)
        try:
            client = self._ensure_client()
            response = client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "messages": messages, "temperature": 0.2},
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload: dict[str, Any] = response.json()
            answer = payload["choices"][0]["message"]["content"].strip()
        except Exception:
            answer = (
                "LLM is configured but unreachable, so here are the most relevant "
                f"local chunks:\n\n{context}"
            )
        return SearchResult(answer=answer, citations=citations)

    def stream_answer(self, query: str, hits: list[RetrievalHit]) -> Iterator[str]:
        """Stream answer token deltas from the LLM (OpenAI-compatible SSE).

        Falls back to yielding the raw context if the LLM is unreachable, so the
        streaming endpoint never hard-fails.
        """
        if not hits:
            yield f"No local context found for: {query}"
            return

        messages, context = self._messages(query, hits)
        try:
            client = self._ensure_client()
            with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0.2,
                    "stream": True,
                },
                timeout=self.timeout,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data = line[len("data: ") :]
                    if data.strip() == "[DONE]":
                        break
                    chunk = json.loads(data)
                    delta = (
                        chunk.get("choices", [{}])[0].get("delta", {}).get("content")
                    )
                    if delta:
                        yield delta
        except Exception:
            yield (
                "LLM is configured but unreachable, so here are the most relevant "
                f"local chunks:\n\n{context}"
            )


def make_answer_generator(settings: Settings) -> SimpleAnswerGenerator | LLMAnswerGenerator:
    """Pick the answer generator based on settings; SimpleAnswerGenerator is the local fallback."""
    if (
        settings.llm_api_base_url
        and settings.llm_api_key
        and settings.llm_model
        and settings.llm_api_base_url.strip()
    ):
        return LLMAnswerGenerator(
            base_url=settings.llm_api_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
        )
    return SimpleAnswerGenerator()
