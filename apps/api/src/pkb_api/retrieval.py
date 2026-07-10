from __future__ import annotations

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


class HybridRetriever:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_search: VectorSearchBackend,
        keyword_search: KeywordSearchBackend,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.vector_search = vector_search
        self.keyword_search = keyword_search

    def search(self, query: str, limit: int = 8) -> list[RetrievalHit]:
        query_vector = self.embedding_provider.embed([query])[0]
        vector_hits = self.vector_search.search(query_vector, limit)
        keyword_hits = self.keyword_search.search(query, limit)
        return merge_hybrid_hits(vector_hits, keyword_hits, limit=limit)


def merge_hybrid_hits(
    vector_hits: list[RetrievalHit],
    keyword_hits: list[RetrievalHit],
    limit: int,
    vector_weight: float = 0.6,
    keyword_weight: float = 0.4,
) -> list[RetrievalHit]:
    vector_scores = _normalize_scores(vector_hits)
    keyword_scores = _normalize_scores(keyword_hits)
    by_chunk: dict[str, RetrievalHit] = {}
    combined_scores: dict[str, float] = {}

    for hit in vector_hits:
        by_chunk.setdefault(hit.chunk_id, hit)
        combined_scores[hit.chunk_id] = combined_scores.get(hit.chunk_id, 0.0) + (
            vector_scores[hit.chunk_id] * vector_weight
        )

    for hit in keyword_hits:
        by_chunk.setdefault(hit.chunk_id, hit)
        combined_scores[hit.chunk_id] = combined_scores.get(hit.chunk_id, 0.0) + (
            keyword_scores[hit.chunk_id] * keyword_weight
        )

    ranked = sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)
    return [
        RetrievalHit(
            document_id=by_chunk[chunk_id].document_id,
            chunk_id=chunk_id,
            title=by_chunk[chunk_id].title,
            section=by_chunk[chunk_id].section,
            page=by_chunk[chunk_id].page,
            text=by_chunk[chunk_id].text,
            score=score,
            source="hybrid",
        )
        for chunk_id, score in ranked[:limit]
    ]


def _normalize_scores(hits: list[RetrievalHit]) -> dict[str, float]:
    if not hits:
        return {}
    max_score = max(hit.score for hit in hits)
    if max_score <= 0:
        return {hit.chunk_id: 0.0 for hit in hits}
    return {hit.chunk_id: hit.score / max_score for hit in hits}


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
            citations=[
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
            ],
        )


class LLMAnswerGenerator:
    """Synthesizes a cited answer via an OpenAI-compatible ``/chat/completions`` endpoint.

    Falls back to returning the raw context on transport errors so the search endpoint
    never hard-fails purely because the LLM is unreachable.
    """

    def __init__(self, base_url: str, api_key: str, model: str, *, timeout: float = 60.0) -> None:
        if not base_url or not api_key or not model:
            raise ValueError("base_url, api_key, and model are required for LLM answers")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def answer(self, query: str, hits: list[RetrievalHit]) -> SearchResult:
        citations = self._citations(hits)
        if not hits:
            return SearchResult(answer=f"No local context found for: {query}", citations=[])

        import httpx

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
        try:
            response = httpx.post(
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

    @staticmethod
    def _citations(hits: list[RetrievalHit]) -> list[ChunkCitation]:
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
