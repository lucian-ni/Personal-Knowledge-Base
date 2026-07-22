from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

from pkb_api.retrieval import RetrievalHit

logger = logging.getLogger(__name__)


class QwenReranker:
    """Cross-encoder reranker backed by Qwen3-Reranker-0.6B (or compatible).

    Scores each (query, document) pair via the model's ``yes``/``no`` token
    logits and re-sorts by P(yes). Loads lazily on the first ``rerank`` call;
    on Apple Silicon it uses MPS, otherwise CPU. Implements the
    ``Reranker`` Protocol from ``pkb_api.retrieval`` structurally.
    """

    _PREFIX = (
        "<|im_start|>system\nJudge whether the Document meets the requirements "
        'based on the Query and the Instruct provided. Note that the answer can '
        'only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    )
    _SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Reranker-0.6B",
        *,
        device: str | None = None,
        max_length: int = 512,
        batch_size: int = 8,
        instruction: str = "Rank the document based on its relevance to the query.",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self.instruction = instruction
        self._model: Any = None
        self._tokenizer: Any = None
        self._yes_id: int | None = None
        self._no_id: int | None = None

    def _ensure_loaded(self) -> tuple[Any, Any]:
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, padding_side="left"
            )
            model = AutoModelForCausalLM.from_pretrained(self.model_name).eval()
            device = self.device or ("mps" if torch.backends.mps.is_available() else "cpu")
            self._model = model.to(device)
            self.device = device
            self._yes_id = self._tokenizer.convert_tokens_to_ids("yes")
            self._no_id = self._tokenizer.convert_tokens_to_ids("no")
        return self._model, self._tokenizer

    def _format(self, query: str, document: str) -> str:
        content = (
            f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {document}"
        )
        return f"{self._PREFIX}{content}{self._SUFFIX}"

    def _score_pairs(self, pairs: list[str]) -> list[float]:
        """Run the model over (query, doc) prompt pairs; return P(yes) per pair."""
        model, tokenizer = self._ensure_loaded()
        import torch

        scores: list[float] = []
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
                add_special_tokens=False,
            ).to(self.device)
            with torch.no_grad():
                logits = model(**inputs).logits[:, -1, :]
            yes = logits[:, self._yes_id]
            no = logits[:, self._no_id]
            probs = torch.softmax(torch.stack([yes, no], dim=-1), dim=-1)[:, 0]
            scores.extend(probs.tolist())
        return scores

    def rerank(
        self, query: str, hits: list[RetrievalHit], limit: int
    ) -> list[RetrievalHit]:
        if not hits:
            return []
        pairs = [self._format(query, h.text) for h in hits]
        scores = self._score_pairs(pairs)
        ranked = sorted(zip(hits, scores, strict=True), key=lambda x: x[1], reverse=True)
        return [
            RetrievalHit(
                document_id=h.document_id,
                chunk_id=h.chunk_id,
                title=h.title,
                section=h.section,
                page=h.page,
                text=h.text,
                score=score,
                source="reranked",
            )
            for h, score in ranked[:limit]
        ]


class ApiReranker:
    """Reranker backed by an Aliyun Model Studio ``/reranks`` endpoint.

    Sends the query plus the candidate chunk texts to ``POST {base_url}/reranks``
    (Aliyun ``compatible-api/v1/reranks``) and re-orders hits by the returned
    relevance scores (``{"results": [{"index": i, "relevance_score": s}, ...]}``).
    Falls back to
    returning the hits in their existing order (truncated to ``limit``) on any
    transport or HTTP error, so search never breaks just because the rerank
    endpoint is unavailable - matching the graceful-degradation pattern of the
    Qdrant/OpenSearch backends. Implements the ``Reranker`` Protocol structurally.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        timeout: float = 30.0,
        instruct: str = (
            "Given a web search query, retrieve relevant passages that answer the query."
        ),
    ) -> None:
        if not base_url or not api_key or not model:
            raise ValueError("base_url, api_key, and model are required for the API reranker")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.instruct = instruct
        self._client: Any = None

    def _ensure_client(self) -> Any:
        # A shared client reuses the HTTP connection across rerank() calls
        # (which run on every search). Created lazily so import needs no network.
        if self._client is None:
            import httpx

            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def rerank(
        self, query: str, hits: list[RetrievalHit], limit: int
    ) -> list[RetrievalHit]:
        if not hits:
            return []
        try:
            client = self._ensure_client()
            response = client.post(
                f"{self.base_url}/reranks",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "query": query,
                    "documents": [h.text for h in hits],
                    "top_n": limit,
                    "instruct": self.instruct,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload: dict[str, Any] = response.json()
            results = payload.get("results") or payload.get("data") or []
        except Exception as exc:
            # Endpoint down / 404 / bad model name: keep the RRF order so the
            # search endpoint stays up. Reranking resumes once a working
            # /reranks endpoint is configured.
            logger.warning(
                "API rerank failed (%s); returning RRF order unchanged: %s",
                type(exc).__name__,
                exc,
            )
            return hits[:limit]

        ranked: list[RetrievalHit] = []
        seen: set[int] = set()
        for item in results:
            idx = item.get("index")
            if idx is None or idx in seen or not (0 <= idx < len(hits)):
                continue
            seen.add(idx)
            score = float(item.get("relevance_score", item.get("score", 0.0)))
            ranked.append(replace(hits[idx], score=score, source="reranked"))
        # Append any indices the endpoint omitted (e.g. it only returned top_n),
        # preserving the original order before the final truncate.
        for idx, hit in enumerate(hits):
            if idx not in seen:
                ranked.append(replace(hit, source="reranked"))
        return ranked[:limit]
