from __future__ import annotations

from typing import Any

from pkb_api.retrieval import RetrievalHit


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
