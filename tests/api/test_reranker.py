from pkb_api.reranker import QwenReranker
from pkb_api.retrieval import RetrievalHit


def _hit(chunk_id: str, text: str = "doc") -> RetrievalHit:
    return RetrievalHit("doc", chunk_id, "title", None, None, text, 0.0, "rrf")


def test_qwen_reranker_format_contains_query_and_document() -> None:
    reranker = QwenReranker()
    prompt = reranker._format("what is a lock", "ReentrantLock is a lock")
    assert "what is a lock" in prompt
    assert "ReentrantLock is a lock" in prompt


def test_qwen_reranker_returns_empty_for_no_hits() -> None:
    reranker = QwenReranker()
    assert reranker.rerank("q", [], limit=5) == []


def test_qwen_reranker_sorts_descending_and_tags_source(monkeypatch) -> None:
    """Scoring is mocked: verify rerank sorts by score and relabels source='reranked'."""
    reranker = QwenReranker()
    monkeypatch.setattr(reranker, "_score_pairs", lambda pairs: [0.1, 0.9, 0.5])
    ranked = reranker.rerank("q", [_hit("a"), _hit("b"), _hit("c")], limit=3)

    assert [h.chunk_id for h in ranked] == ["b", "c", "a"]  # 0.9, 0.5, 0.1
    assert ranked[0].source == "reranked"
    assert ranked[0].score == 0.9


def test_qwen_reranker_respects_limit(monkeypatch) -> None:
    reranker = QwenReranker()
    monkeypatch.setattr(reranker, "_score_pairs", lambda pairs: [0.9, 0.1])
    ranked = reranker.rerank("q", [_hit("a"), _hit("b")], limit=1)

    assert [h.chunk_id for h in ranked] == ["a"]
