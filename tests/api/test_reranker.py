import pytest
from pkb_api.reranker import ApiReranker, QwenReranker
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


# -- ApiReranker (OpenAI-compatible /rerank endpoint) -----------------------


def _api_reranker_with_client(captured: dict, response) -> ApiReranker:
    """Build an ApiReranker whose /rerank call is intercepted by a fake client."""

    class FakeClient:
        def post(self, url, *, headers=None, json=None, timeout=None):
            captured["url"] = url
            captured["headers"] = headers
            captured["body"] = json
            return response

    reranker = ApiReranker(
        base_url="https://rerank.example.com/", api_key="sk-test", model="qwen3-rerank"
    )
    reranker._client = FakeClient()  # reuse the injected connection instead of httpx
    return reranker


def test_api_reranker_requires_full_config() -> None:
    with pytest.raises(ValueError):
        ApiReranker(base_url="", api_key="k", model="m")
    with pytest.raises(ValueError):
        ApiReranker(base_url="https://x", api_key="", model="m")


def test_api_reranker_returns_empty_for_no_hits() -> None:
    reranker = ApiReranker(base_url="https://x", api_key="k", model="m")
    assert reranker.rerank("q", [], limit=5) == []


def test_api_reranker_reranks_by_relevance_score_and_appends_omitted() -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            # Returns index 2 first (highest score), then index 0; index 1 omitted.
            return {
                "results": [
                    {"index": 2, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.5},
                ]
            }

    captured: dict = {}
    reranker = _api_reranker_with_client(captured, FakeResponse())
    ranked = reranker.rerank("RAG", [_hit("a"), _hit("b"), _hit("c")], limit=3)

    # Scored hits first (by relevance desc), then the omitted index appended.
    assert [h.chunk_id for h in ranked] == ["c", "a", "b"]
    assert all(h.source == "reranked" for h in ranked)
    assert ranked[0].score == 0.9
    assert ranked[1].score == 0.5
    # Request shape: endpoint, bearer auth, model/query/documents/top_n.
    assert captured["url"] == "https://rerank.example.com/reranks"
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
    assert captured["body"]["model"] == "qwen3-rerank"
    assert captured["body"]["query"] == "RAG"
    assert captured["body"]["documents"] == ["doc", "doc", "doc"]
    assert captured["body"]["top_n"] == 3
    assert "instruct" in captured["body"]


def test_api_reranker_degrades_to_rrf_order_on_http_error() -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            raise RuntimeError("HTTP 404 not found")

        def json(self) -> dict:
            raise AssertionError("json() should not be called on a failed response")

    reranker = _api_reranker_with_client({}, FakeResponse())
    ranked = reranker.rerank("q", [_hit("a"), _hit("b"), _hit("c")], limit=2)

    # Pass-through: original order, truncated to limit, sources untouched.
    assert [h.chunk_id for h in ranked] == ["a", "b"]
    assert all(h.source == "rrf" for h in ranked)


def test_api_reranker_respects_limit() -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {"results": [{"index": 1, "relevance_score": 0.9}]}

    reranker = _api_reranker_with_client({}, FakeResponse())
    ranked = reranker.rerank("q", [_hit("a"), _hit("b"), _hit("c")], limit=1)

    assert [h.chunk_id for h in ranked] == ["b"]
