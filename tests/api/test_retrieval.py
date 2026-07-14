from pkb_api.retrieval import (
    HybridRetriever,
    RetrievalHit,
    SimpleAnswerGenerator,
    rrf_fuse,
)


def _hit(chunk_id: str, *, score: float = 1.0, source: str = "s", text: str = "t") -> RetrievalHit:
    return RetrievalHit("doc", chunk_id, "title", None, None, text, score, source)


def test_rrf_fuse_combines_ranks_across_lists() -> None:
    """A chunk ranked #1 in both lists scores higher than one in only one list."""
    vector = [_hit("c1", source="vector"), _hit("c2", source="vector")]
    keyword = [_hit("c1", source="keyword"), _hit("c3", source="keyword")]

    fused = rrf_fuse(vector, keyword, k=60, limit=5)

    assert fused[0].chunk_id == "c1"
    assert fused[0].source == "rrf"
    assert abs(fused[0].score - (1 / 61 + 1 / 61)) < 1e-9
    assert {fused[1].chunk_id, fused[2].chunk_id} == {"c2", "c3"}


def test_rrf_fuse_single_list_preserves_order() -> None:
    fused = rrf_fuse([_hit("a"), _hit("b"), _hit("c")], k=60, limit=5)

    assert [h.chunk_id for h in fused] == ["a", "b", "c"]


def test_rrf_fuse_respects_limit() -> None:
    fused = rrf_fuse([_hit("a"), _hit("b"), _hit("c")], k=60, limit=2)

    assert [h.chunk_id for h in fused] == ["a", "b"]


def test_rrf_fuse_empty_returns_empty() -> None:
    assert rrf_fuse([], [], limit=5) == []


def test_rrf_fuse_later_rank_scores_lower() -> None:
    fused = rrf_fuse([_hit("a"), _hit("b")], k=60, limit=5)

    assert fused[0].score > fused[1].score


def test_rrf_fuse_ignores_score_magnitude() -> None:
    """RRF uses rank only; huge BM25 score vs tiny cosine must not skew ordering."""
    vector = [_hit("c1", score=0.01, source="vector")]
    keyword = [_hit("c2", score=999.0, source="keyword"), _hit("c1", score=1.0, source="keyword")]

    fused = rrf_fuse(vector, keyword, k=60, limit=5)

    # c1 appears in both -> ranks first despite keyword's large score on c2.
    assert fused[0].chunk_id == "c1"


def test_simple_answer_generator_returns_citations_when_llm_is_not_configured() -> None:
    generator = SimpleAnswerGenerator()
    hits = [_hit("java:000001", text="ReentrantLock is reentrant.")]

    result = generator.answer("What is ReentrantLock?", hits)

    assert "ReentrantLock is reentrant." in result.answer
    assert result.citations[0].chunk_id == "java:000001"


def test_hybrid_retriever_fuses_backends_without_reranker() -> None:
    class FakeEmbed:
        dimensions = 2

        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0]]

    class FakeVector:
        def search(self, vector: list[float], limit: int) -> list[RetrievalHit]:
            return [_hit("c1", source="vector")]

    class FakeKeyword:
        def search(self, query: str, limit: int) -> list[RetrievalHit]:
            return [_hit("c1", source="keyword"), _hit("c2", source="keyword")]

    retriever = HybridRetriever(
        FakeEmbed(), FakeVector(), FakeKeyword(), reranker=None, rerank_top_k=20
    )
    hits = retriever.search("q", limit=5)

    assert hits[0].chunk_id == "c1"
    assert hits[0].source == "rrf"


def test_hybrid_retriever_applies_reranker_when_configured() -> None:
    class FakeEmbed:
        dimensions = 2

        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0]]

    class FakeVector:
        def search(self, vector: list[float], limit: int) -> list[RetrievalHit]:
            return [_hit("c1", source="vector"), _hit("c2", source="vector")]

    class FakeKeyword:
        def search(self, query: str, limit: int) -> list[RetrievalHit]:
            return []

    class FakeReranker:
        def __init__(self) -> None:
            self.called = False

        def rerank(self, query: str, hits: list[RetrievalHit], limit: int) -> list[RetrievalHit]:
            self.called = True
            # Reverse the order to prove the reranker actually ran.
            return list(reversed(hits))[:limit]

    reranker = FakeReranker()
    retriever = HybridRetriever(
        FakeEmbed(), FakeVector(), FakeKeyword(), reranker=reranker, rerank_top_k=20
    )
    hits = retriever.search("q", limit=2)

    assert reranker.called
    assert [h.chunk_id for h in hits] == ["c2", "c1"]
