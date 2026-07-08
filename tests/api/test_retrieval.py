from pkb_api.retrieval import (
    HybridRetriever,
    RetrievalHit,
    SimpleAnswerGenerator,
    merge_hybrid_hits,
)


def test_merge_hybrid_hits_deduplicates_and_combines_scores() -> None:
    vector_hits = [
        RetrievalHit(
            document_id="java",
            chunk_id="java:000001",
            title="Java",
            section="Lock",
            page=12,
            text="ReentrantLock is reentrant.",
            score=0.8,
            source="vector",
        )
    ]
    keyword_hits = [
        RetrievalHit(
            document_id="java",
            chunk_id="java:000001",
            title="Java",
            section="Lock",
            page=12,
            text="ReentrantLock is reentrant.",
            score=4.0,
            source="keyword",
        ),
        RetrievalHit(
            document_id="redis",
            chunk_id="redis:000001",
            title="Redis",
            section=None,
            page=None,
            text="Redis uses an event loop.",
            score=2.0,
            source="keyword",
        ),
    ]

    merged = merge_hybrid_hits(vector_hits, keyword_hits, limit=2)

    assert [hit.chunk_id for hit in merged] == ["java:000001", "redis:000001"]
    assert merged[0].score > merged[1].score
    assert merged[0].source == "hybrid"


def test_simple_answer_generator_returns_citations_when_llm_is_not_configured() -> None:
    generator = SimpleAnswerGenerator()
    hits = [
        RetrievalHit(
            document_id="java",
            chunk_id="java:000001",
            title="Java",
            section="Lock",
            page=12,
            text="ReentrantLock is reentrant.",
            score=0.9,
            source="hybrid",
        )
    ]

    result = generator.answer("What is ReentrantLock?", hits)

    assert "ReentrantLock is reentrant." in result.answer
    assert result.citations[0].chunk_id == "java:000001"


def test_hybrid_retriever_embeds_query_and_merges_backend_results() -> None:
    class FakeEmbeddingProvider:
        dimensions = 2

        def embed(self, texts: list[str]) -> list[list[float]]:
            assert texts == ["lock"]
            return [[1.0, 0.0]]

    class FakeVectorSearch:
        def search(self, vector: list[float], limit: int) -> list[RetrievalHit]:
            assert vector == [1.0, 0.0]
            assert limit == 3
            return [
                RetrievalHit(
                    "java",
                    "java:000001",
                    "Java",
                    "Lock",
                    12,
                    "Vector hit",
                    0.9,
                    "vector",
                )
            ]

    class FakeKeywordSearch:
        def search(self, query: str, limit: int) -> list[RetrievalHit]:
            assert query == "lock"
            assert limit == 3
            return [
                RetrievalHit(
                    "java",
                    "java:000001",
                    "Java",
                    "Lock",
                    12,
                    "Keyword hit",
                    3.0,
                    "keyword",
                )
            ]

    retriever = HybridRetriever(
        embedding_provider=FakeEmbeddingProvider(),
        vector_search=FakeVectorSearch(),
        keyword_search=FakeKeywordSearch(),
    )

    hits = retriever.search("lock", limit=3)

    assert len(hits) == 1
    assert hits[0].chunk_id == "java:000001"
