from pkb_api.schemas import ChunkCitation, SearchResult


def test_search_result_returns_answer_with_cited_chunks() -> None:
    result = SearchResult(
        answer="ReentrantLock is reentrant.",
        citations=[
            ChunkCitation(
                document_id="java-concurrency",
                chunk_id="java-concurrency:000001",
                title="Java Concurrency",
                section="Lock",
                page=12,
                text="ReentrantLock is a reentrant lock.",
                score=0.91,
            )
        ],
    )

    assert result.citations[0].text == "ReentrantLock is a reentrant lock."
    assert result.citations[0].page == 12
