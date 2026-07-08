from pkb_ingestion.chunker import MarkdownChunker


def test_chunker_preserves_section_and_page_metadata() -> None:
    markdown = """# Java Concurrency

<!-- page: 12 -->

## Lock

ReentrantLock is a reentrant lock. It supports fair and unfair modes.

## AQS

AQS coordinates threads with a state value and a FIFO queue.
"""

    chunks = MarkdownChunker(max_chars=90, overlap_chars=15).chunk(
        document_id="java-concurrency",
        markdown=markdown,
        title="Java Concurrency",
    )

    assert chunks[0].document_id == "java-concurrency"
    assert chunks[0].title == "Java Concurrency"
    assert chunks[0].section == "Lock"
    assert chunks[0].page == 12
    assert "ReentrantLock" in chunks[0].text
    assert chunks[0].chunk_id == "java-concurrency:000001"


def test_chunker_uses_overlap_between_long_chunks() -> None:
    markdown = "## Notes\n\n" + "alpha beta gamma delta " * 30

    chunks = MarkdownChunker(max_chars=120, overlap_chars=24).chunk(
        document_id="notes",
        markdown=markdown,
        title="Notes",
    )

    assert len(chunks) > 1
    assert chunks[0].text[-24:] == chunks[1].text[:24]
