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


def test_chunker_prefers_sentence_boundary_over_hard_cut() -> None:
    # Two sentences with no spaces; the 。 between them falls inside the first window,
    # so the first chunk should end at that boundary instead of cutting mid-sentence.
    s1 = "alpha" * 10  # 50 chars, no boundary
    s2 = "beta" * 10  # 40 chars, no boundary
    markdown = "## Notes\n\n" + s1 + "。" + s2 + "。" + s1

    chunks = MarkdownChunker(max_chars=60, overlap_chars=10).chunk(
        document_id="notes",
        markdown=markdown,
        title="Notes",
    )

    assert len(chunks) > 1
    # First chunk ends right after the 。 that follows the first sentence.
    assert chunks[0].text == s1 + "。"


def test_chunker_falls_back_to_hard_cut_for_run_on_text() -> None:
    # No sentence boundaries anywhere: must still split (hard char cut) and overlap.
    markdown = "## Notes\n\n" + "abcdefgh" * 40

    chunks = MarkdownChunker(max_chars=50, overlap_chars=10).chunk(
        document_id="notes",
        markdown=markdown,
        title="Notes",
    )

    assert len(chunks) > 1
    assert chunks[0].text[-10:] == chunks[1].text[:10]
