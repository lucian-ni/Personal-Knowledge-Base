from pkb_ingestion.chunker import MarkdownChunker
from pkb_ingestion.tokens import HeuristicTokenCounter


def _longest_common_overlap(a: str, b: str) -> int:
    """Largest L with a[-L:] == b[:L]; 0 if they share no overlap window."""
    m = min(len(a), len(b))
    for length in range(m, 0, -1):
        if a[-length:] == b[:length]:
            return length
    return 0


def test_chunker_preserves_section_and_page_metadata() -> None:
    markdown = """# Java Concurrency

<!-- page: 12 -->

## Lock

ReentrantLock is a reentrant lock. It supports fair and unfair modes.

## AQS

AQS coordinates threads with a state value and a FIFO queue.
"""

    chunks = MarkdownChunker(max_tokens=512).chunk(
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
    # The next heading flushes, so the AQS content is its own chunk.
    assert chunks[1].section == "AQS"
    assert chunks[1].chunk_id == "java-concurrency:000002"


def test_chunker_h1_is_dropped_and_does_not_straddle_sections() -> None:
    markdown = """# Doc Title

Content before any section.

## First

First body.

## Second

Second body.
"""

    chunks = MarkdownChunker().chunk(document_id="d", markdown=markdown, title="Doc Title")

    # The H1 line is the document title (passed in separately) and is dropped.
    assert all("# Doc Title" not in chunk.text for chunk in chunks)
    # Each section's body lands in its own chunk (headings flush the buffer).
    assert [chunk.section for chunk in chunks] == [None, "First", "Second"]
    assert "First body." in chunks[1].text
    assert "Second body." in chunks[2].text


def test_chunker_coalesces_paragraphs_until_budget_then_cuts_at_node_boundary() -> None:
    # Ten short paragraphs in one section. They should pack into chunks up to the
    # token budget, and a cut should land between paragraphs (never mid-paragraph).
    markdown = "## S\n\n" + "\n\n".join(f"Paragraph {i} short." for i in range(10))

    chunks = MarkdownChunker(max_tokens=30, overlap_tokens=4).chunk(
        document_id="c", markdown=markdown, title="S"
    )

    assert len(chunks) > 1
    for chunk in chunks:
        # No chunk starts with a fragment left over from a mid-paragraph cut.
        assert not chunk.text.startswith("short.")
        assert chunk.text.startswith("Paragraph")


def test_chunker_keeps_code_blocks_and_tables_atomic_when_they_fit() -> None:
    markdown = """## Code

A paragraph.

```python
def f():
    return 1
```

| a | b |
|---|---|
| 1 | 2 |
"""

    chunks = MarkdownChunker().chunk(document_id="c", markdown=markdown, title="Code")

    joined = "\n".join(chunk.text for chunk in chunks)
    # The code fence and the table survive whole (not split mid-block).
    assert "```python\ndef f():\n    return 1\n```" in joined
    assert "| a | b |\n|---|---|\n| 1 | 2 |" in joined


def test_chunker_hard_splits_oversized_code_block_with_overlap() -> None:
    big_code = "```python\n" + ("x = 1  # comment line\n" * 200) + "```"
    markdown = "## Big\n\n" + big_code

    chunks = MarkdownChunker(max_tokens=40, overlap_tokens=8).chunk(
        document_id="c", markdown=markdown, title="Big"
    )

    # One over-budget code block becomes many hard-split pieces, with overlap.
    assert len(chunks) > 1
    assert _longest_common_overlap(chunks[0].text, chunks[1].text) > 0


def test_chunker_uses_overlap_between_long_chunks() -> None:
    markdown = "## Notes\n\n" + "alpha beta gamma delta " * 30

    chunks = MarkdownChunker(max_tokens=30, overlap_tokens=8).chunk(
        document_id="notes", markdown=markdown, title="Notes"
    )

    assert len(chunks) > 1
    assert _longest_common_overlap(chunks[0].text, chunks[1].text) > 0


def test_chunker_prefers_sentence_boundary_over_hard_cut() -> None:
    # Two CJK-terminated sentences with no spaces; the 。 between them must be a
    # split point, so the first chunk ends exactly after the first sentence.
    s1 = "alpha" * 10  # 50 chars, no boundary
    s2 = "beta" * 10  # 40 chars, no boundary
    markdown = "## Notes\n\n" + s1 + "。" + s2 + "。" + s1

    chunks = MarkdownChunker(max_tokens=20, overlap_tokens=4).chunk(
        document_id="notes", markdown=markdown, title="Notes"
    )

    assert len(chunks) > 1
    # Sentence Splitter packs one sentence per piece; no overlap between them.
    assert chunks[0].text == s1 + "。"
    assert chunks[1].text == s2 + "。"


def test_chunker_falls_back_to_hard_cut_for_run_on_text() -> None:
    # No sentence boundaries anywhere: must still split (hard cut) and overlap.
    markdown = "## Notes\n\n" + "abcdefgh" * 40

    chunks = MarkdownChunker(max_tokens=12, overlap_tokens=2).chunk(
        document_id="notes", markdown=markdown, title="Notes"
    )

    assert len(chunks) > 1
    assert _longest_common_overlap(chunks[0].text, chunks[1].text) > 0


def test_chunker_chunk_indices_are_monotonic_and_stable() -> None:
    markdown = "## S\n\n" + "\n\n".join(f"Para {i}." for i in range(5))
    chunks = MarkdownChunker(max_tokens=15, overlap_tokens=4).chunk(
        document_id="doc", markdown=markdown, title="S"
    )

    assert [chunk.chunk_index for chunk in chunks] == list(range(1, len(chunks) + 1))
    assert all(chunk.chunk_id == f"doc:{chunk.chunk_index:06d}" for chunk in chunks)
    # checksums are content-derived and unique across distinct chunks.
    assert len({chunk.checksum for chunk in chunks}) == len(chunks)


def test_chunker_token_count_matches_heuristic_counter() -> None:
    # Sanity: the chunker's default token counter is the heuristic one, and a
    # chunk's piece never exceeds max_tokens under it.
    markdown = "## S\n\n" + "alpha beta gamma delta " * 50
    counter = HeuristicTokenCounter()
    chunks = MarkdownChunker(max_tokens=30, overlap_tokens=8, token_counter=counter).chunk(
        document_id="d", markdown=markdown, title="S"
    )
    # Hard-split pieces can overlap, so the raw char count may slightly exceed the
    # budget; but the *new* content per piece (minus the overlap window) is in budget.
    # The structural guarantee we assert: every chunk is bounded by a few x budget.
    for chunk in chunks:
        assert counter.count(chunk.text) <= 30 + 8  # budget + overlap slack
