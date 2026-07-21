from __future__ import annotations

from pathlib import Path

from pkb_ingestion.chunker import MarkdownChunker
from pkb_ingestion.pdf_converter import PyMuPdfConverter

# A4-ish page; wide enough to host a clear two-column layout.
_PAGE_W = 595.0
_PAGE_H = 842.0
_LEFT_X = 72.0
_RIGHT_X = 320.0


def _make_pdf(path: Path, pages: list[list[tuple[float, float, str, float]]]) -> None:
    """Write a PDF whose pages are lists of (x, y, text, fontsize) text inserts.

    Synthesizing the PDF in-process keeps the test cross-platform and free of a
    committed binary fixture. PyMuPDF is deterministic and local (no model load,
    no network), so this runs under the normal pytest suite, not the integration one.
    """
    import pymupdf

    doc = pymupdf.open()
    for inserts in pages:
        page = doc.new_page(width=_PAGE_W, height=_PAGE_H)
        for x, y, text, size in inserts:
            page.insert_text((x, y), text, fontsize=size, fontname="helv")
    doc.save(str(path))
    doc.close()


def _convert(path: Path) -> str:
    artifact = PyMuPdfConverter().convert("doc-1", path, title="Doc Title")
    assert artifact.title == "Doc Title"
    return artifact.markdown


def test_headings_and_page_markers_are_emitted(tmp_path: Path) -> None:
    pdf = tmp_path / "doc.pdf"
    _make_pdf(
        pdf,
        [
            [(_LEFT_X, 80, "My Document Title", 24), (_LEFT_X, 160, "First body line.", 11)],
            [(_LEFT_X, 80, "Section Two", 16), (_LEFT_X, 160, "Second page content.", 11)],
        ],
    )

    md = _convert(pdf)

    # Page markers advance per physical page and are recognized by the chunker.
    assert "<!-- page: 1 -->" in md
    assert "<!-- page: 2 -->" in md
    # Font-size inference: 24pt over an 11pt body -> H1; 16pt -> H2.
    assert "# My Document Title" in md
    assert "## Section Two" in md
    # Body text survives as plain paragraphs.
    assert "First body line." in md
    assert "Second page content." in md
    # The H1 is NOT re-emitted as a section (chunker drops H1); sanity: no stray H4+.
    assert "####" not in md


def test_scanned_page_emits_marker_and_note(tmp_path: Path) -> None:
    pdf = tmp_path / "scan.pdf"
    _make_pdf(
        pdf,
        [
            [],  # blank page: no text layer -> scanned
            [(_LEFT_X, 160, "Has text on page two.", 11)],
        ],
    )

    md = _convert(pdf)

    # Page 1 marker still emitted so page numbering stays consistent, plus the note.
    assert "<!-- page: 1 -->" in md
    assert "scanned page 1" in md
    # Page 2 is normal.
    assert "<!-- page: 2 -->" in md
    assert "Has text on page two." in md


def test_two_column_reading_order_is_left_then_right(tmp_path: Path) -> None:
    pdf = tmp_path / "cols.pdf"
    # Left column: top (y=100) and bottom (y=400). Right column: one block at y=250.
    # A naive global y-sort would interleave as top, right, bottom - wrong.
    _make_pdf(
        pdf,
        [
            [
                (_LEFT_X, 100, "AAAA top", 11),
                (_LEFT_X, 400, "AAAA bottom", 11),
                (_RIGHT_X, 250, "BBBB mid", 11),
            ],
        ],
    )

    md = _convert(pdf)

    i_top = md.index("AAAA top")
    i_bottom = md.index("AAAA bottom")
    i_right = md.index("BBBB mid")
    # Both left-column blocks must come before the right-column block.
    assert i_top < i_bottom < i_right


def test_end_to_end_markdown_feeds_the_chunker(tmp_path: Path) -> None:
    pdf = tmp_path / "e2e.pdf"
    _make_pdf(
        pdf,
        [
            [
                (_LEFT_X, 80, "Some Title", 24),
                (_LEFT_X, 160, "Intro text.", 11),
            ],
            [
                (_LEFT_X, 80, "Section Two", 16),
                (_LEFT_X, 160, "ReentrantLock is a reentrant lock.", 11),
            ],
        ],
    )

    md = _convert(pdf)
    chunks = MarkdownChunker().chunk("doc-1", md, title="Some Title")

    assert chunks, "expected at least one chunk"
    # The page-2 marker propagates page metadata to chunks on that page.
    assert any(c.page == 2 for c in chunks)
    # The H2 becomes a section boundary carried as chunk metadata.
    assert any(c.section == "Section Two" for c in chunks)
    assert any("ReentrantLock" in c.text for c in chunks)


def test_missing_title_falls_back_to_filename_stem(tmp_path: Path) -> None:
    pdf = tmp_path / "report.pdf"
    _make_pdf(pdf, [[(_LEFT_X, 80, "Body only.", 11)]])

    artifact = PyMuPdfConverter().convert("doc-1", pdf, title=None)

    assert artifact.title == "report"
