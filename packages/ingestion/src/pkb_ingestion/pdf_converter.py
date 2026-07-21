"""PDF -> markdown conversion via PyMuPDF.

Replaces the previous Docling converter. PyMuPDF ships self-contained wheels
(no native compile step), which is what makes it work cleanly on Windows where
Docling's C++/pdfium chain was breaking.

Phase 1 scope: **text-layer extraction only.** Born-digital PDFs (papers, ebooks,
Word exports) have a real text layer and are handled well - blocks are extracted
with their bounding boxes and font sizes, sorted into reading order, and emitted
as markdown with headings inferred from font size. Scanned pages (no text layer)
degrade gracefully: a page marker is still emitted (so page numbering stays
consistent for the chunker) plus a short note; OCR is not wired up yet. Tables,
lists, and code blocks are not specially reconstructed - they pass through as text
blocks / paragraphs. See ``docs/architecture.md``.
"""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any, Protocol

from pkb_ingestion.models import DocumentArtifact


class PdfConverter(Protocol):
    """Convert a PDF file to a :class:`DocumentArtifact` carrying its markdown."""

    def convert(
        self,
        document_id: str,
        source_path: Path,
        title: str | None = None,
    ) -> DocumentArtifact:
        ...


class PyMuPdfConverter:
    """PDF -> markdown using PyMuPDF (fitz).

    Two passes over the document:

    1. Collect every text span's font size across all pages and take the mode as
       the ``body_size``. Headings are inferred relative to this, so the detector
       adapts to each document's actual type ladder instead of hard-coded points.
    2. Emit markdown per page: a ``<!-- page: N -->`` marker, then blocks sorted
       into reading order (single column top-to-bottom, or left-then-right for
       two-column pages), each block mapped to a heading or paragraph.

    Headings are inferred conservatively from the block's median span size relative
    to ``body_size`` (median, not max, so a single bold word inside a paragraph
    does not promote it to a heading) AND a short-line guard (a real heading is
    short; a long line at a larger size stays a paragraph):

        size >= body * 1.50 -> H1   (chunker drops H1 as the document title)
        size >= body * 1.30 -> H2   (section boundary)
        size >= body * 1.08 -> H3

    PyMuPDF is imported lazily inside ``convert`` so the app starts even if the
    wheel is somehow absent, mirroring the lazy-load pattern used for the
    embedding/reranker models.
    """

    def __init__(
        self,
        *,
        min_text_chars: int = 10,
        max_heading_chars: int = 100,
    ) -> None:
        self.min_text_chars = min_text_chars
        self.max_heading_chars = max_heading_chars

    def convert(
        self,
        document_id: str,
        source_path: Path,
        title: str | None = None,
    ) -> DocumentArtifact:
        try:
            import pymupdf
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "PyMuPDF is required for PDF conversion. Install 'pymupdf'."
            ) from exc

        doc = pymupdf.open(str(source_path))
        try:
            pages: list[tuple[int, float, list[dict[str, Any]], bool]] = []
            sizes: list[float] = []
            for page_no, page in enumerate(doc, start=1):
                width = page.rect.width
                raw_text = page.get_text("text")
                is_scanned = len(raw_text.strip()) < self.min_text_chars
                blocks = [] if is_scanned else page.get_text("dict")["blocks"]
                if not is_scanned:
                    sizes.extend(self._collect_span_sizes(blocks))
                pages.append((page_no, width, blocks, is_scanned))

            body_size = statistics.multimode(sizes)[0] if sizes else 11.0

            parts: list[str] = []
            for page_no, width, blocks, is_scanned in pages:
                page_md = self._render_page(page_no, width, blocks, is_scanned, body_size)
                if page_md.strip():
                    parts.append(page_md)
        finally:
            doc.close()

        markdown = "\n\n".join(parts)
        return DocumentArtifact(
            document_id=document_id,
            title=title or source_path.stem,
            source_path=source_path,
            markdown=markdown,
        )

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _collect_span_sizes(blocks: list[dict[str, Any]]) -> list[float]:
        sizes: list[float] = []
        for block in blocks:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if span.get("text", "").strip():
                        sizes.append(float(span.get("size", 0.0)))
        return sizes

    def _render_page(
        self,
        page_no: int,
        width: float,
        blocks: list[dict[str, Any]],
        is_scanned: bool,
        body_size: float,
    ) -> str:
        parts: list[str] = [f"<!-- page: {page_no} -->"]
        if is_scanned:
            parts.append(f"*[scanned page {page_no}: no text layer; OCR not configured]*")
            return "\n\n".join(parts)

        text_blocks = [b for b in blocks if b.get("type") == 0 and b.get("lines")]
        for md in self._blocks_to_markdown(text_blocks, width, body_size):
            if md:
                parts.append(md)
        return "\n\n".join(parts)

    def _blocks_to_markdown(
        self,
        blocks: list[dict[str, Any]],
        width: float,
        body_size: float,
    ) -> list[str]:
        mid = width / 2.0
        left = [b for b in blocks if b["bbox"][2] < mid]
        right = [b for b in blocks if b["bbox"][0] > mid]
        two_column = bool(left) and bool(right)

        def sort_key(block: dict[str, Any]) -> tuple[int, float, float]:
            x0, y0, x1, _y1 = block["bbox"]
            if not two_column:
                col = 0
            elif x0 < mid and x1 > mid:
                # Full-width block (title / table) straddling the gutter: treat as
                # left column so it sorts by vertical position with the left flow.
                col = 0
            else:
                col = 0 if (x0 + x1) / 2.0 < mid else 1
            return (col, y0, x0)

        ordered = sorted(blocks, key=sort_key)
        return [self._block_to_markdown(block, body_size) for block in ordered]

    def _block_to_markdown(self, block: dict[str, Any], body_size: float) -> str:
        line_texts: list[str] = []
        span_sizes: list[float] = []
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            line_text = "".join(span.get("text", "") for span in spans).strip()
            for span in spans:
                if span.get("text", "").strip():
                    span_sizes.append(float(span.get("size", 0.0)))
            if line_text:
                line_texts.append(line_text)
        if not line_texts:
            return ""

        flat = " ".join(line_texts)
        # A real heading is short. A long line at a larger font (a pull-quote, a
        # code-heavy paragraph) stays a paragraph rather than becoming a section.
        if len(flat) > self.max_heading_chars:
            return "\n".join(line_texts)

        block_size = statistics.median(span_sizes) if span_sizes else 0.0
        level = self._heading_level(block_size, body_size)
        if level:
            return f"{'#' * level} {flat}"
        return "\n".join(line_texts)

    @staticmethod
    def _heading_level(size: float, body_size: float) -> int:
        if body_size <= 0:
            return 0
        ratio = size / body_size
        if ratio >= 1.50:
            return 1
        if ratio >= 1.30:
            return 2
        if ratio >= 1.08:
            return 3
        return 0
