from __future__ import annotations

import re

from pkb_ingestion.ids import stable_chunk_id, text_checksum
from pkb_ingestion.models import Chunk

PAGE_MARKER_RE = re.compile(r"<!--\s*page:\s*(\d+)\s*-->", re.IGNORECASE)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


class MarkdownChunker:
    def __init__(self, max_chars: int = 1400, overlap_chars: int = 180) -> None:
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        if overlap_chars < 0:
            raise ValueError("overlap_chars must be non-negative")
        if overlap_chars >= max_chars:
            raise ValueError("overlap_chars must be smaller than max_chars")
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars

    def chunk(self, document_id: str, markdown: str, title: str) -> list[Chunk]:
        sections = self._split_sections(markdown)
        chunks: list[Chunk] = []

        for section_title, page, text in sections:
            for piece in self._split_text(text):
                if not piece.strip():
                    continue
                chunk_index = len(chunks) + 1
                chunks.append(
                    Chunk(
                        document_id=document_id,
                        chunk_id=stable_chunk_id(document_id, chunk_index),
                        chunk_index=chunk_index,
                        text=piece,
                        title=title,
                        section=section_title,
                        page=page,
                        checksum=text_checksum(piece),
                    )
                )

        return chunks

    def _split_sections(self, markdown: str) -> list[tuple[str | None, int | None, str]]:
        sections: list[tuple[str | None, int | None, list[str]]] = []
        current_section: str | None = None
        current_page: int | None = None
        current_lines: list[str] = []

        for raw_line in markdown.splitlines():
            line = raw_line.rstrip()
            page_match = PAGE_MARKER_RE.match(line)
            if page_match:
                current_page = int(page_match.group(1))
                continue

            heading_match = HEADING_RE.match(line)
            if heading_match:
                level = len(heading_match.group(1))
                heading = heading_match.group(2).strip()
                if level == 1:
                    continue
                if current_lines:
                    sections.append((current_section, current_page, current_lines))
                    current_lines = []
                current_section = heading
                continue

            if line:
                current_lines.append(line)

        if current_lines:
            sections.append((current_section, current_page, current_lines))

        return [
            (section, page, "\n".join(lines).strip())
            for section, page, lines in sections
            if "\n".join(lines).strip()
        ]

    def _split_text(self, text: str) -> list[str]:
        if len(text) <= self.max_chars:
            return [text]

        pieces: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + self.max_chars, len(text))
            pieces.append(text[start:end])
            if end == len(text):
                break
            start = end - self.overlap_chars
        return pieces
