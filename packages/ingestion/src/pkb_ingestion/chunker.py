from __future__ import annotations

import re

from pkb_ingestion.ids import stable_chunk_id, text_checksum
from pkb_ingestion.markdown_ast import (
    KIND_HEADING,
    KIND_HR,
    KIND_PAGE_MARKER,
)
from pkb_ingestion.markdown_ast import (
    parse as parse_markdown,
)
from pkb_ingestion.models import Chunk
from pkb_ingestion.tokens import HeuristicTokenCounter, TokenCounter, _is_cjk


class MarkdownChunker:
    """AST-based markdown chunker.

    Parses markdown into block nodes (markdown-it-py), walks them in document order,
    and accumulates bufferable nodes (paragraphs, lists, blockquotes) into a chunk
    until the token budget is hit, then cuts at a node boundary. Headings update
    section metadata (and flush the buffer, so a chunk never straddles sections).
    Code blocks and tables are atomic. A single node larger than the budget is split
    by the Sentence Splitter, with a hard Token Splitter fallback for any sentence
    that is still too large.

    Token sizes come from an injectable ``TokenCounter`` (heuristic by default, tuned
    for bge-small-zh / BERT-family tokenizers); ``max_tokens`` defaults to 512 to
    match bge-small-zh-v1.5's max sequence length so chunks embed without truncation.
    """

    def __init__(
        self,
        max_tokens: int = 512,
        overlap_tokens: int = 64,
        token_counter: TokenCounter | None = None,
        parser=None,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if overlap_tokens < 0:
            raise ValueError("overlap_tokens must be non-negative")
        if overlap_tokens >= max_tokens:
            raise ValueError("overlap_tokens must be smaller than max_tokens")
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.token_counter = token_counter or HeuristicTokenCounter()
        self._parser = parser

    def chunk(self, document_id: str, markdown: str, title: str) -> list[Chunk]:
        nodes = parse_markdown(markdown, self._parser)
        chunks: list[Chunk] = []
        buffer: list[str] = []
        buffer_tokens = 0
        section: str | None = None
        page: int | None = None
        next_index = 1

        def flush() -> None:
            nonlocal buffer, buffer_tokens, next_index
            if not buffer:
                return
            text = "\n\n".join(buffer).strip()
            buffer = []
            buffer_tokens = 0
            if not text:
                return
            chunks.append(self._make_chunk(document_id, title, section, page, text, next_index))
            next_index += 1

        for node in nodes:
            kind = node.kind

            if kind == KIND_HEADING:
                # H1 is the document title (passed in separately); drop it.
                if node.level == 1:
                    continue
                # A heading is a section boundary: flush so a chunk never mixes
                # sections, then adopt the new section.
                flush()
                section = node.heading
                continue

            if kind == KIND_PAGE_MARKER:
                page = node.page
                continue

            if kind == KIND_HR:
                flush()
                continue

            src = node.source
            if not src.strip():
                continue
            node_tokens = self.token_counter.count(src)

            # A single node larger than the budget can never fit - split it directly.
            if node_tokens > self.max_tokens:
                flush()
                for piece in self._split_oversized(src):
                    if piece.strip():
                        chunks.append(
                            self._make_chunk(document_id, title, section, page, piece, next_index)
                        )
                        next_index += 1
                continue

            # Fits alone. If it would overflow the current buffer, flush first.
            if buffer and buffer_tokens + node_tokens > self.max_tokens:
                flush()
            buffer.append(src)
            buffer_tokens += node_tokens
            if buffer_tokens >= self.max_tokens:
                flush()

        flush()
        return chunks

    # -- Oversized-node splitting: Sentence Splitter -> Token Splitter -------------

    def _split_oversized(self, text: str) -> list[str]:
        """Split a too-large node into budget-sized pieces.

        First the Sentence Splitter carves the text into sentence units and greedily
        packs them up to ``max_tokens``. Any single sentence still over the budget is
        hard-cut by the Token Splitter (with overlap, so continuity is preserved).
        """
        pieces: list[str] = []
        pack: list[str] = []
        pack_tokens = 0
        for sentence in _split_sentences(text):
            st = self.token_counter.count(sentence)
            if st > self.max_tokens:
                # Flush the current pack, then hard-split this over-long sentence.
                if pack:
                    pieces.append("".join(pack).strip())
                    pack = []
                    pack_tokens = 0
                pieces.extend(self._token_split(sentence))
                continue
            if pack and pack_tokens + st > self.max_tokens:
                pieces.append("".join(pack).strip())
                pack = [sentence]
                pack_tokens = st
            else:
                pack.append(sentence)
                pack_tokens += st
        if pack:
            pieces.append("".join(pack).strip())
        return [p for p in pieces if p]

    def _token_split(self, text: str) -> list[str]:
        """Hard-cut text into pieces each <= max_tokens, with overlap_tokens overlap.

        Prefers a sentence/paragraph boundary in the back half of each window (so a
        cut lands on a clean break when possible); falls back to a hard char cut when
        no boundary is found there (e.g. one long run-on sentence).
        """
        pieces: list[str] = []
        n = len(text)
        if n == 0:
            return []
        start = 0
        while start < n:
            hard_end = self._char_offset_for_budget(text, start, self.max_tokens)
            if hard_end <= start:
                # Single char already exceeds budget (pathological config); advance one
                # char to guarantee progress.
                hard_end = start + 1
            end = self._best_break(text, start, hard_end)
            if end <= start:
                end = hard_end
            pieces.append(text[start:end])
            if end >= n:
                break
            # Slide forward by overlap (in tokens). Always make progress.
            overlap_start = self._char_offset_back(text, end, self.overlap_tokens)
            start = max(overlap_start, start + 1)
        return pieces

    def _char_offset_for_budget(self, text: str, start: int, budget: int) -> int:
        """Largest ``end`` (<= len) with ``count(text[start:end]) <= budget``."""
        cjk = 0
        other = 0
        end = start
        n = len(text)
        while end < n:
            if _is_cjk(text[end]):
                cjk += 1
            else:
                other += 1
            if cjk + (other + 3) // 4 > budget:
                break
            end += 1
        return end

    def _char_offset_back(self, text: str, end: int, budget: int) -> int:
        """Smallest ``start`` (>= 0) with ``count(text[start:end]) <= budget``.

        Used to find the overlap window: the start that yields ~``budget`` tokens of
        trailing text to repeat at the head of the next piece.
        """
        cjk = 0
        other = 0
        start = end
        while start > 0:
            if _is_cjk(text[start - 1]):
                cjk += 1
            else:
                other += 1
            if cjk + (other + 3) // 4 > budget:
                break
            start -= 1
        return start

    # Sentence/paragraph boundaries, strongest first. A blank line beats a CJK
    # sentence terminator beats an ASCII one beats a newline beats a space.
    # ``_best_break`` takes the *last* matching boundary in the back half of the
    # window so chunks stay as large as possible without exceeding ``max_tokens``.
    _BOUNDARIES: tuple[str, ...] = (
        "\n\n",
        "。",
        "．",
        ".",
        "!",
        "?",
        "！",
        "？",
        "\n",
        " ",
    )

    def _best_break(self, text: str, start: int, hard_end: int) -> int:
        if hard_end >= len(text):
            return len(text)
        window = text[start:hard_end]
        threshold = len(window) // 2
        best = -1
        sep_len = 0
        for sep in self._BOUNDARIES:
            idx = window.rfind(sep)
            if idx >= threshold and idx > best:
                best = idx
                sep_len = len(sep)
        if best != -1:
            return start + best + sep_len
        return hard_end

    def _make_chunk(
        self,
        document_id: str,
        title: str,
        section: str | None,
        page: int | None,
        text: str,
        chunk_index: int,
    ) -> Chunk:
        return Chunk(
            document_id=document_id,
            chunk_id=stable_chunk_id(document_id, chunk_index),
            chunk_index=chunk_index,
            text=text,
            title=title,
            section=section,
            page=page,
            token_count=self.token_counter.count(text),
            checksum=text_checksum(text),
        )


# Sentence Splitter: split on blank lines first, then on sentence-ending
# punctuation. CJK terminators (。．) always end a sentence; ASCII .!? only when
# followed by whitespace or end-of-string (so "3.14" / "example.com" stay intact).
# The lookahead lives inside the ASCII alternative only - 。． must split even when
# immediately followed by the next CJK character (the normal case).
_SENTENCE_END = re.compile(r"([。．]|[.!?]+(?=\s|$))")


def _split_sentences(text: str) -> list[str]:
    units: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        cur = ""
        for bit in _SENTENCE_END.split(para):
            if not bit:
                continue
            if _SENTENCE_END.fullmatch(bit):
                cur += bit
                units.append(cur)
                cur = ""
            else:
                cur += bit
        if cur.strip():
            units.append(cur)
    return units
