"""Markdown AST built from markdown-it-py tokens.

The chunker walks the ordered sequence of top-level block nodes this module
produces. Each node carries the block's reconstructed source text (sliced from the
original markdown via the token ``map``), so lists/tables/code blocks are kept
whole and verbatim - no re-serialization, no lost formatting.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from markdown_it import MarkdownIt

if TYPE_CHECKING:
    from markdown_it.token import Token

PAGE_MARKER_RE = re.compile(r"<!--\s*page:\s*(\d+)\s*-->", re.IGNORECASE)

# Block kinds emitted by ``parse``.
KIND_HEADING = "heading"
KIND_PARAGRAPH = "paragraph"
KIND_LIST = "list"
KIND_CODE = "code"
KIND_TABLE = "table"
KIND_BLOCKQUOTE = "blockquote"
KIND_HR = "hr"
KIND_PAGE_MARKER = "page_marker"
KIND_OTHER = "other"


@dataclass(frozen=True)
class MdNode:
    """One top-level markdown block.

    ``source`` is the block's verbatim markdown (reconstructed from the parser's
    line map). Lists, tables, and code blocks are atomic - their internal structure
    lives inside ``source``, not as separate nodes - so the chunker treats them as
    single units (kept whole when they fit, hard-split only when oversized).
    """

    kind: str
    tag: str
    source: str
    level: int = 0
    heading: str | None = None
    page: int | None = None


_PARSER: MarkdownIt | None = None


def _default_parser() -> MarkdownIt:
    global _PARSER
    if _PARSER is None:
        # commonmark + raw HTML (so ``<!-- page: N -->`` becomes an html_block) +
        # GFM tables. Commonmark alone has no table rule.
        _PARSER = MarkdownIt("commonmark", {"html": True}).enable("table")
    return _PARSER


def parse(markdown: str, parser: MarkdownIt | None = None) -> list[MdNode]:
    """Return the document's top-level block nodes, in document order."""
    md = parser or _default_parser()
    # Normalize line endings so token ``map`` line numbers line up with our split.
    normalized = markdown.replace("\r\n", "\n").replace("\r", "\n")
    tokens = md.parse(normalized)
    lines = normalized.split("\n")

    roots: list[MdNode] = []
    depth = 0
    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]
        t = tok.type
        if depth == 0:
            if t.endswith("_open"):
                # An inline token (heading text / paragraph text) immediately
                # follows the open token at depth+1; peek it for heading text.
                inline = tokens[i + 1] if (i + 1 < n and tokens[i + 1].type == "inline") else None
                roots.append(_block_node(tok, lines, inline))
                depth += 1
            elif t in ("fence", "hr", "code_block", "html_block"):
                roots.append(_block_node(tok, lines, None))
            # Other leaf tokens (inline/text) do not appear at depth 0; skip defensively.
        elif t.endswith("_open"):
            depth += 1
        elif t.endswith("_close"):
            depth -= 1
        i += 1
    return roots


def _block_node(tok: Token, lines: list[str], inline: Token | None) -> MdNode:
    t = tok.type
    tag = tok.tag
    source = _source(tok, lines)
    if t == "heading_open":
        level = int(tag[1]) if len(tag) == 2 and tag[1:].isdigit() else 0
        heading = inline.content if inline else None
        return MdNode(KIND_HEADING, tag, source, level=level, heading=heading)
    if t in ("fence", "code_block"):
        return MdNode(KIND_CODE, tag, source)
    if t == "table_open":
        return MdNode(KIND_TABLE, tag, source)
    if t in ("bullet_list_open", "ordered_list_open"):
        return MdNode(KIND_LIST, tag, source)
    if t == "paragraph_open":
        return MdNode(KIND_PARAGRAPH, tag, source)
    if t == "blockquote_open":
        return MdNode(KIND_BLOCKQUOTE, tag, source)
    if t == "hr":
        return MdNode(KIND_HR, tag, source)
    if t == "html_block":
        match = PAGE_MARKER_RE.search(source)
        if match:
            return MdNode(KIND_PAGE_MARKER, tag, source, page=int(match.group(1)))
        return MdNode(KIND_OTHER, tag, source)
    return MdNode(KIND_OTHER, tag, source)


def _source(tok: Token, lines: list[str]) -> str:
    """Reconstruct the block's markdown from the token's ``[start, end)`` line map."""
    mp = tok.map
    if not mp:
        return tok.content
    start, end = mp[0], mp[1]
    end = min(end, len(lines))
    if start >= len(lines) or start >= end:
        return ""
    return "\n".join(lines[start:end]).rstrip()
