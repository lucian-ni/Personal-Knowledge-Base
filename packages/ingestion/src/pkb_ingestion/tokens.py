from __future__ import annotations

from typing import Protocol


class TokenCounter(Protocol):
    """Estimates how many tokens a piece of text will cost the embedding model.

    The chunker uses this to size chunks; it stays a pure, ML-free estimate so the
    ingestion library remains a data-transformation layer (no transformers import).
    A real tokenizer could be plugged in by implementing this Protocol.
    """

    def count(self, text: str) -> int: ...


def _is_cjk(ch: str) -> bool:
    """True for CJK ideographs, kana, hangul, CJK punctuation, and fullwidth forms.

    BERT-family tokenizers (bge-small-zh included) split CJK text one token per
    character, so each of these costs ~1 token. ASCII/Latin/digits/punctuation are
    WordPiece-subworded at roughly 1 token per 4 characters.
    """
    cp = ord(ch)
    return (
        0x2E80 <= cp <= 0x9FFF        # CJK radicals + Kangxi + CJK Unified + Ext A
        or 0xAC00 <= cp <= 0xD7AF     # Hangul Syllables
        or 0xF900 <= cp <= 0xFAFF     # CJK Compatibility Ideographs
        or 0xFB00 <= cp <= 0xFB4F     # Hebrew presentation (rare) - keep narrow
        or 0xFF00 <= cp <= 0xFFEF     # Fullwidth forms (！？．，　…)
        or 0x3000 <= cp <= 0x30FF     # CJK Symbols/Punct + Hiragana + Katakana
        or 0x3040 <= cp <= 0x309F     # Hiragana
        or 0x20000 <= cp <= 0x2FA1F   # CJK Ext B-F + Compatibility Supplement
    )


class HeuristicTokenCounter:
    """Token estimate tuned for bge-small-zh / BERT-family tokenizers.

    CJK and fullwidth characters count 1 token each; everything else (ASCII, Latin,
    digits, punctuation, whitespace) counts ``ceil(chars / 4)``. This is the same
    proxy rationale the old character-count used (Chinese has no spaces, so
    ``split()`` would yield ~1), but now it also discounts Latin text the way a real
    WordPiece tokenizer would - so mixed-language documents size correctly.
    """

    def count(self, text: str) -> int:
        cjk = 0
        other = 0
        for ch in text:
            if _is_cjk(ch):
                cjk += 1
            else:
                other += 1
        # ceil(other / 4): 4 non-CJK chars ~ 1 subword token.
        return cjk + (other + 3) // 4
