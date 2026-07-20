from pkb_api.backends import OPENSEARCH_INDEX_MAPPING, OpenSearchKeywordBackend


def test_opensearch_keyword_backend_queries_top_level_text_field() -> None:
    """Regression: the match field must be ``text`` (top-level), not ``document.text``.

    Indexed docs store fields at the top level (bulk_index uses the inner ``document``
    dict as the _source), so ``document.text`` matched nothing and BM25 was silently
    empty - leaving "hybrid" retrieval effectively vector-only.
    """
    captured: dict = {}

    class FakeClient:
        def search(self, *, index, body):
            captured["body"] = body
            return {"hits": {"hits": []}}

    backend = OpenSearchKeywordBackend(FakeClient(), "pkb_chunks")
    backend.search("reentrant lock", limit=5)

    match = captured["body"]["query"]["match"]
    assert "text" in match
    assert "document.text" not in match


def test_opensearch_mapping_uses_cjk_analyzer_for_text_and_title() -> None:
    """Chinese BM25 needs the cjk analyzer; the default standard analyzer tokenizes
    ideograms poorly. The built-in cjk analyzer needs no plugin."""
    props = OPENSEARCH_INDEX_MAPPING["mappings"]["properties"]
    assert props["text"]["analyzer"] == "cjk"
    assert props["title"]["analyzer"] == "cjk"
    assert OPENSEARCH_INDEX_MAPPING["settings"]["analysis"]["analyzer"]["cjk"]["type"] == "cjk"


def test_opensearch_keyword_backend_degrades_to_empty_on_error() -> None:
    class BrokenClient:
        def search(self, *, index, body):
            raise RuntimeError("opensearch unreachable")

    backend = OpenSearchKeywordBackend(BrokenClient(), "pkb_chunks")

    assert backend.search("anything", limit=5) == []
