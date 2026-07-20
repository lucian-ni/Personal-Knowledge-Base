from pkb_ingestion.markdown_ast import (
    KIND_CODE,
    KIND_HEADING,
    KIND_HR,
    KIND_LIST,
    KIND_PAGE_MARKER,
    KIND_PARAGRAPH,
    KIND_TABLE,
    parse,
)


def test_parse_classifies_block_kinds() -> None:
    markdown = """# Title

Intro.

## Section

Para.

- a
- b

```python
x = 1
```

| x | y |
|---|---|
| 1 | 2 |

---
"""
    nodes = parse(markdown)

    assert [(n.kind, n.tag) for n in nodes] == [
        (KIND_HEADING, "h1"),
        (KIND_PARAGRAPH, "p"),
        (KIND_HEADING, "h2"),
        (KIND_PARAGRAPH, "p"),
        (KIND_LIST, "ul"),
        (KIND_CODE, "code"),
        (KIND_TABLE, "table"),
        (KIND_HR, "hr"),
    ]


def test_parse_extracts_heading_level_and_text() -> None:
    nodes = parse("# H1\n\n## H2\n\n### H3\n")
    assert [(n.level, n.heading) for n in nodes] == [(1, "H1"), (2, "H2"), (3, "H3")]


def test_parse_source_reconstruction_matches_original_slice() -> None:
    markdown = """# Title

## Section

First paragraph.

- list item one
- list item two
  with continuation
"""
    nodes = parse(markdown)
    list_node = next(n for n in nodes if n.kind == KIND_LIST)
    # The list source covers the whole list verbatim (including the continued item).
    assert "- list item one\n- list item two\n  with continuation" in list_node.source

    para_node = next(n for n in nodes if n.kind == KIND_PARAGRAPH)
    assert para_node.source == "First paragraph."


def test_parse_detects_page_markers() -> None:
    markdown = """<!-- page: 7 -->

## Section

text

<!-- page: 12 -->

more text
"""
    nodes = parse(markdown)
    markers = [n for n in nodes if n.kind == KIND_PAGE_MARKER]
    assert [m.page for m in markers] == [7, 12]


def test_parse_non_page_html_block_is_other() -> None:
    nodes = parse("<div>raw html</div>\n")
    assert nodes[0].kind != KIND_PAGE_MARKER
    assert "raw html" in nodes[0].source


def test_parse_keeps_table_and_code_verbatim() -> None:
    markdown = """## S

```python
def f():
    return 1
```

| a | b |
|---|---|
| 1 | 2 |
"""
    nodes = parse(markdown)
    code_node = next(n for n in nodes if n.kind == KIND_CODE)
    table_node = next(n for n in nodes if n.kind == KIND_TABLE)
    assert code_node.source == "```python\ndef f():\n    return 1\n```"
    assert table_node.source == "| a | b |\n|---|---|\n| 1 | 2 |"
