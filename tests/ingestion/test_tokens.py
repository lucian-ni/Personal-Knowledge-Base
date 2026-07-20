from pkb_ingestion.tokens import HeuristicTokenCounter


def test_cjk_chars_count_one_token_each() -> None:
    counter = HeuristicTokenCounter()
    # 6 CJK chars -> 6 tokens.
    assert counter.count("你好世界测试") == 6
    # CJK punctuation is also 1 token (it is a CJK code point).
    assert counter.count("。") == 1


def test_latin_text_uses_ceil_div_4() -> None:
    counter = HeuristicTokenCounter()
    assert counter.count("a") == 1  # ceil(1/4)
    assert counter.count("abcd") == 1  # ceil(4/4)
    assert counter.count("abcde") == 2  # ceil(5/4)
    assert counter.count("abcdefgh") == 2  # ceil(8/4)


def test_mixed_cjk_and_latin() -> None:
    counter = HeuristicTokenCounter()
    # "你好" = 2 CJK tokens; "hello" = ceil(5/4) = 2 tokens -> 4 total.
    assert counter.count("你好hello") == 4


def test_empty_string_is_zero() -> None:
    assert HeuristicTokenCounter().count("") == 0
