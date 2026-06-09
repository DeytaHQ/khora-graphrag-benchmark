"""Unit tests for harness.token_counting.

tiktoken is a real local dependency and is used for real here (no network).
"""

from __future__ import annotations

import tiktoken

from khora_graphrag_bench.harness.token_counting import count_context_tokens, count_tokens


def _expected(text: str, encoding_name: str = "cl100k_base") -> int:
    return len(tiktoken.get_encoding(encoding_name).encode(text))


def test_count_tokens_empty_string_is_zero():
    assert count_tokens("") == 0


def test_count_tokens_simple_matches_encoder():
    text = "Hello, world!"
    assert count_tokens(text) == _expected(text)


def test_count_tokens_is_positive_for_nonempty():
    assert count_tokens("Ovid held the office of judex selectus.") > 0


def test_count_tokens_unicode():
    text = "日本語のテキスト 😀 café"
    assert count_tokens(text) == _expected(text)


def test_count_tokens_respects_encoding_name():
    text = "tokenization differs across encoders"
    # p50k_base is a different real local tiktoken encoding.
    assert count_tokens(text, encoding_name="p50k_base") == _expected(text, "p50k_base")


def test_count_tokens_longer_text_more_tokens():
    short = count_tokens("one")
    longer = count_tokens("one two three four five six seven eight")
    assert longer > short


def test_count_context_tokens_empty_list_is_zero():
    assert count_context_tokens([]) == 0


def test_count_context_tokens_single_chunk():
    chunks = ["Hello world"]
    assert count_context_tokens(chunks) == _expected("Hello world")


def test_count_context_tokens_is_sum_of_chunks():
    chunks = ["alpha beta", "gamma delta epsilon"]
    expected = _expected("alpha beta") + _expected("gamma delta epsilon")
    assert count_context_tokens(chunks) == expected


def test_count_context_tokens_skips_empty_entries():
    chunks = ["alpha beta", "", "gamma"]
    # Empty string contributes nothing.
    expected = _expected("alpha beta") + _expected("gamma")
    assert count_context_tokens(chunks) == expected


def test_count_context_tokens_all_empty_strings_is_zero():
    assert count_context_tokens(["", "", ""]) == 0


def test_count_context_tokens_respects_encoding_name():
    chunks = ["alpha", "beta gamma"]
    expected = _expected("alpha", "p50k_base") + _expected("beta gamma", "p50k_base")
    assert count_context_tokens(chunks, encoding_name="p50k_base") == expected


def test_count_context_tokens_matches_concatenation_components():
    # Per-chunk counting equals summing each chunk's own count (not joined text).
    chunks = ["The quick brown fox", "jumps over the lazy dog"]
    assert count_context_tokens(chunks) == sum(count_tokens(c) for c in chunks)
