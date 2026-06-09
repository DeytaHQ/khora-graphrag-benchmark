"""Unit tests for harness.text_utils.sanitize_text."""

from __future__ import annotations

from khora_graphrag_bench.harness.text_utils import sanitize_text


def test_empty_string_returned_unchanged():
    assert sanitize_text("") == ""


def test_none_returned_unchanged():
    # The guard is ``if not text`` so None passes straight through.
    assert sanitize_text(None) is None


def test_plain_text_unchanged():
    text = "Ovid held the office of judex selectus."
    assert sanitize_text(text) == text


def test_whitespace_preserved():
    text = "line one\tcol\nline two\r\nend"
    assert sanitize_text(text) == text


def test_null_byte_stripped():
    assert sanitize_text("a\x00b") == "ab"


def test_low_control_chars_stripped():
    # \x01..\x08 are control chars that must be removed.
    raw = "x\x01\x02\x03\x04\x05\x06\x07\x08y"
    assert sanitize_text(raw) == "xy"


def test_vertical_tab_and_form_feed_stripped():
    # \x0b (vertical tab) and \x0c (form feed) are stripped; \t \n \r are kept.
    assert sanitize_text("a\x0bb\x0cc") == "abc"


def test_chars_between_0e_and_1f_stripped():
    raw = "a\x0e\x0f\x10\x1f b"
    assert sanitize_text(raw) == "a b"


def test_del_char_stripped():
    assert sanitize_text("a\x7fb") == "ab"


def test_tab_newline_carriage_return_kept():
    raw = "a\tb\nc\rd"
    assert sanitize_text(raw) == raw


def test_unicode_preserved():
    text = "Café — naïve Ovídiō 日本語 \U0001f600"
    assert sanitize_text(text) == text


def test_unicode_with_control_chars_mixed():
    assert sanitize_text("日\x00本\x01語") == "日本語"


def test_only_control_chars_yields_empty():
    assert sanitize_text("\x00\x01\x07\x7f") == ""


def test_idempotent():
    raw = "a\x00b\x01c"
    once = sanitize_text(raw)
    assert sanitize_text(once) == once
