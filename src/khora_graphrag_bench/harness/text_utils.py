"""Small text-sanitisation helpers used before sending content to LLM judges."""

from __future__ import annotations

import re

# Control characters except whitespace (\t \n \r) cause some LLM APIs to reject
# input or return malformed JSON. Strip them before any judge call.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]")


def sanitize_text(text: str) -> str:
    """Remove control characters from a string, preserving whitespace."""
    if not text:
        return text
    return _CONTROL_CHARS.sub("", text)
