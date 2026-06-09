"""Token counter for the ``mean_context_tokens`` metric.

Counts the size, in tokens, of the retrieved context the adapter injects into
its answer-LLM call. It's a useful comparative signal for memory systems:
small, relevant context is better than a verbose dump.

Encoding defaults to ``cl100k_base`` (the GPT-4o / GPT-4o-mini tokenizer that
the judge uses). Counting with the wrong encoder still gives a useful
*relative* number across runs since they all use the same tokenizer here.
"""

from __future__ import annotations

from functools import lru_cache

import tiktoken

_DEFAULT_ENCODING_NAME = "cl100k_base"


@lru_cache(maxsize=4)
def _get_encoder(encoding_name: str) -> tiktoken.Encoding:
    return tiktoken.get_encoding(encoding_name)


def count_tokens(text: str, encoding_name: str = _DEFAULT_ENCODING_NAME) -> int:
    """Return the token count of ``text`` under the given tiktoken encoding."""
    if not text:
        return 0
    return len(_get_encoder(encoding_name).encode(text))


def count_context_tokens(chunks: list[str], encoding_name: str = _DEFAULT_ENCODING_NAME) -> int:
    """Total tokens across a list of retrieved chunks. Empty entries skipped."""
    if not chunks:
        return 0
    enc = _get_encoder(encoding_name)
    return sum(len(enc.encode(c)) for c in chunks if c)
