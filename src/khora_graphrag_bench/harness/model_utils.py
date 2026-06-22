"""Reasoning-model detection shared by the judge and the adapter.

OpenAI's GPT-5 and o-series reasoning models reject ``temperature`` != 1 and
meter output via ``max_completion_tokens`` instead of ``max_tokens``. Both the
judge (harness.evaluation) and answer generation (adapters.khora) branch on
this, and the CLI rejects them for extraction, so the prefix list lives in one
place to stay in sync.
"""

from __future__ import annotations

REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def is_reasoning_model(model: str) -> bool:
    """True for OpenAI reasoning models (GPT-5, o-series)."""
    return model.startswith(REASONING_MODEL_PREFIXES)
