"""Unit tests for harness.model_utils.is_reasoning_model."""

from __future__ import annotations

import pytest

from khora_graphrag_bench.harness.model_utils import is_reasoning_model


@pytest.mark.parametrize("model", ["gpt-5", "gpt-5-mini", "o1-mini", "o3-mini", "o4-mini"])
def test_reasoning_models_detected(model: str) -> None:
    assert is_reasoning_model(model) is True


@pytest.mark.parametrize("model", ["gpt-4o-mini", "gpt-4o", "gpt-4.1", "text-embedding-3-small"])
def test_non_reasoning_models_not_detected(model: str) -> None:
    assert is_reasoning_model(model) is False
