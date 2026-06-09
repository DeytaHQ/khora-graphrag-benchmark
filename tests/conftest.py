"""Shared pytest fixtures for the benchmark test suite.

All unit tests must run without external services (no Postgres, Neo4j, network,
or API keys). Anything that needs those belongs under the ``integration`` marker.
"""

from __future__ import annotations

import pytest

from khora_graphrag_bench.harness.base import GraphSearchResult


@pytest.fixture
def sample_search_results() -> list[GraphSearchResult]:
    """A small, deterministic set of graph-search results for adapter/reporter tests."""
    return [
        GraphSearchResult(
            document_id="doc-1",
            content="Ovid held the office of judex selectus in Roman society.",
            score=0.92,
            evidence=["Ovid held the office of judex selectus in Roman society."],
            source_nodes=["Ovid", "judex selectus"],
            source_edges=["Ovid -held-> judex selectus"],
        ),
        GraphSearchResult(
            document_id="doc-2",
            content="Corinna is described as Ovid's beloved.",
            score=0.71,
            evidence=["Corinna is described as Ovid's beloved."],
            source_nodes=["Corinna", "Ovid"],
        ),
    ]
