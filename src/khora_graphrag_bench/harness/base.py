"""Core types and the GraphRAGAdapter protocol.

A benchmark adapter wraps a memory system (here: Khora) and exposes a uniform
three-phase interface — graph construction, retrieval, answer generation —
that the GraphRAG-Bench evaluation pipeline drives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class Document:
    """A document to ingest into the memory system."""

    doc_id: str
    content: str
    title: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GraphConstructionResult:
    """Result of building a knowledge graph from source documents."""

    num_nodes: int
    num_edges: int
    num_communities: int
    construction_time_ms: float
    cost_usd: float | None = None
    node_types: dict[str, int] = field(default_factory=dict)
    edge_types: dict[str, int] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GraphSearchResult:
    """A graph-retrieval result with attached evidence and source attribution."""

    document_id: str
    content: str
    score: float
    evidence: list[str] = field(default_factory=list)
    source_nodes: list[str] = field(default_factory=list)
    source_edges: list[str] = field(default_factory=list)
    metadata: dict | None = None


@dataclass(frozen=True, slots=True)
class GeneratedAnswer:
    """A generated answer with focused supporting evidence.

    ``evidence`` should be a short list of 1–5 atomic factual statements that
    justify the answer (not a dump of the raw retrieved context). The
    statement-level F-beta r_score depends on this being apples-to-apples
    with the gold rationale; large evidence lists destroy precision and
    deflate the rationale score.
    """

    answer: str
    evidence: list[str] = field(default_factory=list)
    context: str = ""
    confidence: float | None = None
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class GraphRAGAdapter(Protocol):
    """Three-phase protocol that all GraphRAG-Bench adapters implement."""

    @property
    def name(self) -> str:
        """Human-readable adapter name (used in reports)."""
        ...

    async def setup(self) -> None:
        """Open connections, prepare schemas. Called once per run."""
        ...

    async def teardown(self) -> None:
        """Release connections and resources."""
        ...

    async def build_graph(self, documents: list[Document]) -> GraphConstructionResult:
        """Phase 1: ingest documents AND build the knowledge graph."""
        ...

    async def graph_search(self, query: str, top_k: int = 10) -> list[GraphSearchResult]:
        """Phase 2: graph-augmented retrieval with evidence attribution."""
        ...

    async def generate_answer(
        self, query: str, context: list[GraphSearchResult], question_type: str | None = None
    ) -> GeneratedAnswer:
        """Phase 3: generate an answer + a focused rationale list from context.

        ``question_type`` is the coarse GraphRAG-Bench category (``FB``, ``OE``,
        ...). Adapters may use it to select a label-blind answer style - e.g. a
        fewest-words prompt for short-answer ``FB`` vs. a complete-coverage
        prompt for open-ended ``OE`` - the same brevity/coverage intent a
        deployed system would infer from the question text. It MUST NOT be used
        to leak the gold answer or gold facts into generation.
        """
        ...

    async def get_graph_stats(self) -> dict[str, Any]:
        """Return structural graph statistics (num_nodes, num_edges, ...)."""
        ...
