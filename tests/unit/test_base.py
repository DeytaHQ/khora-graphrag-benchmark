"""Unit tests for harness.base dataclasses and the GraphRAGAdapter protocol."""

from __future__ import annotations

import dataclasses

import pytest

from khora_graphrag_bench.harness.base import (
    Document,
    GeneratedAnswer,
    GraphConstructionResult,
    GraphRAGAdapter,
    GraphSearchResult,
)

# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------


def test_document_required_fields_and_defaults():
    doc = Document(doc_id="d1", content="hello")
    assert doc.doc_id == "d1"
    assert doc.content == "hello"
    assert doc.title == ""
    assert doc.metadata == {}


def test_document_full_construction():
    doc = Document(doc_id="d1", content="c", title="t", metadata={"k": "v"})
    assert doc.title == "t"
    assert doc.metadata == {"k": "v"}


def test_document_is_frozen():
    doc = Document(doc_id="d1", content="c")
    with pytest.raises(dataclasses.FrozenInstanceError):
        doc.content = "changed"


def test_document_metadata_default_not_shared():
    a = Document(doc_id="a", content="a")
    b = Document(doc_id="b", content="b")
    assert a.metadata is not b.metadata


def test_document_has_slots():
    doc = Document(doc_id="d1", content="c")
    assert not hasattr(doc, "__dict__")


# ---------------------------------------------------------------------------
# GraphConstructionResult
# ---------------------------------------------------------------------------


def test_graph_construction_result_required_and_defaults():
    res = GraphConstructionResult(num_nodes=10, num_edges=20, num_communities=3, construction_time_ms=123.4)
    assert res.num_nodes == 10
    assert res.num_edges == 20
    assert res.num_communities == 3
    assert res.construction_time_ms == 123.4
    assert res.cost_usd is None
    assert res.node_types == {}
    assert res.edge_types == {}
    assert res.metadata == {}


def test_graph_construction_result_full():
    res = GraphConstructionResult(
        num_nodes=1,
        num_edges=0,
        num_communities=0,
        construction_time_ms=0.0,
        cost_usd=1.5,
        node_types={"Person": 1},
        edge_types={"loves": 0},
        metadata={"run": "x"},
    )
    assert res.cost_usd == 1.5
    assert res.node_types == {"Person": 1}
    assert res.edge_types == {"loves": 0}
    assert res.metadata == {"run": "x"}


def test_graph_construction_result_frozen():
    res = GraphConstructionResult(num_nodes=1, num_edges=1, num_communities=1, construction_time_ms=1.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.num_nodes = 2


def test_graph_construction_result_dict_defaults_not_shared():
    a = GraphConstructionResult(num_nodes=0, num_edges=0, num_communities=0, construction_time_ms=0.0)
    b = GraphConstructionResult(num_nodes=0, num_edges=0, num_communities=0, construction_time_ms=0.0)
    assert a.node_types is not b.node_types
    assert a.edge_types is not b.edge_types


# ---------------------------------------------------------------------------
# GraphSearchResult
# ---------------------------------------------------------------------------


def test_graph_search_result_defaults():
    res = GraphSearchResult(document_id="d1", content="c", score=0.5)
    assert res.evidence == []
    assert res.source_nodes == []
    assert res.source_edges == []
    assert res.metadata is None


def test_graph_search_result_full():
    res = GraphSearchResult(
        document_id="d1",
        content="c",
        score=0.9,
        evidence=["e1"],
        source_nodes=["n1"],
        source_edges=["n1 -r-> n2"],
        metadata={"k": 1},
    )
    assert res.evidence == ["e1"]
    assert res.source_nodes == ["n1"]
    assert res.source_edges == ["n1 -r-> n2"]
    assert res.metadata == {"k": 1}


def test_graph_search_result_frozen():
    res = GraphSearchResult(document_id="d1", content="c", score=0.5)
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.content = "x"


def test_graph_search_result_list_defaults_not_shared():
    a = GraphSearchResult(document_id="a", content="a", score=0.1)
    b = GraphSearchResult(document_id="b", content="b", score=0.1)
    assert a.evidence is not b.evidence
    assert a.source_nodes is not b.source_nodes


def test_sample_search_results_fixture(sample_search_results):
    assert len(sample_search_results) == 2
    assert all(isinstance(r, GraphSearchResult) for r in sample_search_results)
    assert sample_search_results[0].score == 0.92
    assert sample_search_results[1].source_edges == []


# ---------------------------------------------------------------------------
# GeneratedAnswer
# ---------------------------------------------------------------------------


def test_generated_answer_defaults():
    ans = GeneratedAnswer(answer="42")
    assert ans.answer == "42"
    assert ans.evidence == []
    assert ans.context == ""
    assert ans.confidence is None
    assert ans.metadata == {}


def test_generated_answer_full():
    ans = GeneratedAnswer(
        answer="42",
        evidence=["fact"],
        context="ctx",
        confidence=0.8,
        metadata={"k": "v"},
    )
    assert ans.evidence == ["fact"]
    assert ans.context == "ctx"
    assert ans.confidence == 0.8
    assert ans.metadata == {"k": "v"}


def test_generated_answer_frozen():
    ans = GeneratedAnswer(answer="a")
    with pytest.raises(dataclasses.FrozenInstanceError):
        ans.answer = "b"


def test_generated_answer_defaults_not_shared():
    a = GeneratedAnswer(answer="a")
    b = GeneratedAnswer(answer="b")
    assert a.evidence is not b.evidence
    assert a.metadata is not b.metadata


# ---------------------------------------------------------------------------
# GraphRAGAdapter protocol (runtime_checkable)
# ---------------------------------------------------------------------------


class _CompleteAdapter:
    @property
    def name(self) -> str:
        return "complete"

    async def setup(self) -> None: ...

    async def teardown(self) -> None: ...

    async def build_graph(self, documents):
        return GraphConstructionResult(num_nodes=0, num_edges=0, num_communities=0, construction_time_ms=0.0)

    async def graph_search(self, query, top_k=10):
        return []

    async def generate_answer(self, query, context):
        return GeneratedAnswer(answer="")

    async def get_graph_stats(self):
        return {}


class _IncompleteAdapter:
    @property
    def name(self) -> str:
        return "incomplete"

    async def setup(self) -> None: ...


def test_complete_adapter_satisfies_protocol():
    assert isinstance(_CompleteAdapter(), GraphRAGAdapter)


def test_incomplete_adapter_does_not_satisfy_protocol():
    assert not isinstance(_IncompleteAdapter(), GraphRAGAdapter)


def test_arbitrary_object_not_adapter():
    assert not isinstance(object(), GraphRAGAdapter)


async def test_complete_adapter_methods_callable():
    adapter = _CompleteAdapter()
    assert adapter.name == "complete"
    await adapter.setup()
    result = await adapter.build_graph([])
    assert isinstance(result, GraphConstructionResult)
    assert await adapter.graph_search("q") == []
    answer = await adapter.generate_answer("q", [])
    assert isinstance(answer, GeneratedAnswer)
    assert await adapter.get_graph_stats() == {}
    await adapter.teardown()
