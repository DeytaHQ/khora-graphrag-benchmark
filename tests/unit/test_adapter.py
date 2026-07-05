"""Unit tests for the Khora adapter.

These exercise the pure routing/rendering logic that does not require a live
khora engine or database. The module-level LLM helper and ``self._lake.recall``
are mocked so no network/DB/API key is ever touched.
"""

from __future__ import annotations

import types
from unittest.mock import AsyncMock, patch

import pytest

from khora_graphrag_bench.adapters.khora import (
    _DEFAULT_ENTITY_TYPES,
    _DEFAULT_RELATIONSHIP_TYPES,
    KhoraAdapter,
    _generation_params,
)
from khora_graphrag_bench.harness.base import GraphRAGAdapter, GraphSearchResult

ADAPTER_MOD = "khora_graphrag_bench.adapters.khora"


# ---------------------------------------------------------------------------
# __init__ param handling + constants
# ---------------------------------------------------------------------------


def test_default_ontology_constants() -> None:
    assert "PERSON" in _DEFAULT_ENTITY_TYPES
    assert "ORGANIZATION" in _DEFAULT_ENTITY_TYPES
    assert "CREATURE" in _DEFAULT_ENTITY_TYPES
    assert "INTERACTS_WITH" in _DEFAULT_RELATIONSHIP_TYPES
    assert "RELATES_TO" in _DEFAULT_RELATIONSHIP_TYPES
    # No duplicates in the shipped ontology.
    assert len(_DEFAULT_ENTITY_TYPES) == len(set(_DEFAULT_ENTITY_TYPES))
    assert len(_DEFAULT_RELATIONSHIP_TYPES) == len(set(_DEFAULT_RELATIONSHIP_TYPES))


def test_init_defaults() -> None:
    a = KhoraAdapter()
    assert a._params == {}
    assert a._entity_types is _DEFAULT_ENTITY_TYPES
    assert a._relationship_types is _DEFAULT_RELATIONSHIP_TYPES
    assert a._max_concurrent_llm_calls == 10
    assert a._lake is None
    assert a._namespace_id is None
    assert a._doc_id_map == {}
    assert a._last_ingestion_entities == 0
    assert a._last_ingestion_relationships == 0


def test_init_none_params_is_empty_dict() -> None:
    assert KhoraAdapter(None)._params == {}


def test_init_param_overrides() -> None:
    a = KhoraAdapter(
        params={
            "entity_types": ["FOO"],
            "relationship_types": ["BAR"],
            "max_concurrent_documents": 3,
            "max_concurrent_llm_calls": 7,
        }
    )
    assert a._entity_types == ["FOO"]
    assert a._relationship_types == ["BAR"]
    assert a._max_concurrent_documents == 3
    assert a._max_concurrent_llm_calls == 7


def test_max_concurrent_documents_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KGB_MAX_CONCURRENT_DOCUMENTS", "42")
    assert KhoraAdapter()._max_concurrent_documents == 42


def test_max_concurrent_documents_param_beats_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KGB_MAX_CONCURRENT_DOCUMENTS", "42")
    assert KhoraAdapter(params={"max_concurrent_documents": 9})._max_concurrent_documents == 9


def test_name_property() -> None:
    assert KhoraAdapter().name == "khora"


# ---------------------------------------------------------------------------
# generate_answer - context assembly, uniform prompt
# ---------------------------------------------------------------------------


def _capture_llm() -> AsyncMock:
    """An AsyncMock standing in for ``_call_llm_for_answer_with_rationale``."""
    return AsyncMock(return_value=("the answer", ["fact one", "fact two"]))


def _captured_context(mock: AsyncMock) -> str:
    """The assembled context text, passed positionally as the 2nd arg."""
    return mock.await_args.args[1]


async def test_generate_answer_returns_answer_and_rationale(sample_search_results) -> None:
    mock = _capture_llm()
    with patch(f"{ADAPTER_MOD}._call_llm_for_answer_with_rationale", mock):
        result = await KhoraAdapter().generate_answer("Who held the office?", sample_search_results)
    assert result.answer == "the answer"
    assert result.evidence == ["fact one", "fact two"]
    # context is the assembled context block, surfaced for provenance.
    assert "--- Source ---" in result.context


async def test_generate_answer_context_block_assembly(sample_search_results) -> None:
    mock = _capture_llm()
    with patch(f"{ADAPTER_MOD}._call_llm_for_answer_with_rationale", mock):
        await KhoraAdapter().generate_answer("Who held the office?", sample_search_results)

    ctx = _captured_context(mock)
    # Each source gets a "--- Source ---" header with its content.
    assert ctx.count("--- Source ---") == 2
    assert "Ovid held the office of judex selectus" in ctx
    assert "Corinna is described as Ovid's beloved" in ctx
    # source_nodes are rendered as an "Entities mentioned" line.
    assert "Entities mentioned: Ovid, judex selectus" in ctx
    assert "Entities mentioned: Corinna, Ovid" in ctx
    # source_edges are rendered once under the relationships header.
    assert "--- Relationships among the entities ---" in ctx
    assert "Ovid -held-> judex selectus" in ctx


async def test_generate_answer_no_entities_no_edges() -> None:
    """A result without source_nodes/source_edges renders just the bare source."""
    ctx_only = [GraphSearchResult(document_id="d", content="bare content", score=0.5)]
    mock = _capture_llm()
    with patch(f"{ADAPTER_MOD}._call_llm_for_answer_with_rationale", mock):
        await KhoraAdapter().generate_answer("q", ctx_only)
    ctx = _captured_context(mock)
    assert ctx == "--- Source ---\nbare content"
    assert "Entities mentioned" not in ctx
    assert "Relationships among the entities" not in ctx


async def test_generate_answer_dedups_edges() -> None:
    dupes = [
        GraphSearchResult(document_id="d1", content="a", score=0.5, source_edges=["X -r-> Y", "P -q-> Q"]),
        GraphSearchResult(document_id="d2", content="b", score=0.4, source_edges=["X -r-> Y"]),
    ]
    mock = _capture_llm()
    with patch(f"{ADAPTER_MOD}._call_llm_for_answer_with_rationale", mock):
        await KhoraAdapter().generate_answer("q", dupes)
    ctx = _captured_context(mock)
    rel_section = ctx.split("--- Relationships among the entities ---\n")[1]
    assert rel_section.count("X -r-> Y") == 1
    assert "P -q-> Q" in rel_section


async def test_generate_answer_entities_capped_at_five() -> None:
    many = [
        GraphSearchResult(
            document_id="d",
            content="c",
            score=0.5,
            source_nodes=["A", "B", "C", "D", "E", "F", "G"],
        )
    ]
    mock = _capture_llm()
    with patch(f"{ADAPTER_MOD}._call_llm_for_answer_with_rationale", mock):
        await KhoraAdapter().generate_answer("q", many)
    ctx = _captured_context(mock)
    assert "Entities mentioned: A, B, C, D, E" in ctx
    assert "F" not in ctx.split("Entities mentioned:")[1]


# ----- uniform answer prompt (no question_type routing) -----


async def _system_and_tokens(query: str, params: dict | None = None) -> tuple[str, int]:
    """Run generate_answer and return (system_prompt, max_tokens) seen by the LLM helper."""
    mock = _capture_llm()
    ctx = [GraphSearchResult(document_id="d", content="c", score=0.5)]
    with patch(f"{ADAPTER_MOD}._call_llm_for_answer_with_rationale", mock):
        await KhoraAdapter(params=params).generate_answer(query, ctx)
    kwargs = mock.await_args.kwargs
    return kwargs["system"], kwargs["max_tokens"]


async def test_uniform_prompt_same_for_every_question() -> None:
    # Every question gets the one neutral prompt + budget — no per-type routing,
    # and the adapter never sees the benchmark's question_type.
    fb_system, fb_tokens = await _system_and_tokens("Who killed the duke?")
    oe_system, oe_tokens = await _system_and_tokens("Describe the city and its people.")
    creative_system, _ = await _system_and_tokens("Write a diary entry as Ovid.")
    assert fb_system == oe_system == creative_system
    assert fb_tokens == oe_tokens == 512
    # Neutral wording — no scorer-rubric / per-type language.
    assert "ONLY the provided context" in fb_system
    assert "EVERY distinct entity" not in fb_system
    assert "ONE short sentence" not in fb_system


def test_generate_answer_does_not_accept_question_type() -> None:
    # Regression guard: the gold question_type must not be threadable into generation,
    # on either the adapter OR the protocol it implements (both were reverted once).
    import inspect

    assert "question_type" not in inspect.signature(KhoraAdapter.generate_answer).parameters
    assert "question_type" not in inspect.signature(GraphRAGAdapter.generate_answer).parameters


async def test_generate_answer_passes_llm_model_param() -> None:
    mock = _capture_llm()
    ctx = [GraphSearchResult(document_id="d", content="c", score=0.5)]
    with patch(f"{ADAPTER_MOD}._call_llm_for_answer_with_rationale", mock):
        await KhoraAdapter(params={"llm_model": "gpt-4o"}).generate_answer("Who?", ctx)
    assert mock.await_args.kwargs["model"] == "gpt-4o"


async def test_generate_answer_default_llm_model() -> None:
    mock = _capture_llm()
    ctx = [GraphSearchResult(document_id="d", content="c", score=0.5)]
    with patch(f"{ADAPTER_MOD}._call_llm_for_answer_with_rationale", mock):
        await KhoraAdapter().generate_answer("Who?", ctx)
    assert mock.await_args.kwargs["model"] == "gpt-4o-mini"


async def test_route_generation_model_param() -> None:
    mock = _capture_llm()
    ctx = [GraphSearchResult(document_id="d", content="c", score=0.5)]
    with patch(f"{ADAPTER_MOD}._call_llm_for_answer_with_rationale", mock):
        await KhoraAdapter(params={"generation_model": "gpt-5-mini"}).generate_answer("Who?", ctx)
    assert mock.await_args.kwargs["model"] == "gpt-5-mini"


async def test_route_generation_model_overrides_llm_model() -> None:
    # generation_model wins over the legacy single-knob llm_model fallback.
    mock = _capture_llm()
    ctx = [GraphSearchResult(document_id="d", content="c", score=0.5)]
    params = {"llm_model": "gpt-4o", "generation_model": "gpt-5-mini"}
    with patch(f"{ADAPTER_MOD}._call_llm_for_answer_with_rationale", mock):
        await KhoraAdapter(params=params).generate_answer("Who?", ctx)
    assert mock.await_args.kwargs["model"] == "gpt-5-mini"


# ---------------------------------------------------------------------------
# _generation_params (GPT-5 / o-series reasoning-model compatibility)
# ---------------------------------------------------------------------------


def test_generation_params_non_reasoning_keeps_temperature() -> None:
    assert _generation_params("gpt-4o-mini", 256) == {"temperature": 0.0, "max_tokens": 256}


@pytest.mark.parametrize("model", ["gpt-5-mini", "gpt-5", "o1-mini", "o3-mini", "o4-mini"])
def test_generation_params_reasoning_models_drop_temperature(model: str) -> None:
    params = _generation_params(model, 256)
    assert "temperature" not in params
    assert "max_tokens" not in params
    # Floor protects the answer from being eaten by reasoning tokens.
    assert params["max_completion_tokens"] >= 8192
    assert params["reasoning_effort"] == "low"


# ---------------------------------------------------------------------------
# graph_search - rendering with a faked recall result
# ---------------------------------------------------------------------------


def _ns(**kw) -> types.SimpleNamespace:
    return types.SimpleNamespace(**kw)


def _fake_recall_result():
    """A RecallResult-like object: chunks, documents, entities, relationships."""
    chunks = [
        _ns(id="c1", document_id="uuid-doc-1", content="Chunk one content.", score=0.9),
        _ns(id="c2", document_id="uuid-doc-1", content="Chunk two content.", score=0.8),
        # Duplicate chunk id -> should be deduped out.
        _ns(id="c1", document_id="uuid-doc-1", content="Chunk one content.", score=0.7),
        _ns(id="c3", document_id="uuid-doc-2", content="Chunk three content.", score=0.6),
    ]
    documents = [
        _ns(id="uuid-doc-1", metadata={"bench_doc_id": "bench-1"}),
        _ns(id="uuid-doc-2", metadata={"bench_doc_id": "bench-2"}),
    ]
    entities = [
        _ns(id="e1", name="Ovid"),
        _ns(id="e2", name="Corinna"),
        # A UUID-looking name is filtered from the surfaced entity list.
        _ns(id="e3", name="0badf00d-1234-5678-9abc-def012345678"),
    ]
    relationships = [
        _ns(source_entity_id="e1", target_entity_id="e2", relationship_type="LOVES", description="devoted"),
    ]
    return _ns(chunks=chunks, documents=documents, entities=entities, relationships=relationships)


def _adapter_with_fake_recall(result):
    a = KhoraAdapter()
    a._namespace_id = "ns-1"
    a._lake = _ns(recall=AsyncMock(return_value=result))
    return a


async def test_graph_search_dedup_and_mapping() -> None:
    a = _adapter_with_fake_recall(_fake_recall_result())
    out = await a.graph_search("who loves whom?", top_k=5)

    # 4 input chunks, one is a duplicate id -> 3 unique results.
    assert [r.content for r in out] == ["Chunk one content.", "Chunk two content.", "Chunk three content."]
    # Original bench doc ids are mapped back from the UUIDs.
    assert out[0].document_id == "bench-1"
    assert out[2].document_id == "bench-2"
    assert out[0].score == pytest.approx(0.9)


async def test_graph_search_top_k_truncation() -> None:
    a = _adapter_with_fake_recall(_fake_recall_result())
    out = await a.graph_search("q", top_k=2)
    assert len(out) == 2


async def test_graph_search_entities_and_edges_on_first_result() -> None:
    a = _adapter_with_fake_recall(_fake_recall_result())
    out = await a.graph_search("q", top_k=5)

    # UUID-looking entity name is filtered out of source_nodes.
    assert out[0].source_nodes == ["Ovid", "Corinna"]
    # Every result shares the same entity list.
    assert out[1].source_nodes == ["Ovid", "Corinna"]
    # Relationship triples only attach to the first result.
    assert out[0].source_edges == ["Ovid —loves→ Corinna (devoted)"]
    assert out[1].source_edges == []
    assert out[2].source_edges == []


async def test_graph_search_recall_pool_default_and_override() -> None:
    result = _fake_recall_result()
    a = _adapter_with_fake_recall(result)
    await a.graph_search("q", top_k=5)
    assert a._lake.recall.await_args.kwargs["limit"] == 50  # max(top_k, default 50)

    a2 = KhoraAdapter(params={"recall_pool": 7})
    a2._namespace_id = "ns-1"
    a2._lake = _ns(recall=AsyncMock(return_value=_fake_recall_result()))
    await a2.graph_search("q", top_k=3)
    assert a2._lake.recall.await_args.kwargs["limit"] == 7


async def test_graph_search_recall_pool_floored_at_top_k() -> None:
    a = KhoraAdapter(params={"recall_pool": 2})
    a._namespace_id = "ns-1"
    a._lake = _ns(recall=AsyncMock(return_value=_fake_recall_result()))
    await a.graph_search("q", top_k=10)
    assert a._lake.recall.await_args.kwargs["limit"] == 10  # max(top_k, recall_pool)


async def test_graph_search_evidence_is_chunk_content() -> None:
    a = _adapter_with_fake_recall(_fake_recall_result())
    out = await a.graph_search("q", top_k=5)
    assert out[0].evidence == ["Chunk one content."]


async def test_graph_search_handles_empty_result() -> None:
    empty = _ns(chunks=[], documents=[], entities=[], relationships=[])
    a = _adapter_with_fake_recall(empty)
    assert await a.graph_search("q") == []
