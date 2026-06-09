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
)
from khora_graphrag_bench.harness.base import GraphSearchResult

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
# _detect_question_type - keyword routing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        "Write a diary entry about the journey.",
        "Compose a poem about the sea.",
        "Create a letter to the king.",
        "Please WRITE A short note.",  # case-insensitive
        "Write a letter as the protagonist.",
    ],
)
def test_detect_creative(query: str) -> None:
    assert KhoraAdapter._detect_question_type(query) == "creative"


@pytest.mark.parametrize(
    "query",
    [
        "How is the city depicted in the novel?",
        "Describe the relationship between the two families.",
        "What role does fate play in the story?",
        "Summarize the third chapter.",
        "Give an overview of the plot.",
        "How does the motif of water recur?",
    ],
)
def test_detect_summary(query: str) -> None:
    assert KhoraAdapter._detect_question_type(query) == "summary"


@pytest.mark.parametrize(
    "query",
    [
        "Who killed the duke?",
        "What is the capital named in chapter two?",
        "Which year did the war begin?",
    ],
)
def test_detect_factual(query: str) -> None:
    assert KhoraAdapter._detect_question_type(query) == "factual"


def test_detect_creative_beats_summary() -> None:
    # Has both a creative keyword ("write a") and a summary keyword ("describe").
    assert KhoraAdapter._detect_question_type("Write a paragraph that describes the hero.") == "creative"


# ---------------------------------------------------------------------------
# generate_answer - routing, context assembly, token limits
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


# ----- question_type routing -----


async def _route(query: str, question_type: str | None, params: dict | None = None) -> tuple[str, int]:
    """Run generate_answer and return (system_prompt, max_tokens) seen by the LLM helper."""
    mock = _capture_llm()
    ctx = [GraphSearchResult(document_id="d", content="c", score=0.5)]
    with patch(f"{ADAPTER_MOD}._call_llm_for_answer_with_rationale", mock):
        await KhoraAdapter(params=params).generate_answer(query, ctx, question_type=question_type)
    kwargs = mock.await_args.kwargs
    return kwargs["system"], kwargs["max_tokens"]


async def test_route_fb_is_factual() -> None:
    system, max_tokens = await _route("Who killed the duke?", "FB")
    assert "ONE short sentence" in system
    assert max_tokens == 256


async def test_route_oe_is_coverage() -> None:
    system, max_tokens = await _route("List the relationships involved.", "OE")
    assert "open-ended question" in system
    assert "EVERY distinct entity" in system
    assert max_tokens == 384


async def test_route_creative_from_wording_overrides_question_type() -> None:
    # Creative wording wins even when question_type says FB.
    system, max_tokens = await _route("Write a diary entry as Ovid.", "FB")
    assert "composing the requested piece" in system
    assert max_tokens == 1024


async def test_route_summary_fallback_when_type_missing() -> None:
    system, max_tokens = await _route("Describe the city.", None)
    assert "CONCISE and PRECISE" in system
    assert max_tokens == 512


async def test_route_factual_fallback_for_unknown_type() -> None:
    # MC/TF/MS or unknown type with non-summary wording falls back to factual.
    system, max_tokens = await _route("Who is the heir?", "MC")
    assert "ONE short sentence" in system
    assert max_tokens == 256


async def test_route_passes_llm_model_param() -> None:
    mock = _capture_llm()
    ctx = [GraphSearchResult(document_id="d", content="c", score=0.5)]
    with patch(f"{ADAPTER_MOD}._call_llm_for_answer_with_rationale", mock):
        await KhoraAdapter(params={"llm_model": "gpt-4o"}).generate_answer("Who?", ctx, question_type="FB")
    assert mock.await_args.kwargs["model"] == "gpt-4o"


async def test_route_default_llm_model() -> None:
    mock = _capture_llm()
    ctx = [GraphSearchResult(document_id="d", content="c", score=0.5)]
    with patch(f"{ADAPTER_MOD}._call_llm_for_answer_with_rationale", mock):
        await KhoraAdapter().generate_answer("Who?", ctx, question_type="FB")
    assert mock.await_args.kwargs["model"] == "gpt-4o-mini"


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
