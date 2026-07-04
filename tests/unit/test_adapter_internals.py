"""Unit tests for the Khora adapter's previously-untested internals.

These cover the connection/DB/LLM-touching methods by mocking every external
boundary: the khora engine (``Khora``), the schema migration's DB session
(``khora.db.get_db`` / ``init_db``), ``litellm.acompletion``, and the lake's
storage. No Postgres, Neo4j, network, or API key is ever touched.
"""

from __future__ import annotations

import contextlib
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from khora_graphrag_bench.adapters.khora import (
    KhoraAdapter,
    _call_llm_for_answer_with_rationale,
    _ensure_khora_schema,
)
from khora_graphrag_bench.harness.base import Document, GraphConstructionResult

ADAPTER_MOD = "khora_graphrag_bench.adapters.khora"


def _ns(**kw) -> types.SimpleNamespace:
    return types.SimpleNamespace(**kw)


# ---------------------------------------------------------------------------
# _ensure_khora_schema - DB session fully mocked, no real DB touched
# ---------------------------------------------------------------------------


def _patch_schema_db(*, constraint_exists: bool, delete_rowcount: int = 0):
    """Patch khora.db.get_db/init_db with a fake async session.

    Returns (init_db_mock, session_mock) and a context manager to install them.
    The session's ``execute`` returns a result whose ``.first()`` reflects
    whether the constraint already exists, and whose ``.rowcount`` reflects the
    dedup DELETE count.
    """
    init_db = AsyncMock()

    session = MagicMock()
    # First execute() is the SELECT pg_constraint; subsequent are DELETE/ALTER.
    select_result = MagicMock()
    select_result.first.return_value = (1,) if constraint_exists else None
    delete_result = MagicMock()
    delete_result.rowcount = delete_rowcount
    alter_result = MagicMock()
    session.execute = AsyncMock(side_effect=[select_result, delete_result, alter_result])
    session.commit = AsyncMock()

    @contextlib.asynccontextmanager
    async def fake_get_db():
        yield session

    fake_db_mod = types.SimpleNamespace(get_db=fake_get_db, init_db=init_db)
    return init_db, session, fake_db_mod


async def test_ensure_schema_constraint_already_exists() -> None:
    init_db, session, fake_db_mod = _patch_schema_db(constraint_exists=True)
    with patch.dict("sys.modules", {"khora.db": fake_db_mod}):
        await _ensure_khora_schema()

    init_db.assert_awaited_once()
    # Early return: only the SELECT runs, no DELETE/ALTER, no commit.
    assert session.execute.await_count == 1
    session.commit.assert_not_awaited()


async def test_ensure_schema_creates_constraint() -> None:
    init_db, session, fake_db_mod = _patch_schema_db(constraint_exists=False, delete_rowcount=3)
    with patch.dict("sys.modules", {"khora.db": fake_db_mod}):
        await _ensure_khora_schema()

    init_db.assert_awaited_once()
    # SELECT + DELETE + ALTER all run, then commit.
    assert session.execute.await_count == 3
    session.commit.assert_awaited_once()


async def test_ensure_schema_creates_constraint_no_dedup() -> None:
    """rowcount == 0 skips the info log branch but still adds the constraint."""
    init_db, session, fake_db_mod = _patch_schema_db(constraint_exists=False, delete_rowcount=0)
    with patch.dict("sys.modules", {"khora.db": fake_db_mod}):
        await _ensure_khora_schema()
    assert session.execute.await_count == 3
    session.commit.assert_awaited_once()


# ---------------------------------------------------------------------------
# _call_llm_for_answer_with_rationale - litellm.acompletion mocked
# ---------------------------------------------------------------------------


def _resp(content: str):
    """A litellm-style completion response with one choice."""
    return _ns(choices=[_ns(message=_ns(content=content))])


def _patch_litellm(acompletion: AsyncMock):
    """Patch the ``litellm`` module imported inside the helper."""
    fake = types.SimpleNamespace(acompletion=acompletion)
    return patch.dict("sys.modules", {"litellm": fake})


async def test_llm_json_mode_success_list_rationale() -> None:
    raw = '{"answer": "Ovid", "rationale": ["he held office", "he wrote poems"]}'
    aco = AsyncMock(return_value=_resp(raw))
    with _patch_litellm(aco):
        answer, rationale = await _call_llm_for_answer_with_rationale("q", "ctx", system="sys")
    assert answer == "Ovid"
    assert rationale == ["he held office", "he wrote poems"]
    # JSON mode used response_format on the (single) call.
    assert aco.await_count == 1
    assert aco.await_args.kwargs["response_format"] == {"type": "json_object"}


async def test_llm_json_rationale_as_str() -> None:
    raw = '{"answer": "Ovid", "rationale": "a single statement"}'
    aco = AsyncMock(return_value=_resp(raw))
    with _patch_litellm(aco):
        answer, rationale = await _call_llm_for_answer_with_rationale("q", "ctx", system="sys")
    assert answer == "Ovid"
    assert rationale == ["a single statement"]


async def test_llm_json_rationale_empty_str() -> None:
    raw = '{"answer": "Ovid", "rationale": "   "}'
    aco = AsyncMock(return_value=_resp(raw))
    with _patch_litellm(aco):
        answer, rationale = await _call_llm_for_answer_with_rationale("q", "ctx", system="sys")
    assert answer == "Ovid"
    assert rationale == []


async def test_llm_regex_fallback_parse() -> None:
    """Non-JSON prose wrapping a JSON object is recovered via regex."""
    raw = 'Sure! Here is the result:\n{"answer": "Corinna", "rationale": ["beloved of Ovid"]}\nDone.'
    aco = AsyncMock(return_value=_resp(raw))
    with _patch_litellm(aco):
        answer, rationale = await _call_llm_for_answer_with_rationale("q", "ctx", system="sys")
    assert answer == "Corinna"
    assert rationale == ["beloved of Ovid"]


async def test_llm_unparseable_falls_back_to_raw_answer() -> None:
    """No JSON at all: answer becomes the stripped raw text, rationale empty."""
    raw = "  just plain text, no json here  "
    aco = AsyncMock(return_value=_resp(raw))
    with _patch_litellm(aco):
        answer, rationale = await _call_llm_for_answer_with_rationale("q", "ctx", system="sys")
    assert answer == "just plain text, no json here"
    assert rationale == []


async def test_llm_regex_match_still_invalid_json() -> None:
    """A brace-bounded blob that is not valid JSON -> data={}, raw fallback."""
    raw = "prefix {not: valid, json} suffix"
    aco = AsyncMock(return_value=_resp(raw))
    with _patch_litellm(aco):
        answer, rationale = await _call_llm_for_answer_with_rationale("q", "ctx", system="sys")
    assert answer == raw.strip()
    assert rationale == []


async def test_llm_empty_answer_in_json_falls_back_to_raw() -> None:
    """JSON parses but answer is blank -> fall back to the raw blob."""
    raw = '{"answer": "", "rationale": ["x"]}'
    aco = AsyncMock(return_value=_resp(raw))
    with _patch_litellm(aco):
        answer, rationale = await _call_llm_for_answer_with_rationale("q", "ctx", system="sys")
    assert answer == raw  # raw is the whole JSON string, stripped (no surrounding ws)
    assert rationale == ["x"]


async def test_llm_json_mode_exception_then_plaintext_fallback() -> None:
    """JSON-mode call raises; plain-text retry succeeds."""
    aco = AsyncMock(side_effect=[RuntimeError("json mode unsupported"), _resp("plain answer")])
    with _patch_litellm(aco):
        answer, rationale = await _call_llm_for_answer_with_rationale("q", "ctx", system="sys")
    assert answer == "plain answer"
    assert rationale == []
    assert aco.await_count == 2
    # Second (fallback) call must NOT pass response_format.
    assert "response_format" not in aco.await_args_list[1].kwargs


async def test_llm_both_calls_fail_returns_empty() -> None:
    aco = AsyncMock(side_effect=[RuntimeError("boom1"), RuntimeError("boom2")])
    with _patch_litellm(aco):
        answer, rationale = await _call_llm_for_answer_with_rationale("q", "ctx", system="sys")
    assert answer == ""
    assert rationale == []
    assert aco.await_count == 2


async def test_llm_none_content_unparseable_empty() -> None:
    """response content None -> raw '' -> answer '' (empty)."""
    aco = AsyncMock(return_value=_resp(None))
    with _patch_litellm(aco):
        answer, rationale = await _call_llm_for_answer_with_rationale("q", "ctx", system="sys")
    assert answer == ""
    assert rationale == []


# ---------------------------------------------------------------------------
# adapter_version property
# ---------------------------------------------------------------------------


async def test_adapter_version_returns_installed_version() -> None:
    # version() is imported inside the property from importlib.metadata.
    with patch("importlib.metadata.version", return_value="9.9.9"):
        assert KhoraAdapter().adapter_version == "9.9.9"


async def test_adapter_version_missing_package_returns_empty() -> None:
    from importlib.metadata import PackageNotFoundError

    with patch("importlib.metadata.version", side_effect=PackageNotFoundError("khora")):
        assert KhoraAdapter().adapter_version == ""


# ---------------------------------------------------------------------------
# setup() - real khora config objects, mocked Khora engine + schema
# ---------------------------------------------------------------------------


def _patch_khora_engine():
    """Build a fake ``Khora`` class whose instance has an async lifecycle.

    Returns (KhoraClass_mock, lake_instance_mock). Patch the real ``khora``
    module's ``Khora`` attribute (not the whole module) so the sibling
    ``khora.config`` / ``khora.engines`` imports in ``setup`` still resolve.
    """
    lake = MagicMock()
    lake.__aenter__ = AsyncMock(return_value=lake)
    lake.create_namespace = AsyncMock(return_value=_ns(namespace_id="ns-abc"))
    khora_cls = MagicMock(return_value=lake)
    return khora_cls, lake


async def test_setup_builds_config_and_sets_namespace() -> None:
    khora_cls, lake = _patch_khora_engine()
    adapter = KhoraAdapter()
    with (
        patch("khora.Khora", khora_cls),
        patch(f"{ADAPTER_MOD}._ensure_khora_schema", AsyncMock()) as schema,
    ):
        await adapter.setup()

    schema.assert_awaited_once()
    lake.__aenter__.assert_awaited_once()
    lake.create_namespace.assert_awaited_once()
    assert adapter._lake is lake
    assert adapter._namespace_id == "ns-abc"

    # Khora constructed with engine="vectorcypher" and a real VectorCypherConfig.
    assert khora_cls.call_args.kwargs["engine"] == "vectorcypher"
    from khora.engines.vectorcypher.engine import VectorCypherConfig

    vc = khora_cls.call_args.kwargs["engine_kwargs"]["vectorcypher_config"]
    assert isinstance(vc, VectorCypherConfig)


async def test_setup_applies_param_overrides() -> None:
    khora_cls, lake = _patch_khora_engine()
    adapter = KhoraAdapter(
        params={
            "llm_model": "gpt-4o",
            "embedding_dimension": 3072,
            "selective_extraction": True,
            "stage1_recall_limit": 99,
            "skeleton_core_ratio": 0.5,
            "min_extraction_tokens": 16,
            "enable_reranking": False,
            "max_concurrent_llm_calls": 4,
        }
    )
    with (
        patch("khora.Khora", khora_cls),
        patch(f"{ADAPTER_MOD}._ensure_khora_schema", AsyncMock()),
    ):
        await adapter.setup()

    config = khora_cls.call_args.args[0]
    assert config.llm.model == "gpt-4o"
    assert config.llm.embedding_dimension == 3072
    assert config.llm.max_concurrent_llm_calls == 4
    assert config.pipelines.selective_extraction is True
    assert config.query.stage1_recall_limit == 99

    vc = khora_cls.call_args.kwargs["engine_kwargs"]["vectorcypher_config"]
    assert vc.skeleton_core_ratio == 0.5
    assert vc.min_extraction_tokens == 16
    assert vc.enable_reranking is False


# ---------------------------------------------------------------------------
# teardown()
# ---------------------------------------------------------------------------


async def test_teardown_closes_lake_and_resets_state() -> None:
    adapter = KhoraAdapter()
    lake = MagicMock()
    lake.__aexit__ = AsyncMock()
    adapter._lake = lake
    adapter._namespace_id = "ns-1"
    adapter._doc_id_map = {"a": "b"}

    await adapter.teardown()

    lake.__aexit__.assert_awaited_once_with(None, None, None)
    assert adapter._lake is None
    assert adapter._namespace_id is None
    assert adapter._doc_id_map == {}


async def test_teardown_noop_when_lake_none() -> None:
    adapter = KhoraAdapter()
    await adapter.teardown()  # should not raise
    assert adapter._lake is None


# ---------------------------------------------------------------------------
# build_graph / _ingest_documents - mocked lake + storage
# ---------------------------------------------------------------------------


def _adapter_with_ingest_lake(*, entities: int, relationships: int, stored_docs):
    """Adapter wired to a fake lake supporting remember_batch + storage.list_*."""
    adapter = KhoraAdapter()
    adapter._namespace_id = "ns-1"

    remember_result = _ns(processed=2, chunks=5, entities=entities, relationships=relationships)
    storage = _ns(
        list_documents=AsyncMock(return_value=stored_docs),
        count_entities=AsyncMock(return_value=entities),
        count_relationships=AsyncMock(return_value=relationships),
        get_communities=AsyncMock(return_value=[]),
    )
    adapter._lake = _ns(
        remember_batch=AsyncMock(return_value=remember_result),
        storage=storage,
        _resolve_namespace=AsyncMock(return_value="ns-1"),
    )
    return adapter


async def test_build_graph_returns_construction_result() -> None:
    stored = [
        _ns(id="uuid-1", metadata={"bench_doc_id": "bench-1"}),
        _ns(id="uuid-2", metadata={"bench_doc_id": "bench-2"}),
    ]
    adapter = _adapter_with_ingest_lake(entities=10, relationships=4, stored_docs=stored)
    docs = [
        Document(doc_id="bench-1", content="content one", title="T1", metadata={"k": "v"}),
        Document(doc_id="bench-2", content="content two"),
    ]
    result = await adapter.build_graph(docs)

    assert isinstance(result, GraphConstructionResult)
    assert result.num_nodes == 10
    assert result.num_edges == 4
    assert result.construction_time_ms >= 0.0

    # remember_batch fed the ontology types and concurrency.
    kwargs = adapter._lake.remember_batch.await_args.kwargs
    assert kwargs["namespace"] == "ns-1"
    assert kwargs["entity_types"] == adapter._entity_types
    assert kwargs["relationship_types"] == adapter._relationship_types

    # doc_id_map built from stored documents (UUID -> bench id, bench id -> bench id).
    assert adapter._doc_id_map["uuid-1"] == "bench-1"
    assert adapter._doc_id_map["bench-1"] == "bench-1"


async def test_build_graph_falls_back_to_ingestion_counts() -> None:
    """When storage reports 0 entities, fall back to remember_batch counts."""
    adapter = KhoraAdapter()
    adapter._namespace_id = "ns-1"
    remember_result = _ns(processed=1, chunks=2, entities=7, relationships=3)
    storage = _ns(
        list_documents=AsyncMock(return_value=[]),
        count_entities=AsyncMock(return_value=0),  # 0 -> triggers fallback
        count_relationships=AsyncMock(return_value=0),
        get_communities=AsyncMock(return_value=[]),
    )
    adapter._lake = _ns(
        remember_batch=AsyncMock(return_value=remember_result),
        storage=storage,
        _resolve_namespace=AsyncMock(return_value="ns-1"),
    )

    result = await adapter.build_graph([Document(doc_id="d1", content="x")])
    assert result.num_nodes == 7
    assert result.num_edges == 3


async def test_ingest_skips_docs_without_bench_id() -> None:
    stored = [
        _ns(id="uuid-1", metadata={"bench_doc_id": "bench-1"}),
        _ns(id="uuid-2", metadata={}),  # no bench id -> skipped
        _ns(id="uuid-3", metadata="not-a-dict"),  # non-dict metadata -> {} -> skipped
    ]
    adapter = _adapter_with_ingest_lake(entities=1, relationships=0, stored_docs=stored)
    await adapter.build_graph([Document(doc_id="bench-1", content="x")])
    assert "uuid-1" in adapter._doc_id_map
    assert "uuid-2" not in adapter._doc_id_map
    assert "uuid-3" not in adapter._doc_id_map


# ---------------------------------------------------------------------------
# get_graph_stats - mocked storage + _resolve_namespace
# ---------------------------------------------------------------------------


def _stats_storage(*, entities: int, relationships: int, communities: int = 0) -> object:
    """Fake storage exposing the server-side count APIs get_graph_stats now uses."""
    return _ns(
        count_entities=AsyncMock(return_value=entities),
        count_relationships=AsyncMock(return_value=relationships),
        get_communities=AsyncMock(return_value=[object()] * communities),
    )


async def test_get_graph_stats_success() -> None:
    adapter = KhoraAdapter()
    adapter._namespace_id = "ns-1"
    storage = _stats_storage(entities=4, relationships=3)
    adapter._lake = _ns(storage=storage, _resolve_namespace=AsyncMock(return_value="resolved-ns"))

    stats = await adapter.get_graph_stats()
    assert stats["num_nodes"] == 4
    assert stats["num_edges"] == 3
    assert stats["num_communities"] == 0
    assert stats["avg_degree"] == pytest.approx((2 * 3) / 4)
    assert stats["connectivity"] == pytest.approx(3 / (4 * 3))
    # resolved id was used for the server-side count queries (no limit cap).
    storage.count_entities.assert_awaited_once_with("resolved-ns")
    storage.count_relationships.assert_awaited_once_with("resolved-ns")


async def test_get_graph_stats_counts_communities() -> None:
    """num_communities reflects the real materialized-community count, not a literal 0."""
    adapter = KhoraAdapter()
    adapter._namespace_id = "ns-1"
    storage = _stats_storage(entities=10, relationships=8, communities=3)
    adapter._lake = _ns(storage=storage, _resolve_namespace=AsyncMock(return_value="ns-1"))

    stats = await adapter.get_graph_stats()
    assert stats["num_communities"] == 3
    # Paginated (not the capped default limit) so the count can't be silently truncated.
    storage.get_communities.assert_awaited_once_with("ns-1", limit=500, offset=0)


async def test_get_graph_stats_paginates_communities() -> None:
    """A full first page triggers a second get_communities call; counts sum across pages."""
    adapter = KhoraAdapter()
    adapter._namespace_id = "ns-1"
    pages = [[object()] * 500, [object()] * 7]  # full page -> partial page -> stop
    storage = _ns(
        count_entities=AsyncMock(return_value=100),
        count_relationships=AsyncMock(return_value=80),
        get_communities=AsyncMock(side_effect=pages),
    )
    adapter._lake = _ns(storage=storage, _resolve_namespace=AsyncMock(return_value="ns-1"))

    stats = await adapter.get_graph_stats()
    assert stats["num_communities"] == 507
    assert storage.get_communities.await_count == 2
    assert storage.get_communities.await_args_list[1].kwargs == {"limit": 500, "offset": 500}


async def test_get_graph_stats_community_failure_degrades_to_zero() -> None:
    """A failing community query still yields valid node/edge counts."""
    adapter = KhoraAdapter()
    adapter._namespace_id = "ns-1"
    storage = _ns(
        count_entities=AsyncMock(return_value=5),
        count_relationships=AsyncMock(return_value=4),
        get_communities=AsyncMock(side_effect=RuntimeError("no dream graph")),
    )
    adapter._lake = _ns(storage=storage, _resolve_namespace=AsyncMock(return_value="ns-1"))

    stats = await adapter.get_graph_stats()
    assert stats["num_nodes"] == 5
    assert stats["num_edges"] == 4
    assert stats["num_communities"] == 0


async def test_get_graph_stats_resolve_failure_uses_stable_id() -> None:
    """_resolve_namespace raising falls back to the stored namespace id."""
    adapter = KhoraAdapter()
    adapter._namespace_id = "ns-stable"
    storage = _stats_storage(entities=2, relationships=1)
    adapter._lake = _ns(
        storage=storage,
        _resolve_namespace=AsyncMock(side_effect=RuntimeError("cannot resolve")),
    )

    stats = await adapter.get_graph_stats()
    assert stats["num_nodes"] == 2
    assert stats["num_edges"] == 1
    storage.count_entities.assert_awaited_once_with("ns-stable")


async def test_get_graph_stats_empty_graph_zero_division_guards() -> None:
    adapter = KhoraAdapter()
    adapter._namespace_id = "ns-1"
    storage = _stats_storage(entities=0, relationships=0)
    adapter._lake = _ns(storage=storage, _resolve_namespace=AsyncMock(return_value="ns-1"))

    stats = await adapter.get_graph_stats()
    assert stats["num_nodes"] == 0
    assert stats["avg_degree"] == 0.0
    assert stats["connectivity"] == 0.0


async def test_get_graph_stats_storage_failure_fallback() -> None:
    """storage.count_entities raising -> the outer except returns the empty dict."""
    adapter = KhoraAdapter()
    adapter._namespace_id = "ns-1"
    storage = _ns(
        count_entities=AsyncMock(side_effect=RuntimeError("db down")),
        count_relationships=AsyncMock(return_value=0),
        get_communities=AsyncMock(return_value=[]),
    )
    adapter._lake = _ns(storage=storage, _resolve_namespace=AsyncMock(return_value="ns-1"))

    stats = await adapter.get_graph_stats()
    assert stats == {"num_nodes": 0, "num_edges": 0, "num_communities": 0}
