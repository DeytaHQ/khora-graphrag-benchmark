"""Unit tests for the benchmark runner orchestration.

The runner is driven against a fully mocked ``GraphRAGAdapter`` and patched
LLM-judge evaluation functions so no DB, network, or LLM is used. The pure
scorers (exact-match MC/TF, rouge_l, ar_metric) run for real.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from khora_graphrag_bench.datasets.schema import (
    DatasetDocument,
    GraphRAGDataset,
    GraphRAGQuestion,
)
from khora_graphrag_bench.harness import runner as runner_mod
from khora_graphrag_bench.harness.base import (
    GeneratedAnswer,
    GraphConstructionResult,
    GraphSearchResult,
)
from khora_graphrag_bench.harness.results import BenchmarkRunResult
from khora_graphrag_bench.harness.runner import BenchmarkRunner

# ---------------------------------------------------------------------------
# Fixtures: fake adapter + tiny in-memory dataset
# ---------------------------------------------------------------------------


def _make_adapter(*, num_nodes: int = 12, num_edges: int = 20) -> AsyncMock:
    """A fully mocked adapter implementing the GraphRAGAdapter protocol."""
    adapter = AsyncMock()
    adapter.name = "fake-adapter"
    adapter.adapter_version = "9.9.9"

    adapter.setup = AsyncMock(return_value=None)
    adapter.teardown = AsyncMock(return_value=None)
    adapter.build_graph = AsyncMock(
        return_value=GraphConstructionResult(
            num_nodes=num_nodes,
            num_edges=num_edges,
            num_communities=3,
            construction_time_ms=123.0,
        )
    )
    adapter.graph_search = AsyncMock(
        return_value=[
            GraphSearchResult(
                document_id="d1",
                content="Paris is the capital of France.",
                score=0.9,
                evidence=["Paris is the capital of France."],
                source_nodes=["Paris", "France"],
            ),
            GraphSearchResult(
                document_id="d2",
                content="France is a country in Europe.",
                score=0.7,
                evidence=[],
            ),
        ]
    )
    adapter.generate_answer = AsyncMock(
        return_value=GeneratedAnswer(
            answer="Paris",
            evidence=["Paris is the capital of France."],
        )
    )
    adapter.get_graph_stats = AsyncMock(return_value={"num_nodes": num_nodes})
    return adapter


def _make_dataset() -> GraphRAGDataset:
    """Dataset spanning all four question types and several difficulties."""
    docs = [
        DatasetDocument(doc_id="d1", content="Paris is the capital of France.", title="Geo"),
        DatasetDocument(doc_id="d2", content="France is a country in Europe.", title="Geo2"),
    ]
    questions = [
        GraphRAGQuestion(
            question_id="q-mc",
            question="What is the capital of France?",
            question_type="MC",
            difficulty="fact_retrieval",
            discipline="geography",
            gold_answer="A",
            options=["A. Paris", "B. London", "C. Rome", "D. Berlin"],
            evidence=["Paris is the capital of France."],
            relevant_doc_ids=["d1"],
        ),
        GraphRAGQuestion(
            question_id="q-tf",
            question="Is Paris in France?",
            question_type="TF",
            difficulty="complex_reasoning",
            discipline="geography",
            gold_answer="True",
            evidence=["Paris is the capital of France."],
            relevant_doc_ids=["d1"],
        ),
        GraphRAGQuestion(
            question_id="q-fb",
            question="The capital of France is ____.",
            question_type="FB",
            difficulty="contextual_summarization",
            discipline="geography",
            gold_answer="Paris",
            evidence=["Paris is the capital of France."],
            relevant_doc_ids=["d1"],
        ),
        GraphRAGQuestion(
            question_id="q-oe",
            question="Describe the capital of France.",
            question_type="OE",
            difficulty="creative_generation",
            discipline="geography",
            gold_answer="Paris is the capital of France.",
            evidence=["Paris is the capital of France."],
            relevant_doc_ids=["d1", "d2"],
        ),
    ]
    return GraphRAGDataset(
        name="tiny",
        documents=docs,
        questions=questions,
        entity_types=["LOCATION"],
        relationship_types=["CAPITAL_OF"],
    )


@pytest.fixture
def patch_judges(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch every async LLM-judge function the runner calls to fixed values."""
    monkeypatch.setattr(runner_mod, "compute_answer_correctness_llm", AsyncMock(return_value=0.8))
    monkeypatch.setattr(runner_mod, "compute_r_score", AsyncMock(return_value=0.6))
    monkeypatch.setattr(runner_mod, "compute_context_relevance", AsyncMock(return_value=0.7))
    monkeypatch.setattr(runner_mod, "compute_evidence_recall", AsyncMock(return_value=0.5))
    monkeypatch.setattr(runner_mod, "compute_coverage_score", AsyncMock(return_value=0.4))
    monkeypatch.setattr(runner_mod, "compute_faithfulness_score", AsyncMock(return_value=0.9))


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_invalid_sample_mode_raises() -> None:
    with pytest.raises(ValueError, match="sample_mode"):
        BenchmarkRunner(_make_adapter(), _make_dataset(), sample_mode="bogus")


# ---------------------------------------------------------------------------
# Happy path: full orchestration
# ---------------------------------------------------------------------------


async def test_run_full_pipeline(patch_judges: None) -> None:
    adapter = _make_adapter()
    runner = BenchmarkRunner(adapter, _make_dataset(), sample_mode="full", top_k=3)

    result = await runner.run()

    assert isinstance(result, BenchmarkRunResult)
    assert result.adapter_name == "fake-adapter"
    assert result.dataset_name == "tiny"
    assert result.khora_version == "9.9.9"
    assert result.num_questions == 4
    assert result.num_documents == 2
    assert len(result.per_question) == 4
    assert result.errors == []

    # Lifecycle calls.
    adapter.setup.assert_awaited_once()
    adapter.teardown.assert_awaited_once()
    adapter.build_graph.assert_awaited_once()
    assert adapter.graph_search.await_count == 4
    # top_k forwarded.
    for call in adapter.graph_search.await_args_list:
        assert call.kwargs["top_k"] == 3

    # Construction detail populated with derived metrics.
    assert result.construction is not None
    assert result.construction.num_nodes == 12
    assert result.construction.avg_degree > 0

    # Aggregates computed.
    agg = result.aggregate_metrics
    assert "accuracy" in agg
    assert "mean_answer_score" in agg
    assert agg["mean_r_score"] == pytest.approx(0.6)
    assert "mean_context_tokens" in agg
    # Auxiliary metrics averaged across questions that produced them.
    assert agg["context_relevance"] == pytest.approx(0.7)
    assert agg["evidence_recall"] == pytest.approx(0.5)


async def test_question_type_branches(patch_judges: None) -> None:
    """MC/TF use deterministic scoring; FB/OE go through the LLM judge."""
    adapter = _make_adapter()

    # generate_answer now receives the label-blind question_type; shape the
    # answer per type so the deterministic MC/TF scorers can match. Routing on
    # question_type also guards that the runner actually threads it through.
    async def _gen(query, context, question_type=None):  # noqa: ARG001
        qt = (question_type or "").upper()
        if qt == "MC":
            return GeneratedAnswer(answer="A", evidence=["Paris is the capital of France."])
        if qt == "TF":
            return GeneratedAnswer(answer="True", evidence=["Paris is the capital of France."])
        return GeneratedAnswer(answer="Paris", evidence=["Paris is the capital of France."])

    adapter.generate_answer = AsyncMock(side_effect=_gen)
    runner = BenchmarkRunner(adapter, _make_dataset(), sample_mode="full")
    result = await runner.run()

    by_id = {r.question_id: r for r in result.per_question}

    # MC: answer letter "A" exact-matches gold "A".
    assert by_id["q-mc"].answer_score == pytest.approx(1.0)
    # TF: answer "True" normalises to gold "True".
    assert by_id["q-tf"].answer_score == pytest.approx(1.0)
    # FB/OE: LLM judge mocked to 0.8.
    assert by_id["q-fb"].answer_score == pytest.approx(0.8)
    assert by_id["q-oe"].answer_score == pytest.approx(0.8)

    # ar_metric is derived from answer_score + r_score for every question.
    for r in result.per_question:
        assert r.ar_metric >= 0.0

    # Difficulty-driven aux metrics: creative_generation gets faithfulness.
    assert "faithfulness" in by_id["q-oe"].retrieval_metrics
    # contextual_summarization gets coverage.
    assert "coverage" in by_id["q-fb"].retrieval_metrics
    # fact_retrieval gets rouge_l.
    assert "rouge_l" in by_id["q-mc"].retrieval_metrics


async def test_runner_threads_question_type_to_generate(patch_judges: None) -> None:
    """The runner passes each question's label-blind question_type to generate_answer."""
    adapter = _make_adapter()
    seen: list[str | None] = []

    async def _gen(query, context, question_type=None):  # noqa: ARG001
        seen.append(question_type)
        return GeneratedAnswer(answer="Paris", evidence=["Paris is the capital of France."])

    adapter.generate_answer = AsyncMock(side_effect=_gen)
    runner = BenchmarkRunner(adapter, _make_dataset(), sample_mode="full")
    await runner.run()

    assert set(seen) == {"MC", "TF", "FB", "OE"}


async def test_breakdowns_populated(patch_judges: None) -> None:
    runner = BenchmarkRunner(_make_adapter(), _make_dataset(), sample_mode="full")
    result = await runner.run()

    assert set(result.by_question_type) == {"MC", "TF", "FB", "OE"}
    assert set(result.by_difficulty) == {
        "fact_retrieval",
        "complex_reasoning",
        "contextual_summarization",
        "creative_generation",
    }
    for grp in result.by_question_type.values():
        assert grp["n"] == 1.0
        assert "accuracy" in grp


# ---------------------------------------------------------------------------
# Error path: a question raises during retrieval
# ---------------------------------------------------------------------------


async def test_question_error_recorded(patch_judges: None) -> None:
    adapter = _make_adapter()
    adapter.graph_search = AsyncMock(side_effect=RuntimeError("boom"))
    runner = BenchmarkRunner(adapter, _make_dataset(), sample_mode="full")

    result = await runner.run()

    # Every question failed -> recorded as errors, none valid.
    assert len(result.errors) == 4
    assert all("RuntimeError: boom" in e for e in result.errors)
    assert all(r.error is not None for r in result.per_question)
    assert all(not r.answer_correct for r in result.per_question)
    # No valid rows -> only the reliability counters remain (quality/cost skipped).
    assert result.aggregate_metrics == {"error_count": 4.0, "error_rate": 1.0}


async def test_partial_error_keeps_valid_rows(patch_judges: None) -> None:
    adapter = _make_adapter()
    good = adapter.graph_search.return_value
    adapter.graph_search = AsyncMock(side_effect=[RuntimeError("boom"), good, good, good])
    runner = BenchmarkRunner(adapter, _make_dataset(), sample_mode="full", query_concurrency=1)

    result = await runner.run()

    errored = [r for r in result.per_question if r.error]
    valid = [r for r in result.per_question if r.error is None]
    assert len(errored) == 1
    assert len(valid) == 3
    assert len(result.errors) == 1
    assert result.aggregate_metrics  # computed from the 3 valid rows
    assert result.aggregate_metrics["error_count"] == 1.0
    assert result.aggregate_metrics["error_rate"] == 0.25


# ---------------------------------------------------------------------------
# Edge: empty graph short-circuits retrieval
# ---------------------------------------------------------------------------


async def test_zero_node_graph_skips_questions(patch_judges: None) -> None:
    adapter = _make_adapter(num_nodes=0, num_edges=0)
    runner = BenchmarkRunner(adapter, _make_dataset(), sample_mode="full")

    result = await runner.run()

    assert result.per_question == []
    assert any("0 entities" in e for e in result.errors)
    adapter.graph_search.assert_not_awaited()


# ---------------------------------------------------------------------------
# Edge: empty generated answer becomes an explicit refusal
# ---------------------------------------------------------------------------


async def test_empty_answer_becomes_refusal(patch_judges: None) -> None:
    adapter = _make_adapter()
    adapter.generate_answer = AsyncMock(return_value=GeneratedAnswer(answer="   ", evidence=[]))
    runner = BenchmarkRunner(adapter, _make_dataset(), sample_mode="full")

    result = await runner.run()

    for r in result.per_question:
        assert r.generated_answer == "I don't have enough information to answer this."


# ---------------------------------------------------------------------------
# teardown failure must not break the run
# ---------------------------------------------------------------------------


async def test_teardown_failure_swallowed(patch_judges: None) -> None:
    adapter = _make_adapter()
    adapter.teardown = AsyncMock(side_effect=RuntimeError("teardown boom"))
    runner = BenchmarkRunner(adapter, _make_dataset(), sample_mode="full")

    result = await runner.run()  # should not raise
    assert isinstance(result, BenchmarkRunResult)
    assert result.num_questions == 4


# ---------------------------------------------------------------------------
# Per-phase cost attribution
# ---------------------------------------------------------------------------


def test_cost_tracker_buckets_cost_by_phase(monkeypatch: pytest.MonkeyPatch) -> None:
    import litellm

    from khora_graphrag_bench.harness.runner import _cost_phase, _CostTracker

    monkeypatch.setattr(litellm, "success_callback", [], raising=False)
    monkeypatch.setattr(litellm, "_async_success_callback", [], raising=False)

    tracker = _CostTracker()
    tracker.start()
    sync_cb, _async_cb = tracker._handler  # type: ignore[misc]

    with _cost_phase("judge"):
        sync_cb({"response_cost": 0.5}, None, None, None)
    with _cost_phase("generation"):
        sync_cb({"response_cost": 0.2}, None, None, None)
    # A call made outside any tagged phase lands in "other".
    sync_cb({"response_cost": 0.1}, None, None, None)
    tracker.stop()

    assert tracker.total_cost == pytest.approx(0.8)
    assert tracker.by_phase["judge"] == pytest.approx(0.5)
    assert tracker.by_phase["generation"] == pytest.approx(0.2)
    assert tracker.by_phase["other"] == pytest.approx(0.1)
    assert tracker.by_phase["construction"] == 0.0


async def test_cost_tracker_async_callback_records(monkeypatch: pytest.MonkeyPatch) -> None:
    import litellm

    from khora_graphrag_bench.harness.runner import _cost_phase, _CostTracker

    monkeypatch.setattr(litellm, "success_callback", [], raising=False)
    monkeypatch.setattr(litellm, "_async_success_callback", [], raising=False)

    tracker = _CostTracker()
    tracker.start()
    _sync_cb, async_cb = tracker._handler  # type: ignore[misc]
    with _cost_phase("construction"):
        await async_cb({"response_cost": 0.3}, None, None, None)
    tracker.stop()

    assert tracker.by_phase["construction"] == pytest.approx(0.3)


async def test_run_surfaces_cost_by_phase(patch_judges: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """A run with recorded per-phase cost surfaces it on the result + aggregate."""

    class _FakeCost:
        def __init__(self) -> None:
            self.total_cost = 1.0
            self.by_phase = {
                "construction": 0.6,
                "retrieval": 0.0,
                "generation": 0.0,
                "judge": 0.4,
                "other": 0.0,
            }

        def start(self) -> None: ...

        def stop(self) -> None: ...

    monkeypatch.setattr(runner_mod, "_CostTracker", _FakeCost)
    result = await BenchmarkRunner(_make_adapter(), _make_dataset(), sample_mode="full").run()

    # Zero-cost phases are dropped; non-zero ones surface on the result + aggregate.
    assert result.cost_by_phase == {"construction": 0.6, "judge": 0.4}
    assert result.aggregate_metrics["cost_construction_usd"] == pytest.approx(0.6)
    assert result.aggregate_metrics["cost_judge_usd"] == pytest.approx(0.4)
    assert result.aggregate_metrics["cost_usd"] == pytest.approx(1.0)
