"""Unit tests for the multi-run aggregation helpers."""

from __future__ import annotations

import statistics

import pytest

from khora_graphrag_bench.reporters.aggregate import aggregate_runs, format_comparison_markdown


def _run(**metrics: float) -> dict:
    """A minimal run ``result`` dict with the given aggregate metrics."""
    runtime = metrics.pop("runtime_seconds", None)
    result: dict = {"aggregate_metrics": dict(metrics)}
    if runtime is not None:
        result["runtime_seconds"] = runtime
    return result


def test_aggregate_runs_mean_stdev_n() -> None:
    runs = [
        _run(mean_answer_score=0.70, accuracy=0.80),
        _run(mean_answer_score=0.72, accuracy=0.79),
        _run(mean_answer_score=0.74, accuracy=0.81),
    ]
    agg = aggregate_runs(runs)

    assert agg["mean_answer_score"]["n"] == 3.0
    assert agg["mean_answer_score"]["mean"] == pytest.approx(0.72)
    assert agg["mean_answer_score"]["stdev"] == pytest.approx(statistics.stdev([0.70, 0.72, 0.74]))
    assert agg["accuracy"]["mean"] == pytest.approx(0.80)


def test_single_run_has_zero_stdev() -> None:
    agg = aggregate_runs([_run(mean_answer_score=0.7)])
    assert agg["mean_answer_score"]["n"] == 1.0
    assert agg["mean_answer_score"]["stdev"] == 0.0


def test_metric_missing_in_one_run_reflected_in_n() -> None:
    runs = [
        _run(mean_answer_score=0.7, coverage=0.6),
        _run(mean_answer_score=0.8),  # no coverage
    ]
    agg = aggregate_runs(runs)
    assert agg["mean_answer_score"]["n"] == 2.0
    assert agg["coverage"]["n"] == 1.0
    assert agg["coverage"]["mean"] == pytest.approx(0.6)


def test_runtime_seconds_becomes_runtime_min() -> None:
    agg = aggregate_runs([_run(mean_answer_score=0.7, runtime_seconds=120.0)])
    assert agg["runtime_min"]["mean"] == pytest.approx(2.0)


def test_non_numeric_and_bool_values_ignored() -> None:
    # error_rate is numeric and kept; a bool/str would be dropped.
    runs = [{"aggregate_metrics": {"mean_answer_score": 0.7, "flag": True, "note": "x", "error_rate": 0.0}}]
    agg = aggregate_runs(runs)
    assert "mean_answer_score" in agg
    assert "error_rate" in agg
    assert "flag" not in agg
    assert "note" not in agg


def test_empty_runs_raises() -> None:
    with pytest.raises(ValueError, match="at least one run"):
        aggregate_runs([])


def test_format_comparison_markdown_table() -> None:
    agg = aggregate_runs(
        [
            _run(mean_answer_score=0.70, accuracy=0.80, cost_usd=3.0, runtime_seconds=1740.0),
            _run(mean_answer_score=0.72, accuracy=0.80, cost_usd=3.4, runtime_seconds=1860.0),
        ]
    )
    table = format_comparison_markdown(
        agg, reference={"mean_answer_score": 0.694, "cost_usd": 3.59, "runtime_minutes": 478}
    )

    assert "| metric | mean | stdev | n |" in table
    # mean_answer_score row: mean 0.71, reference 0.694, delta +0.0160.
    assert "`mean_answer_score`" in table
    assert "0.7100" in table
    assert "0.6940" in table
    assert "+0.0160" in table
    # cost renders as money; no reference-less em dash for a metric with a ref.
    assert "$3.20" in table
    # runtime_min is derived from runtime_seconds and keyed to runtime_minutes.
    assert "`runtime_min`" in table
    assert "30.0" in table  # mean of 29.0 and 31.0 min
    # accuracy has no reference here -> em dash in the reference/delta columns.
    assert "—" in table
