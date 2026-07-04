"""Average several benchmark runs into mean +/- stdev per metric.

Single-run deltas on this benchmark are noise: answer generation is not seeded,
so the generated answers (and therefore the judge inputs) differ run to run.
Two runs of the same build have spread ~1pt on ``mean_answer_score``, and the
published reference baseline is itself a multi-run mean. Averaging >= 3 runs and
reporting the spread is what lets a real regression be told apart from variance.

This module holds the pure aggregation + rendering logic; ``scripts/aggregate_runs.py``
is the thin CLI wrapper over it.
"""

from __future__ import annotations

import statistics
from typing import Any

# Headline metrics shown in the comparison table, in display order.
_HEADLINE_METRICS = (
    "mean_answer_score",
    "accuracy",
    "coverage",
    "rouge_l",
    "faithfulness",
    "context_relevance",
    "evidence_recall",
    "cost_usd",
)

# Metrics whose reference key differs from the metric name, or that render
# with non-default precision.
_REFERENCE_KEY = {"runtime_min": "runtime_minutes"}
_MONEY_METRICS = {"cost_usd"}
_TIME_METRICS = {"runtime_min"}


def _run_metrics(result: dict[str, Any]) -> dict[str, float]:
    """Pull the flat numeric metric map for one run from its ``result`` dict."""
    agg = dict(result.get("aggregate_metrics") or {})
    # runtime_min isn't in aggregate_metrics; derive it from runtime_seconds.
    runtime_s = result.get("runtime_seconds")
    if isinstance(runtime_s, int | float) and not isinstance(runtime_s, bool) and runtime_s > 0:
        agg.setdefault("runtime_min", runtime_s / 60.0)
    return {k: float(v) for k, v in agg.items() if isinstance(v, int | float) and not isinstance(v, bool)}


def aggregate_runs(results: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Return ``{metric: {"mean", "stdev", "n"}}`` across the given runs.

    ``results`` are the ``result`` sub-dicts of each run's ``report.json``.
    Metrics are the union of numeric keys across runs; a run missing a metric
    simply doesn't contribute to it (``n`` reflects how many runs reported it).
    ``stdev`` is the sample standard deviation, or ``0.0`` for a single value.
    """
    if not results:
        raise ValueError("aggregate_runs needs at least one run")
    per_metric: dict[str, list[float]] = {}
    for result in results:
        for key, value in _run_metrics(result).items():
            per_metric.setdefault(key, []).append(value)
    out: dict[str, dict[str, float]] = {}
    for metric, values in per_metric.items():
        out[metric] = {
            "mean": statistics.fmean(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "n": float(len(values)),
        }
    return out


def _fmt_value(metric: str, value: float) -> str:
    if metric in _MONEY_METRICS:
        return f"${value:.2f}"
    if metric in _TIME_METRICS:
        return f"{value:.1f}"
    return f"{value:.4f}"


def format_comparison_markdown(
    agg: dict[str, dict[str, float]],
    *,
    reference: dict[str, Any] | None = None,
) -> str:
    """Render ``aggregate_runs`` output as a Markdown ``mean +/- stdev`` table."""
    reference = reference or {}
    lines = [
        "| metric | mean | stdev | n | reference | Δ (mean − ref) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    metrics = [m for m in _HEADLINE_METRICS if m in agg]
    if "runtime_min" in agg:
        metrics.append("runtime_min")
    for metric in metrics:
        stats = agg[metric]
        mean, stdev, n = stats["mean"], stats["stdev"], int(stats["n"])
        ref = reference.get(_REFERENCE_KEY.get(metric, metric))
        mean_s = _fmt_value(metric, mean)
        if ref is None:
            ref_s, delta_s = "—", "—"
        else:
            ref_s = _fmt_value(metric, float(ref))
            delta_s = f"{mean - float(ref):+.4f}"
        lines.append(f"| `{metric}` | {mean_s} | {stdev:.4f} | {n} | {ref_s} | {delta_s} |")
    return "\n".join(lines)
