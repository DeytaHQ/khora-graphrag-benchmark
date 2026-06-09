"""Markdown reporter — produces a human-readable summary."""

from __future__ import annotations

from pathlib import Path

from khora_graphrag_bench.harness.results import BenchmarkRunResult
from khora_graphrag_bench.reporters._reference import KHORA_BASELINE


def _fmt(v: object) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _delta(local: float | None, ref: float) -> str:
    if local is None:
        return "—"
    diff = local - ref
    pct = (diff / ref * 100) if ref else 0.0
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff:.3f} ({sign}{pct:.1f}%)"


def write_markdown_report(result: BenchmarkRunResult, out_dir: str | Path) -> Path:
    """Render a Markdown report to ``{out_dir}/report.md``."""
    out_path = Path(out_dir) / "report.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    agg = result.aggregate_metrics
    lines: list[str] = []
    lines.append(f"# GraphRAG-Bench results — {result.adapter_name}")
    lines.append("")
    lines.append(f"- **Run id**: `{result.run_id}`")
    lines.append(f"- **Started**: {result.started_at.isoformat()}")
    lines.append(f"- **Completed**: {result.completed_at.isoformat()}")
    lines.append(f"- **Runtime**: {result.runtime_seconds / 60:.1f} min")
    lines.append(f"- **Sample mode**: `{result.sample_mode}`")
    lines.append(f"- **Documents**: {result.num_documents}")
    lines.append(f"- **Questions**: {result.num_questions}")
    lines.append(f"- **Dataset**: `{result.dataset_name}` (hash `{result.dataset_hash}`)")
    lines.append(f"- **Judge model**: `{result.judge_model}`")
    lines.append(f"- **Cost**: ${result.cost_usd:.2f}")
    lines.append("")

    error_rate = agg.get("error_rate", 0.0)
    if error_rate > 0:
        error_count = int(agg.get("error_count", 0))
        lines.append(
            f"> ⚠️ **{error_count}/{result.num_questions} questions errored "
            f"({error_rate * 100:.1f}%)** and were excluded from the metrics below — "
            "aggregates cover the successful questions only and are **not comparable to the "
            "reference baseline** if this rate is non-trivial."
        )
        lines.append("")

    if result.construction is not None:
        c = result.construction
        lines.append("## Phase 1 — graph construction")
        lines.append("")
        lines.append(f"- **Nodes**: {c.num_nodes:,}")
        lines.append(f"- **Edges**: {c.num_edges:,}")
        lines.append(f"- **Communities**: {c.num_communities}")
        lines.append(f"- **Avg degree**: {c.avg_degree:.2f}")
        lines.append(f"- **Construction time**: {c.construction_time_ms / 1000:.1f}s")
        lines.append("")

    # ----- Headline metrics vs Khora baseline ----------------------------
    lines.append("## Headline metrics — your run vs Khora reference baseline")
    lines.append("")
    lines.append("| metric | your run | Khora reference | Δ |")
    lines.append("|---|---:|---:|---:|")
    metric_keys = [
        ("mean_r_score", "mean_r_score"),
        ("mean_ar_metric", "mean_ar_metric"),
        ("mean_answer_score", "mean_answer_score"),
        ("accuracy", "accuracy"),
        ("coverage", "coverage"),
        ("rouge_l", "rouge_l"),
        ("faithfulness", "faithfulness"),
        ("context_relevance", "context_relevance"),
        ("evidence_recall", "evidence_recall"),
    ]
    for label, key in metric_keys:
        local = agg.get(key)
        ref = KHORA_BASELINE.get(key)
        if ref is None:
            continue
        lines.append(
            f"| `{label}` | {_fmt(local) if local is not None else '—'} | {_fmt(ref)} | {_delta(local, ref)} |"
        )
    if "cost_usd" in agg:
        lines.append(
            f"| `cost_usd` | ${agg['cost_usd']:.2f} | ${KHORA_BASELINE['cost_usd']:.2f} | "
            f"{_delta(agg['cost_usd'], KHORA_BASELINE['cost_usd'])} |"
        )
    lines.append(
        f"| `runtime_min` | {result.runtime_seconds / 60:.1f} | {KHORA_BASELINE['runtime_minutes']} | "
        f"{_delta(result.runtime_seconds / 60, float(KHORA_BASELINE['runtime_minutes']))} |"
    )
    lines.append("")
    lines.append(
        "> _Reference baseline: khora "
        f"`{KHORA_BASELINE['khora_version']}` on "
        f"`{KHORA_BASELINE['dataset']}` at `full` sampling with `{KHORA_BASELINE['judge_model']}` judge._"
    )
    lines.append("")

    # ----- Breakdown by difficulty / question type ----------------------
    if result.by_difficulty:
        lines.append("## Breakdown by difficulty")
        lines.append("")
        lines.append("| difficulty | n | accuracy | r_score | ar_metric |")
        lines.append("|---|---:|---:|---:|---:|")
        for diff, m in sorted(result.by_difficulty.items()):
            lines.append(
                f"| `{diff}` | {int(m.get('n', 0))} | {_fmt(m.get('accuracy', 0))} | "
                f"{_fmt(m.get('mean_r_score', 0))} | {_fmt(m.get('mean_ar_metric', 0))} |"
            )
        lines.append("")

    if result.by_question_type:
        lines.append("## Breakdown by question type")
        lines.append("")
        lines.append("| type | n | accuracy | r_score | ar_metric |")
        lines.append("|---|---:|---:|---:|---:|")
        for qt, m in sorted(result.by_question_type.items()):
            lines.append(
                f"| `{qt}` | {int(m.get('n', 0))} | {_fmt(m.get('accuracy', 0))} | "
                f"{_fmt(m.get('mean_r_score', 0))} | {_fmt(m.get('mean_ar_metric', 0))} |"
            )
        lines.append("")

    if result.errors:
        lines.append("## Errors")
        lines.append("")
        for err in result.errors[:20]:
            lines.append(f"- {err}")
        if len(result.errors) > 20:
            lines.append(f"- ... and {len(result.errors) - 20} more")
        lines.append("")

    out_path.write_text("\n".join(lines))
    return out_path
