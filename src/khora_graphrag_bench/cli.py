"""Command-line entry point: ``khora-graphrag-bench run`` / ``report``.

Surface:

    khora-graphrag-bench run --sample {small|medium|full} [--top-k 5]
    khora-graphrag-bench report [--run-id ID] [--format {json,md,html,all}]

The Make targets in ``Makefile`` call this CLI; you can use it directly too.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

import click

from khora_graphrag_bench import __version__
from khora_graphrag_bench.adapters.khora import KhoraAdapter
from khora_graphrag_bench.datasets.loader import load_graphrag_bench
from khora_graphrag_bench.harness.model_utils import is_reasoning_model
from khora_graphrag_bench.harness.runner import BenchmarkRunner
from khora_graphrag_bench.reporters import (
    write_html_report,
    write_json_report,
    write_markdown_report,
)
from khora_graphrag_bench.reporters._reference import KHORA_BASELINE

logger = logging.getLogger("khora_graphrag_bench")


RESULTS_ROOT = Path(os.environ.get("BENCH_RESULTS_DIR", "results"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.group(help="GraphRAG-Bench evaluation of Khora.")
@click.version_option(__version__, prog_name="khora-graphrag-bench")
@click.option("-v", "--verbose", count=True, help="Increase log verbosity (-v info, -vv debug).")
def main(verbose: int) -> None:
    level = logging.WARNING
    if verbose >= 2:
        level = logging.DEBUG
    elif verbose >= 1:
        level = logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@main.command(help="Run the benchmark end-to-end and produce JSON/Markdown/HTML reports.")
@click.option(
    "--sample",
    type=click.Choice(["small", "medium", "full"], case_sensitive=False),
    default="full",
    show_default=True,
    help="Sampling mode. 'small' is a fast smoke test, 'full' is the published baseline.",
)
@click.option(
    "--top-k",
    type=int,
    default=5,
    show_default=True,
    help="Number of chunks retrieved per question.",
)
@click.option(
    "--judge-model",
    default="gpt-4o-mini",
    show_default=True,
    help="LLM used by the judge prompts. Paper-aligned default is gpt-4o-mini.",
)
@click.option(
    "--gen-model",
    default="gpt-4o-mini",
    show_default=True,
    help="LLM that writes each answer from retrieved context. GPT-5/o-series supported.",
)
@click.option(
    "--extract-model",
    default="gpt-4o-mini",
    show_default=True,
    help=(
        "LLM that builds the graph during indexing (changing it forces a re-index). "
        "Use a non-reasoning model (gpt-4o-mini, gpt-4o, gpt-4.1); khora's extractor "
        "rejects GPT-5/o-series reasoning models."
    ),
)
@click.option(
    "--no-report",
    is_flag=True,
    help="Skip writing the JSON/MD/HTML report files (still prints summary).",
)
def run(sample: str, top_k: int, judge_model: str, gen_model: str, extract_model: str, no_report: bool) -> None:
    _require_openai_key()
    # Fail fast: khora 0.18.5's extractor hardcodes temperature/max_tokens, so a
    # reasoning extract model would 400 mid-ingestion - after reset-db has wiped
    # the DB and we've spent time/money on a partial run.
    if is_reasoning_model(extract_model):
        raise click.BadParameter(
            f"{extract_model!r} is a reasoning model; khora's extractor rejects these. "
            "Use a non-reasoning model (gpt-4o-mini, gpt-4o, gpt-4.1).",
            param_hint="--extract-model",
        )
    asyncio.run(
        _run_async(
            sample=sample.lower(),
            top_k=top_k,
            judge_model=judge_model,
            gen_model=gen_model,
            extract_model=extract_model,
            write_reports=not no_report,
        )
    )


@main.command(help="Regenerate JSON/MD/HTML reports from a previous run (defaults to the latest).")
@click.option("--run-id", help="Specific run id under results/. Defaults to the most recent.")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["json", "md", "html", "all"]),
    default="all",
    show_default=True,
)
def report(run_id: str | None, fmt: str) -> None:
    run_dir = _resolve_run_dir(run_id)
    src = run_dir / "report.json"
    if not src.exists():
        click.echo(f"✗ No report.json found in {run_dir}", err=True)
        sys.exit(1)
    raw = json.loads(src.read_text())
    result = _result_from_json(raw["result"])
    if fmt in ("json", "all"):
        write_json_report(result, run_dir)
        click.echo(f"  → {run_dir / 'report.json'}")
    if fmt in ("md", "all"):
        write_markdown_report(result, run_dir)
        click.echo(f"  → {run_dir / 'report.md'}")
    if fmt in ("html", "all"):
        write_html_report(result, run_dir)
        click.echo(f"  → {run_dir / 'report.html'}")


# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------


async def _run_async(
    *,
    sample: str,
    top_k: int,
    judge_model: str,
    gen_model: str = "gpt-4o-mini",
    extract_model: str = "gpt-4o-mini",
    write_reports: bool,
) -> None:
    click.echo(f"📦 Loading GraphRAG-Bench dataset (sample={sample})...")
    dataset = load_graphrag_bench()
    click.echo(f"   {len(dataset.documents)} documents, {len(dataset.questions)} questions")

    adapter = KhoraAdapter(
        params={
            "entity_types": dataset.entity_types,
            "relationship_types": dataset.relationship_types,
            "generation_model": gen_model,
            "extraction_model": extract_model,
        }
    )
    runner = BenchmarkRunner(
        adapter=adapter,
        dataset=dataset,
        sample_mode=sample,
        top_k=top_k,
        judge_model=judge_model,
    )
    click.echo(
        f"🚀 Running with adapter={adapter.name}, judge={judge_model}, gen={gen_model}, extract={extract_model}..."
    )
    result = await runner.run()

    # Persist
    run_dir = RESULTS_ROOT / result.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    if write_reports:
        write_json_report(result, run_dir)
        write_markdown_report(result, run_dir)
        write_html_report(result, run_dir)

    # Update "latest" symlink so `make report` finds the right dir
    latest_link = RESULTS_ROOT / "latest"
    try:
        if latest_link.is_symlink() or latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(result.run_id)
    except OSError as e:
        logger.warning("Could not update latest symlink: %s", e)

    # Console summary
    click.echo("")
    click.echo("=" * 64)
    click.echo(f"✓ Run complete — {result.runtime_seconds / 60:.1f} min, ${result.cost_usd:.2f}")
    click.echo("=" * 64)
    _print_summary(result.aggregate_metrics)
    click.echo("")
    click.echo(f"Reports: {run_dir}")


def _print_summary(agg: dict) -> None:
    metrics = [
        ("mean_answer_score", "mean_answer_score"),
        ("accuracy", "accuracy"),
        ("faithfulness", "faithfulness"),
        ("coverage", "coverage"),
        ("rouge_l", "rouge_l"),
    ]
    click.echo(f"  {'metric':<22} {'your run':>10} {'Khora ref':>12}")
    click.echo(f"  {'-' * 22} {'-' * 10} {'-' * 12}")
    for label, key in metrics:
        ref = KHORA_BASELINE.get(key)
        if ref is None:
            continue
        local = agg.get(key)
        local_s = f"{local:.4f}" if local is not None else "—"
        click.echo(f"  {label:<22} {local_s:>10} {ref:>12.4f}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_openai_key() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        click.echo(
            "✗ OPENAI_API_KEY is not set.\n"
            "  Copy .env.example to .env and fill it in, or export it directly:\n"
            "      export OPENAI_API_KEY=sk-...",
            err=True,
        )
        sys.exit(1)


def _resolve_run_dir(run_id: str | None) -> Path:
    """Find the run directory for ``run_id``, or the most recent run."""
    if not RESULTS_ROOT.exists():
        click.echo(f"✗ No results directory at {RESULTS_ROOT.resolve()}", err=True)
        sys.exit(1)

    if run_id:
        run_dir = RESULTS_ROOT / run_id
        if not run_dir.exists():
            click.echo(f"✗ Run {run_id} not found in {RESULTS_ROOT}", err=True)
            sys.exit(1)
        return run_dir

    # Prefer the `latest` symlink if present, otherwise pick the most recent
    # by modification time among run-id-shaped directories.
    latest = RESULTS_ROOT / "latest"
    if latest.exists():
        return latest.resolve()

    candidates = [d for d in RESULTS_ROOT.iterdir() if d.is_dir() and d.name not in {"latest"}]
    if not candidates:
        click.echo("✗ No prior runs found. Run `make run-small` first.", err=True)
        sys.exit(1)
    return max(candidates, key=lambda d: d.stat().st_mtime)


def _result_from_json(d: dict):
    """Round-trip the JSON dict back into a ``BenchmarkRunResult``.

    Only used by the report regeneration command. Tolerant of missing fields
    so older report formats keep working.
    """
    from datetime import datetime

    from khora_graphrag_bench.harness.results import (
        BenchmarkRunResult,
        GraphConstructionDetail,
        QuestionResult,
    )

    def _dt(s):
        return datetime.fromisoformat(s) if isinstance(s, str) else s

    construction = None
    if d.get("construction"):
        c = d["construction"]
        construction = GraphConstructionDetail(
            num_nodes=c["num_nodes"],
            num_edges=c["num_edges"],
            num_communities=c.get("num_communities", 0),
            construction_time_ms=c.get("construction_time_ms", 0.0),
            avg_degree=c.get("avg_degree", 0.0),
            density=c.get("density", 0.0),
        )

    return BenchmarkRunResult(
        run_id=d["run_id"],
        started_at=_dt(d["started_at"]),
        completed_at=_dt(d["completed_at"]),
        adapter_name=d["adapter_name"],
        dataset_name=d["dataset_name"],
        dataset_hash=d["dataset_hash"],
        sample_mode=d["sample_mode"],
        num_documents=d["num_documents"],
        num_questions=d["num_questions"],
        judge_model=d.get("judge_model", "gpt-4o-mini"),
        construction=construction,
        aggregate_metrics=d.get("aggregate_metrics", {}),
        by_difficulty=d.get("by_difficulty", {}),
        by_question_type=d.get("by_question_type", {}),
        per_question=[QuestionResult(**q) for q in d.get("per_question", [])],
        cost_usd=d.get("cost_usd", 0.0),
        runtime_seconds=d.get("runtime_seconds", 0.0),
        errors=d.get("errors", []),
    )


if __name__ == "__main__":
    main()
