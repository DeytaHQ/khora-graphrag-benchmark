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
from khora_graphrag_bench.harness.evaluation import DEFAULT_EVIDENCE_COSINE_THRESHOLD
from khora_graphrag_bench.harness.model_utils import is_reasoning_model
from khora_graphrag_bench.harness.runner import DEFAULT_QUERY_CONCURRENCY, BenchmarkRunner
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
    "--second-pass",
    is_flag=True,
    default=False,
    help=(
        "Enable khora's second-pass relationship extraction (#1409): a denser "
        "relationship graph at extra ingest cost. Off by default; forces a re-index."
    ),
)
@click.option(
    "--min-chunk-similarity",
    type=click.FloatRange(0.0, 1.0),
    default=0.0,
    show_default=True,
    help=(
        "Cosine floor on retrieved chunks (0.0 = off, khora default). Drops weak "
        "matches from context instead of padding it."
    ),
)
@click.option(
    "--query-concurrency",
    type=click.IntRange(1, 200),
    default=DEFAULT_QUERY_CONCURRENCY,
    envvar="KGB_QUERY_CONCURRENCY",
    show_default=True,
    help=(
        "Max questions answered concurrently (env: KGB_QUERY_CONCURRENCY). Caps the "
        "OpenAI request burst; higher = faster query phase until rate limits bite. "
        "Independent per question, so it changes only wall-clock, never results."
    ),
)
@click.option(
    "--retrieval-only",
    is_flag=True,
    default=False,
    help=(
        "Score retrieval quality only: skip answer generation + LLM judging and "
        "compute embedding-cosine evidence_recall@k of the retrieved chunks "
        "against the gold evidence. ~$0.09/run (embeddings only) vs the full "
        "~$4.17. Use with McNemar A/Bs (`analyze`) to detect small retrieval fixes."
    ),
)
@click.option(
    "--evidence-cosine-threshold",
    type=click.FloatRange(0.0, 1.0),
    default=None,
    help=(
        "Cosine above which a gold-evidence statement counts as covered by a "
        "retrieved chunk (retrieval-only mode). Defaults to the calibrated "
        f"{DEFAULT_EVIDENCE_COSINE_THRESHOLD}."
    ),
)
@click.option(
    "--no-report",
    is_flag=True,
    help="Skip writing the JSON/MD/HTML report files (still prints summary).",
)
def run(
    sample: str,
    top_k: int,
    judge_model: str,
    gen_model: str,
    extract_model: str,
    second_pass: bool,
    min_chunk_similarity: float,
    query_concurrency: int,
    retrieval_only: bool,
    evidence_cosine_threshold: float | None,
    no_report: bool,
) -> None:
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
            second_pass=second_pass,
            min_chunk_similarity=min_chunk_similarity,
            query_concurrency=query_concurrency,
            retrieval_only=retrieval_only,
            evidence_cosine_threshold=(
                evidence_cosine_threshold
                if evidence_cosine_threshold is not None
                else DEFAULT_EVIDENCE_COSINE_THRESHOLD
            ),
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


@main.command(help="Paired McNemar A/B of two runs (baseline vs candidate) on per-question flips.")
@click.argument("baseline")
@click.argument("candidate")
def analyze(baseline: str, candidate: str) -> None:
    """Compare two runs by their run-id (or path) with McNemar's paired test.

    Reads each run's ``report.json``, pairs per-question correctness on
    ``question_id``, and reports the discordant-pair counts, the test statistic,
    p-value, and the net flip count. Detects a ~1.5pt effect (~30 flips on 2010q)
    in a single paired run despite ~0.73pt run-to-run mean-accuracy noise.
    """
    from khora_graphrag_bench.harness.analysis import compare_runs

    base_report = _load_report_json(baseline)
    cand_report = _load_report_json(candidate)
    result = compare_runs(base_report, cand_report)
    t = result.table

    click.echo("=" * 64)
    click.echo(f"McNemar paired A/B — {baseline} (baseline) vs {candidate} (candidate)")
    click.echo("=" * 64)
    click.echo(f"  paired questions           {t.n}")
    click.echo(f"  both correct               {t.both_correct}")
    click.echo(f"  both wrong                 {t.both_wrong}")
    click.echo(f"  baseline-only correct (b)  {t.baseline_only}   (candidate regressed)")
    click.echo(f"  candidate-only correct (c) {t.candidate_only}   (candidate improved)")
    click.echo(f"  discordant pairs (b+c)     {t.discordant}")
    click.echo("  " + "-" * 40)
    click.echo(f"  net flips (c - b)          {result.net_flips:+d}")
    click.echo(f"  accuracy delta             {result.accuracy_delta * 100:+.2f} pt")
    click.echo(f"  test                       {result.method}")
    click.echo(f"  statistic                  {result.statistic:.4f}")
    click.echo(f"  p-value                    {result.p_value:.4g}")
    verdict = "SIGNIFICANT (p < 0.05)" if result.significant_at_05 else "not significant (p >= 0.05)"
    click.echo(f"  verdict                    {verdict}")
    click.echo("=" * 64)


def _load_report_json(run_id_or_path: str) -> dict:
    """Load a run's ``report.json`` given a run-id under results/ or a direct path."""
    p = Path(run_id_or_path)
    # A bare run-id resolves under results/; an explicit path (dir or file) is used as-is.
    if not p.exists():
        p = RESULTS_ROOT / run_id_or_path
    if p.is_dir():
        p = p / "report.json"
    if not p.exists():
        click.echo(f"✗ No report.json found for {run_id_or_path!r} (looked at {p})", err=True)
        sys.exit(1)
    return json.loads(p.read_text())


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
    second_pass: bool = False,
    min_chunk_similarity: float = 0.0,
    query_concurrency: int = DEFAULT_QUERY_CONCURRENCY,
    retrieval_only: bool = False,
    evidence_cosine_threshold: float = DEFAULT_EVIDENCE_COSINE_THRESHOLD,
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
            "extraction_second_pass": second_pass,
            "min_chunk_similarity": min_chunk_similarity,
        }
    )
    runner = BenchmarkRunner(
        adapter=adapter,
        dataset=dataset,
        sample_mode=sample,
        top_k=top_k,
        judge_model=judge_model,
        query_concurrency=query_concurrency,
        retrieval_only=retrieval_only,
        evidence_cosine_threshold=evidence_cosine_threshold,
    )
    knobs = []
    if second_pass:
        knobs.append("second_pass=on")
    if min_chunk_similarity > 0.0:
        knobs.append(f"min_chunk_sim={min_chunk_similarity}")
    if query_concurrency != DEFAULT_QUERY_CONCURRENCY:
        knobs.append(f"query_conc={query_concurrency}")
    if retrieval_only:
        knobs.append(f"retrieval_only (evidence_cosine>={evidence_cosine_threshold})")
    knobs_s = (", " + ", ".join(knobs)) if knobs else ""
    mode_label = "retrieval-only (embeddings, no judge)" if retrieval_only else f"judge={judge_model}, gen={gen_model}"
    click.echo(f"🚀 Running with adapter={adapter.name}, {mode_label}, extract={extract_model}{knobs_s}...")
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
    # In retrieval-only mode the headline is evidence_recall_at_k (+ the pass-rate
    # "accuracy"); the judged metrics are absent. Show it first when present.
    metrics = [
        ("evidence_recall_at_k", "evidence_recall_at_k"),
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
        local = agg.get(key)
        # Skip a row only when there's nothing to show for it (no local value AND
        # no reference). retrieval_only metrics have no reference baseline yet, so
        # they'd otherwise be dropped despite being the headline number.
        if ref is None and local is None:
            continue
        local_s = f"{local:.4f}" if local is not None else "—"
        ref_s = f"{ref:>12.4f}" if ref is not None else f"{'—':>12}"
        click.echo(f"  {label:<22} {local_s:>10} {ref_s}")


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
        cost_by_phase=d.get("cost_by_phase", {}),
        runtime_seconds=d.get("runtime_seconds", 0.0),
        errors=d.get("errors", []),
    )


if __name__ == "__main__":
    main()
