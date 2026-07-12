"""End-to-end test of the CLI ``run`` command's ``_run_async`` pipeline.

The dataset loader, adapter, and reporters are monkeypatched so the whole
``run`` path executes against mocks - no services, no network, no API key
material beyond a dummy env var.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from click.testing import CliRunner

from khora_graphrag_bench import cli
from khora_graphrag_bench.datasets.schema import DatasetDocument, GraphRAGDataset, GraphRAGQuestion
from khora_graphrag_bench.harness.results import BenchmarkRunResult


def _dataset() -> GraphRAGDataset:
    return GraphRAGDataset(
        name="tiny",
        documents=[DatasetDocument(doc_id="d1", content="Paris is the capital of France.")],
        questions=[
            GraphRAGQuestion(
                question_id="q1",
                question="Capital of France?",
                question_type="FB",
                difficulty="fact_retrieval",
                gold_answer="Paris",
                evidence=["Paris is the capital of France."],
                relevant_doc_ids=["d1"],
            )
        ],
        entity_types=["LOCATION"],
        relationship_types=["CAPITAL_OF"],
    )


def _fake_result() -> BenchmarkRunResult:
    from datetime import UTC, datetime

    now = datetime.now(UTC)
    return BenchmarkRunResult(
        run_id="run-test",
        started_at=now,
        completed_at=now,
        adapter_name="fake",
        dataset_name="tiny",
        dataset_hash="abc",
        sample_mode="small",
        num_documents=1,
        num_questions=1,
        judge_model="gpt-4o-mini",
        construction=None,
        aggregate_metrics={"mean_r_score": 0.5, "accuracy": 1.0},
        by_difficulty={},
        by_question_type={},
        per_question=[],
        cost_usd=1.23,
        runtime_seconds=120.0,
    )


@pytest.fixture
def patched_pipeline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Patch loader, adapter, runner, and reporters in the cli namespace."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(cli, "RESULTS_ROOT", tmp_path / "results")

    monkeypatch.setattr(cli, "load_graphrag_bench", lambda: _dataset())

    fake_adapter = MagicMock()
    fake_adapter.name = "fake"
    monkeypatch.setattr(cli, "KhoraAdapter", lambda *a, **k: fake_adapter)

    result = _fake_result()
    fake_runner = MagicMock()
    fake_runner.run = AsyncMock(return_value=result)
    monkeypatch.setattr(cli, "BenchmarkRunner", lambda *a, **k: fake_runner)

    json_w = MagicMock()
    md_w = MagicMock()
    html_w = MagicMock()
    monkeypatch.setattr(cli, "write_json_report", json_w)
    monkeypatch.setattr(cli, "write_markdown_report", md_w)
    monkeypatch.setattr(cli, "write_html_report", html_w)

    return {
        "adapter": fake_adapter,
        "runner": fake_runner,
        "result": result,
        "reporters": (json_w, md_w, html_w),
    }


def test_run_end_to_end_writes_reports(patched_pipeline: dict) -> None:
    runner = CliRunner()
    res = runner.invoke(cli.main, ["run", "--sample", "small", "--top-k", "3"])

    assert res.exit_code == 0, res.output
    assert "Loading GraphRAG-Bench dataset" in res.output
    assert "Run complete" in res.output
    assert "Reports:" in res.output

    patched_pipeline["runner"].run.assert_awaited_once()
    for w in patched_pipeline["reporters"]:
        w.assert_called_once()

    # Run dir + latest symlink created.
    run_dir = cli.RESULTS_ROOT / "run-test"
    assert run_dir.exists()
    assert (cli.RESULTS_ROOT / "latest").resolve() == run_dir.resolve()


def test_run_no_report_skips_reporters(patched_pipeline: dict) -> None:
    runner = CliRunner()
    res = runner.invoke(cli.main, ["run", "--sample", "small", "--no-report"])

    assert res.exit_code == 0, res.output
    for w in patched_pipeline["reporters"]:
        w.assert_not_called()
    # Summary still printed even without report files.
    assert "Run complete" in res.output


def test_run_forwards_options_to_runner(patched_pipeline: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def _capture(*args, **kwargs):
        captured.update(kwargs)
        return patched_pipeline["runner"]

    monkeypatch.setattr(cli, "BenchmarkRunner", _capture)

    runner = CliRunner()
    res = runner.invoke(
        cli.main,
        ["run", "--sample", "MEDIUM", "--top-k", "7", "--judge-model", "gpt-4o"],
    )

    assert res.exit_code == 0, res.output
    assert captured["sample_mode"] == "medium"  # lowercased by the command
    assert captured["top_k"] == 7
    assert captured["judge_model"] == "gpt-4o"


def test_run_forwards_quality_knobs_to_adapter(patched_pipeline: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def _capture_adapter(*args, **kwargs):
        captured.update(kwargs.get("params", {}))
        return patched_pipeline["adapter"]

    monkeypatch.setattr(cli, "KhoraAdapter", _capture_adapter)

    runner = CliRunner()
    res = runner.invoke(
        cli.main,
        ["run", "--sample", "small", "--second-pass", "--min-chunk-similarity", "0.2"],
    )
    assert res.exit_code == 0, res.output
    assert captured["extraction_second_pass"] is True
    assert captured["min_chunk_similarity"] == 0.2


def test_run_quality_knobs_default_off(patched_pipeline: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def _capture_adapter(*args, **kwargs):
        captured.update(kwargs.get("params", {}))
        return patched_pipeline["adapter"]

    monkeypatch.setattr(cli, "KhoraAdapter", _capture_adapter)

    runner = CliRunner()
    res = runner.invoke(cli.main, ["run", "--sample", "small"])
    assert res.exit_code == 0, res.output
    assert captured["extraction_second_pass"] is False
    assert captured["min_chunk_similarity"] == 0.0


def test_run_rejects_reasoning_extract_model(patched_pipeline: dict) -> None:
    """A reasoning --extract-model fails fast at CLI entry, before reset-db/ingestion."""
    runner = CliRunner()
    res = runner.invoke(cli.main, ["run", "--sample", "small", "--extract-model", "gpt-5-mini"])

    assert res.exit_code != 0
    assert "reasoning model" in res.output
    assert "--extract-model" in res.output
    # Validation happens before the pipeline runs.
    patched_pipeline["runner"].run.assert_not_awaited()


def test_run_allows_reasoning_gen_model(patched_pipeline: dict) -> None:
    """A reasoning --gen-model is allowed (the adapter handles its params)."""
    runner = CliRunner()
    res = runner.invoke(cli.main, ["run", "--sample", "small", "--gen-model", "gpt-5-mini"])

    assert res.exit_code == 0, res.output
    patched_pipeline["runner"].run.assert_awaited_once()


def test_run_handles_symlink_oserror(patched_pipeline: dict) -> None:
    """A failing latest-symlink update is logged, not fatal.

    Pre-creating ``latest`` as a non-empty directory makes ``unlink()`` raise
    ``OSError`` (IsADirectory/NotEmpty), exercising the swallowed error branch
    without globally patching ``Path``.
    """
    latest = cli.RESULTS_ROOT / "latest"
    latest.mkdir(parents=True)
    (latest / "blocker").write_text("x")

    runner = CliRunner()
    res = runner.invoke(cli.main, ["run", "--sample", "small"])
    assert res.exit_code == 0, res.output
    assert "Run complete" in res.output
    assert "Could not update latest symlink" not in res.output  # warning is logged, not echoed


def test_run_forwards_retrieval_only_to_runner(patched_pipeline: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def _capture(*args, **kwargs):
        captured.update(kwargs)
        return patched_pipeline["runner"]

    monkeypatch.setattr(cli, "BenchmarkRunner", _capture)

    runner = CliRunner()
    res = runner.invoke(
        cli.main,
        ["run", "--sample", "small", "--retrieval-only", "--evidence-cosine-threshold", "0.42"],
    )
    assert res.exit_code == 0, res.output
    assert captured["retrieval_only"] is True
    assert captured["evidence_cosine_threshold"] == 0.42


def test_run_retrieval_only_defaults_threshold(patched_pipeline: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    from khora_graphrag_bench.harness.evaluation import DEFAULT_EVIDENCE_COSINE_THRESHOLD

    captured = {}

    def _capture(*args, **kwargs):
        captured.update(kwargs)
        return patched_pipeline["runner"]

    monkeypatch.setattr(cli, "BenchmarkRunner", _capture)

    runner = CliRunner()
    res = runner.invoke(cli.main, ["run", "--sample", "small", "--retrieval-only"])
    assert res.exit_code == 0, res.output
    assert captured["retrieval_only"] is True
    assert captured["evidence_cosine_threshold"] == DEFAULT_EVIDENCE_COSINE_THRESHOLD
