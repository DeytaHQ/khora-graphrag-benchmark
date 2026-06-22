"""Unit tests for the CLI.

These cover argument parsing, the pure helpers (``_print_summary``,
``_resolve_run_dir``, ``_require_openai_key``, ``_result_from_json``) and the
``report`` command's filesystem handling. No benchmark ``run`` that needs
services is executed.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from khora_graphrag_bench import cli


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def results_root(tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the CLI at a temp results dir for the duration of a test."""
    root = Path(tmp_path) / "results"
    monkeypatch.setattr(cli, "RESULTS_ROOT", root)
    return root


# ---------------------------------------------------------------------------
# Top-level group
# ---------------------------------------------------------------------------


def test_version(runner: CliRunner) -> None:
    result = runner.invoke(cli.main, ["--version"])
    assert result.exit_code == 0
    assert "khora-graphrag-bench" in result.output


def test_help_lists_commands(runner: CliRunner) -> None:
    result = runner.invoke(cli.main, ["--help"])
    assert result.exit_code == 0
    assert "run" in result.output
    assert "report" in result.output


def test_run_rejects_bad_sample(runner: CliRunner) -> None:
    result = runner.invoke(cli.main, ["run", "--sample", "bogus"])
    assert result.exit_code != 0
    assert "Invalid value" in result.output


# ---------------------------------------------------------------------------
# _require_openai_key
# ---------------------------------------------------------------------------


def test_require_openai_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SystemExit) as exc:
        cli._require_openai_key()
    assert exc.value.code == 1


def test_require_openai_key_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    # Should not raise / exit.
    assert cli._require_openai_key() is None


def test_run_aborts_without_key(runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = runner.invoke(cli.main, ["run"])
    assert result.exit_code == 1
    assert "OPENAI_API_KEY is not set" in result.output


# ---------------------------------------------------------------------------
# _print_summary
# ---------------------------------------------------------------------------


def test_print_summary_renders_known_metrics(capsys: pytest.CaptureFixture) -> None:
    agg = {
        "mean_answer_score": 0.5,
        "accuracy": 0.75,
        "rouge_l": 0.40,
    }
    cli._print_summary(agg)
    out = capsys.readouterr().out
    assert "metric" in out
    assert "Khora ref" in out
    assert "mean_answer_score" in out
    assert "0.5000" in out  # local value formatted to 4 dp
    assert "accuracy" in out
    # R-Score/AR were dropped (non-native metric from a different benchmark).
    assert "mean_r_score" not in out


def test_print_summary_missing_local_shows_dash(capsys: pytest.CaptureFixture) -> None:
    cli._print_summary({})  # no local metrics at all
    out = capsys.readouterr().out
    # Reference column still printed; local column shows the em-dash placeholder.
    assert "mean_answer_score" in out
    assert "—" in out
    assert "mean_r_score" not in out


# ---------------------------------------------------------------------------
# _resolve_run_dir
# ---------------------------------------------------------------------------


def test_resolve_run_dir_no_results_dir(results_root: Path) -> None:
    assert not results_root.exists()
    with pytest.raises(SystemExit) as exc:
        cli._resolve_run_dir(None)
    assert exc.value.code == 1


def test_resolve_run_dir_explicit_id(results_root: Path) -> None:
    target = results_root / "run-123"
    target.mkdir(parents=True)
    assert cli._resolve_run_dir("run-123") == target


def test_resolve_run_dir_explicit_id_missing(results_root: Path) -> None:
    results_root.mkdir(parents=True)
    with pytest.raises(SystemExit) as exc:
        cli._resolve_run_dir("nope")
    assert exc.value.code == 1


def test_resolve_run_dir_latest_symlink(results_root: Path) -> None:
    results_root.mkdir(parents=True)
    real = results_root / "run-abc"
    real.mkdir()
    (results_root / "latest").symlink_to("run-abc")
    assert cli._resolve_run_dir(None).resolve() == real.resolve()


def test_resolve_run_dir_most_recent_by_mtime(results_root: Path) -> None:
    results_root.mkdir(parents=True)
    old = results_root / "run-old"
    new = results_root / "run-new"
    old.mkdir()
    new.mkdir()
    import os

    os.utime(old, (1000, 1000))
    os.utime(new, (2000, 2000))
    assert cli._resolve_run_dir(None) == new


def test_resolve_run_dir_no_candidates(results_root: Path) -> None:
    results_root.mkdir(parents=True)  # empty
    with pytest.raises(SystemExit) as exc:
        cli._resolve_run_dir(None)
    assert exc.value.code == 1


# ---------------------------------------------------------------------------
# report command
# ---------------------------------------------------------------------------


def test_report_no_report_json(runner: CliRunner, results_root: Path) -> None:
    run_dir = results_root / "run-x"
    run_dir.mkdir(parents=True)
    result = runner.invoke(cli.main, ["report", "--run-id", "run-x"])
    assert result.exit_code == 1
    assert "No report.json found" in result.output


def _minimal_report_dict() -> dict:
    return {
        "result": {
            "run_id": "run-x",
            "started_at": "2026-06-01T00:00:00",
            "completed_at": "2026-06-01T00:10:00",
            "adapter_name": "khora",
            "dataset_name": "graphrag_bench_novel",
            "dataset_hash": "deadbeef",
            "sample_mode": "small",
            "num_documents": 1,
            "num_questions": 2,
            "aggregate_metrics": {"mean_r_score": 0.5, "accuracy": 0.75},
        }
    }


def test_report_regenerates_json(runner: CliRunner, results_root: Path) -> None:
    run_dir = results_root / "run-x"
    run_dir.mkdir(parents=True)
    (run_dir / "report.json").write_text(json.dumps(_minimal_report_dict()))

    result = runner.invoke(cli.main, ["report", "--run-id", "run-x", "--format", "json"])
    assert result.exit_code == 0, result.output
    assert (run_dir / "report.json").exists()
    assert "report.json" in result.output


def test_report_all_formats(runner: CliRunner, results_root: Path) -> None:
    run_dir = results_root / "run-x"
    run_dir.mkdir(parents=True)
    (run_dir / "report.json").write_text(json.dumps(_minimal_report_dict()))

    result = runner.invoke(cli.main, ["report", "--run-id", "run-x", "--format", "all"])
    assert result.exit_code == 0, result.output
    assert (run_dir / "report.json").exists()
    assert (run_dir / "report.md").exists()
    assert (run_dir / "report.html").exists()


def test_report_rejects_bad_format(runner: CliRunner) -> None:
    result = runner.invoke(cli.main, ["report", "--format", "xml"])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# _result_from_json
# ---------------------------------------------------------------------------


def test_result_from_json_minimal() -> None:
    result = cli._result_from_json(_minimal_report_dict()["result"])
    assert result.run_id == "run-x"
    assert result.adapter_name == "khora"
    assert result.judge_model == "gpt-4o-mini"  # default for old reports
    assert result.construction is None
    assert result.cost_usd == 0.0
    assert result.aggregate_metrics["accuracy"] == 0.75


def test_result_from_json_with_construction() -> None:
    d = _minimal_report_dict()["result"]
    d["construction"] = {
        "num_nodes": 10,
        "num_edges": 20,
        "num_communities": 2,
        "construction_time_ms": 500.0,
        "avg_degree": 4.0,
        "density": 0.1,
    }
    result = cli._result_from_json(d)
    assert result.construction is not None
    assert result.construction.num_nodes == 10
    assert result.construction.num_edges == 20


# ---------------------------------------------------------------------------
# RESULTS_ROOT env wiring
# ---------------------------------------------------------------------------


def test_results_root_honors_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BENCH_RESULTS_DIR", "/tmp/kgb-custom-results")
    reloaded = importlib.reload(cli)
    try:
        assert reloaded.RESULTS_ROOT == Path("/tmp/kgb-custom-results")
    finally:
        monkeypatch.delenv("BENCH_RESULTS_DIR", raising=False)
        importlib.reload(cli)
