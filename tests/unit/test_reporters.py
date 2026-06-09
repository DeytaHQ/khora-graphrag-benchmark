"""Unit tests for the JSON / Markdown / HTML reporters.

All tests run offline and write only into ``tmp_path``.
"""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from khora_graphrag_bench.harness.results import (
    BenchmarkRunResult,
    GraphConstructionDetail,
    QuestionResult,
)
from khora_graphrag_bench.reporters._reference import KHORA_BASELINE
from khora_graphrag_bench.reporters.html import write_html_report
from khora_graphrag_bench.reporters.json_writer import write_json_report
from khora_graphrag_bench.reporters.markdown import write_markdown_report


def _question(**overrides) -> QuestionResult:
    base = {
        "question_id": "q1",
        "question": "Who held the office of judex selectus?",
        "question_type": "FB",
        "difficulty": "easy",
        "discipline": "literature",
        "gold_answer": "Ovid",
        "generated_answer": "Ovid",
        "evidence_retrieved": ["Ovid held the office of judex selectus."],
        "evidence_expected": ["Ovid held the office of judex selectus."],
        "answer_correct": True,
        "answer_score": 1.0,
        "r_score": 0.9,
        "ar_metric": 0.8,
        "retrieval_metrics": {"precision": 0.7, "recall": 0.6},
        "latency_ms": 42.0,
        "context_tokens": 256,
    }
    base.update(overrides)
    return QuestionResult(**base)


def _full_result(**overrides) -> BenchmarkRunResult:
    """A realistic, fully-populated run result."""
    base = {
        "run_id": "run-2026",
        "started_at": datetime(2026, 6, 1, 12, 0, 0),
        "completed_at": datetime(2026, 6, 1, 12, 30, 0),
        "adapter_name": "khora",
        "dataset_name": "graphrag_bench_novel",
        "dataset_hash": "deadbeef",
        "sample_mode": "medium",
        "num_documents": 3,
        "num_questions": 2,
        "judge_model": "gpt-4o-mini",
        "construction": GraphConstructionDetail(
            num_nodes=1234,
            num_edges=5678,
            num_communities=12,
            construction_time_ms=90000.0,
            avg_degree=4.6,
            density=0.01,
        ),
        "aggregate_metrics": {
            "mean_r_score": 0.60,
            "mean_ar_metric": 0.45,
            "mean_answer_score": 0.70,
            "accuracy": 0.80,
            "coverage": 0.65,
            "rouge_l": 0.48,
            "faithfulness": 0.79,
            "context_relevance": 0.33,
            "evidence_recall": 0.84,
            "cost_usd": 6.10,
        },
        "by_difficulty": {
            "easy": {"n": 1, "accuracy": 1.0, "mean_r_score": 0.9, "mean_ar_metric": 0.8},
            "hard": {"n": 1, "accuracy": 0.0, "mean_r_score": 0.3, "mean_ar_metric": 0.2},
        },
        "by_question_type": {
            "FB": {"n": 1, "accuracy": 1.0, "mean_r_score": 0.9, "mean_ar_metric": 0.8},
            "OE": {"n": 1, "accuracy": 0.0, "mean_r_score": 0.3, "mean_ar_metric": 0.2},
        },
        "per_question": [
            _question(),
            _question(
                question_id="q2",
                difficulty="hard",
                question_type="OE",
                answer_correct=False,
                answer_score=0.0,
                error="judge timeout",
            ),
        ],
        "cost_usd": 6.10,
        "runtime_seconds": 1800.0,
        "errors": ["q2: judge timeout"],
        "khora_version": "0.17.0",
    }
    base.update(overrides)
    return BenchmarkRunResult(**base)


def _minimal_result(**overrides) -> BenchmarkRunResult:
    """An edge-case run: no construction, empty metrics/breakdowns/questions."""
    base = {
        "run_id": "run-min",
        "started_at": datetime(2026, 6, 1, 12, 0, 0),
        "completed_at": datetime(2026, 6, 1, 12, 0, 5),
        "adapter_name": "khora",
        "dataset_name": "graphrag_bench_novel",
        "dataset_hash": "0",
        "sample_mode": "small",
        "num_documents": 0,
        "num_questions": 0,
        "judge_model": "gpt-4o-mini",
        "construction": None,
        "aggregate_metrics": {},
        "by_difficulty": {},
        "by_question_type": {},
        "per_question": [],
    }
    base.update(overrides)
    return BenchmarkRunResult(**base)


# --------------------------------------------------------------------------- #
# JSON reporter
# --------------------------------------------------------------------------- #
class TestJsonReporter:
    def test_writes_file_and_returns_path(self, tmp_path):
        out = write_json_report(_full_result(), tmp_path)
        assert out == tmp_path / "report.json"
        assert out.exists()

    def test_creates_nested_out_dir(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c"
        out = write_json_report(_full_result(), nested)
        assert out.exists()

    def test_round_trips_and_top_level_keys(self, tmp_path):
        out = write_json_report(_full_result(), tmp_path)
        payload = json.loads(out.read_text())
        assert payload["schema_version"] == "1.0"
        assert payload["khora_reference_baseline"] == KHORA_BASELINE
        assert "result" in payload

    def test_result_section_has_expected_keys(self, tmp_path):
        out = write_json_report(_full_result(), tmp_path)
        result = json.loads(out.read_text())["result"]
        for key in (
            "run_id",
            "adapter_name",
            "aggregate_metrics",
            "by_difficulty",
            "by_question_type",
            "per_question",
            "construction",
            "khora_version",
        ):
            assert key in result

    def test_datetime_serialised_as_isoformat(self, tmp_path):
        out = write_json_report(_full_result(), tmp_path)
        result = json.loads(out.read_text())["result"]
        assert result["started_at"] == "2026-06-01T12:00:00"
        assert result["completed_at"] == "2026-06-01T12:30:00"

    def test_per_question_serialised_with_nested_fields(self, tmp_path):
        out = write_json_report(_full_result(), tmp_path)
        result = json.loads(out.read_text())["result"]
        assert len(result["per_question"]) == 2
        first = result["per_question"][0]
        assert first["question_id"] == "q1"
        assert first["retrieval_metrics"] == {"precision": 0.7, "recall": 0.6}
        assert first["context_tokens"] == 256

    def test_construction_serialised(self, tmp_path):
        out = write_json_report(_full_result(), tmp_path)
        construction = json.loads(out.read_text())["result"]["construction"]
        assert construction["num_nodes"] == 1234
        assert construction["num_edges"] == 5678

    def test_minimal_result_round_trips(self, tmp_path):
        out = write_json_report(_minimal_result(), tmp_path)
        result = json.loads(out.read_text())["result"]
        assert result["construction"] is None
        assert result["per_question"] == []
        assert result["aggregate_metrics"] == {}

    def test_string_out_dir_accepted(self, tmp_path):
        out = write_json_report(_full_result(), str(tmp_path))
        assert out.exists()

    def test_unserialisable_value_raises_typeerror(self, tmp_path):
        result = _minimal_result(aggregate_metrics={"bad": object()})
        with pytest.raises(TypeError):
            write_json_report(result, tmp_path)


# --------------------------------------------------------------------------- #
# Markdown reporter
# --------------------------------------------------------------------------- #
class TestMarkdownReporter:
    def test_writes_file_and_returns_path(self, tmp_path):
        out = write_markdown_report(_full_result(), tmp_path)
        assert out == tmp_path / "report.md"
        assert out.exists()
        assert out.read_text().strip()

    def test_contains_header_and_metadata(self, tmp_path):
        text = write_markdown_report(_full_result(), tmp_path).read_text()
        assert "# GraphRAG-Bench results — khora" in text
        assert "`run-2026`" in text
        assert "2026-06-01T12:00:00" in text
        assert "Sample mode" in text and "`medium`" in text

    def test_contains_construction_section(self, tmp_path):
        text = write_markdown_report(_full_result(), tmp_path).read_text()
        assert "## Phase 1 — graph construction" in text
        assert "1,234" in text  # nodes thousands-formatted
        assert "5,678" in text

    def test_headline_metric_rows_present(self, tmp_path):
        text = write_markdown_report(_full_result(), tmp_path).read_text()
        assert "## Headline metrics" in text
        for label in (
            "mean_r_score",
            "mean_ar_metric",
            "mean_answer_score",
            "accuracy",
            "coverage",
            "rouge_l",
            "faithfulness",
            "context_relevance",
            "evidence_recall",
        ):
            assert f"`{label}`" in text
        # local value formatted to 4 dp
        assert "0.8000" in text
        # reference baseline value visible (derived from the source of truth so
        # this stays correct when the baseline is refreshed)
        assert f"{KHORA_BASELINE['accuracy']:.4f}" in text

    def test_cost_and_runtime_rows(self, tmp_path):
        text = write_markdown_report(_full_result(), tmp_path).read_text()
        assert "`cost_usd`" in text
        assert "`runtime_min`" in text
        # runtime: 1800s -> 30.0 min
        assert "30.0" in text

    def test_delta_signs(self, tmp_path):
        # A local value above the baseline must render a positive delta with a
        # leading "+". Derive the local value from the baseline so this holds
        # regardless of the baseline's current value.
        res = _full_result()
        res.aggregate_metrics["accuracy"] = KHORA_BASELINE["accuracy"] + 0.05
        text = write_markdown_report(res, tmp_path).read_text()
        assert "+0.050" in text

    def test_breakdown_tables(self, tmp_path):
        text = write_markdown_report(_full_result(), tmp_path).read_text()
        assert "## Breakdown by difficulty" in text
        assert "`easy`" in text and "`hard`" in text
        assert "## Breakdown by question type" in text
        assert "`FB`" in text and "`OE`" in text

    def test_errors_section(self, tmp_path):
        text = write_markdown_report(_full_result(), tmp_path).read_text()
        assert "## Errors" in text
        assert "q2: judge timeout" in text

    def test_errors_truncated_over_twenty(self, tmp_path):
        errs = [f"err-{i}" for i in range(25)]
        result = _full_result(errors=errs)
        text = write_markdown_report(result, tmp_path).read_text()
        assert "and 5 more" in text
        assert "err-24" not in text  # beyond the first 20

    def test_minimal_result_omits_optional_sections(self, tmp_path):
        text = write_markdown_report(_minimal_result(), tmp_path).read_text()
        assert "## Phase 1" not in text
        assert "## Breakdown by difficulty" not in text
        assert "## Breakdown by question type" not in text
        assert "## Errors" not in text
        # missing local metrics rendered as em-dash placeholder
        assert "—" in text

    def test_missing_metric_renders_dash(self, tmp_path):
        # accuracy absent -> the accuracy row uses the em-dash for the local value
        result = _full_result(aggregate_metrics={"mean_r_score": 0.60})
        text = write_markdown_report(result, tmp_path).read_text()
        # find the accuracy row line
        accuracy_line = next(ln for ln in text.splitlines() if ln.startswith("| `accuracy`"))
        assert "—" in accuracy_line


# --------------------------------------------------------------------------- #
# HTML reporter
# --------------------------------------------------------------------------- #
class TestHtmlReporter:
    def test_writes_file_and_returns_path(self, tmp_path):
        out = write_html_report(_full_result(), tmp_path)
        assert out == tmp_path / "report.html"
        assert out.exists()

    def test_non_empty_well_formed_shell(self, tmp_path):
        html = write_html_report(_full_result(), tmp_path).read_text()
        assert html.strip()
        assert html.startswith("<!DOCTYPE html>")
        assert "</html>" in html

    def test_contains_metadata(self, tmp_path):
        html = write_html_report(_full_result(), tmp_path).read_text()
        assert "GraphRAG-Bench — khora" in html
        assert "run-2026" in html
        assert "medium" in html
        assert "gpt-4o-mini" in html

    def test_headline_table_rows(self, tmp_path):
        html = write_html_report(_full_result(), tmp_path).read_text()
        assert "Headline metrics" in html
        for label in (
            "mean_r_score",
            "accuracy",
            "evidence_recall",
            "cost_usd",
            "runtime_min",
        ):
            assert f"<code>{label}</code>" in html

    def test_delta_classes(self, tmp_path):
        html = write_html_report(_full_result(), tmp_path).read_text()
        assert "delta-pos" in html  # accuracy beats baseline
        assert "delta-neg" in html  # context_relevance/coverage below baseline

    def test_construction_section(self, tmp_path):
        html = write_html_report(_full_result(), tmp_path).read_text()
        assert "Phase 1 — graph construction" in html
        assert "1,234" in html
        assert "5,678" in html

    def test_breakdown_sections(self, tmp_path):
        html = write_html_report(_full_result(), tmp_path).read_text()
        assert "Breakdown by difficulty" in html
        assert "Breakdown by question type" in html
        assert "<code>easy</code>" in html
        assert "<code>FB</code>" in html

    def test_errors_section(self, tmp_path):
        html = write_html_report(_full_result(), tmp_path).read_text()
        assert "<h2>Errors</h2>" in html
        assert "q2: judge timeout" in html

    def test_errors_truncated_over_twenty(self, tmp_path):
        result = _full_result(errors=[f"err-{i}" for i in range(25)])
        html = write_html_report(result, tmp_path).read_text()
        assert "and 5 more" in html

    def test_reference_footnote(self, tmp_path):
        html = write_html_report(_full_result(), tmp_path).read_text()
        assert KHORA_BASELINE["khora_version"] in html
        assert KHORA_BASELINE["dataset"] in html

    def test_minimal_result_omits_optional_sections(self, tmp_path):
        html = write_html_report(_minimal_result(), tmp_path).read_text()
        assert html.startswith("<!DOCTYPE html>")
        assert "Phase 1 — graph construction" not in html
        assert "Breakdown by difficulty" not in html
        assert "Breakdown by question type" not in html
        assert "<h2>Errors</h2>" not in html
        # missing local metric -> em-dash placeholder in headline rows
        assert "—" in html

    def test_no_cost_row_when_absent(self, tmp_path):
        # cost_usd absent from aggregate_metrics -> no cost_usd headline code cell
        result = _full_result(aggregate_metrics={"accuracy": 0.80})
        html = write_html_report(result, tmp_path).read_text()
        assert "<code>cost_usd</code>" not in html
        # runtime row is always present
        assert "<code>runtime_min</code>" in html


def test_reports_show_reliability_banner_when_questions_errored(tmp_path):
    """An elevated error_rate must surface a 'not comparable to baseline' banner."""
    result = _full_result(
        aggregate_metrics={"mean_answer_score": 0.70, "accuracy": 0.80, "error_count": 1.0, "error_rate": 0.5}
    )
    md = write_markdown_report(result, tmp_path).read_text()
    assert "questions errored" in md
    assert "not comparable to the reference baseline" in md

    html = write_html_report(result, tmp_path).read_text()
    assert "questions errored" in html
    assert "not comparable to the reference baseline" in html


def test_reports_no_reliability_banner_on_clean_run(tmp_path):
    """A clean run (no error_rate) shows no reliability banner."""
    md = write_markdown_report(_full_result(), tmp_path).read_text()
    assert "questions errored" not in md
    html = write_html_report(_full_result(), tmp_path).read_text()
    assert "questions errored" not in html
