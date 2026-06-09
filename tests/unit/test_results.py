"""Unit tests for the result dataclasses in ``harness/results.py``."""

from __future__ import annotations

from dataclasses import fields
from datetime import datetime

import pytest

from khora_graphrag_bench.harness.results import (
    BenchmarkRunResult,
    GraphConstructionDetail,
    QuestionResult,
)


def _make_question(**overrides) -> QuestionResult:
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
    }
    base.update(overrides)
    return QuestionResult(**base)


class TestQuestionResult:
    def test_required_fields_and_defaults(self):
        q = _make_question()
        assert q.question_id == "q1"
        assert q.answer_correct is True
        # defaulted optional fields
        assert q.retrieval_metrics == {}
        assert q.latency_ms == 0.0
        assert q.context_tokens is None
        assert q.error is None

    def test_default_factory_is_independent(self):
        a = _make_question(question_id="a")
        b = _make_question(question_id="b")
        a.retrieval_metrics["precision"] = 0.5
        assert b.retrieval_metrics == {}

    def test_optional_overrides(self):
        q = _make_question(
            retrieval_metrics={"precision": 0.7, "recall": 0.6},
            latency_ms=123.4,
            context_tokens=512,
            error="boom",
        )
        assert q.retrieval_metrics["precision"] == 0.7
        assert q.latency_ms == 123.4
        assert q.context_tokens == 512
        assert q.error == "boom"

    def test_missing_required_field_raises(self):
        with pytest.raises(TypeError):
            QuestionResult(question_id="q1")  # type: ignore[call-arg]


class TestGraphConstructionDetail:
    def test_required_and_defaults(self):
        c = GraphConstructionDetail(
            num_nodes=10,
            num_edges=20,
            num_communities=3,
            construction_time_ms=1500.0,
        )
        assert c.num_nodes == 10
        assert c.num_edges == 20
        assert c.num_communities == 3
        assert c.construction_time_ms == 1500.0
        assert c.avg_degree == 0.0
        assert c.density == 0.0

    def test_optional_overrides(self):
        c = GraphConstructionDetail(
            num_nodes=1,
            num_edges=1,
            num_communities=1,
            construction_time_ms=1.0,
            avg_degree=2.5,
            density=0.33,
        )
        assert c.avg_degree == 2.5
        assert c.density == 0.33


class TestBenchmarkRunResult:
    def _make_run(self, **overrides) -> BenchmarkRunResult:
        base = {
            "run_id": "run-1",
            "started_at": datetime(2026, 6, 1, 12, 0, 0),
            "completed_at": datetime(2026, 6, 1, 12, 30, 0),
            "adapter_name": "khora",
            "dataset_name": "graphrag_bench_novel",
            "dataset_hash": "abc123",
            "sample_mode": "small",
            "num_documents": 1,
            "num_questions": 2,
            "judge_model": "gpt-4o-mini",
            "construction": None,
            "aggregate_metrics": {"accuracy": 0.5},
            "by_difficulty": {},
            "by_question_type": {},
            "per_question": [],
        }
        base.update(overrides)
        return BenchmarkRunResult(**base)

    def test_required_fields_and_defaults(self):
        run = self._make_run()
        assert run.run_id == "run-1"
        assert run.construction is None
        assert run.per_question == []
        # defaulted optional fields
        assert run.cost_usd == 0.0
        assert run.runtime_seconds == 0.0
        assert run.errors == []
        assert run.khora_version == ""

    def test_default_factory_errors_independent(self):
        a = self._make_run(run_id="a")
        b = self._make_run(run_id="b")
        a.errors.append("oops")
        assert b.errors == []

    def test_holds_nested_objects(self):
        construction = GraphConstructionDetail(num_nodes=5, num_edges=4, num_communities=1, construction_time_ms=10.0)
        run = self._make_run(
            construction=construction,
            per_question=[_make_question()],
            cost_usd=1.23,
            runtime_seconds=600.0,
            errors=["a", "b"],
            khora_version="0.17.0",
        )
        assert run.construction is construction
        assert len(run.per_question) == 1
        assert run.cost_usd == 1.23
        assert run.runtime_seconds == 600.0
        assert run.errors == ["a", "b"]
        assert run.khora_version == "0.17.0"

    def test_field_names_match_reporter_expectations(self):
        names = {f.name for f in fields(BenchmarkRunResult)}
        expected = {
            "run_id",
            "started_at",
            "completed_at",
            "adapter_name",
            "sample_mode",
            "num_documents",
            "num_questions",
            "judge_model",
            "construction",
            "aggregate_metrics",
            "by_difficulty",
            "by_question_type",
            "per_question",
            "cost_usd",
            "runtime_seconds",
            "errors",
        }
        assert expected.issubset(names)
