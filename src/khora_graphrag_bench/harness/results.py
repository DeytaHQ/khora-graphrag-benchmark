"""Result types produced by the benchmark runner."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class QuestionResult:
    """Per-question result row written into the JSON report."""

    question_id: str
    question: str
    question_type: str
    difficulty: str
    discipline: str
    gold_answer: str
    generated_answer: str
    evidence_retrieved: list[str]
    evidence_expected: list[str]
    answer_correct: bool
    answer_score: float
    r_score: float
    ar_metric: float
    retrieval_metrics: dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0
    context_tokens: int | None = None
    error: str | None = None


@dataclass
class GraphConstructionDetail:
    """Result of phase 1 (graph build)."""

    num_nodes: int
    num_edges: int
    num_communities: int
    construction_time_ms: float
    avg_degree: float = 0.0
    density: float = 0.0


@dataclass
class BenchmarkRunResult:
    """Top-level run result containing aggregates + per-question detail.

    Reporters consume this; JSON/Markdown/HTML are all functions of this
    object. ``aggregate_metrics`` carries the headline scores; ``by_difficulty``
    and ``by_question_type`` are convenience breakdowns; ``per_question`` is the
    full detail for debugging and reproducibility checks.
    """

    run_id: str
    started_at: datetime
    completed_at: datetime
    adapter_name: str
    dataset_name: str
    dataset_hash: str
    sample_mode: str  # "small" | "medium" | "full"
    num_documents: int
    num_questions: int
    judge_model: str
    construction: GraphConstructionDetail | None
    aggregate_metrics: dict[str, float]
    by_difficulty: dict[str, dict[str, float]]
    by_question_type: dict[str, dict[str, float]]
    per_question: list[QuestionResult]
    cost_usd: float = 0.0
    runtime_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)
    khora_version: str = ""
