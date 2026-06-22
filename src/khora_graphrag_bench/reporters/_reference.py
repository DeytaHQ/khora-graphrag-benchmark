"""Canonical Khora reference numbers — published baseline for comparison.

These are the headline graphrag_full / khora numbers produced with the same
code and config that ships in this repo. The reporters render them alongside
the user's local results so contributors can see how their machine + Khora
version compare to the published baseline.

Quality metrics (accuracy, answer_score, coverage, rouge_l, faithfulness,
context_relevance, evidence_recall) are the mean of
two independent full-sampling runs against khora 0.18.5 on
graphrag_bench_novel (2010 questions) with a gpt-4o-mini judge and
paper-aligned prompts. A third full run on the same khora version executed on
a Fedora 44 x86_64 host returned numbers within run-to-run LLM-judge noise of
these means on 8 of 9 quality metrics, confirming cross-platform
reproducibility.

runtime_minutes and cost_usd are the mean of the two developer-machine
(Apple Silicon) full runs, since those reflect what you'll see locally;
both depend on hardware and on OpenAI pricing. Cost in particular varies
~30-40% run-to-run because LiteLLM's judge cache (``.cache/khora-graphrag-bench/llm_judge/``)
hits on the deterministic-seed questions in subsequent runs.

Retrieval-side metrics can land a few points lower on slower machines where
Neo4j transaction contention bites during ingestion; see the README
"Reproducibility" section.

Date captured: 2026-06-06. Refreshed after the khora 0.18.4 / 0.18.5 release
cycle (retrieval correctness + reranker robustness fixes) and harness updates
for a stronger reranker with a wider candidate pool and full entity extraction
at skeleton_core_ratio=1.0.

NOTE (2026-06-09): ``r_score`` / ``ar_metric`` were removed. They reproduce a
metric from a different, name-colliding benchmark (arXiv 2506.02404), not the
GraphRAG-Bench paper this harness runs (arXiv 2506.05690), and our dataset
carries no gold rationale field for that metric to score against. The values
below also predate the uniform answer-generation prompt and the
coverage/evidence_recall denominator fix, both of which shift several numbers;
regenerate this baseline from a fresh full run before republishing.
"""

KHORA_BASELINE = {
    "khora_version": "0.18.5",
    "judge_model": "gpt-4o-mini",
    "dataset": "graphrag_bench_novel",
    "sampling": "full",
    "num_questions": 2010,
    "mean_answer_score": 0.702,
    "accuracy": 0.809,
    "coverage": 0.760,
    "rouge_l": 0.470,
    "faithfulness": 0.788,
    "context_relevance": 0.348,
    "evidence_recall": 0.887,
    "runtime_minutes": 514,
    "cost_usd": 4.56,
}
