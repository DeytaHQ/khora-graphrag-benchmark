"""Canonical Khora reference numbers, the published baseline for comparison.

These are the headline graphrag_full / khora numbers produced with the same
code and config that ships in this repo. The reporters render them alongside
the user's local results so contributors can see how their machine + Khora
version compare to the published baseline.

Quality metrics (accuracy, mean_answer_score, coverage, rouge_l, faithfulness,
context_relevance, evidence_recall) are the mean of two independent full
sampling runs against ``khora==0.21.0`` on graphrag_bench_novel (2010
questions) with a gpt-4o-mini judge and paper-aligned prompts. Both runs
share the same bench commit (the post PR #3 main: uniform answer prompt and
the coverage/evidence_recall denominator fix). 6 of 7 quality metrics agreed
to within +/- 0.005 across the two runs; only ``faithfulness`` showed a wider
+/- 0.02 band, a property of the metric itself (the judge decomposes the
generated answer, and answer generation is not seeded so the statement set
varies run to run).

runtime_minutes and cost_usd are the mean of the same two developer-machine
(Apple Silicon) full runs, since those reflect what you'll see locally;
both depend on hardware and on OpenAI pricing. Cost in particular varies
roughly 10-40% run to run because LiteLLM's judge cache
(``.cache/khora-graphrag-bench/llm_judge/``) hits on the deterministic-seed
questions in subsequent runs.

Retrieval-side metrics can land a few points lower on slower machines where
Neo4j transaction contention bites during ingestion; see the README
"Reproducibility" section.

Date captured: 2026-06-25. Refreshed after khora 0.21.0 released. Supersedes
the 2026-06-06 0.18.5 baseline, which predated the harness PR #3 uniform
answer prompt and the coverage/evidence_recall denominator fix.
"""

KHORA_BASELINE = {
    "khora_version": "0.21.0",
    "judge_model": "gpt-4o-mini",
    "dataset": "graphrag_bench_novel",
    "sampling": "full",
    "num_questions": 2010,
    "mean_answer_score": 0.694,
    "accuracy": 0.799,
    "coverage": 0.711,
    "rouge_l": 0.439,
    "faithfulness": 0.748,
    "context_relevance": 0.352,
    "evidence_recall": 0.891,
    "runtime_minutes": 478,
    "cost_usd": 3.59,
}
