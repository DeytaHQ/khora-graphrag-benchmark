"""Benchmark execution loop for the graphrag pipeline.

Orchestrates the three-phase evaluation against a single ``GraphRAGAdapter``:

  1. ``build_graph(documents)`` once
  2. ``graph_search(question)`` + ``generate_answer(question, context)`` per question
  3. Score: deterministic for MC/MS/TF, LLM-judged for FB/OE; plus ``r_score``,
     ``ar_metric``, and the per-difficulty auxiliaries (context_relevance,
     evidence_recall, coverage, faithfulness, rouge_l)

Single adapter, single suite.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from datetime import UTC, datetime
from uuid import uuid4

from khora_graphrag_bench.datasets.schema import GraphRAGDataset, GraphRAGQuestion
from khora_graphrag_bench.harness.base import Document, GraphRAGAdapter
from khora_graphrag_bench.harness.evaluation import (
    compute_answer_correctness,
    compute_answer_correctness_llm,
    compute_ar_metric,
    compute_context_relevance,
    compute_coverage_score,
    compute_evidence_recall,
    compute_faithfulness_score,
    compute_graph_construction_metrics,
    compute_r_score,
    compute_rouge_l,
    get_metrics_for_level,
)
from khora_graphrag_bench.harness.results import (
    BenchmarkRunResult,
    GraphConstructionDetail,
    QuestionResult,
)
from khora_graphrag_bench.harness.token_counting import count_context_tokens

logger = logging.getLogger(__name__)


# Sampling fractions chosen to match published smoke-test conventions.
SAMPLE_FRACTIONS = {"small": 0.05, "medium": 0.30, "full": 1.0}

# When doc-first sampling drops below this number of questions, fall back to
# question-first sampling so smoke tests still produce meaningful signal.
MIN_SAMPLE_QUESTIONS = 30

# Maximum questions answered concurrently. Caps OpenAI request burst.
DEFAULT_QUERY_CONCURRENCY = 5

# A run with more than this fraction of errored questions is flagged unreliable:
# aggregates over the survivors may not be comparable to the reference baseline.
ERROR_RATE_RELIABILITY_THRESHOLD = 0.02


# ---------------------------------------------------------------------------
# Cost tracking via litellm success callback
# ---------------------------------------------------------------------------


class _CostTracker:
    """Sums per-call cost from litellm's success callbacks for this run.

    Registers on **both** ``litellm.success_callback`` (sync ``completion``
    path) and ``litellm._async_success_callback`` (async ``acompletion``
    path). The async path is the load-bearing one for this benchmark —
    every judge call and the adapter's answer-generation call goes through
    ``acompletion``, so registering only on the sync list (the obvious
    public API) silently drops 100% of costs.

    Cost is read defensively from a few well-known fields because litellm
    populates ``response_cost`` in different places depending on version /
    code path:

    * ``kwargs["response_cost"]`` — most builds
    * ``kwargs["standard_logging_object"]["response_cost"]`` — newer logging payloads
    * ``getattr(completion_response, "_hidden_params", {})["response_cost"]`` — fallback
    """

    def __init__(self) -> None:
        self.total_cost: float = 0.0
        self._registered = False
        self._handler: object | None = None

    def _extract_cost(self, kwargs: dict, completion_response) -> float:
        cost = kwargs.get("response_cost")
        if cost is None:
            payload = kwargs.get("standard_logging_object") or {}
            cost = payload.get("response_cost") if isinstance(payload, dict) else None
        if cost is None:
            hidden = getattr(completion_response, "_hidden_params", None) or {}
            if isinstance(hidden, dict):
                cost = hidden.get("response_cost")
        try:
            return float(cost) if cost is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

    def start(self) -> None:
        import litellm

        tracker = self

        def _on_success(kwargs, completion_response, start_time, end_time):  # noqa: ARG001
            try:
                tracker.total_cost += tracker._extract_cost(kwargs, completion_response)
            except Exception as e:  # noqa: BLE001 — must not break the run, but log so cost gaps aren't silent
                logger.warning("cost tracker: failed to extract cost from a completion: %s", e)

        async def _on_async_success(kwargs, completion_response, start_time, end_time):  # noqa: ARG001
            try:
                tracker.total_cost += tracker._extract_cost(kwargs, completion_response)
            except Exception as e:  # noqa: BLE001
                logger.warning("cost tracker: failed to extract cost from a completion: %s", e)

        # Store both handlers so stop() can remove them cleanly.
        self._handler = (_on_success, _on_async_success)
        litellm.success_callback = (litellm.success_callback or []) + [_on_success]
        litellm._async_success_callback = (litellm._async_success_callback or []) + [_on_async_success]
        self._registered = True

    def stop(self) -> None:
        if not self._registered:
            return
        try:
            import litellm

            sync_handler, async_handler = self._handler  # type: ignore[misc]
            if sync_handler in (litellm.success_callback or []):
                litellm.success_callback.remove(sync_handler)
            if async_handler in (litellm._async_success_callback or []):
                litellm._async_success_callback.remove(async_handler)
        except Exception:  # noqa: S110
            pass
        self._registered = False


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def _apply_sampling(dataset: GraphRAGDataset, sample_mode: str) -> tuple[list[GraphRAGQuestion], list[Document]]:
    """Trim the dataset to a sampled subset, returning ``(questions, documents)``.

    Sampling strategy:

    * **Doc-first**: pick a fraction of documents (deterministic seed=42),
      then retain only questions whose ``relevant_doc_ids`` are a subset of
      the sampled doc ids. This matches `q.relevant_doc_ids` ⊆ retained_ids
      so questions whose specific evidence chunks weren't sampled get dropped.
    * **Question-first fallback**: if doc-first leaves fewer than
      ``MIN_SAMPLE_QUESTIONS``, swap strategies: pick the questions first,
      include the docs they need, then pad with up to 4× as many distractor
      docs from the rest of the corpus.

    The seed is fixed at 42 so reruns at the same sample mode hit the same
    questions / docs.
    """
    fraction = SAMPLE_FRACTIONS.get(sample_mode, 1.0)
    all_docs = dataset.documents
    all_qs = dataset.questions

    if fraction >= 1.0:
        documents = [Document(d.doc_id, d.content, d.title, dict(d.metadata)) for d in all_docs]
        return list(all_qs), documents

    rng = random.Random(42)  # noqa: S311 — deterministic sampling is intentional
    n_docs = max(1, int(len(all_docs) * fraction))
    sampled_docs = rng.sample(list(all_docs), n_docs)
    retained_ids = {d.doc_id for d in sampled_docs}
    sampled_qs = [q for q in all_qs if set(q.relevant_doc_ids).issubset(retained_ids)]

    if len(sampled_qs) < MIN_SAMPLE_QUESTIONS and len(all_qs) >= MIN_SAMPLE_QUESTIONS:
        # Doc-first under-sampled; switch to question-first.
        n_q = MIN_SAMPLE_QUESTIONS
        sampled_qs = rng.sample(list(all_qs), n_q)
        required_ids = {did for q in sampled_qs for did in q.relevant_doc_ids}
        required_docs = [d for d in all_docs if d.doc_id in required_ids]
        other_docs = [d for d in all_docs if d.doc_id not in required_ids]
        distractor_budget = min(len(other_docs), len(required_docs) * 4)
        distractors = rng.sample(other_docs, distractor_budget) if distractor_budget > 0 else []
        sampled_docs = required_docs + distractors
        logger.warning(
            "sample=%s yielded <%d questions; switched to question-first: %d questions, %d docs",
            sample_mode,
            MIN_SAMPLE_QUESTIONS,
            len(sampled_qs),
            len(sampled_docs),
        )

    documents = [Document(d.doc_id, d.content, d.title, dict(d.metadata)) for d in sampled_docs]
    logger.info("sample=%s: %d docs, %d questions", sample_mode, len(documents), len(sampled_qs))
    return sampled_qs, documents


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class BenchmarkRunner:
    """Drives the three-phase graphrag pipeline against a single adapter."""

    def __init__(
        self,
        adapter: GraphRAGAdapter,
        dataset: GraphRAGDataset,
        *,
        sample_mode: str = "full",
        top_k: int = 5,
        judge_model: str = "gpt-4o-mini",
        query_concurrency: int = DEFAULT_QUERY_CONCURRENCY,
    ) -> None:
        if sample_mode not in SAMPLE_FRACTIONS:
            raise ValueError(f"sample_mode must be one of {sorted(SAMPLE_FRACTIONS)}, got {sample_mode!r}")
        self.adapter = adapter
        self.dataset = dataset
        self.sample_mode = sample_mode
        self.top_k = top_k
        self.judge_model = judge_model
        self._sem = asyncio.Semaphore(query_concurrency)

    async def run(self) -> BenchmarkRunResult:
        """Execute the benchmark and return a populated ``BenchmarkRunResult``."""
        questions, documents = _apply_sampling(self.dataset, self.sample_mode)

        run_id = uuid4().hex[:12]
        started_at = datetime.now(UTC)
        suite_start = time.perf_counter()

        cost = _CostTracker()
        cost.start()

        await self.adapter.setup()
        errors: list[str] = []
        construction: GraphConstructionDetail | None = None
        per_question: list[QuestionResult] = []
        try:
            # ----- Phase 1: build the graph --------------------------------
            logger.info("Phase 1: building graph from %d documents", len(documents))
            build_t0 = time.perf_counter()
            build_result = await self.adapter.build_graph(documents)
            build_elapsed = (time.perf_counter() - build_t0) * 1000
            graph_metrics = compute_graph_construction_metrics(
                build_result.num_nodes, build_result.num_edges, num_chunks=len(documents)
            )
            construction = GraphConstructionDetail(
                num_nodes=build_result.num_nodes,
                num_edges=build_result.num_edges,
                num_communities=build_result.num_communities,
                construction_time_ms=build_result.construction_time_ms or build_elapsed,
                avg_degree=graph_metrics.get("avg_degree", 0.0),
                density=graph_metrics.get("density", 0.0),
            )
            logger.info(
                "Graph built: %d nodes, %d edges in %.0fms",
                build_result.num_nodes,
                build_result.num_edges,
                build_elapsed,
            )

            if build_result.num_nodes == 0:
                errors.append(
                    "Graph construction produced 0 entities — extraction may have failed; "
                    "skipping retrieval and generation"
                )
                logger.error(errors[-1])
            else:
                # ----- Phase 2 + 3: per-question loop ----------------------
                logger.info("Phase 2+3: answering %d questions", len(questions))
                per_question = list(await asyncio.gather(*[self._answer_question(q) for q in questions]))
                # Surface per-question exceptions into the run-level errors list
                errors.extend(r.error for r in per_question if r.error)
        finally:
            try:
                await self.adapter.teardown()
            except Exception as e:  # noqa: BLE001
                logger.warning("adapter teardown raised: %s", e)
            cost.stop()

        completed_at = datetime.now(UTC)
        runtime_seconds = time.perf_counter() - suite_start

        # ----- Aggregation -----------------------------------------------------
        valid = [r for r in per_question if r.error is None]
        n = len(valid) or 1
        error_count = len(per_question) - len(valid)
        error_rate = error_count / len(per_question) if per_question else 0.0
        if error_rate > ERROR_RATE_RELIABILITY_THRESHOLD:
            logger.warning(
                "%d/%d questions errored (%.1f%%); aggregate metrics are computed over the %d "
                "successful questions only and may not be comparable to the reference baseline.",
                error_count,
                len(per_question),
                error_rate * 100,
                len(valid),
            )

        aggregate: dict[str, float] = {}
        if valid:
            aggregate = {
                "accuracy": sum(1 for r in valid if r.answer_correct) / n,
                "mean_answer_score": sum(r.answer_score for r in valid) / n,
                "mean_r_score": sum(r.r_score for r in valid) / n,
                "mean_ar_metric": sum(r.ar_metric for r in valid) / n,
            }
            ct = [r.context_tokens for r in valid if r.context_tokens is not None]
            if ct:
                aggregate["mean_context_tokens"] = sum(ct) / len(ct)
            # Average the auxiliary metrics across questions that produced them.
            for key in ("context_relevance", "evidence_recall", "coverage", "faithfulness", "rouge_l"):
                values = [r.retrieval_metrics[key] for r in valid if key in r.retrieval_metrics]
                if values:
                    aggregate[key] = sum(values) / len(values)

        # ----- Breakdowns ------------------------------------------------------
        by_difficulty = _group_aggregate(valid, key=lambda r: r.difficulty)
        by_question_type = _group_aggregate(valid, key=lambda r: r.question_type)

        # ----- Cost ------------------------------------------------------------
        if cost.total_cost > 0:
            aggregate["cost_usd"] = cost.total_cost
            aggregate["cost_per_query_usd"] = cost.total_cost / max(len(questions), 1)
            correct = sum(1 for r in valid if r.answer_correct)
            if correct:
                aggregate["cost_per_correct_answer_usd"] = cost.total_cost / correct

        # Always surface failure rate so a degraded run is visible rather than
        # silently averaging over a shrunken denominator.
        aggregate["error_count"] = float(error_count)
        aggregate["error_rate"] = round(error_rate, 4)

        return BenchmarkRunResult(
            run_id=run_id,
            started_at=started_at,
            completed_at=completed_at,
            adapter_name=self.adapter.name,
            dataset_name=self.dataset.name,
            dataset_hash=self.dataset.compute_hash(),
            sample_mode=self.sample_mode,
            num_documents=len(documents),
            num_questions=len(questions),
            judge_model=self.judge_model,
            construction=construction,
            aggregate_metrics=aggregate,
            by_difficulty=by_difficulty,
            by_question_type=by_question_type,
            per_question=per_question,
            cost_usd=cost.total_cost,
            runtime_seconds=runtime_seconds,
            errors=errors,
            khora_version=getattr(self.adapter, "adapter_version", ""),
        )

    # ----- Per-question pipeline --------------------------------------------

    async def _answer_question(self, q: GraphRAGQuestion) -> QuestionResult:
        """Phase 2+3 for a single question, with full per-question try/except."""
        async with self._sem:
            t0 = time.perf_counter()
            try:
                search_results = await self.adapter.graph_search(q.question, top_k=self.top_k)

                evidence_retrieved: list[str] = []
                for sr in search_results:
                    evidence_retrieved.extend(sr.evidence)
                    if not sr.evidence:
                        evidence_retrieved.append(sr.content[:200])

                # Per-question proxy for "retrieved context size" — sum of
                # tiktoken counts across each search result's content. The
                # adapter constructs the actual prompt inside generate_answer;
                # this is the closest harness-side approximation and feeds
                # the aggregate mean_context_tokens metric.
                q_context_tokens: int | None = None
                if search_results:
                    q_context_tokens = count_context_tokens([sr.content for sr in search_results])

                # Generate
                generated_answer = ""
                gen_evidence: list[str] = []
                if search_results:
                    gen = await self.adapter.generate_answer(q.question, search_results)
                    generated_answer = gen.answer
                    gen_evidence = list(gen.evidence)
                if not generated_answer.strip():
                    # Don't let judges score "" as wrong — produce an explicit refusal.
                    generated_answer = "I don't have enough information to answer this."

                # Answer correctness
                qt = q.question_type.upper()
                answer_score = compute_answer_correctness(generated_answer, q.gold_answer, qt)
                if qt in ("FB", "OE") and generated_answer:
                    answer_score = await compute_answer_correctness_llm(
                        question=q.question,
                        generated=generated_answer,
                        gold=q.gold_answer,
                        judge_model=self.judge_model,
                    )

                # R score: prefer the adapter's focused rationale when present,
                # else fall back to the joined raw evidence (older adapter contract).
                generated_rationale = "\n".join(gen_evidence) if gen_evidence else "\n".join(evidence_retrieved)
                gold_rationale = "\n".join(q.evidence)
                r_score_val = 0.0
                if generated_rationale and gold_rationale:
                    r_score_val = await compute_r_score(
                        generated_rationale=generated_rationale,
                        gold_rationale=gold_rationale,
                        question=q.question,
                        judge_model=self.judge_model,
                    )
                ar = compute_ar_metric(answer_score, r_score_val)

                # Auxiliaries
                extra: dict[str, float] = {}
                if search_results:
                    context_text = "\n\n".join(sr.content for sr in search_results)
                    if context_text.strip():
                        extra["context_relevance"] = await compute_context_relevance(
                            q.question, context_text, q.evidence, self.judge_model
                        )
                        extra["evidence_recall"] = await compute_evidence_recall(
                            q.question, context_text, q.evidence, self.judge_model
                        )

                if generated_answer:
                    applicable = get_metrics_for_level(q.difficulty)
                    if "rouge_l" in applicable:
                        extra["rouge_l"] = compute_rouge_l(generated_answer, q.gold_answer)
                    if "coverage" in applicable:
                        extra["coverage"] = await compute_coverage_score(
                            q.question, q.gold_answer, generated_answer, self.judge_model
                        )
                    if "faithfulness" in applicable:
                        context_text = "\n\n".join(sr.content for sr in search_results)
                        extra["faithfulness"] = await compute_faithfulness_score(
                            q.question, generated_answer, context_text, self.judge_model
                        )

                latency_ms = (time.perf_counter() - t0) * 1000
                return QuestionResult(
                    question_id=q.question_id,
                    question=q.question,
                    question_type=q.question_type,
                    difficulty=q.difficulty,
                    discipline=q.discipline,
                    gold_answer=q.gold_answer,
                    generated_answer=generated_answer,
                    evidence_retrieved=evidence_retrieved,
                    evidence_expected=list(q.evidence),
                    answer_correct=answer_score >= 0.5,
                    answer_score=answer_score,
                    r_score=r_score_val,
                    ar_metric=ar,
                    retrieval_metrics=extra,
                    latency_ms=latency_ms,
                    context_tokens=q_context_tokens,
                )
            except Exception as e:  # noqa: BLE001
                latency_ms = (time.perf_counter() - t0) * 1000
                logger.warning("Question %s failed: %s", q.question_id, e)
                return QuestionResult(
                    question_id=q.question_id,
                    question=q.question,
                    question_type=q.question_type,
                    difficulty=q.difficulty,
                    discipline=q.discipline,
                    gold_answer=q.gold_answer,
                    generated_answer="",
                    evidence_retrieved=[],
                    evidence_expected=list(q.evidence),
                    answer_correct=False,
                    answer_score=0.0,
                    r_score=0.0,
                    ar_metric=0.0,
                    retrieval_metrics={},
                    latency_ms=latency_ms,
                    error=f"{type(e).__name__}: {e}",
                )


def _group_aggregate(rows: list[QuestionResult], *, key) -> dict[str, dict[str, float]]:
    """Group ``rows`` by ``key(row)`` and compute per-group metric means."""
    out: dict[str, dict[str, float]] = {}
    groups: dict[str, list[QuestionResult]] = {}
    for r in rows:
        groups.setdefault(key(r), []).append(r)
    for label, group in groups.items():
        n = len(group)
        agg = {
            "n": float(n),
            "accuracy": sum(1 for r in group if r.answer_correct) / n,
            "mean_answer_score": sum(r.answer_score for r in group) / n,
            "mean_r_score": sum(r.r_score for r in group) / n,
            "mean_ar_metric": sum(r.ar_metric for r in group) / n,
        }
        for mk in ("context_relevance", "evidence_recall", "coverage", "faithfulness", "rouge_l"):
            values = [r.retrieval_metrics[mk] for r in group if mk in r.retrieval_metrics]
            if values:
                agg[mk] = sum(values) / len(values)
        out[label] = agg
    return out
