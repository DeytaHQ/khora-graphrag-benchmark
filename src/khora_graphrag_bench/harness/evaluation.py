"""GraphRAG-Bench (ICLR'26) evaluation metrics.

Implements the published evaluation methodology end-to-end:

* Answer correctness for the five question types (MC, MS, TF, FB, OE)
* R Score — rationale quality vs gold rationale
* AR Metric — answer × rationale combined score
* Context relevance, evidence recall, coverage, faithfulness
* ROUGE-L
* Graph construction structural metrics

The LLM-judged prompts include the few-shot examples from the paper's reference
implementation (https://github.com/GraphRAG-Bench/GraphRAG-Benchmark) — they
materially stabilise verdicts vs zero-shot variants.

Judge calls are cached to disk by (PROMPT_VERSION, model, prompt) SHA-256 so
re-running a benchmark on the same dataset is cheap. Bump ``PROMPT_VERSION``
when you change any prompt body — it invalidates cached entries cleanly.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

from khora_graphrag_bench.harness.model_utils import is_reasoning_model
from khora_graphrag_bench.harness.text_utils import sanitize_text

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Disk-backed cache for LLM judge calls
# ---------------------------------------------------------------------------

_judge_cache: dict[str, dict] = {}
_judge_cache_dir: Path | None = None

# Bump whenever any judge prompt body or response shape changes. Every existing
# cache entry becomes a stale key and is naturally re-computed.
PROMPT_VERSION = "v3"


def _init_cache(cache_dir: str = ".cache/khora-graphrag-bench/llm_judge") -> Path:
    global _judge_cache_dir
    if _judge_cache_dir is None:
        _judge_cache_dir = Path(cache_dir)
        _judge_cache_dir.mkdir(parents=True, exist_ok=True)
    return _judge_cache_dir


def _cache_key(model: str, prompt: str) -> str:
    payload = f"{PROMPT_VERSION}|{model}|{prompt}"
    return hashlib.sha256(payload.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Deterministic scoring for MC / MS / TF question types
# ---------------------------------------------------------------------------


def extract_option_letter(text: str) -> str:
    """Extract a single option letter (A-D) from generated text."""
    text = text.strip()
    patterns = [
        r"\b([A-Da-d])\)",
        r"\(([A-Da-d])\)",
        r"\b([A-Da-d])\.",
        r"(?:answer|choice|option)\s+(?:is\s+)?([A-Da-d])\b",
        r"^([A-Da-d])$",
        r"^([A-Da-d])\b",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    m = re.search(r"\b([A-D])\b", text)
    if m:
        return m.group(1)
    return text.strip()[:1].upper()


def extract_option_set(text: str) -> set[str]:
    """Extract a set of option letters from multi-select answer text."""
    return {c.upper() for c in re.findall(r"[A-Da-d]", text)}


def extract_tf_answer(text: str) -> str:
    """Normalise a true/false answer to lowercase ``"true"`` / ``"false"``."""
    text = text.strip().lower()
    if text in ("true", "t", "yes", "correct"):
        return "true"
    if text in ("false", "f", "no", "incorrect"):
        return "false"
    if "true" in text:
        return "true"
    if "false" in text:
        return "false"
    return text


def score_mc(generated: str, gold: str) -> float:
    """Multiple choice: binary exact match on option letter."""
    return 1.0 if extract_option_letter(generated) == extract_option_letter(gold) else 0.0


def score_ms(generated: str, gold: str) -> float:
    """Multi-select: 1.0 for exact-set match, 0.5 for strict subset, else 0."""
    gen_set = extract_option_set(generated)
    gold_set = extract_option_set(gold)
    if gen_set == gold_set:
        return 1.0
    if gen_set.issubset(gold_set) and len(gen_set) > 0:
        return 0.5
    return 0.0


def score_tf(generated: str, gold: str) -> float:
    """True/false: binary exact match after normalisation."""
    return 1.0 if extract_tf_answer(generated) == extract_tf_answer(gold) else 0.0


def compute_answer_correctness(generated: str, gold: str, question_type: str) -> float:
    """Deterministic correctness for MC/MS/TF; returns 0 for FB/OE (LLM-judged)."""
    qt = question_type.upper().strip()
    if qt == "MC":
        return score_mc(generated, gold)
    if qt == "MS":
        return score_ms(generated, gold)
    if qt == "TF":
        return score_tf(generated, gold)
    return 0.0  # FB and OE require LLM-judged scoring; see compute_answer_correctness_llm


# ---------------------------------------------------------------------------
# Judge prompts (paper-aligned with GraphRAG-Bench reference implementation)
# ---------------------------------------------------------------------------

_STATEMENT_GENERATION_PROMPT = """Given a question and an answer, analyze the complexity of each sentence in the answer. Break down each sentence into one or more fully understandable statements. Ensure that no pronouns are used in any statement. Format the outputs in JSON.

Example Input:
Question: Who was Albert Einstein and what is he best known for?
Answer: He was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time. He was best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics.

Example Output:
["Albert Einstein was a German-born theoretical physicist.", "Albert Einstein is recognized as one of the greatest and most influential physicists of all time.", "Albert Einstein was best known for developing the theory of relativity.", "Albert Einstein also made important contributions to the development of the theory of quantum mechanics."]

Input Text:
Question: {question}
Answer: {answer}

Generated Statements:"""

# Few-shot examples adapted from GraphRAG-Bench `CORRECTNESS_EXAMPLES`
# (Evaluation/metrics/answer_accuracy.py). The worked TP/FP/FN walkthroughs
# materially stabilise the F-beta classification used by r_score and ar_metric.
_CORRECTNESS_CLASSIFICATION_PROMPT = """Given a ground truth and answer statements, analyze each statement and classify them in one of the following categories:

- **TP (true positive)**: statements present in the answer that are also directly supported by one or more statements in ground truth
- **FP (false positive)**: statements present in the answer but not directly supported by any statement in ground truth
- **FN (false negative)**: statements found in the ground truth but not present in the answer

Each statement can only belong to one category. Provide a reason for each classification.

Respond with a JSON object containing:
- "TP": list of {{"statement": "...", "reason": "..."}}
- "FP": list of {{"statement": "...", "reason": "..."}}
- "FN": list of {{"statement": "...", "reason": "..."}}

### Examples

Example 1:
Question: What powers the sun and what is its primary function?
Answer Statements: ["The sun is powered by nuclear fission, similar to nuclear reactors on Earth.", "The primary function of the sun is to provide light to the solar system."]
Ground Truth Statements: ["The sun is powered by nuclear fusion, where hydrogen atoms fuse to form helium.", "This fusion process in the sun's core releases a tremendous amount of energy.", "The energy from the sun provides heat and light, which are essential for life on Earth.", "The sun's light plays a critical role in Earth's climate system.", "Sunlight helps to drive the weather and ocean currents."]
Output:
{{
  "TP": [
    {{"statement": "The primary function of the sun is to provide light to the solar system.", "reason": "Somewhat supported by the ground truth mentioning the sun providing light, though the truth focuses more broadly on energy."}}
  ],
  "FP": [
    {{"statement": "The sun is powered by nuclear fission, similar to nuclear reactors on Earth.", "reason": "Contradicts the ground truth, which states the sun is powered by nuclear fusion."}}
  ],
  "FN": [
    {{"statement": "The sun is powered by nuclear fusion, where hydrogen atoms fuse to form helium.", "reason": "Not included in the answer."}},
    {{"statement": "This fusion process in the sun's core releases a tremendous amount of energy.", "reason": "Not mentioned in the answer."}},
    {{"statement": "The energy from the sun provides heat and light, which are essential for life on Earth.", "reason": "Heat and life-essential aspects are omitted."}},
    {{"statement": "The sun's light plays a critical role in Earth's climate system.", "reason": "Not addressed in the answer."}},
    {{"statement": "Sunlight helps to drive the weather and ocean currents.", "reason": "Omitted in the answer."}}
  ]
}}

Example 2:
Question: What is the boiling point of water?
Answer Statements: ["The boiling point of water is 100 degrees Celsius at sea level"]
Ground Truth Statements: ["The boiling point of water is 100 degrees Celsius (212 degrees Fahrenheit) at sea level.", "The boiling point of water can change with altitude."]
Output:
{{
  "TP": [
    {{"statement": "The boiling point of water is 100 degrees Celsius at sea level", "reason": "Directly supported by the ground truth statement specifying 100 degrees Celsius at sea level."}}
  ],
  "FP": [],
  "FN": [
    {{"statement": "The boiling point of water can change with altitude.", "reason": "Not mentioned in the answer."}}
  ]
}}

### Current Analysis
Question: {question}
Answer Statements: {answer_statements}
Ground Truth Statements: {ground_truth_statements}"""

# Worked example adapted from GraphRAG-Bench
# (Evaluation/metrics/context_relevance_v2.py).
_CONTEXT_RELEVANCE_PROMPT = """You are evaluating the relevance of a retrieved context passage to a question.

### Task
Given a question, associated evidence, and a retrieved context, score the context's relevance on a 0-2 scale:

- **2 (Highly Relevant)**: The context directly answers the question or is essential for understanding the evidence.
- **1 (Partially Relevant)**: The context provides related background information but does not directly answer the question.
- **0 (Not Relevant)**: The context covers completely different topics and is not useful.

Respond ONLY with a JSON object containing:
- "reason": A brief explanation (1 sentence)
- "relevance_score": An integer (0, 1, or 2)

### Example
Input:
Question: "What is the capital of Australia?"
Evidence: ["Canberra is the capital of Australia"]
Context: "The capital of Australia is Canberra, a planned city located between Sydney and Melbourne."

Output:
{{"reason": "The context directly confirms that Canberra is the capital of Australia, matching the question and evidence.", "relevance_score": 2}}

### Actual Input
Question: "{question}"
Evidence: {evidence}
Context: "{context}"

### Your Response:"""

# Worked example adapted from GraphRAG-Bench (Evaluation/metrics/evidence_recall.py).
_EVIDENCE_RECALL_PROMPT = """### Task
You are given a list of evidence statements and a retrieved Context. For each evidence statement, determine whether it can be attributed to the Context.

Respond ONLY with a JSON object containing a "classifications" list. Each item should include:
- "statement": the exact evidence string
- "reason": a brief explanation (1 sentence)
- "attributed": 1 if the evidence can be attributed to the Context, otherwise 0

### Example
Input:
Context: "Einstein won the Nobel Prize in 1921 for physics."
Evidence: ["Einstein received the Nobel Prize", "He was born in Germany"]

Output:
{{
  "classifications": [
    {{"statement": "Einstein received the Nobel Prize", "reason": "Matches the context mentioning the Nobel Prize.", "attributed": 1}},
    {{"statement": "He was born in Germany", "reason": "Birth information is not in the context.", "attributed": 0}}
  ]
}}

### Actual Input
Context: "{context}"
Evidence: {evidence}
Question: "{question}" (for reference only)

### Your Response:"""

# Worked example adapted from GraphRAG-Bench (Evaluation/metrics/coverage.py).
_COVERAGE_FACT_EXTRACTION_PROMPT = """You are given a question and a reference answer. Break down the reference answer into a list of distinct factual statements (facts) that could be independently verified. Output them as a JSON list of strings under the "facts" field.

### Example
Input:
Question: "What causes seasons?"
Reference Answer: "Seasonal changes result from Earth's axial tilt. This tilt causes different hemispheres to receive varying sunlight."

Output:
{{"facts": ["Seasonal changes result from Earth's axial tilt", "The axial tilt causes different hemispheres to receive varying sunlight"]}}

### Actual Input
Question: {question}
Reference Answer: {reference}

Output:"""

# The paper's FACT_COVERAGE_PROMPT threads ``question`` through to the LLM so
# the model can disambiguate facts whose coverage depends on question intent;
# this prompt and its caller (compute_coverage_score) follow that pattern.
_COVERAGE_FACT_CHECK_PROMPT = """For each factual statement from the reference, determine if it is covered in the response.

Respond ONLY with a JSON object containing a "classifications" list. Each item should have:
- "statement": the exact fact
- "attributed": 1 if the fact is covered in the response, 0 if not

### Example
Question: "What causes seasons?"
Response: "Seasons are caused by Earth's tilted axis"
Reference Facts: ["Seasonal changes result from Earth's axial tilt", "The axial tilt causes different hemispheres to receive varying sunlight"]

Output:
{{
  "classifications": [
    {{"statement": "Seasonal changes result from Earth's axial tilt", "attributed": 1}},
    {{"statement": "The axial tilt causes different hemispheres to receive varying sunlight", "attributed": 0}}
  ]
}}

### Actual Input
Question: "{question}"
Response: "{response}"
Reference Facts: {facts}

Output:"""

# Worked examples adapted from GraphRAG-Bench `FAITHFULNESS_EXAMPLES`
# (Evaluation/metrics/faithfulness.py). The pair shows a mixed-faithfulness
# case (4 statements, 1 supported) and a fully orthogonal case.
_FAITHFULNESS_VERDICT_PROMPT = """Given a list of statements and a context, determine whether each statement can be inferred from the context.

For each statement, provide:
- "statement": the exact statement
- "verdict": 1 if the statement is supported by the context, 0 if not
- "reason": brief explanation

Respond with a JSON object containing a "verdicts" list.

### Examples

Example 1:
Context: John is a student at XYZ University. He is pursuing a degree in Computer Science. He is enrolled in several courses this semester, including Data Structures, Algorithms, and Database Management. John is a diligent student and spends a significant amount of time studying and completing assignments. He often stays late in the library to work on his projects.
Statements: ["John is majoring in Biology.", "John is taking a course on Artificial Intelligence.", "John is a dedicated student.", "John has a part-time job."]
Output:
{{
  "verdicts": [
    {{"statement": "John is majoring in Biology.", "verdict": 0, "reason": "John's major is Computer Science, not Biology."}},
    {{"statement": "John is taking a course on Artificial Intelligence.", "verdict": 0, "reason": "AI is not among the courses listed."}},
    {{"statement": "John is a dedicated student.", "verdict": 1, "reason": "The context describes John as diligent and staying late in the library, implying dedication."}},
    {{"statement": "John has a part-time job.", "verdict": 0, "reason": "No information about a part-time job is provided."}}
  ]
}}

Example 2:
Context: Photosynthesis is a process used by plants, algae, and certain bacteria to convert light energy into chemical energy.
Statements: ["Albert Einstein was a genius."]
Output:
{{
  "verdicts": [
    {{"statement": "Albert Einstein was a genius.", "verdict": 0, "reason": "The context and statement are unrelated."}}
  ]
}}

### Current Analysis
Context: {context}
Statements: {statements}

Output:"""


# ---------------------------------------------------------------------------
# LLM judge call (with disk + memory cache)
# ---------------------------------------------------------------------------


def _judge_completion_params(model: str) -> dict[str, Any]:
    """Per-model kwargs for the judge completion call.

    Reasoning models (GPT-5, o-series) reject ``temperature`` != 1 and ``seed``,
    and meter output via ``max_completion_tokens`` - which also funds the hidden
    reasoning tokens, so it needs more headroom than gpt-4o-mini's 4096. We cap
    reasoning at ``low`` since judging is bounded classification. gpt-4o-mini
    keeps the original deterministic params so the baseline series stays
    reproducible; reverting is just passing ``--judge-model gpt-4o-mini``.
    """
    if is_reasoning_model(model):
        return {"max_completion_tokens": 16384, "reasoning_effort": "low"}
    return {"temperature": 0.0, "max_tokens": 4096, "seed": 42}


async def llm_judge(
    prompt: str,
    model: str = "gpt-4o-mini",
    cache_dir: str = ".cache/khora-graphrag-bench/llm_judge",
) -> dict[str, Any]:
    """Call an LLM and parse the JSON response, with two-tier caching.

    Cached by (PROMPT_VERSION, model, prompt) SHA-256 so changing the judge
    model or any prompt body produces a fresh cache key automatically.

    Returns ``{}`` for a successful-but-unparseable response (callers handle that
    gracefully). Raises after ``max_retries`` consecutive API failures so the
    caller's question is flagged as errored rather than silently scored 0.
    """
    import litellm

    prompt = sanitize_text(prompt)
    cache_path = _init_cache(cache_dir)
    key = _cache_key(model, prompt)

    if key in _judge_cache:
        return _judge_cache[key]

    disk_path = cache_path / f"{key}.json"
    if disk_path.exists():
        try:
            result = json.loads(disk_path.read_text())
            _judge_cache[key] = result
            return result
        except Exception:  # noqa: S110
            pass  # corrupt entry — re-compute

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = await litellm.acompletion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                **_judge_completion_params(model),
            )
            text = response.choices[0].message.content or ""
            result = _parse_json_response(text)
            _judge_cache[key] = result
            try:
                disk_path.write_text(json.dumps(result))
            except Exception:  # noqa: S110
                pass  # cache write is best-effort
            return result
        except Exception as e:
            if attempt < max_retries - 1:
                import asyncio

                wait = 2 ** (attempt + 1)
                logger.warning("LLM judge attempt %d failed, retrying in %ds: %s", attempt + 1, wait, e)
                await asyncio.sleep(wait)
            else:
                # Don't silently return {}: a persistent judge failure would zero
                # the question's scores invisibly. Raise so the runner flags the
                # question as errored (surfaced via error_rate), not fake-scored 0.
                logger.error("LLM judge failed after %d attempts: %s", max_retries, e)
                raise
    raise RuntimeError("llm_judge: retries exhausted")  # unreachable


def _parse_json_response(text: str) -> dict[str, Any]:
    """Parse a JSON object from an LLM response, tolerating markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        match = re.search(r"\[[\s\S]*\]", text)
        if match:
            try:
                return {"items": json.loads(match.group())}
            except json.JSONDecodeError:
                pass
    return {}


# ---------------------------------------------------------------------------
# LLM-judged answer scoring (FB / OE) and R Score
# ---------------------------------------------------------------------------


async def _generate_statements(question: str, answer: str, model: str = "gpt-4o-mini") -> list[str]:
    """Decompose an answer into atomic statements via the LLM."""
    prompt = _STATEMENT_GENERATION_PROMPT.format(question=question, answer=answer)
    result = await llm_judge(prompt, model=model)
    if isinstance(result, dict) and "items" in result:
        return result["items"]
    if isinstance(result, list):
        return result
    for v in result.values():
        if isinstance(v, list):
            return v
    return [answer]


async def _classify_statements(
    question: str,
    answer_statements: list[str],
    ground_truth_statements: list[str],
    model: str = "gpt-4o-mini",
) -> dict[str, list]:
    """Classify each generated statement as TP, FP, or FN."""
    prompt = _CORRECTNESS_CLASSIFICATION_PROMPT.format(
        question=question,
        answer_statements=json.dumps(answer_statements),
        ground_truth_statements=json.dumps(ground_truth_statements),
    )
    result = await llm_judge(prompt, model=model)
    return {
        "TP": result.get("TP", []),
        "FP": result.get("FP", []),
        "FN": result.get("FN", []),
    }


def _compute_f_beta(tp: int, fp: int, fn: int, beta: float = 1.0) -> float:
    """Compute F-beta from TP/FP/FN counts."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)


async def _compute_semantic_similarity(
    text_a: str,
    text_b: str,
    embedding_model: str = "text-embedding-3-small",
) -> float:
    """Cosine similarity between two texts, rescaled from [-1, 1] to [0, 1]."""
    import litellm
    import numpy as np

    # No try/except: a persistent embedding failure must propagate so the runner
    # flags the question as errored (excluded from aggregates, surfaced via
    # error_rate) rather than silently scoring a fabricated 0.5.
    resp_a = await litellm.aembedding(model=embedding_model, input=[text_a], num_retries=2)
    resp_b = await litellm.aembedding(model=embedding_model, input=[text_b], num_retries=2)
    vec_a = np.array(resp_a.data[0]["embedding"])
    vec_b = np.array(resp_b.data[0]["embedding"])
    cosine = float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b) + 1e-10))
    return (cosine + 1) / 2


async def compute_answer_correctness_llm(
    question: str,
    generated: str,
    gold: str,
    judge_model: str = "gpt-4o-mini",
    w_factuality: float = 0.75,
    w_semantic: float = 0.25,
    beta: float = 1.0,
    embedding_model: str = "text-embedding-3-small",
) -> float:
    """LLM-judged correctness for FB / OE answers.

    Combines an F-beta score over decomposed statements (weight 0.75) with the
    embedding cosine of full answer vs gold (weight 0.25). Matches the paper.
    """
    gen_statements = await _generate_statements(question, generated, model=judge_model)
    gold_statements = await _generate_statements(question, gold, model=judge_model)

    classification = await _classify_statements(question, gen_statements, gold_statements, model=judge_model)
    tp = len(classification.get("TP", []))
    fp = len(classification.get("FP", []))
    fn = len(classification.get("FN", []))

    factuality = _compute_f_beta(tp, fp, fn, beta=beta)
    semantic_sim = await _compute_semantic_similarity(generated, gold, embedding_model=embedding_model)

    return w_factuality * factuality + w_semantic * semantic_sim


# ---------------------------------------------------------------------------
# R Score (rationale quality)
# ---------------------------------------------------------------------------


async def compute_r_score(
    generated_rationale: str,
    gold_rationale: str,
    question: str,
    judge_model: str = "gpt-4o-mini",
    w_factuality: float = 0.75,
    w_semantic: float = 0.25,
    beta: float = 1.0,
    embedding_model: str = "text-embedding-3-small",
) -> float:
    """Reuse the answer-correctness methodology on rationales rather than answers.

    The adapter is expected to return a focused, short rationale (1–5 atomic
    statements) from ``generate_answer`` — apples-to-apples comparison with the
    gold rationale's size. Stuffing the raw retrieved context here destroys
    precision in the F-beta scorer and deflates the score.
    """
    if not generated_rationale or not gold_rationale:
        return 0.0
    return await compute_answer_correctness_llm(
        question=question,
        generated=generated_rationale,
        gold=gold_rationale,
        judge_model=judge_model,
        w_factuality=w_factuality,
        w_semantic=w_semantic,
        beta=beta,
        embedding_model=embedding_model,
    )


# ---------------------------------------------------------------------------
# AR Metric
# ---------------------------------------------------------------------------


def compute_ar_metric(answer_score: float, r_score: float) -> float:
    """AR = answer_score × r_score."""
    return answer_score * r_score


# ---------------------------------------------------------------------------
# Embedding-cosine evidence recall@k (retrieval-only mode, #12)
# ---------------------------------------------------------------------------

# Default cosine threshold above which a gold-evidence statement counts as
# "covered" by a retrieved chunk. Calibrated against the LLM-judge
# evidence_recall on a labelled slice: the embedding proxy correlated far better
# than the tested lexical proxy (r=0.34, too weak). Tunable via the run mode's
# --evidence-cosine-threshold.
DEFAULT_EVIDENCE_COSINE_THRESHOLD = 0.55


async def _embed_texts(texts: list[str], embedding_model: str = "text-embedding-3-small") -> list[list[float]]:
    """Embed ``texts`` in a single batched request, returning row-aligned vectors.

    No try/except: a persistent embedding failure must propagate so the runner
    flags the question as errored (surfaced via error_rate) rather than silently
    scoring a fabricated value. ``num_retries`` handles transient blips.
    """
    import litellm

    resp = await litellm.aembedding(model=embedding_model, input=texts, num_retries=2)
    # litellm preserves input order in resp.data; sort by index defensively in
    # case a provider ever returns them out of order.
    rows = sorted(resp.data, key=lambda d: d.get("index", 0)) if resp.data else []
    return [r["embedding"] for r in rows]


def _cosine_matrix_max(evidence_vecs: list[list[float]], chunk_vecs: list[list[float]]) -> list[float]:
    """For each evidence vector, the max cosine similarity to any chunk vector.

    Returns a list aligned with ``evidence_vecs``; ``0.0`` for an evidence row
    when there are no chunks to compare against.
    """
    import numpy as np

    if not evidence_vecs:
        return []
    if not chunk_vecs:
        return [0.0] * len(evidence_vecs)

    ev = np.asarray(evidence_vecs, dtype=np.float64)
    ch = np.asarray(chunk_vecs, dtype=np.float64)
    ev /= np.linalg.norm(ev, axis=1, keepdims=True) + 1e-10
    ch /= np.linalg.norm(ch, axis=1, keepdims=True) + 1e-10
    sims = ev @ ch.T  # (n_evidence, n_chunks)
    return [float(row.max()) for row in sims]


async def compute_evidence_recall_at_k(
    evidence: list[str],
    retrieved_chunks: list[str],
    *,
    threshold: float = DEFAULT_EVIDENCE_COSINE_THRESHOLD,
    pass_threshold: float = 0.5,
    embedding_model: str = "text-embedding-3-small",
) -> dict[str, Any]:
    """Embedding-cosine evidence_recall@k for retrieval-only scoring (#12).

    Embeds the gold ``evidence`` statements and the ``retrieved_chunks`` (one
    batched embedding call), then for each evidence statement checks whether its
    best cosine match among the retrieved chunks clears ``threshold``. This is
    the cheap ($embeddings-only), judge-free retrieval-quality signal; the tested
    lexical proxy correlated too weakly (r=0.34) so we use embeddings.

    Returns a dict with:

    * ``evidence_recall_at_k`` — fraction of gold statements covered in [0, 1].
    * ``covered`` / ``total`` — the numerator and denominator.
    * ``max_cosines`` — per-statement best cosine (for calibration).
    * ``passed`` — the per-question correct/incorrect flag McNemar A/Bs pair on:
      ``evidence_recall_at_k >= pass_threshold`` (default 0.5 = a majority of the
      gold evidence surfaced in the top-k).

    An empty gold-evidence list yields ``evidence_recall_at_k=0.0`` and
    ``passed=False`` (nothing to attribute retrieval against).
    """
    total = len(evidence)
    if total == 0:
        return {"evidence_recall_at_k": 0.0, "covered": 0, "total": 0, "max_cosines": [], "passed": False}

    # One batched request for gold evidence + chunks; split the returned rows.
    all_vecs = await _embed_texts(list(evidence) + list(retrieved_chunks), embedding_model=embedding_model)
    evidence_vecs = all_vecs[:total]
    chunk_vecs = all_vecs[total:]

    max_cosines = _cosine_matrix_max(evidence_vecs, chunk_vecs)
    covered = sum(1 for c in max_cosines if c >= threshold)
    recall = covered / total
    return {
        "evidence_recall_at_k": recall,
        "covered": covered,
        "total": total,
        "max_cosines": [round(c, 4) for c in max_cosines],
        "passed": recall >= pass_threshold,
    }


# ---------------------------------------------------------------------------
# Context relevance & evidence recall
# ---------------------------------------------------------------------------


async def compute_context_relevance(
    question: str,
    context: str,
    evidence: list[str],
    judge_model: str = "gpt-4o-mini",
) -> float:
    """0-2 LLM scoring of retrieved context relevance, normalised to [0, 1]."""
    # Chunk long contexts so each judge call stays within prompt-budget limits.
    max_len = 5000
    if len(context) > max_len:
        chunks = [context[i : i + 3000] for i in range(0, len(context), 2500)]
    else:
        chunks = [context]

    scores = []
    for chunk in chunks:
        prompt = _CONTEXT_RELEVANCE_PROMPT.format(question=question, evidence=json.dumps(evidence), context=chunk)
        result = await llm_judge(prompt, model=judge_model)
        raw = result.get("relevance_score", 0)
        scores.append(min(float(raw), 2.0) / 2.0)

    return sum(scores) / len(scores) if scores else 0.0


async def compute_evidence_recall(
    question: str,
    context: str,
    evidence: list[str],
    judge_model: str = "gpt-4o-mini",
) -> float:
    """Fraction of gold evidence statements attributable to retrieved context."""
    if not evidence:
        return 0.0
    prompt = _EVIDENCE_RECALL_PROMPT.format(context=context, evidence=json.dumps(evidence), question=question)
    result = await llm_judge(prompt, model=judge_model)
    classifications = result.get("classifications", [])
    if not classifications:
        return 0.0
    attributed = sum(1 for c in classifications if c.get("attributed", 0) == 1)
    # Denominator is the count the judge actually classified, not len(evidence):
    # the LLM occasionally returns a different count than the gold list we sent,
    # and dividing by the sent-in count diverges from the paper's reference impl
    # (Evaluation/metrics/evidence_recall.py), which uses len(classifications).
    return attributed / len(classifications)


# ---------------------------------------------------------------------------
# Coverage & faithfulness (difficulty levels 3 & 4)
# ---------------------------------------------------------------------------


async def compute_coverage_score(
    question: str,
    reference: str,
    response: str,
    judge_model: str = "gpt-4o-mini",
) -> float:
    """Fraction of gold-answer facts that appear in the generated answer.

    Two-step: extract facts from the reference, then check coverage in the
    response. ``question`` is passed to both calls to match the paper.
    """
    prompt = _COVERAGE_FACT_EXTRACTION_PROMPT.format(question=question, reference=reference)
    result = await llm_judge(prompt, model=judge_model)
    facts = result.get("facts", [])
    if not facts:
        return 0.0

    prompt2 = _COVERAGE_FACT_CHECK_PROMPT.format(question=question, facts=json.dumps(facts), response=response)
    result2 = await llm_judge(prompt2, model=judge_model)
    classifications = result2.get("classifications", [])
    if not classifications:
        return 0.0

    covered = sum(1 for c in classifications if c.get("attributed", 0) == 1)
    # Denominator is the count the judge actually classified, not len(facts):
    # matches the paper's reference impl (Evaluation/metrics/coverage.py) and
    # avoids divergence when the judge returns a different count than we sent.
    return covered / len(classifications)


async def compute_faithfulness_score(
    question: str,
    generated: str,
    context: str,
    judge_model: str = "gpt-4o-mini",
) -> float:
    """Fraction of generated statements supported by retrieved context.

    Note the denominator: ``len(verdicts)``, not ``len(statements)``. The LLM
    occasionally returns more verdicts than we sent (re-decomposing or
    duplicating); dividing by the statement count we sent in lets the score
    exceed 1.0 and corrupts the aggregate. The paper's reference impl
    (Evaluation/metrics/faithfulness.py) also uses ``len(verdicts)``.
    """
    statements = await _generate_statements(question, generated, model=judge_model)
    if not statements:
        return 0.0
    prompt = _FAITHFULNESS_VERDICT_PROMPT.format(context=context, statements=json.dumps(statements))
    result = await llm_judge(prompt, model=judge_model)
    verdicts = result.get("verdicts", [])
    if not verdicts:
        return 0.0
    supported = sum(1 for v in verdicts if v.get("verdict", 0) == 1)
    return supported / len(verdicts)


# ---------------------------------------------------------------------------
# ROUGE-L
# ---------------------------------------------------------------------------


def compute_rouge_l(generated: str, reference: str) -> float:
    """ROUGE-L F-measure between generated and reference text."""
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = scorer.score(reference, generated)
        return scores["rougeL"].fmeasure
    except ImportError:
        # Defensive fallback only: rouge-score is a hard dependency, so this never
        # fires in a locked install. It uses plain LCS (no Porter stemming), so it
        # would diverge slightly from the real library if it ever ran.
        return _simple_rouge_l(generated, reference)


def _simple_rouge_l(generated: str, reference: str) -> float:
    """LCS-based ROUGE-L fallback used when rouge_score isn't installed."""
    gen_tokens = generated.lower().split()
    ref_tokens = reference.lower().split()
    if not gen_tokens or not ref_tokens:
        return 0.0
    m, n = len(ref_tokens), len(gen_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == gen_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs_len = dp[m][n]
    precision = lcs_len / n if n > 0 else 0.0
    recall = lcs_len / m if m > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Graph construction metrics
# ---------------------------------------------------------------------------


def compute_graph_construction_metrics(
    num_nodes: int,
    num_edges: int,
    node_degrees: list[int] | None = None,
    num_chunks: int = 0,
) -> dict[str, float]:
    """Structural quality metrics for a constructed knowledge graph."""
    metrics: dict[str, float] = {"num_nodes": float(num_nodes), "num_edges": float(num_edges)}

    if num_nodes > 0:
        metrics["avg_degree"] = (2 * num_edges) / num_nodes
        max_edges_directed = num_nodes * (num_nodes - 1)
        metrics["density"] = num_edges / max_edges_directed if max_edges_directed > 0 else 0.0
    else:
        metrics["avg_degree"] = 0.0
        metrics["density"] = 0.0

    if num_nodes >= 2:
        max_edges_undirected = num_nodes * (num_nodes - 1) / 2
        metrics["graph_density"] = min(num_edges / max_edges_undirected, 1.0)
    else:
        metrics["graph_density"] = 0.0

    if num_chunks > 0:
        metrics["entities_per_chunk"] = num_nodes / num_chunks
    if num_nodes > 0:
        metrics["relationships_per_entity"] = num_edges / num_nodes

    if node_degrees is not None and len(node_degrees) > 0:
        non_isolated = sum(1 for d in node_degrees if d >= 1)
        metrics["non_isolated_node_ratio"] = non_isolated / len(node_degrees)
    elif num_nodes > 0 and num_edges > 0:
        metrics["non_isolated_node_ratio"] = min(1.0, (2 * num_edges) / num_nodes)
    else:
        metrics["non_isolated_node_ratio"] = 0.0

    return metrics


def get_metrics_for_level(difficulty: str) -> list[str]:
    """Applicable metrics for a given difficulty level (per the paper)."""
    base = ["answer_score", "r_score", "ar_metric"]
    if difficulty in ("fact_retrieval", "complex_reasoning"):
        return base + ["rouge_l"]
    if difficulty == "contextual_summarization":
        return base + ["coverage"]
    if difficulty == "creative_generation":
        return base + ["coverage", "faithfulness"]
    return base
