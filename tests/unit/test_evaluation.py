"""Unit tests for harness.evaluation.

Pure functions are tested directly. Functions that call an LLM judge or
embedding API are tested by mocking the network-bound calls
(``llm_judge`` / ``_compute_semantic_similarity``) so NO network happens.
We verify the scoring math (e.g. faithfulness denominator uses len(verdicts)).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from khora_graphrag_bench.harness import evaluation as ev

MODULE = "khora_graphrag_bench.harness.evaluation"


# ---------------------------------------------------------------------------
# extract_option_letter / extract_option_set / extract_tf_answer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("A)", "A"),
        ("(B)", "B"),
        ("C.", "C"),
        ("The answer is D", "D"),
        ("answer is a", "A"),
        ("b", "B"),
        ("A", "A"),
        ("D something", "D"),
    ],
)
def test_extract_option_letter(text, expected):
    assert ev.extract_option_letter(text) == expected


def test_extract_option_letter_fallback_first_char_uppercased():
    # No recognisable pattern: falls back to first char uppercased.
    assert ev.extract_option_letter("zebra") == "Z"


def test_extract_option_letter_empty_string():
    assert ev.extract_option_letter("") == ""


def test_extract_option_set():
    assert ev.extract_option_set("A, C and d") == {"A", "C", "D"}


def test_extract_option_set_empty():
    assert ev.extract_option_set("none here 1 2 3") == set()


@pytest.mark.parametrize(
    "text,expected",
    [
        ("true", "true"),
        ("True", "true"),
        ("T", "true"),
        ("yes", "true"),
        ("correct", "true"),
        ("false", "false"),
        ("F", "false"),
        ("no", "false"),
        ("incorrect", "false"),
        ("It is true that...", "true"),
        ("definitely false here", "false"),
    ],
)
def test_extract_tf_answer(text, expected):
    assert ev.extract_tf_answer(text) == expected


def test_extract_tf_answer_unrecognised_returns_normalised_text():
    assert ev.extract_tf_answer("Maybe") == "maybe"


# ---------------------------------------------------------------------------
# score_mc / score_ms / score_tf / compute_answer_correctness
# ---------------------------------------------------------------------------


def test_score_mc_match_and_mismatch():
    assert ev.score_mc("The answer is B", "(B)") == 1.0
    assert ev.score_mc("A", "C") == 0.0


def test_score_ms_exact_set():
    # Note: extract_option_set greedily matches every [A-Da-d] char, so prose
    # words ("and") would inject stray letters. Use bare letter tokens.
    assert ev.score_ms("A C", "C A") == 1.0


def test_score_ms_extracts_stray_letters_from_prose():
    # "and" contains 'a' and 'd' -> set becomes {A, C, D}, mismatching gold {A, C}.
    assert ev.score_ms("A and C", "A C") == 0.0


def test_score_ms_strict_subset_half_credit():
    assert ev.score_ms("A", "A B") == 0.5


def test_score_ms_empty_generated_is_zero():
    # Empty gen set is technically a subset but len==0 guard rejects it.
    assert ev.score_ms("no letters", "A B") == 0.0


def test_score_ms_superset_is_zero():
    assert ev.score_ms("A B C", "A B") == 0.0


def test_score_tf():
    assert ev.score_tf("true", "yes") == 1.0
    assert ev.score_tf("false", "true") == 0.0


def test_compute_answer_correctness_dispatch():
    assert ev.compute_answer_correctness("B", "(B)", "MC") == 1.0
    assert ev.compute_answer_correctness("A B", "A B", "MS") == 1.0
    assert ev.compute_answer_correctness("true", "yes", "TF") == 1.0


def test_compute_answer_correctness_case_and_whitespace_insensitive_type():
    assert ev.compute_answer_correctness("B", "B", "  mc  ") == 1.0


def test_compute_answer_correctness_fb_oe_return_zero():
    assert ev.compute_answer_correctness("anything", "gold", "FB") == 0.0
    assert ev.compute_answer_correctness("anything", "gold", "OE") == 0.0
    assert ev.compute_answer_correctness("anything", "gold", "unknown") == 0.0


# ---------------------------------------------------------------------------
# _compute_f_beta
# ---------------------------------------------------------------------------


def test_f_beta_perfect():
    assert ev._compute_f_beta(tp=5, fp=0, fn=0) == 1.0


def test_f_beta_all_zero():
    assert ev._compute_f_beta(tp=0, fp=0, fn=0) == 0.0


def test_f_beta_f1_value():
    # precision = 2/3, recall = 2/4 = 0.5 -> F1 = 2*(2/3*0.5)/(2/3+0.5)
    p, r = 2 / 3, 0.5
    expected = 2 * (p * r) / (p + r)
    assert ev._compute_f_beta(tp=2, fp=1, fn=2) == pytest.approx(expected)


def test_f_beta_only_fp_is_zero():
    assert ev._compute_f_beta(tp=0, fp=3, fn=0) == 0.0


def test_f_beta_only_fn_is_zero():
    assert ev._compute_f_beta(tp=0, fp=0, fn=3) == 0.0


def test_f_beta_beta_weights_recall():
    # With beta=2 (recall-weighted), tp=1 fp=1 fn=0: precision=0.5, recall=1.0
    p, r, beta = 0.5, 1.0, 2.0
    expected = (1 + beta**2) * (p * r) / (beta**2 * p + r)
    assert ev._compute_f_beta(tp=1, fp=1, fn=0, beta=beta) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# compute_ar_metric
# ---------------------------------------------------------------------------


def test_compute_ar_metric():
    assert ev.compute_ar_metric(0.8, 0.5) == pytest.approx(0.4)
    assert ev.compute_ar_metric(0.0, 1.0) == 0.0
    assert ev.compute_ar_metric(1.0, 1.0) == 1.0


# ---------------------------------------------------------------------------
# ROUGE-L
# ---------------------------------------------------------------------------


def test_rouge_l_identical_is_one():
    text = "the cat sat on the mat"
    assert ev.compute_rouge_l(text, text) == pytest.approx(1.0)


def test_rouge_l_disjoint_is_zero():
    assert ev.compute_rouge_l("apple banana", "carrot potato") == 0.0


def test_rouge_l_partial_between_zero_and_one():
    score = ev.compute_rouge_l("the cat sat", "the cat sat on the mat")
    assert 0.0 < score < 1.0


def test_simple_rouge_l_empty_inputs():
    assert ev._simple_rouge_l("", "anything") == 0.0
    assert ev._simple_rouge_l("anything", "") == 0.0


def test_simple_rouge_l_identical():
    assert ev._simple_rouge_l("a b c", "a b c") == pytest.approx(1.0)


def test_simple_rouge_l_lcs_value():
    # ref="a b c d", gen="a c d": LCS = a c d (len 3)
    # precision = 3/3 = 1.0, recall = 3/4 = 0.75 -> F = 2*1*0.75/1.75
    expected = 2 * 1.0 * 0.75 / 1.75
    assert ev._simple_rouge_l("a c d", "a b c d") == pytest.approx(expected)


def test_compute_rouge_l_falls_back_when_rouge_score_missing():
    # Force the ImportError branch by hiding rouge_score from import machinery.
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "rouge_score" or name.startswith("rouge_score."):
            raise ImportError("simulated missing rouge_score")
        return real_import(name, *args, **kwargs)

    with patch.object(builtins, "__import__", side_effect=fake_import):
        score = ev.compute_rouge_l("a b c", "a b c")
    assert score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# compute_graph_construction_metrics
# ---------------------------------------------------------------------------


def test_graph_metrics_empty_graph():
    m = ev.compute_graph_construction_metrics(num_nodes=0, num_edges=0)
    assert m["num_nodes"] == 0.0
    assert m["num_edges"] == 0.0
    assert m["avg_degree"] == 0.0
    assert m["density"] == 0.0
    assert m["graph_density"] == 0.0
    assert m["non_isolated_node_ratio"] == 0.0


def test_graph_metrics_basic_values():
    m = ev.compute_graph_construction_metrics(num_nodes=4, num_edges=3)
    assert m["avg_degree"] == pytest.approx(2 * 3 / 4)
    # directed density = edges / (n*(n-1)) = 3 / 12
    assert m["density"] == pytest.approx(3 / 12)
    # undirected graph_density = edges / (n*(n-1)/2) = 3 / 6
    assert m["graph_density"] == pytest.approx(3 / 6)
    assert m["relationships_per_entity"] == pytest.approx(3 / 4)


def test_graph_metrics_single_node_no_graph_density():
    m = ev.compute_graph_construction_metrics(num_nodes=1, num_edges=0)
    assert m["graph_density"] == 0.0
    assert m["avg_degree"] == 0.0


def test_graph_metrics_graph_density_capped_at_one():
    # More edges than max for an undirected simple graph -> capped to 1.0.
    m = ev.compute_graph_construction_metrics(num_nodes=3, num_edges=100)
    assert m["graph_density"] == 1.0


def test_graph_metrics_entities_per_chunk():
    m = ev.compute_graph_construction_metrics(num_nodes=10, num_edges=5, num_chunks=2)
    assert m["entities_per_chunk"] == pytest.approx(5.0)


def test_graph_metrics_no_chunks_omits_entities_per_chunk():
    m = ev.compute_graph_construction_metrics(num_nodes=10, num_edges=5, num_chunks=0)
    assert "entities_per_chunk" not in m


def test_graph_metrics_node_degrees_ratio():
    m = ev.compute_graph_construction_metrics(num_nodes=4, num_edges=2, node_degrees=[0, 1, 2, 0])
    # 2 of 4 nodes have degree >= 1.
    assert m["non_isolated_node_ratio"] == pytest.approx(0.5)


def test_graph_metrics_node_degrees_empty_list_uses_fallback():
    # Empty list falls through to the edge-based estimate branch.
    m = ev.compute_graph_construction_metrics(num_nodes=4, num_edges=2, node_degrees=[])
    assert m["non_isolated_node_ratio"] == pytest.approx(min(1.0, (2 * 2) / 4))


def test_graph_metrics_fallback_non_isolated_ratio_capped():
    m = ev.compute_graph_construction_metrics(num_nodes=2, num_edges=10)
    # (2*10)/2 = 10, capped to 1.0
    assert m["non_isolated_node_ratio"] == 1.0


# ---------------------------------------------------------------------------
# get_metrics_for_level
# ---------------------------------------------------------------------------


def test_get_metrics_for_level_fact_retrieval():
    assert ev.get_metrics_for_level("fact_retrieval") == [
        "answer_score",
        "r_score",
        "ar_metric",
        "rouge_l",
    ]


def test_get_metrics_for_level_complex_reasoning_has_rouge():
    assert "rouge_l" in ev.get_metrics_for_level("complex_reasoning")


def test_get_metrics_for_level_contextual_summarization_has_coverage():
    metrics = ev.get_metrics_for_level("contextual_summarization")
    assert "coverage" in metrics
    assert "faithfulness" not in metrics


def test_get_metrics_for_level_creative_generation_has_coverage_and_faithfulness():
    metrics = ev.get_metrics_for_level("creative_generation")
    assert "coverage" in metrics
    assert "faithfulness" in metrics


def test_get_metrics_for_level_unknown_returns_base():
    assert ev.get_metrics_for_level("nonsense") == ["answer_score", "r_score", "ar_metric"]


# ---------------------------------------------------------------------------
# _parse_json_response (pure, no network)
# ---------------------------------------------------------------------------


def test_parse_json_response_plain_object():
    assert ev._parse_json_response('{"a": 1}') == {"a": 1}


def test_parse_json_response_markdown_fenced():
    text = '```json\n{"a": 1, "b": 2}\n```'
    assert ev._parse_json_response(text) == {"a": 1, "b": 2}


def test_parse_json_response_object_embedded_in_prose():
    text = 'Sure! Here is the result: {"verdict": 1} hope it helps'
    assert ev._parse_json_response(text) == {"verdict": 1}


def test_parse_json_response_bare_array_returned_as_list():
    # A clean array parses directly via json.loads, so it is returned as-is
    # (the {"items": ...} wrapping only happens in the regex fallback branch).
    assert ev._parse_json_response('["x", "y"]') == ["x", "y"]


def test_parse_json_response_array_embedded_in_prose_wrapped_in_items():
    # Surrounding prose defeats json.loads; the array regex fallback wraps it.
    text = 'Output: ["a", "b"]'
    assert ev._parse_json_response(text) == {"items": ["a", "b"]}


def test_parse_json_response_unparseable_returns_empty():
    assert ev._parse_json_response("no json at all") == {}


# ---------------------------------------------------------------------------
# _cache_key (pure)
# ---------------------------------------------------------------------------


def test_cache_key_deterministic_and_model_sensitive():
    k1 = ev._cache_key("gpt-4o-mini", "prompt")
    k2 = ev._cache_key("gpt-4o-mini", "prompt")
    k3 = ev._cache_key("gpt-4o", "prompt")
    assert k1 == k2
    assert k1 != k3
    assert len(k1) == 64  # sha256 hexdigest


# ---------------------------------------------------------------------------
# _generate_statements (mock llm_judge)
# ---------------------------------------------------------------------------


async def test_generate_statements_from_items_key():
    with patch(f"{MODULE}.llm_judge", new=AsyncMock(return_value={"items": ["s1", "s2"]})):
        out = await ev._generate_statements("q", "a")
    assert out == ["s1", "s2"]


async def test_generate_statements_from_first_list_value():
    with patch(f"{MODULE}.llm_judge", new=AsyncMock(return_value={"statements": ["x"]})):
        out = await ev._generate_statements("q", "a")
    assert out == ["x"]


async def test_generate_statements_empty_dict_falls_back_to_answer():
    with patch(f"{MODULE}.llm_judge", new=AsyncMock(return_value={})):
        out = await ev._generate_statements("q", "the answer")
    assert out == ["the answer"]


# ---------------------------------------------------------------------------
# _classify_statements (mock llm_judge)
# ---------------------------------------------------------------------------


async def test_classify_statements_normalises_missing_keys():
    with patch(f"{MODULE}.llm_judge", new=AsyncMock(return_value={"TP": [{"statement": "s"}]})):
        out = await ev._classify_statements("q", ["s"], ["g"])
    assert out["TP"] == [{"statement": "s"}]
    assert out["FP"] == []
    assert out["FN"] == []


# ---------------------------------------------------------------------------
# compute_answer_correctness_llm (mock statements + classification + embedding)
# ---------------------------------------------------------------------------


async def test_answer_correctness_llm_combines_factuality_and_semantic():
    # 2 TP, 0 FP, 0 FN -> factuality F1 = 1.0; semantic mocked to 0.4.
    classification = {"TP": [{}, {}], "FP": [], "FN": []}

    async def fake_judge(prompt, **kwargs):
        if "classify" in prompt or "TP (true positive)" in prompt:
            return classification
        return {"items": ["a", "b"]}

    with (
        patch(f"{MODULE}.llm_judge", new=AsyncMock(side_effect=fake_judge)),
        patch(f"{MODULE}._compute_semantic_similarity", new=AsyncMock(return_value=0.4)),
    ):
        score = await ev.compute_answer_correctness_llm("q", "gen", "gold")

    # 0.75 * 1.0 + 0.25 * 0.4 = 0.85
    assert score == pytest.approx(0.85)


async def test_answer_correctness_llm_custom_weights():
    classification = {"TP": [{}], "FP": [{}], "FN": []}  # precision .5, recall 1 -> F1 = 2/3

    with (
        patch(f"{MODULE}._generate_statements", new=AsyncMock(return_value=["s"])),
        patch(f"{MODULE}._classify_statements", new=AsyncMock(return_value=classification)),
        patch(f"{MODULE}._compute_semantic_similarity", new=AsyncMock(return_value=1.0)),
    ):
        score = await ev.compute_answer_correctness_llm("q", "gen", "gold", w_factuality=0.5, w_semantic=0.5)

    f1 = 2 / 3
    assert score == pytest.approx(0.5 * f1 + 0.5 * 1.0)


# ---------------------------------------------------------------------------
# compute_r_score (mock the underlying correctness call)
# ---------------------------------------------------------------------------


async def test_r_score_empty_rationale_returns_zero_without_llm():
    judge = AsyncMock()
    with patch(f"{MODULE}.llm_judge", new=judge):
        assert await ev.compute_r_score("", "gold", "q") == 0.0
        assert await ev.compute_r_score("gen", "", "q") == 0.0
    judge.assert_not_called()


async def test_r_score_delegates_to_answer_correctness():
    with patch(f"{MODULE}.compute_answer_correctness_llm", new=AsyncMock(return_value=0.66)) as m:
        score = await ev.compute_r_score("gen rationale", "gold rationale", "q")
    assert score == 0.66
    assert m.await_args.kwargs["generated"] == "gen rationale"
    assert m.await_args.kwargs["gold"] == "gold rationale"


# ---------------------------------------------------------------------------
# compute_context_relevance (mock llm_judge)
# ---------------------------------------------------------------------------


async def test_context_relevance_normalises_score():
    with patch(f"{MODULE}.llm_judge", new=AsyncMock(return_value={"relevance_score": 2})):
        score = await ev.compute_context_relevance("q", "short context", ["e"])
    assert score == pytest.approx(1.0)


async def test_context_relevance_partial():
    with patch(f"{MODULE}.llm_judge", new=AsyncMock(return_value={"relevance_score": 1})):
        score = await ev.compute_context_relevance("q", "short context", ["e"])
    assert score == pytest.approx(0.5)


async def test_context_relevance_clamps_above_two():
    with patch(f"{MODULE}.llm_judge", new=AsyncMock(return_value={"relevance_score": 5})):
        score = await ev.compute_context_relevance("q", "short context", ["e"])
    assert score == pytest.approx(1.0)


async def test_context_relevance_missing_score_defaults_zero():
    with patch(f"{MODULE}.llm_judge", new=AsyncMock(return_value={})):
        score = await ev.compute_context_relevance("q", "short", ["e"])
    assert score == 0.0


async def test_context_relevance_long_context_averages_chunks():
    # >5000 chars triggers chunking; alternate scores per call and average.
    long_context = "x" * 6000
    judge = AsyncMock(side_effect=[{"relevance_score": 2}, {"relevance_score": 0}, {"relevance_score": 2}])
    with patch(f"{MODULE}.llm_judge", new=judge):
        score = await ev.compute_context_relevance("q", long_context, ["e"])
    # chunks at step 2500 over 6000 -> starts at 0,2500,5000 => 3 chunks
    assert judge.await_count == 3
    assert score == pytest.approx((1.0 + 0.0 + 1.0) / 3)


# ---------------------------------------------------------------------------
# compute_evidence_recall (mock llm_judge)
# ---------------------------------------------------------------------------


async def test_evidence_recall_empty_evidence_returns_zero():
    judge = AsyncMock()
    with patch(f"{MODULE}.llm_judge", new=judge):
        assert await ev.compute_evidence_recall("q", "ctx", []) == 0.0
    judge.assert_not_called()


async def test_evidence_recall_fraction():
    result = {
        "classifications": [
            {"statement": "a", "attributed": 1},
            {"statement": "b", "attributed": 0},
        ]
    }
    with patch(f"{MODULE}.llm_judge", new=AsyncMock(return_value=result)):
        score = await ev.compute_evidence_recall("q", "ctx", ["a", "b"])
    # 1 attributed / len(evidence)=2
    assert score == pytest.approx(0.5)


async def test_evidence_recall_no_classifications_returns_zero():
    with patch(f"{MODULE}.llm_judge", new=AsyncMock(return_value={"classifications": []})):
        assert await ev.compute_evidence_recall("q", "ctx", ["a"]) == 0.0


async def test_evidence_recall_denominator_is_evidence_count():
    # Fewer classifications than evidence: denominator stays len(evidence).
    result = {"classifications": [{"statement": "a", "attributed": 1}]}
    with patch(f"{MODULE}.llm_judge", new=AsyncMock(return_value=result)):
        score = await ev.compute_evidence_recall("q", "ctx", ["a", "b", "c", "d"])
    assert score == pytest.approx(1 / 4)


# ---------------------------------------------------------------------------
# compute_coverage_score (mock llm_judge, two calls)
# ---------------------------------------------------------------------------


async def test_coverage_score_fraction():
    extraction = {"facts": ["f1", "f2"]}
    check = {"classifications": [{"attributed": 1}, {"attributed": 0}]}
    judge = AsyncMock(side_effect=[extraction, check])
    with patch(f"{MODULE}.llm_judge", new=judge):
        score = await ev.compute_coverage_score("q", "reference", "response")
    assert score == pytest.approx(0.5)
    assert judge.await_count == 2


async def test_coverage_score_no_facts_returns_zero_without_second_call():
    judge = AsyncMock(side_effect=[{"facts": []}])
    with patch(f"{MODULE}.llm_judge", new=judge):
        score = await ev.compute_coverage_score("q", "reference", "response")
    assert score == 0.0
    assert judge.await_count == 1


async def test_coverage_score_no_classifications_returns_zero():
    judge = AsyncMock(side_effect=[{"facts": ["f1"]}, {"classifications": []}])
    with patch(f"{MODULE}.llm_judge", new=judge):
        score = await ev.compute_coverage_score("q", "reference", "response")
    assert score == 0.0


async def test_coverage_denominator_is_fact_count():
    # 3 facts, only 2 classifications returned, 2 covered -> 2/3.
    extraction = {"facts": ["f1", "f2", "f3"]}
    check = {"classifications": [{"attributed": 1}, {"attributed": 1}]}
    judge = AsyncMock(side_effect=[extraction, check])
    with patch(f"{MODULE}.llm_judge", new=judge):
        score = await ev.compute_coverage_score("q", "reference", "response")
    assert score == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# compute_faithfulness_score (mock llm_judge; denominator = len(verdicts))
# ---------------------------------------------------------------------------


async def test_faithfulness_no_statements_returns_zero():
    with patch(f"{MODULE}._generate_statements", new=AsyncMock(return_value=[])):
        assert await ev.compute_faithfulness_score("q", "gen", "ctx") == 0.0


async def test_faithfulness_fraction_supported():
    verdicts = {
        "verdicts": [
            {"verdict": 1},
            {"verdict": 0},
            {"verdict": 1},
            {"verdict": 0},
        ]
    }
    with (
        patch(f"{MODULE}._generate_statements", new=AsyncMock(return_value=["s1", "s2", "s3", "s4"])),
        patch(f"{MODULE}.llm_judge", new=AsyncMock(return_value=verdicts)),
    ):
        score = await ev.compute_faithfulness_score("q", "gen", "ctx")
    # 2 supported / 4 verdicts
    assert score == pytest.approx(0.5)


async def test_faithfulness_denominator_uses_len_verdicts_not_statements():
    # Sent 2 statements but judge returns 4 verdicts (3 supported).
    # Denominator must be len(verdicts)=4 -> 0.75 (NOT 3/2=1.5).
    verdicts = {"verdicts": [{"verdict": 1}, {"verdict": 1}, {"verdict": 1}, {"verdict": 0}]}
    with (
        patch(f"{MODULE}._generate_statements", new=AsyncMock(return_value=["s1", "s2"])),
        patch(f"{MODULE}.llm_judge", new=AsyncMock(return_value=verdicts)),
    ):
        score = await ev.compute_faithfulness_score("q", "gen", "ctx")
    assert score == pytest.approx(0.75)
    assert score <= 1.0


async def test_faithfulness_no_verdicts_returns_zero():
    with (
        patch(f"{MODULE}._generate_statements", new=AsyncMock(return_value=["s1"])),
        patch(f"{MODULE}.llm_judge", new=AsyncMock(return_value={"verdicts": []})),
    ):
        assert await ev.compute_faithfulness_score("q", "gen", "ctx") == 0.0


# ---------------------------------------------------------------------------
# llm_judge caching path (mock litellm; no network)
# ---------------------------------------------------------------------------


async def test_llm_judge_memory_cache_short_circuits(monkeypatch, tmp_path):
    # Pre-seed the in-memory cache so litellm is never imported/called.
    key = ev._cache_key("gpt-4o-mini", ev.sanitize_text("a prompt"))
    monkeypatch.setitem(ev._judge_cache, key, {"cached": True})
    monkeypatch.setattr(ev, "_judge_cache_dir", tmp_path)

    result = await ev.llm_judge("a prompt", model="gpt-4o-mini", cache_dir=str(tmp_path))
    assert result == {"cached": True}


async def test_llm_judge_raises_after_retries_exhausted(monkeypatch, tmp_path):
    """A persistent judge failure must raise (not silently return {} → fake 0 score)."""
    import uuid

    import litellm

    monkeypatch.setattr(ev, "_judge_cache", {})  # isolate memory cache
    monkeypatch.setattr(ev, "_judge_cache_dir", tmp_path)  # isolate disk cache
    monkeypatch.setattr("asyncio.sleep", AsyncMock())  # no real backoff waits
    calls = {"n": 0}

    async def boom(*_args, **_kwargs):
        calls["n"] += 1
        raise RuntimeError("judge down")

    monkeypatch.setattr(litellm, "acompletion", boom)
    prompt = f"retry-isolation-{uuid.uuid4()}"  # unique → cannot collide with any cache entry
    with pytest.raises(RuntimeError, match="judge down"):
        await ev.llm_judge(prompt, model="gpt-4o-mini", cache_dir=str(tmp_path))
    assert calls["n"] == 3  # all three attempts made before raising


# ---------------------------------------------------------------------------
# _judge_completion_params (per-model kwargs; GPT-5 / o-series compatibility)
# ---------------------------------------------------------------------------


def test_judge_params_gpt4o_mini_keeps_baseline_params():
    # The reverse-back target: gpt-4o-mini must keep the exact original params.
    assert ev._judge_completion_params("gpt-4o-mini") == {
        "temperature": 0.0,
        "max_tokens": 4096,
        "seed": 42,
    }


@pytest.mark.parametrize("model", ["gpt-5-mini", "gpt-5", "o1-mini", "o3-mini", "o4-mini"])
def test_judge_params_reasoning_models_drop_temperature_and_seed(model):
    params = ev._judge_completion_params(model)
    assert "temperature" not in params  # reasoning models reject temperature != 1
    assert "seed" not in params
    assert "max_tokens" not in params  # must use max_completion_tokens instead
    assert params["max_completion_tokens"] >= 4096
    assert params["reasoning_effort"] == "low"


async def test_llm_judge_forwards_reasoning_params_to_litellm(monkeypatch, tmp_path):
    # The wiring that prevents a 400: a gpt-5 judge call must not pass
    # temperature/seed/max_tokens to litellm.
    import litellm

    monkeypatch.setattr(ev, "_judge_cache", {})
    monkeypatch.setattr(ev, "_judge_cache_dir", tmp_path)
    captured = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)

        class _Msg:
            content = '{"ok": 1}'

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        return _Resp()

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    await ev.llm_judge("prompt-x", model="gpt-5-mini", cache_dir=str(tmp_path))
    assert "temperature" not in captured
    assert "seed" not in captured
    assert "max_tokens" not in captured
    assert captured["max_completion_tokens"] >= 4096
    assert captured["reasoning_effort"] == "low"


async def test_semantic_similarity_propagates_embedding_failure(monkeypatch):
    """An embedding failure must propagate (not return a fabricated 0.5)."""
    import litellm

    async def boom(*_args, **_kwargs):
        raise RuntimeError("embeddings down")

    monkeypatch.setattr(litellm, "aembedding", boom)
    with pytest.raises(RuntimeError, match="embeddings down"):
        await ev._compute_semantic_similarity("a", "b")
