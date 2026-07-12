"""Unit tests for harness.analysis (McNemar paired A/B, #12).

The McNemar math is checked against known contingency tables with textbook
p-values so a regression in the exact/chi-square selection or the continuity
correction is caught. ``compare_runs`` is checked on synthetic report dicts.
"""

from __future__ import annotations

import pytest

from khora_graphrag_bench.harness.analysis import (
    ContingencyTable,
    build_contingency,
    compare_runs,
    mcnemar_test,
)

# ---------------------------------------------------------------------------
# mcnemar_test — against known values
# ---------------------------------------------------------------------------


def test_mcnemar_exact_small_discordant():
    # b=1, c=11 -> two-sided exact binomial(min=1, n=12, p=0.5).
    stat, p, method = mcnemar_test(1, 11)
    assert method == "exact"
    assert stat == 12.0  # the binomial n (discordant total)
    assert p == pytest.approx(0.00634765625, abs=1e-9)


def test_mcnemar_chi2_continuity_wikipedia_example():
    # Wikipedia McNemar worked example: b=101, c=121.
    # chi2 = (|101-121|-1)^2 / 222 = 361/222 = 1.62612...
    stat, p, method = mcnemar_test(101, 121)
    assert method == "chi2_continuity"
    assert stat == pytest.approx(1.6261261261, abs=1e-6)
    assert p == pytest.approx(0.20223969808, abs=1e-6)


def test_mcnemar_zero_discordant_is_p1():
    # Runs never disagreed -> no evidence of a difference.
    stat, p, method = mcnemar_test(0, 0)
    assert (stat, p, method) == (0.0, 1.0, "exact")


def test_mcnemar_auto_selects_exact_below_cutoff_chi2_above():
    assert mcnemar_test(2, 10)[2] == "exact"  # discordant 12 < 25
    assert mcnemar_test(10, 20)[2] == "chi2_continuity"  # discordant 30 >= 25


def test_mcnemar_force_method():
    assert mcnemar_test(1, 11, exact=True)[2] == "exact"
    assert mcnemar_test(1, 11, exact=False)[2] == "chi2_continuity"


def test_mcnemar_realistic_retrieval_fix_is_detectable():
    # The issue's premise: a ~1.5pt fix (~30 net flips on 2010q) is significant
    # in ONE paired run. Model it as b=15, c=45 (30 net flips, 60 discordant).
    stat, p, method = mcnemar_test(15, 45)
    assert method == "chi2_continuity"
    assert p < 0.001  # highly significant despite 0.73pt mean-accuracy noise


def test_mcnemar_symmetric_in_magnitude():
    # Swapping b and c leaves the p-value unchanged (two-sided test).
    _, p_forward, _ = mcnemar_test(5, 20)
    _, p_reverse, _ = mcnemar_test(20, 5)
    assert p_forward == pytest.approx(p_reverse)


def test_mcnemar_negative_counts_raise():
    with pytest.raises(ValueError, match="non-negative"):
        mcnemar_test(-1, 3)


# ---------------------------------------------------------------------------
# build_contingency
# ---------------------------------------------------------------------------


def test_build_contingency_classifies_all_four_cells():
    baseline = {"a": True, "b": True, "c": False, "d": False}
    candidate = {"a": True, "b": False, "c": True, "d": False}
    t = build_contingency(baseline, candidate)
    assert t.both_correct == 1  # a
    assert t.baseline_only == 1  # b (regression)
    assert t.candidate_only == 1  # c (improvement)
    assert t.both_wrong == 1  # d
    assert t.n == 4
    assert t.discordant == 2
    assert t.net_flips == 0


def test_build_contingency_only_pairs_shared_questions():
    baseline = {"a": True, "b": False, "x_only_base": True}
    candidate = {"a": True, "b": True, "y_only_cand": False}
    t = build_contingency(baseline, candidate)
    # Only a and b are shared.
    assert t.n == 2
    assert t.both_correct == 1  # a
    assert t.candidate_only == 1  # b: base wrong, cand right


def test_net_flips_sign_favors_candidate():
    # More candidate-only than baseline-only -> positive net flips.
    t = ContingencyTable(both_correct=10, baseline_only=3, candidate_only=8, both_wrong=5)
    assert t.net_flips == 5


# ---------------------------------------------------------------------------
# compare_runs — from report.json dicts
# ---------------------------------------------------------------------------


def _report(pairs: list[tuple[str, bool, bool]]) -> dict:
    """Build a report.json-shaped dict. pairs = (question_id, correct, errored)."""
    return {
        "result": {
            "per_question": [
                {"question_id": q, "answer_correct": c, **({"error": "boom"} if e else {})} for q, c, e in pairs
            ]
        }
    }


def test_compare_runs_pairs_and_excludes_errors():
    base = _report(
        [("q1", True, False), ("q2", True, False), ("q3", False, False), ("q4", True, False), ("q5", True, True)]
    )
    cand = _report(
        [("q1", True, False), ("q2", False, False), ("q3", True, False), ("q4", True, False), ("q5", False, True)]
    )
    r = compare_runs(base, cand)
    # q5 errored in baseline -> excluded. q1/q4 both correct, q2 regression, q3 improvement.
    assert r.table.n == 4
    assert r.table.both_correct == 2
    assert r.table.baseline_only == 1
    assert r.table.candidate_only == 1
    assert r.net_flips == 0


def test_compare_runs_accuracy_delta_and_significance():
    # 40 concordant-correct + 15 regressions + 45 improvements = 100 paired.
    base = _report(
        [(f"c{i}", True, False) for i in range(40)]
        + [(f"r{i}", True, False) for i in range(15)]
        + [(f"i{i}", False, False) for i in range(45)]
    )
    cand = _report(
        [(f"c{i}", True, False) for i in range(40)]
        + [(f"r{i}", False, False) for i in range(15)]
        + [(f"i{i}", True, False) for i in range(45)]
    )
    r = compare_runs(base, cand)
    assert r.table.n == 100
    assert r.net_flips == 30
    assert r.accuracy_delta == pytest.approx(0.30)
    assert r.significant_at_05 is True


def test_compare_runs_accepts_bare_result_dict():
    # _correctness_map tolerates a dict that is already the `result` payload.
    base = _report([("q1", True, False)])["result"]
    cand = _report([("q1", False, False)])["result"]
    r = compare_runs(base, cand)
    assert r.table.baseline_only == 1
