"""Paired A/B analysis of two benchmark runs via McNemar's test (#12).

Mean-accuracy comparison is too noisy to see a realistic retrieval fix: with a
~0.73pt run-to-run stdev, a ~1.5pt effect is invisible at 1-3 runs. McNemar's
test pairs the two runs *per question* and looks only at the discordant pairs
(questions one run got right and the other got wrong), so a ~1.5pt effect
(~30 net flips on the 2010-question suite) is detectable in a single paired run
at standard significance.

This module is pure (no I/O beyond reading the two ``report.json`` dicts the
caller hands it) so the McNemar math is unit-testable against a known
contingency table.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContingencyTable:
    """2x2 paired contingency of per-question correctness (baseline vs candidate).

    * ``both_correct`` - right in both runs.
    * ``baseline_only`` (``b``) - right in baseline, wrong in candidate (a regression).
    * ``candidate_only`` (``c``) - wrong in baseline, right in candidate (an improvement).
    * ``both_wrong`` - wrong in both runs.

    Only ``b`` and ``c`` (the discordant pairs) carry the paired signal.
    """

    both_correct: int
    baseline_only: int
    candidate_only: int
    both_wrong: int

    @property
    def n(self) -> int:
        return self.both_correct + self.baseline_only + self.candidate_only + self.both_wrong

    @property
    def discordant(self) -> int:
        return self.baseline_only + self.candidate_only

    @property
    def net_flips(self) -> int:
        """Net improvement of candidate over baseline (``c - b``)."""
        return self.candidate_only - self.baseline_only


@dataclass(frozen=True)
class McNemarResult:
    """Outcome of McNemar's test on a paired contingency table."""

    table: ContingencyTable
    statistic: float
    p_value: float
    method: str  # "exact" (binomial) or "chi2_continuity"
    net_flips: int
    # Accuracy delta implied by the net flips (candidate - baseline), in points.
    accuracy_delta: float
    significant_at_05: bool


# Below this many discordant pairs the chi-square approximation is unreliable;
# use the exact binomial test instead (standard textbook cutoff).
_EXACT_CUTOFF = 25


def mcnemar_test(baseline_only: int, candidate_only: int, *, exact: bool | None = None) -> tuple[float, float, str]:
    """McNemar's test on the two discordant cell counts.

    Args:
        baseline_only: ``b`` - count of questions baseline got right and candidate wrong.
        candidate_only: ``c`` - count of questions baseline got wrong and candidate right.
        exact: Force the exact binomial (``True``) or the continuity-corrected
            chi-square (``False``). ``None`` (default) auto-selects: exact when
            the discordant total ``b + c`` is small (< 25), chi-square otherwise.

    Returns:
        ``(statistic, p_value, method)``. For the exact test ``statistic`` is the
        discordant total ``b + c`` (the binomial n); for the chi-square test it
        is the continuity-corrected chi-square statistic. ``method`` is
        ``"exact"`` or ``"chi2_continuity"``.

    A zero discordant total (``b == c == 0``) means the runs never disagreed:
    p-value 1.0, no evidence of a difference.
    """
    b, c = int(baseline_only), int(candidate_only)
    if b < 0 or c < 0:
        raise ValueError(f"discordant counts must be non-negative, got b={b}, c={c}")

    n_discordant = b + c
    if n_discordant == 0:
        return (0.0, 1.0, "exact")

    use_exact = (n_discordant < _EXACT_CUTOFF) if exact is None else exact

    if use_exact:
        from scipy.stats import binomtest

        # Two-sided exact binomial: under H0 each discordant pair is equally
        # likely to flip either way (p=0.5).
        p_value = float(binomtest(min(b, c), n_discordant, 0.5, alternative="two-sided").pvalue)
        return (float(n_discordant), p_value, "exact")

    from scipy.stats import chi2

    # Continuity-corrected chi-square (Edwards): (|b - c| - 1)^2 / (b + c).
    statistic = (abs(b - c) - 1) ** 2 / n_discordant
    p_value = float(chi2.sf(statistic, 1))
    return (float(statistic), p_value, "chi2_continuity")


def build_contingency(
    baseline_correct: dict[str, bool],
    candidate_correct: dict[str, bool],
) -> ContingencyTable:
    """Build the paired 2x2 table from two ``question_id -> correct`` maps.

    Only questions present in BOTH runs are paired (McNemar needs pairs); a
    question missing from either run is dropped. Callers should verify the two
    runs cover the same question set for a clean comparison.
    """
    shared = baseline_correct.keys() & candidate_correct.keys()
    both_correct = baseline_only = candidate_only = both_wrong = 0
    for qid in shared:
        b_ok = bool(baseline_correct[qid])
        c_ok = bool(candidate_correct[qid])
        if b_ok and c_ok:
            both_correct += 1
        elif b_ok and not c_ok:
            baseline_only += 1
        elif not b_ok and c_ok:
            candidate_only += 1
        else:
            both_wrong += 1
    return ContingencyTable(both_correct, baseline_only, candidate_only, both_wrong)


def _correctness_map(report: dict) -> dict[str, bool]:
    """Map ``question_id -> answer_correct`` from a report.json ``result`` dict.

    Errored questions (``error`` set) are excluded: a crash is not a retrieval
    verdict and would pollute the paired comparison. Works for both full-judge
    and retrieval-only runs, since both populate ``answer_correct`` per question.
    """
    result = report.get("result", report)
    out: dict[str, bool] = {}
    for q in result.get("per_question", []):
        if q.get("error"):
            continue
        qid = q.get("question_id")
        if qid is not None:
            out[str(qid)] = bool(q.get("answer_correct", False))
    return out


def compare_runs(baseline_report: dict, candidate_report: dict) -> McNemarResult:
    """Paired McNemar comparison of two loaded ``report.json`` payloads.

    Pairs per-question correctness on ``question_id``, builds the contingency,
    and runs McNemar's test. ``accuracy_delta`` is the net-flip-implied accuracy
    change over the paired set (candidate - baseline), in fractional points.
    """
    baseline = _correctness_map(baseline_report)
    candidate = _correctness_map(candidate_report)
    table = build_contingency(baseline, candidate)

    statistic, p_value, method = mcnemar_test(table.baseline_only, table.candidate_only)
    accuracy_delta = (table.net_flips / table.n) if table.n else 0.0

    return McNemarResult(
        table=table,
        statistic=statistic,
        p_value=p_value,
        method=method,
        net_flips=table.net_flips,
        accuracy_delta=accuracy_delta,
        significant_at_05=p_value < 0.05,
    )
