"""Runner + regression gate. The user-facing surface.

A scorer is any callable(case) -> bool (did this single sample pass). You bring
the LLM call + assertion; sigeval brings the statistics. Model-agnostic on
purpose — no provider lock-in, unlike the platform-tier tools.
"""

import json
import os
from dataclasses import dataclass, asdict

from .stats import (
    decide, two_proportion_pvalue, Verdict, _holm_adjusted,
    _validate_positive_integer, _validate_threshold, _validate_z, _validate_alpha,
)


@dataclass
class CaseResult:
    case_id: str
    k: int
    n: int
    verdict: Verdict


def run_case(case_id, scorer, sample, n_samples, threshold, z=1.96):
    """Run one case n_samples times, return a statistical verdict."""
    _validate_run_parameters(n_samples, threshold, z)
    k = sum(1 for _ in range(n_samples) if scorer(sample))
    return CaseResult(case_id, k, n_samples, decide(k, n_samples, threshold, z))


def run_suite(cases, scorer, n_samples=20, threshold=0.8, z=1.96):
    """cases: iterable of (case_id, sample). Returns list[CaseResult]."""
    _validate_run_parameters(n_samples, threshold, z)
    return [run_case(cid, scorer, s, n_samples, threshold, z) for cid, s in cases]


def _validate_run_parameters(n_samples, threshold, z):
    _validate_positive_integer(n_samples, "n_samples")
    _validate_threshold(threshold)
    _validate_z(z)


def save_baseline(results, path):
    """Persist k/n per case so future runs can test for regression."""
    data = {r.case_id: {"k": r.k, "n": r.n} for r in results}
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def check_regression(results, baseline_path, alpha=0.05, correction="holm"):
    """Flag cases whose pass-rate dropped *significantly* vs baseline.

    Returns list of (case_id, p_value). A raw-number diff would fire on noise;
    the two-proportion test only fires when the drop clears statistical
    significance at `alpha`.

    With correction="holm" (default) each case's p-value is Holm-adjusted
    across all compared cases, so one unlucky marginal drop in a large suite
    does not produce a false alarm — the reported p is the adjusted value.
    correction=None compares raw per-case p-values (the v0.1.0 behavior).
    Comparisons across repeated CI runs are still uncorrected.
    """
    _validate_alpha(alpha)
    if correction not in ("holm", None):
        raise ValueError('correction must be "holm" or None')
    if not os.path.exists(baseline_path):
        return []  # ponytail: no baseline yet -> nothing to regress against
    with open(baseline_path) as f:
        base = json.load(f)
    tested = []
    for r in results:
        b = base.get(r.case_id)
        if not b:
            continue
        tested.append((r.case_id, two_proportion_pvalue(b["k"], b["n"], r.k, r.n)))
    if correction == "holm":
        adjusted = _holm_adjusted([p for _, p in tested])
        tested = [(cid, p_adj) for (cid, _), p_adj in zip(tested, adjusted)]
    return [(cid, p) for cid, p in tested if p < alpha]


def assert_eval(case_id, scorer, sample, n_samples=20, threshold=0.8, z=1.96):
    """Drop-in for a pytest test body. Raises AssertionError unless verdict is PASS.

    INCONCLUSIVE also raises — a build that can't prove it's good is not green.
    This is the whole "pytest for LLMs that isn't flaky" surface: one function,
    usable in any existing pytest suite, no plugin required.
    """
    res = run_case(case_id, scorer, sample, n_samples, threshold, z)
    assert res.verdict.name == "PASS", str(res.verdict)
    return res
