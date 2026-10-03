"""Sample budgeting: stop when the current Wilson interval clears the threshold.

The second pain the 2026 research flagged (after flakiness) is COST — evals in
CI run every case a fixed N times at $1-3/run. We sample in batches and bail
early when the current interval has cleared (or fallen below) the threshold.
Later samples can change that verdict. Repeated checks do not retain the
fixed-sample confidence coverage; this is a cost-saving heuristic.

Worst case (borderline rate sitting on the threshold) still runs to max_samples
and returns INCONCLUSIVE — the honest answer, not a guess.
"""

from .stats import decide, _validate_positive_integer, _validate_threshold, _validate_z
from .core import CaseResult


def run_case_budgeted(case_id, scorer, sample, threshold,
                      min_samples=8, max_samples=200, batch=8, z=1.96):
    """Sample until the current verdict is PASS/FAIL, or max_samples is hit.

    min_samples guards against calling it too early on a lucky/unlucky streak.
    Early stopping does not preserve fixed-sample confidence coverage.
    Returns a CaseResult with n = the samples actually spent (often << max).
    """
    _validate_budget_parameters(threshold, min_samples, max_samples, batch, z)
    k = n = 0
    while n < max_samples:
        take = min(batch, max_samples - n)
        k += sum(1 for _ in range(take) if scorer(sample))
        n += take
        if n < min_samples:
            continue
        v = decide(k, n, threshold, z)
        if v.name != "INCONCLUSIVE":
            return CaseResult(case_id, k, n, v)
    return CaseResult(case_id, k, n, decide(k, n, threshold, z))


def run_suite_budgeted(cases, scorer, threshold=0.8,
                       min_samples=8, max_samples=200, batch=8, z=1.96):
    """Budgeted variant of run_suite. Same return shape (list[CaseResult])."""
    _validate_budget_parameters(threshold, min_samples, max_samples, batch, z)
    return [
        run_case_budgeted(cid, scorer, s, threshold, min_samples, max_samples, batch, z)
        for cid, s in cases
    ]


def _validate_budget_parameters(threshold, min_samples, max_samples, batch, z):
    _validate_positive_integer(min_samples, "min_samples")
    _validate_positive_integer(max_samples, "max_samples")
    _validate_positive_integer(batch, "batch")
    if min_samples > max_samples:
        raise ValueError("min_samples must be <= max_samples")
    _validate_threshold(threshold)
    _validate_z(z)
