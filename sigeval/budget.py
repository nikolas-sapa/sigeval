"""Sample budgeting: stop sampling the moment the verdict is statistically locked.

The second pain the 2026 research flagged (after flakiness) is COST — evals in
CI run every case a fixed N times at $1-3/run. Most of those samples are wasted:
once the confidence interval has cleared (or fallen below) the threshold, extra
samples change nothing. We sample in batches and bail early.

Worst case (borderline rate sitting on the threshold) still runs to max_samples
and returns INCONCLUSIVE — the honest answer, not a guess.
"""

from .stats import decide
from .core import CaseResult


def run_case_budgeted(case_id, scorer, sample, threshold,
                      min_samples=8, max_samples=200, batch=8, z=1.96):
    """Sequentially sample until a PASS/FAIL verdict locks, or max_samples hit.

    min_samples guards against calling it too early on a lucky/unlucky streak.
    Returns a CaseResult with n = the samples actually spent (often << max).
    """
    if min_samples > max_samples:
        raise ValueError("min_samples must be <= max_samples")
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
    return [
        run_case_budgeted(cid, scorer, s, threshold, min_samples, max_samples, batch, z)
        for cid, s in cases
    ]
