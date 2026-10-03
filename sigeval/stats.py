"""Statistical primitives for flaky-free LLM evals. Stdlib only — no scipy/numpy.

The whole thesis lives here: an eval result is a *proportion* (k passes out of n
samples), not a single boolean. We reason about the true pass-rate with a
confidence interval instead of trusting one run.
"""

import math
from dataclasses import dataclass


def _validate_positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_counts(k: int, n: int) -> None:
    _validate_positive_integer(n, "n")
    if isinstance(k, bool) or not isinstance(k, int) or not 0 <= k <= n:
        raise ValueError("k must be an integer with 0 <= k <= n")


def _validate_finite(value: float, name: str) -> None:
    try:
        finite = math.isfinite(value)
    except (TypeError, ValueError, OverflowError):
        finite = False
    if not finite:
        raise ValueError(f"{name} must be finite")


def _validate_z(z: float) -> None:
    _validate_finite(z, "z")
    if z <= 0:
        raise ValueError("z must be > 0")


def _validate_threshold(threshold: float) -> None:
    _validate_finite(threshold, "threshold")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")


def _validate_alpha(alpha: float) -> None:
    _validate_finite(alpha, "alpha")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")


def _phi(z: float) -> float:
    """Standard normal CDF via erf. ponytail: exact enough, no scipy dependency."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Chosen over the normal approximation because it stays inside [0,1] and is
    correct at small n and extreme rates (k=0 or k=n) — exactly the cases a
    small eval suite hits constantly. z=1.96 -> 95% confidence.
    """
    _validate_counts(k, n)
    _validate_z(z)
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / denom
    # These endpoints are exact mathematically; avoid a floating-point sliver
    # turning all-fail at threshold 0 into PASS (or all-pass at 1 into FAIL).
    low = 0.0 if k == 0 else max(0.0, center - half)
    high = 1.0 if k == n else min(1.0, center + half)
    return low, high


def two_proportion_pvalue(k1: int, n1: int, k2: int, n2: int) -> float:
    """One-sided p-value that group 2's rate is *below* group 1's rate.

    Used for regression detection: group 1 = saved baseline, group 2 = current.
    Pooled two-proportion z-test. Returns the p-value for H1: p2 < p1.
    """
    _validate_counts(k1, n1)
    _validate_counts(k2, n2)
    p1, p2 = k1 / n1, k2 / n2
    pool = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    if se == 0.0:
        # identical degenerate rates (both all-pass or all-fail) -> no evidence of drop
        return 1.0
    z = (p2 - p1) / se
    return _phi(z)  # left tail = P(observed drop or worse under H0)


@dataclass
class Verdict:
    name: str          # "PASS" | "FAIL" | "INCONCLUSIVE"
    k: int
    n: int
    rate: float
    ci_low: float
    ci_high: float
    threshold: float

    def __bool__(self) -> bool:
        return self.name == "PASS"

    def __str__(self) -> str:
        return (f"{self.name}: {self.k}/{self.n} passed "
                f"(rate={self.rate:.2f}, CI [{self.ci_low:.2f}, {self.ci_high:.2f}], "
                f"threshold={self.threshold:.2f})")


def decide(k: int, n: int, threshold: float, z: float = 1.96) -> Verdict:
    """Verdict from the confidence interval, not the point estimate.

    - PASS:         CI lower bound > threshold  (true rate is *significantly* above)
    - FAIL:         CI upper bound < threshold  (true rate is *significantly* below)
    - INCONCLUSIVE: interval straddles threshold (collect more samples)

    This is the anti-flake mechanism: a run that happens to score just over the
    line returns INCONCLUSIVE, not a green build that flips red next commit.
    """
    _validate_threshold(threshold)
    low, high = wilson_interval(k, n, z)
    rate = k / n
    if low > threshold:
        name = "PASS"
    elif high < threshold:
        name = "FAIL"
    else:
        name = "INCONCLUSIVE"
    return Verdict(name, k, n, rate, low, high, threshold)
