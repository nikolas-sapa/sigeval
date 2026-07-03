"""sigeval — pytest for LLMs that isn't flaky.

Evals are proportions with confidence intervals, not single booleans.
"""

from .stats import wilson_interval, two_proportion_pvalue, decide, Verdict
from .core import (
    run_case,
    run_suite,
    save_baseline,
    check_regression,
    assert_eval,
    CaseResult,
)

__version__ = "0.0.1"
__all__ = [
    "wilson_interval",
    "two_proportion_pvalue",
    "decide",
    "Verdict",
    "run_case",
    "run_suite",
    "save_baseline",
    "check_regression",
    "assert_eval",
    "CaseResult",
]
