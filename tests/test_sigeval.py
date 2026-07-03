"""Runnable check for the correctness-critical path: the statistics.

Run: python -m pytest tests/ -q   (or) python tests/test_sigeval.py
No LLM calls — scorers are stubbed random processes so the suite is offline,
deterministic under a fixed seed, and fast.
"""

import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sigeval.stats import wilson_interval, two_proportion_pvalue, decide
from sigeval import (
    run_suite, save_baseline, check_regression,
    run_case_budgeted, make_judge,
)


def test_wilson_bounds_are_sane():
    low, high = wilson_interval(8, 10)
    assert 0.0 <= low <= 0.8 <= high <= 1.0
    # extreme case k=n must not exceed 1.0 (where normal approx breaks)
    low, high = wilson_interval(10, 10)
    assert high <= 1.0 and low < 1.0
    # k=0 must not go below 0
    low, high = wilson_interval(0, 10)
    assert low >= 0.0 and high > 0.0


def test_bigger_n_shrinks_interval():
    lo_small, hi_small = wilson_interval(8, 10)
    lo_big, hi_big = wilson_interval(80, 100)
    assert (hi_big - lo_big) < (hi_small - lo_small)


def test_decide_is_inconclusive_near_the_line():
    # 17/20 = 0.85 point estimate, threshold 0.8 -> too close to call at n=20
    v = decide(17, 20, 0.8)
    assert v.name == "INCONCLUSIVE", v
    # same 0.85 rate but n=1000 -> interval tightens enough to clear 0.8 -> PASS
    v = decide(850, 1000, 0.8)
    assert v.name == "PASS", v
    # clearly below
    assert decide(2, 20, 0.8).name == "FAIL"


def test_regression_only_fires_on_significant_drop():
    # baseline 95/100, current 93/100 -> noise, must NOT flag
    assert two_proportion_pvalue(95, 100, 93, 100) > 0.05
    # baseline 95/100, current 70/100 -> real drop, MUST flag
    assert two_proportion_pvalue(95, 100, 70, 100) < 0.05
    # identical all-pass baselines -> no false regression
    assert two_proportion_pvalue(50, 50, 50, 50) == 1.0


def test_end_to_end_suite_and_baseline():
    rng = random.Random(42)
    # scorer that passes ~90% of the time
    good = lambda _s: rng.random() < 0.90
    cases = [("greeting", "hi"), ("farewell", "bye")]
    results = run_suite(cases, good, n_samples=200, threshold=0.8)
    assert all(r.verdict.name == "PASS" for r in results), [str(r.verdict) for r in results]

    path = "/tmp/_sigeval_baseline.json"
    save_baseline(results, path)
    # rerun at same quality -> no regression
    results2 = run_suite(cases, good, n_samples=200, threshold=0.8)
    assert check_regression(results2, path) == []
    # rerun with a degraded scorer (~60%) -> regression fires
    bad = lambda _s: rng.random() < 0.60
    results3 = run_suite(cases, bad, n_samples=200, threshold=0.8)
    assert len(check_regression(results3, path)) == 2
    os.remove(path)


def test_budget_stops_early_on_clear_pass():
    # a near-perfect scorer should lock a PASS well before max_samples
    calls = {"n": 0}
    def scorer(_s):
        calls["n"] += 1
        return True  # 100% pass -> interval clears 0.8 fast
    res = run_case_budgeted("easy", scorer, "x", threshold=0.8,
                            min_samples=8, max_samples=200, batch=8)
    assert res.verdict.name == "PASS", res.verdict
    assert res.n < 200, f"should have stopped early, used {res.n}"
    assert calls["n"] == res.n  # never sampled more than it reported


def test_budget_runs_to_max_when_borderline():
    rng = random.Random(7)
    # rate sits right on the threshold -> should never lock -> INCONCLUSIVE at max
    scorer = lambda _s: rng.random() < 0.80
    res = run_case_budgeted("borderline", scorer, "x", threshold=0.8,
                            min_samples=8, max_samples=64, batch=8)
    assert res.n == 64
    assert res.verdict.name == "INCONCLUSIVE", res.verdict


def test_judge_parses_model_verdict():
    # stub "model": PASS only when the output contains "refund"
    def complete(prompt):
        return "PASS - clearly on topic" if "refund" in prompt else "FAIL, off topic"
    scorer = make_judge(complete, criterion="is about money back")
    assert scorer("here is our refund policy") is True
    assert scorer("we sell hats") is False


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all passed")
