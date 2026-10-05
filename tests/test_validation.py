"""Invalid parameters must fail before samples or baseline I/O are spent."""

import subprocess
import sys
from pathlib import Path

import pytest

from sigeval import (
    assert_eval,
    check_regression,
    decide,
    make_judge,
    run_case,
    run_case_budgeted,
    run_suite,
    run_suite_budgeted,
    two_proportion_pvalue,
    wilson_interval,
)


@pytest.mark.parametrize("k,n", [(-1, 5), (6, 5), (0, 0), (0, -1),
                                    (1.0, 5), (1, 5.0), (True, 5), (0, True)])
@pytest.mark.parametrize("function", [wilson_interval, lambda k, n: decide(k, n, 0.8)])
def test_invalid_binomial_counts(k, n, function):
    with pytest.raises(ValueError):
        function(k, n)


@pytest.mark.parametrize("k,n", [(-1, 5), (6, 5), (0, 0), (0, -1),
                                    (1.0, 5), (1, 5.0), (True, 5), (0, True)])
@pytest.mark.parametrize("group", [1, 2])
def test_invalid_regression_counts(k, n, group):
    args = (k, n, 2, 5) if group == 1 else (2, 5, k, n)
    with pytest.raises(ValueError):
        two_proportion_pvalue(*args)


@pytest.mark.parametrize("z", [-1.96, 0, float("nan"), float("inf"), float("-inf")])
def test_invalid_confidence_multiplier(z):
    with pytest.raises(ValueError):
        wilson_interval(8, 10, z)


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_decision_threshold(threshold):
    with pytest.raises(ValueError):
        decide(8, 10, threshold)


@pytest.mark.parametrize("runner", [run_case, assert_eval])
@pytest.mark.parametrize("kwargs", [
    {"n_samples": 0}, {"n_samples": -1}, {"n_samples": 2.5},
    {"n_samples": True}, {"threshold": -0.1}, {"threshold": 1.1},
    {"threshold": float("nan")}, {"threshold": float("inf")},
    {"z": 0}, {"z": -1.96}, {"z": float("nan")}, {"z": float("inf")},
])
def test_fixed_runner_rejects_before_scorer(runner, kwargs):
    calls = []
    params = {"n_samples": 20, "threshold": 0.8, **kwargs}
    with pytest.raises(ValueError):
        runner("case", lambda sample: calls.append(sample) or True, "sample", **params)
    assert calls == []


@pytest.mark.parametrize("kwargs", [
    {"min_samples": 0}, {"min_samples": -1}, {"min_samples": 2.5},
    {"min_samples": True}, {"max_samples": 0}, {"max_samples": -1},
    {"max_samples": 2.5}, {"max_samples": True}, {"batch": 2.5},
    {"batch": True}, {"min_samples": 21, "max_samples": 20},
    {"threshold": -0.1}, {"threshold": 1.1}, {"threshold": float("nan")},
    {"threshold": float("inf")}, {"z": 0}, {"z": -1.96},
    {"z": float("nan")}, {"z": float("inf")},
])
def test_budgeted_runner_rejects_before_scorer(kwargs):
    calls = []
    params = {"threshold": 0.8, **kwargs}
    with pytest.raises(ValueError):
        run_case_budgeted("case", lambda sample: calls.append(sample) or True,
                          "sample", **params)
    assert calls == []


@pytest.mark.parametrize("batch", [0, -1])
def test_nonpositive_batch_terminates_without_sampling(batch):
    # Bound the regression: the original implementation hangs for either value.
    script = (
        "from sigeval import run_case_budgeted\n"
        "def scorer(sample):\n"
        "    raise AssertionError('invalid batch spent a sample')\n"
        "try:\n"
        f"    run_case_budgeted('case', scorer, 'sample', 0.8, batch={batch})\n"
        "except ValueError:\n"
        "    pass\n"
        "else:\n"
        "    raise AssertionError('invalid batch accepted')\n"
    )
    result = subprocess.run([sys.executable, "-c", script],
                            cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, text=True, timeout=1)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("runner,kwargs", [
    (run_suite, {"n_samples": 0}), (run_suite, {"threshold": float("nan")}),
    (run_suite, {"z": -1.96}),
    (run_suite_budgeted, {"min_samples": 0}),
    (run_suite_budgeted, {"max_samples": 0}),
    (run_suite_budgeted, {"batch": 0}),
    (run_suite_budgeted, {"threshold": float("nan")}),
    (run_suite_budgeted, {"z": -1.96}),
])
def test_empty_suites_still_validate_parameters(runner, kwargs):
    with pytest.raises(ValueError):
        runner([], lambda sample: True, **kwargs)


@pytest.mark.parametrize("alpha", [0, 1, -0.1, 1.1, float("nan"),
                                   float("inf"), float("-inf")])
def test_invalid_alpha_rejects_before_baseline_io(alpha, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr("sigeval.core.os.path.exists", lambda path: calls.append(path) or False)
    with pytest.raises(ValueError):
        check_regression([], tmp_path / "missing.json", alpha=alpha)
    assert calls == []


@pytest.mark.parametrize("reply", ["PASSAGE", "PASSING", "PASS_FAIL", "PASS1", "PASSÉ"])
def test_judge_rejects_longer_tokens(reply):
    assert make_judge(lambda prompt: reply, "criterion")("output") is False


@pytest.mark.parametrize("reply", ["PASS", "PASS.", "PASS - because", " pass\n", "FAIL"])
def test_judge_preserves_existing_verdict_formats(reply):
    assert make_judge(lambda prompt: reply, "criterion")("output") is (reply != "FAIL")


@pytest.mark.parametrize("batch", [1, 3, 8, 100])
def test_valid_budget_never_exceeds_maximum(batch):
    calls = []
    result = run_case_budgeted("case", lambda sample: calls.append(sample) or len(calls) % 2 == 0,
                               "sample", threshold=0.5, min_samples=3,
                               max_samples=10, batch=batch)
    assert len(calls) == result.n == 10
    assert result.verdict.name == "INCONCLUSIVE"


def test_valid_fixed_runner_spends_exact_sample_count():
    calls = []
    result = run_case("case", lambda sample: calls.append(sample) or True,
                      "sample", n_samples=20, threshold=0.8)
    assert len(calls) == result.n == result.k == 20
    assert result.verdict.name == "PASS"


def test_custom_confidence_verdict_has_no_incorrect_95_percent_label():
    assert "95%" not in str(decide(8, 10, 0.8, z=1))


@pytest.mark.parametrize("n", range(1, 101))
def test_wilson_extreme_endpoints_are_exact(n):
    assert wilson_interval(0, n)[0] == 0.0
    assert wilson_interval(n, n)[1] == 1.0
    assert decide(0, n, 0).name == "INCONCLUSIVE"
    assert decide(n, n, 1).name == "INCONCLUSIVE"


def test_all_pass_budget_at_threshold_one_never_returns_fail():
    result = run_case_budgeted("case", lambda sample: True, "sample", threshold=1,
                               min_samples=3, max_samples=10, batch=3)
    assert result.n == 10
    assert result.verdict.name == "INCONCLUSIVE"


def test_all_fail_budget_at_threshold_zero_never_returns_pass():
    result = run_case_budgeted("case", lambda sample: False, "sample", threshold=0,
                               min_samples=11, max_samples=20, batch=11)
    assert result.n == 20
    assert result.verdict.name == "INCONCLUSIVE"
