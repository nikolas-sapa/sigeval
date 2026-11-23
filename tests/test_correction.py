"""Suite-wide regression checks: Holm correction across compared cases.

Reference values are hardcoded from statsmodels 0.15.0:
  statsmodels.stats.multitest.multipletests(p, method="holm")
  statsmodels.stats.proportion.proportions_ztest(..., alternative="larger")
so the hand-rolled adjustment is pinned to a real implementation, not just
to itself.
"""

import json

import pytest

from sigeval import CaseResult, check_regression, decide
from sigeval.stats import _holm_adjusted, two_proportion_pvalue


@pytest.mark.parametrize("p_values,expected", [
    # statsmodels multipletests(method="holm") outputs
    ([0.01, 0.04, 0.03, 0.0025], [0.03, 0.06, 0.06, 0.01]),
    ([0.029, 0.038], [0.058, 0.058]),
    ([0.9, 0.95], [1.0, 1.0]),
    ([0.001, 0.5, 0.02], [0.003, 0.5, 0.04]),
    ([0.04, 0.05], [0.08, 0.08]),
    ([], []),
    ([0.03], [0.03]),
])
def test_holm_matches_statsmodels(p_values, expected):
    assert _holm_adjusted(p_values) == pytest.approx(expected, abs=1e-12)


def test_reference_pvalues_match_statsmodels_ztest():
    # proportions_ztest(count, nobs, alternative="larger"), statsmodels 0.15.0
    assert two_proportion_pvalue(18, 20, 13, 20) == pytest.approx(0.0291652323, abs=1e-9)
    assert two_proportion_pvalue(19, 20, 15, 20) == pytest.approx(0.0382612502, abs=1e-9)
    assert two_proportion_pvalue(18, 20, 10, 20) == pytest.approx(0.0028877490, abs=1e-9)


def _result(case_id, k, n):
    return CaseResult(case_id, k, n, decide(k, n, 0.8))


@pytest.fixture
def marginal_suite(tmp_path):
    # two marginal drops (p~0.03, p~0.04) + one real drop (p~0.003)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({
        "marginal_a": {"k": 18, "n": 20},
        "marginal_b": {"k": 19, "n": 20},
        "real_drop": {"k": 18, "n": 20},
        "not_run": {"k": 5, "n": 20},
    }))
    results = [
        _result("marginal_a", 13, 20),   # raw p = 0.0292
        _result("marginal_b", 15, 20),   # raw p = 0.0383
        _result("real_drop", 10, 20),    # raw p = 0.0029
    ]
    return results, baseline


def test_holm_suppresses_marginal_flags_but_keeps_real_drop(marginal_suite):
    results, baseline = marginal_suite
    flagged = check_regression(results, baseline, correction="holm")
    # m=3: adjusted p = [0.0583, 0.0583, 0.0087] -> only the real drop fires
    assert [cid for cid, _ in flagged] == ["real_drop"]
    # reported p is the Holm-adjusted value: 3 * 0.0028877490
    assert flagged[0][1] == pytest.approx(3 * 0.0028877490, abs=1e-9)


def test_correction_none_restores_raw_percase_flags(marginal_suite):
    results, baseline = marginal_suite
    flagged = check_regression(results, baseline, correction=None)
    assert {cid for cid, _ in flagged} == {"marginal_a", "marginal_b", "real_drop"}
    # raw p-values are reported untouched
    assert dict(flagged)["marginal_a"] == pytest.approx(0.0291652323, abs=1e-9)


def test_single_case_is_unchanged_by_correction(tmp_path):
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"only": {"k": 19, "n": 20}}))
    results = [_result("only", 15, 20)]
    flagged = check_regression(results, baseline)
    assert len(flagged) == 1
    assert flagged[0][1] == pytest.approx(0.0382612502, abs=1e-9)


@pytest.mark.parametrize("correction", ["bonferroni", "none", "", 1, 0.05])
def test_invalid_correction_rejects_before_baseline_io(correction, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr("sigeval.core.os.path.exists", lambda path: calls.append(path) or False)
    with pytest.raises(ValueError):
        check_regression([], tmp_path / "missing.json", correction=correction)
    assert calls == []
