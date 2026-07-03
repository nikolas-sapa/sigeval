---
title: "LLM eval regression testing in CI: how to stop chasing phantom regressions"
author: Nikolas Sapalidis
date: 2026-07-03
updated: 2026-07-03
target_query: "LLM eval regression testing CI"
---

# LLM eval regression testing in CI: how to stop chasing phantom regressions

**Short answer:** compare your current eval pass-rate against a saved baseline using a two-proportion significance test, not a raw-number diff. A drop from 95/100 to 92/100 is statistical noise (p ≈ 0.20) and should not fail your build; a drop to 70/100 is a real regression (p < 0.001) and should. Gating CI on raw score differences trains your team to ignore alerts.

## Why raw-diff regression checks fail

LLM outputs are non-deterministic, so eval scores wobble run to run even when nothing changed. If your CI fails whenever the score dips below last week's number, most of those failures are noise. Teams respond the only rational way: they stop trusting the alert and start rubber-stamping merges. The regression gate becomes decorative.

The fix is to only fail when the drop is statistically significant.

## Noise vs. signal, side by side

| Baseline | Current | Raw diff | Two-proportion p-value | Correct action |
|---|---|---|---|---|
| 95 / 100 | 92 / 100 | −3 | 0.20 | **Pass** — noise |
| 95 / 100 | 90 / 100 | −5 | 0.09 | Borderline — sample more |
| 95 / 100 | 70 / 100 | −25 | <0.001 | **Fail** — real regression |

A raw-diff gate fails all three. A significance test fails only the last one. (p-values computed with [sigeval](https://github.com/nikolas-sapa/sigeval).)

## How to set it up

1. **Sample each eval case multiple times** and record `k` passes out of `n`. A single run isn't a measurement.
2. **Save a baseline** once your build is green — store `k/n` per case.
3. **On each PR, run a two-proportion test** comparing current vs. baseline. Fail the build only when p < 0.05.
4. **Refresh the baseline** after an intentional, verified improvement.

```python
from sigeval import run_suite, save_baseline, check_regression

results = run_suite(cases, scorer, n_samples=50, threshold=0.8)
regressions = check_regression(results, "baseline.json")   # [] on first run
assert not regressions, f"significant quality drop: {regressions}"
# after a confirmed improvement, snapshot the new baseline:
save_baseline(results, "baseline.json")
```

This drops into any pytest suite and runs in GitHub Actions like any other test.

## FAQ

**How do you regression-test an LLM in CI?**
Sample each eval case multiple times, save a baseline of pass-rates, and on each change run a two-proportion significance test against that baseline. Fail the build only when the drop is statistically significant (p < 0.05), not on every raw-number dip.

**Why does my LLM eval pass rate change without code changes?**
Because LLMs are non-deterministic — the same prompt yields different outputs run to run, even at temperature zero. Small pass-rate fluctuations are expected noise, which is why regression gates should use significance tests, not exact-match comparisons.

**What's a good threshold for failing an LLM regression test?**
Fail when a two-proportion test shows p < 0.05 that the current pass-rate is below the baseline. This catches real quality drops while ignoring sampling noise.

**Can I use pytest for LLM regression testing?**
Yes. Libraries like sigeval expose the statistics as plain functions that run inside a normal pytest suite and CI pipeline — no separate platform required.

---

*Written by Nikolas Sapalidis, creator of [sigeval](https://github.com/nikolas-sapa/sigeval), an open-source library for statistically rigorous LLM evaluation. Last updated 2026-07-03.*
