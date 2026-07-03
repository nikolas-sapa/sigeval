# sigeval

**pytest for LLMs that isn't flaky.**

Every other eval framework checks *"did this run pass?"* LLMs are
non-deterministic — so that green build flips red on the next commit for no
reason. sigeval treats an eval as what it actually is: a **proportion with a
confidence interval**. Verdicts are `PASS`, `FAIL`, or `INCONCLUSIVE (collect
more samples)` — never a coin-flip.

```python
from sigeval import assert_eval

def test_summarizer_stays_on_topic():
    def scorer(article):
        summary = my_llm(article)              # your model call
        return "refund" in summary.lower()      # your assertion

    # runs 20 samples, PASSES only if the TRUE pass-rate is
    # significantly above 0.8 at 95% confidence
    assert_eval("on_topic", scorer, sample=ARTICLE, n_samples=20, threshold=0.8)
```

## Why it exists

The 2026 eval landscape (DeepEval, RAGAS, Promptfoo, Braintrust, LangSmith…)
is crowded — but they all share one hole the research keeps pointing at:
**they ignore statistical significance.** A prompt that scores 0.82 today and
0.78 tomorrow isn't a regression, it's noise. sigeval is the only one that
knows the difference.

Two guarantees:

1. **No flaky greens.** A verdict is `PASS` only when the confidence-interval
   lower bound clears your threshold. Borderline runs return `INCONCLUSIVE`,
   not a green build that breaks tomorrow.
2. **No noise regressions.** `check_regression()` uses a two-proportion
   significance test against a saved baseline — it fires on a *real* quality
   drop, not on sampling jitter.

## Install

```bash
pip install sigeval   # stdlib-only, no numpy/scipy
```

## Regression gate in CI

```python
from sigeval import run_suite, save_baseline, check_regression

results = run_suite(cases, scorer, n_samples=20, threshold=0.8)
regressions = check_regression(results, "baseline.json")   # [] on first run
assert not regressions, f"significant quality drop: {regressions}"
# once green, snapshot the new baseline:
save_baseline(results, "baseline.json")
```

## Design

- **Model-agnostic.** A scorer is any `callable(sample) -> bool`. Bring any
  provider, any assertion. No lock-in.
- **Stdlib only.** Wilson score intervals + two-proportion z-test, hand-rolled
  on `math`. No heavy deps.
- **Fits your existing pytest suite.** One function, `assert_eval`. No plugin,
  no config, no dashboard required.

## Status

v0.0.1 — core statistics + regression gate, fully tested. Roadmap: sample
budgeting (stop early once a verdict is statistically locked, to cut CI cost),
pytest plugin with per-case reporting, LLM-judge scorer helpers.

MIT.
