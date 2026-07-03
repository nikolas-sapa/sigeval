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

## Sample budgeting (cut CI cost)

Fixed-N sampling wastes money — once the interval clears the threshold, extra
samples change nothing. Budgeting samples in batches and stops the moment the
verdict locks:

```python
from sigeval import run_case_budgeted

res = run_case_budgeted("on_topic", scorer, sample=ARTICLE,
                        threshold=0.8, min_samples=8, max_samples=200)
print(res.n)   # often 8-24 instead of 200 for a clearly-good model
```

A clearly-good or clearly-bad model resolves in a handful of samples; only a
model sitting right on the threshold spends the full budget (and returns
`INCONCLUSIVE` — the honest answer).

## LLM-as-judge, no lock-in

```python
from sigeval import make_judge

judge = make_judge(complete_fn, criterion="answer is grounded in the context")
scorer = lambda case: judge(my_llm(case))
```

`complete_fn` is any `callable(prompt) -> str` — OpenAI, Anthropic, Ollama,
vLLM, or a stub in tests. Judge noise flows through the same statistics as
everything else.

## Design

- **Model-agnostic.** A scorer is any `callable(sample) -> bool`. Bring any
  provider, any assertion. No lock-in.
- **Stdlib only.** Wilson score intervals + two-proportion z-test, hand-rolled
  on `math`. No heavy deps.
- **Fits your existing pytest suite.** One function, `assert_eval`. No plugin,
  no config, no dashboard required.

## Status

v0.1.0 — core statistics, regression gate, sample budgeting, LLM-judge helper,
and an optional pytest reporting plugin. Fully tested, stdlib-only. Roadmap:
richer per-case CI reporting, cost dashboards, adaptive batch sizing.

MIT.
