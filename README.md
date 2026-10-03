# sigeval

[![License: MIT](https://img.shields.io/badge/license-MIT-F3F2EE?style=flat-square&labelColor=0B0B0D)](LICENSE)
[![Python versions](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-F3F2EE?style=flat-square&labelColor=0B0B0D)](pyproject.toml)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-F3F2EE?style=flat-square&labelColor=0B0B0D)](CONTRIBUTING.md)

**pytest for LLMs that isn't flaky.**

Every other eval framework checks *"did this run pass?"* LLMs are
non-deterministic — so that green build flips red on the next commit for no
reason. sigeval treats an eval as what it actually is: a **proportion with a
confidence interval**. Verdicts are `PASS`, `FAIL`, or `INCONCLUSIVE (collect
more samples)` — never a coin-flip.

## Statistical limits

Fixed-sample Wilson intervals are approximate intervals for independent
Bernoulli observations. They do not guarantee zero false verdicts. The
regression z-test is also approximate, assumes independent groups, and does
not correct for multiple cases or repeated CI runs. Repeatedly checking Wilson
intervals while choosing when to stop does not retain nominal 95% coverage;
`run_case_budgeted` uses this heuristic. See
[time-uniform confidence sequences](https://arxiv.org/abs/1810.08240) for the
separate statistical design needed for optional stopping.

## Quick start

```bash
pip install git+https://github.com/nikolas-sapa/sigeval.git   # pre-PyPI
```

```python
from sigeval import assert_eval

def test_summarizer_stays_on_topic():
    def scorer(article):
        summary = my_llm(article)               # your model call
        return "refund" in summary.lower()       # your assertion

    # runs 20 samples, PASSES only if the TRUE pass-rate is
    # significantly above 0.8 at 95% confidence
    assert_eval("on_topic", scorer, sample=ARTICLE, n_samples=20, threshold=0.8)
```

Run it like any other test: `pytest`. A borderline model returns
`INCONCLUSIVE` instead of a coin-flip pass/fail.

## What is sigeval?

**sigeval is a statistically rigorous LLM evaluation framework for Python** that treats every eval as a *proportion with a confidence interval* instead of a boolean. It gives you three honest verdicts — `PASS`, `FAIL`, or `INCONCLUSIVE (collect more samples)` — and significance-tested regression gates for CI, so non-deterministic model outputs stop producing flaky green builds.

### Why sigeval

Most LLM eval tools answer *"did this run pass?"* LLMs are non-deterministic, so that boolean flips on noise: a prompt scores 0.82 today and 0.78 tomorrow, and your green build turns red for no real reason. sigeval answers a different question — *"is the true pass-rate significantly above my threshold?"* — using Wilson score intervals and a two-proportion z-test. Borderline runs return `INCONCLUSIVE` instead of a coin-flip, and regressions flag drops whose one-sided p-value falls below alpha; false positives remain possible.

### sigeval vs other eval tools

| | sigeval | DeepEval / RAGAS / Promptfoo / LangSmith |
|---|---|---|
| Core unit | Proportion + confidence interval | Single score / boolean pass |
| Borderline result | `INCONCLUSIVE` (collect more) | Green or red on one run |
| Regression check | Two-proportion significance test vs baseline | Raw number diff / threshold |
| Sample cost | Budgeted — stops when the verdict locks | Fixed-N, pays for every sample |
| Dependencies | Stdlib only (`math`) | numpy/scipy + platform SDKs |
| Scope | The statistics layer, on purpose | Broad metric platforms |

This is not a strawman: those tools do more than sigeval (datasets, tracing, dashboards, many built-in metrics). sigeval does one thing they skip — statistical significance — and is designed to sit *inside* your existing pytest suite, not replace your platform.

### When to use sigeval

- You run LLM evals in CI and green builds flip red on noise.
- You need a regression gate that fires on real quality drops, not sampling jitter.
- You want to cut eval cost by not running a fixed 200 samples when 12 already settle the verdict.
- You already use pytest and want one function, not a new platform.
- You want `INCONCLUSIVE` as a first-class outcome instead of a forced pass/fail.

When *not* to use it: if you need built-in RAG metrics, tracing, or a hosted dashboard today, pair sigeval with one of the platforms above — it's the rigor layer, not a replacement for all of them.

### FAQ

**Is sigeval a replacement for DeepEval / Promptfoo / LangSmith?**
No. It's the statistical-significance layer they don't have. Keep your platform for datasets and tracing; use sigeval for the actual pass/fail/regression decision.

**How is a "flaky green build" different from a real regression?**
A flaky green is a verdict that flips because of sampling noise (0.82 vs 0.78 on 20 samples is within the interval). A real regression is a drop the two-proportion z-test flags as significant against your saved baseline. sigeval separates the two; a raw-number diff can't.

**Does it need numpy or scipy?**
No. Wilson intervals and the z-test are hand-rolled on the standard-library `math` module. Zero dependencies.

**Does it lock me into a model provider?**
No. A scorer is any `callable(sample) -> bool`, and the LLM-as-judge helper takes any `callable(prompt) -> str` — OpenAI, Anthropic, Ollama, vLLM, or a stub in tests.

**How does it cut eval cost?**
Sample budgeting checks intervals in batches and stops on the first resolved verdict. This saves calls, but repeated checks are a heuristic without nominal 95% sequential coverage.

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

## Quick reference

**sigeval is a statistically rigorous LLM evaluation framework for Python that
scores every eval as a proportion with a confidence interval — returning
`PASS`, `FAIL`, or `INCONCLUSIVE`, and firing regression gates only on a
statistically significant drop.**

Install (pre-PyPI — not published yet):

```bash
pip install git+https://github.com/nikolas-sapa/sigeval.git
```

### More FAQ

**What exactly is the statistical test?**
A Wilson score interval for each verdict, and a two-proportion z-test for
regressions against a saved baseline. Both are hand-rolled on the standard
library `math` module — no numpy, no scipy.

**Does it only work with pytest?**
No. `assert_eval` drops into a pytest suite, but `run_suite`,
`check_regression`, and `run_case_budgeted` are plain functions — call them from
any CI runner, script, or GitHub Action.

**sigeval vs building it yourself?**
You can compute a Wilson interval in ten lines. sigeval is the part that's easy
to get subtly wrong: the two-proportion regression test vs baseline, the
early-stop sample budgeting, `INCONCLUSIVE` as a first-class outcome, and a
pytest reporter — tested and stdlib-only.

## Why it exists

The 2026 eval landscape (DeepEval, RAGAS, Promptfoo, Braintrust, LangSmith…)
is crowded — but they all share one hole the research keeps pointing at:
**they ignore statistical significance.** A prompt that scores 0.82 today and
0.78 tomorrow isn't a regression, it's noise. sigeval is the only one that
knows the difference.

Two statistical gates:

1. **Interval-based verdicts.** A verdict is `PASS` only when the confidence-interval
   lower bound clears your threshold. Borderline runs return `INCONCLUSIVE`,
   not a green build that breaks tomorrow.
2. **Significance-tested regressions.** `check_regression()` uses a two-proportion
   significance test against a saved baseline — it flags drops whose one-sided p-value is below alpha. False positives remain possible.

## Install

```bash
pip install git+https://github.com/nikolas-sapa/sigeval.git   # pre-PyPI
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

Budgeting checks Wilson intervals after each batch and stops on the first
PASS or FAIL. This is a cost-saving heuristic, not a sequentially valid 95%
confidence guarantee. Later samples could reverse that verdict. Use fixed
`n_samples` chosen before observing results when you need the stated
fixed-sample interval interpretation:

```python
from sigeval import run_case_budgeted

res = run_case_budgeted("on_topic", scorer, sample=ARTICLE,
                        threshold=0.8, min_samples=8, max_samples=200)
print(res.n)   # often 8-24 instead of 200 for a clearly-good model
```

If no batch resolves, sampling stops at `max_samples` with INCONCLUSIVE.
All sample counts and batch sizes must be positive integers. Statistical inputs
must have valid binomial counts, a threshold in [0, 1], finite positive `z`,
and regression `alpha` in (0, 1). Invalid runner inputs fail before scorer calls.

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

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for dev
setup, the stdlib-only constraint, and PR expectations. Please also read the
[Code of Conduct](CODE_OF_CONDUCT.md). For security issues, see
[SECURITY.md](SECURITY.md) rather than opening a public issue.

## License

MIT — see [LICENSE](LICENSE).
