# Your LLM evals are lying to you

Here's a test that passes on Monday and fails on Tuesday, with zero code changes in between:

```python
def test_summary_is_on_topic():
    summary = my_llm(article)
    assert judge(summary) == "on_topic"
```

Monday it's green. Tuesday it's red. Wednesday green again. You didn't touch the prompt, the model, or the test. So which day was telling the truth?

Neither. The test was never measuring what you thought it was.

## The bug isn't in your code, it's in the assertion

An LLM is non-deterministic. The same prompt yields different outputs run to run — different phrasing, sometimes different facts. This is true even at temperature zero: batched inference on GPUs isn't bitwise-deterministic, and providers ship silent model updates that shift the output distribution under you.

So when your eval runs once and returns pass/fail, you're not measuring your model's quality. You're sampling a coin once and reading the result as if it were a constant. A prompt that's genuinely 85% on-topic will fail a single-sample eval 15% of the time — and there's no bug to find, because the "bug" is that you asked a probabilistic system a yes/no question.

This isn't a fringe observation. A 2023 research paper on regression-testing LLM APIs put it directly: LLM testing "requires fundamental changes to traditional testing approaches, due to different correctness notions, prompting brittleness, and non-determinism," and it closes by calling for "statistical tests and test minimization strategies" that, as of 2026, no mainstream eval tool ships. ([arxiv.org/abs/2311.11123](https://arxiv.org/abs/2311.11123))

## What every eval tool checks vs. what it should check

The 2026 eval landscape is crowded — DeepEval, RAGAS, Promptfoo, Braintrust, LangSmith, Phoenix, Langfuse. They differ on metrics, dashboards, and hosting. They agree on one thing that's quietly wrong: they answer **"did this run pass?"**

The question that actually keeps your build honest is **"is the true pass-rate above my threshold — and am I sure enough to say so?"**

Those are different questions. The first is a coin flip. The second is statistics.

## The fix is a confidence interval

Treat an eval result as what it is: a proportion. `k` passes out of `n` samples, with uncertainty that shrinks as `n` grows.

- 17/20 passing (85%) against a 0.8 threshold sounds like a pass. Run the numbers: the 95% confidence interval is roughly [0.64, 0.95]. It straddles 0.8. **You cannot call this** — the honest verdict is "collect more samples."
- 850/1000 passing — same 85% rate — gives a CI of about [0.83, 0.87]. Now you're significantly above 0.8. **That's a real pass.**

Same point estimate, opposite conclusions, because sample size is the whole story and single-run evals throw it away.

This is the entire idea behind [sigeval](https://github.com/nikolas-sapa/sigeval), a small library I built after chasing one too many phantom regressions. It runs your eval `n` times and returns one of three verdicts:

- **PASS** — the CI lower bound clears your threshold
- **FAIL** — the CI upper bound is below it
- **INCONCLUSIVE** — the interval straddles it; sample more

```python
from sigeval import assert_eval

def test_summary_is_on_topic():
    def scorer(article):
        return judge(my_llm(article)) == "on_topic"
    # green ONLY if the true rate is significantly above 0.8
    assert_eval("on_topic", scorer, sample=article, n_samples=20, threshold=0.8)
```

A borderline run returns INCONCLUSIVE instead of a green build that flips red on your teammate's laptop.

## Regressions: signal, not noise

The same trap wrecks regression detection. Your baseline scored 95/100. Today's build scores 92/100. Is that a regression?

No. Run a two-proportion test and 95→92 is well within noise (p ≫ 0.05). But 95→70 *is* real (p < 0.05). A raw-number diff fires on both and trains you to ignore the alert. sigeval only flags the drop that clears significance:

```python
from sigeval import check_regression
regressions = check_regression(results, "baseline.json")
assert not regressions, f"significant quality drop: {regressions}"
```

## The part nobody optimizes: cost

Sampling `n` times per case, every commit, costs money — a modest judge-based suite runs $1–3 per CI run and climbs from there. So sigeval samples in batches and stops the moment the verdict is statistically locked. A clearly-good model resolves in ~8 samples instead of 200; only a model sitting right on the threshold spends the full budget. You pay for certainty, not for samples you didn't need.

## Try it

It's stdlib-only (no numpy, no scipy — just Wilson score intervals and a two-proportion z-test), model-agnostic, MIT-licensed, and it drops into any existing pytest suite as one function.

- Repo: [github.com/nikolas-sapa/sigeval](https://github.com/nikolas-sapa/sigeval)
- Interactive calculator (see it on your own numbers, no install): [nikolas-sapa.github.io/sigeval](https://nikolas-sapa.github.io/sigeval/)

If you're gating CI on single-run eval scores today, run your last "flaky" result through the calculator. There's a decent chance the build that flipped on you was never conclusive in the first place.
