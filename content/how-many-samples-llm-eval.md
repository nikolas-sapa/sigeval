---
title: "How many samples do you need for a reliable LLM eval?"
author: Nikolas Sapalidis
date: 2026-07-03
updated: 2026-07-03
target_query: "how many samples for an LLM eval"
---

# How many samples do you need for a reliable LLM eval?

**Short answer:** enough that the confidence interval around your pass-rate no longer crosses your threshold. For a model scoring near an 80% bar, that usually means 100–400 samples per test case, not the 5–20 most teams use. Sample size — not the average score — determines whether an eval result is trustworthy.

That's the counterintuitive part, so here's the concrete version.

## The same score, two opposite verdicts

| Result | Pass rate | 95% confidence interval | Verdict at threshold 0.8 |
|---|---|---|---|
| 17 / 20 | 85% | [0.64, 0.95] | **Inconclusive** — interval crosses 0.8 |
| 170 / 200 | 85% | [0.79, 0.89] | **Inconclusive** — still crosses 0.8 |
| 850 / 1000 | 85% | [0.83, 0.87] | **Pass** — significantly above 0.8 |

All three score 85%. Only the last one proves the model is actually above your bar. At n=20, an 85% result is statistically indistinguishable from a 64% model — you simply haven't collected enough evidence to tell them apart.

(These intervals use the Wilson score method, computed with [sigeval](https://github.com/nikolas-sapa/sigeval); you can check your own numbers in the [confidence calculator](https://nikolas-sapa.github.io/sigeval/).)

## Why so many? Because LLMs are non-deterministic

A single eval run samples a probabilistic system once. The same prompt produces different outputs run to run — different phrasing, sometimes different facts — even at temperature zero, because batched GPU inference isn't bitwise-deterministic and providers ship silent model updates. A 2023 study on regression-testing LLM APIs concluded this "requires fundamental changes to traditional testing approaches" and called for explicit statistical treatment of non-determinism ([arXiv:2311.11123](https://arxiv.org/abs/2311.11123)).

So one run is a coin flip. To estimate the true pass-rate, you need repeated samples — and the number you need depends on how close your model sits to the threshold.

## The rule of thumb

- **Far from the threshold** (model ~95%, bar 0.8): ~30–50 samples is plenty. The interval clears the bar fast.
- **Near the threshold** (model ~82%, bar 0.8): you may need 500+ samples, and you might *never* separate them — which is itself the honest answer ("too close to call").
- **Don't fix N blindly.** Sample sequentially and stop once the interval clears. This is called sample budgeting, and it's how you avoid paying for samples you don't need.

## Don't guess N — let the statistics decide

Fixing a flat `n_samples=20` either wastes money on easy cases or under-samples hard ones. The better pattern samples in batches and stops the moment the verdict is statistically locked:

```python
from sigeval import run_case_budgeted

res = run_case_budgeted("on_topic", scorer, sample=x,
                        threshold=0.8, min_samples=8, max_samples=400)
print(res.n)   # ~8 for a clearly-good model, up to 400 for a borderline one
```

## FAQ

**How many samples for an LLM eval?**
Enough that the confidence interval around the pass-rate no longer crosses your threshold — typically 100–400 near an 80% bar, fewer if the model is clearly good or bad. Sample size, not the raw average, determines reliability.

**Is 20 samples enough for an LLM eval?**
Rarely. At n=20, an 85% pass-rate has a 95% confidence interval of roughly [0.64, 0.95] — wide enough that you can't distinguish a good model from a mediocre one. Use 20 only for clearly-good or clearly-bad cases.

**Does temperature zero remove the need for multiple samples?**
No. Even at temperature zero, batched GPU inference is not bitwise-deterministic and provider model updates shift outputs. You still need repeated sampling to estimate the true pass-rate.

**How do I reduce the cost of running many samples?**
Use sample budgeting: sample in small batches and stop as soon as the confidence interval clears (or falls below) your threshold. Clearly-good models resolve in ~8 samples instead of hundreds.

---

*Written by Nikolas Sapalidis, creator of [sigeval](https://github.com/nikolas-sapa/sigeval), an open-source library for statistically rigorous LLM evaluation. Last updated 2026-07-03.*
