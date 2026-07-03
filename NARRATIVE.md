# sigeval — application narrative (DRAFT)

> This is scaffolding for a YC / a16z Speedrun application, not the repo's
> public README. It exists so the *project* reads as a company, not a portfolio
> piece. Every bracketed field is something only you can fill — do not ship with
> them blank.

## One-liner
sigeval makes LLM evals statistically rigorous, so teams stop shipping on flaky
green builds. It's the missing "statistical significance" layer under the
crowded eval-tooling stack.

## The problem (why now)
Every team building on LLMs runs evals in CI. But LLMs are non-deterministic —
the same prompt scores 0.82 today and 0.78 tomorrow. Existing tools (DeepEval,
RAGAS, Promptfoo, LangSmith) check *"did this run pass?"*, so builds flake:
green flips red on noise, and real regressions hide inside sampling jitter. The
2026 research literature explicitly names statistical rigor and eval cost as the
two open problems — and no shipping tool solves either. Promptfoo's acquisition
by OpenAI (Mar 2026) proves the category matters; the consolidation leaves the
rigor layer unowned.

## The wedge
Treat every eval as a *proportion with a confidence interval*, not a boolean.
- Verdicts are PASS / FAIL / **INCONCLUSIVE** — never a coin flip.
- Regressions fire on a two-proportion significance test vs a saved baseline,
  not a raw-number diff.
- Sample budgeting stops sampling once the verdict locks — directly attacking
  the CI-cost problem.

Narrow on purpose. We are not a 50-metric platform; we are the one thing the
platforms get wrong.

## Why us
[Your builder story — what you've shipped, why you see this pain, your unfair
speed. THIS IS THE REAL EVALUATION CRITERION. Fill it honestly.]

## Traction
[The load-bearing gap. As of now: none. Before you submit you need at least:
GitHub stars, N repos using it in CI, 3 named users who'd give a quote. Do NOT
submit with this section empty — the repo alone is not traction.]

## Business
Open-source core → hosted layer for teams (shared dashboards, historical
baselines, cost analytics across runs). Same wedge the incumbents monetized,
entered from the rigor angle they skipped. [Validate willingness-to-pay with 5
design-partner conversations before claiming this.]

## The ask / 14-week plan
- Wks 1-6: core + budgeting shipped. **[DONE — v0.1.0]**
- Wks 7-10: launch (Show HN, r/LocalLLaMA, X build-in-public), convert usage to
  issues/PRs, land 3 design partners.
- Wks 11-14: hosted MVP, first paying/retained users, apply with real traction.

## Honest risks (state them; investors respect it)
1. Adoption is unproven — a great wedge ≠ distribution. Mitigation: build-in-
   public loop, not a single launch spike.
2. An incumbent could add statistical verdicts in a sprint. Mitigation: move
   first, own the "not flaky" positioning, go deep on cost/budgeting they won't
   prioritize.
3. Solo founder in a crowded space. Mitigation: [your answer].
