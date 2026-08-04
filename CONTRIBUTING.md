# Contributing to sigeval

Thanks for considering a contribution. sigeval is deliberately small and
stdlib-only — keep that in mind before opening a PR.

## Dev setup

```bash
git clone https://github.com/nikolas-sapa/sigeval.git
cd sigeval
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Running tests

```bash
python -m pytest -q
```

CI (`.github/workflows/ci.yml`) runs this same command against Python 3.10,
3.11, and 3.12. Make sure it passes locally before opening a PR.

There is no configured linter or formatter — match the existing code style
(see `sigeval/`) rather than reformatting files wholesale.

## Repo layout

```
sigeval/
  core.py           # assert_eval, run_suite, run_case_budgeted
  stats.py          # Wilson interval, two-proportion z-test
  budget.py         # sample budgeting
  judge.py          # LLM-as-judge helper
  pytest_plugin.py  # optional pytest reporting plugin
tests/              # pytest suite
examples/           # runnable usage examples
```

## The stdlib-only constraint

sigeval has zero runtime dependencies (`dependencies = []` in
`pyproject.toml`), by design — it's meant to drop into any pytest suite
without dragging in numpy/scipy or a platform SDK. A PR that adds a runtime
dependency needs to justify why the stdlib (particularly `math`) can't do the
job. `pytest` (dev-only) is the one exception, and it stays dev-only.

## PR expectations

- Keep changes focused; unrelated cleanup belongs in a separate PR.
- Add or update tests for any behavioral change — see `tests/test_sigeval.py`
  for the existing style.
- Changes to `stats.py` or `budget.py` (the statistical core) need a test that
  demonstrates the math is correct, not just that the function runs.
- Update `README.md` if you change public API behavior.
- Describe *why* the change is needed in the PR description, not just what
  changed.

## Reporting bugs / requesting features

Use the GitHub issue templates. For security issues, see `SECURITY.md`
instead of opening a public issue.
